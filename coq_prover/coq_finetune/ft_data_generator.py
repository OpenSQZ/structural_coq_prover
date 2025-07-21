import os
import json
import asyncio
from utils import get_config, read_jsonl_file
from data_extraction.coq_data.Ps_class import PSItem, ps_object_single
from typing import List, Tuple, Dict
from coq_prover.coq_context.retrieval import Retrieval
from coq_prover.coq_context.prompt_gen import PromptGenerator
from coq_prover.coq_context.llm_method import llm_state_explanation
from coq_prover.coq_context.state_explanation import StateExplanation
from tqdm import tqdm

class FTDataGenerator:
    def __init__(self, config_path):
        self.config_path = config_path
        config = get_config(config_path)
        self.def_table_path = config.paths.def_table_path
        self.def_table = read_jsonl_file(self.def_table_path)
        self.def_table_dict = {item['name']: item for item in self.def_table}
        self.ps_table_path = config.paths.ps_table_path
        self.ps_table = read_jsonl_file(self.ps_table_path)
        self.ps_table_dict = {item['name']: item for item in self.ps_table}
        
        self.tokenizer_path = config.paths.tokenizer_path
        self.emb_model_path = config.paths.emb_model_path
        self.emb_data_path = config.paths.emb_data_path
        self.state_explanation_model_path = config.paths.state_explanation_model_path
        self.ft_data_dir = config.paths.ft_data_dir
        if not os.path.exists(self.ft_data_dir):
            os.makedirs(self.ft_data_dir)
        self.detailed_log_path = os.path.join(self.ft_data_dir, 'detailed_log_reorganize.jsonl')
        self.prompt_tactic_path = os.path.join(self.ft_data_dir, 'prompt_tactic_reorganize.jsonl')

        self.if_background = config.flags.if_background
        self.if_use_intuition = config.flags.if_use_intuition
        self.simplify_ps = config.flags.simplify_ps
        self.plain_prompt = config.flags.plain_prompt
        self.state_encode = config.flags.state_encode
        self.use_origin = config.flags.use_origin
        self.if_explanation = config.flags.if_explanation
        self.reconsider_mode = config.flags.reconsider_mode
        self.use_api = config.flags.use_api

        self.concept_num = config.params.concept_num
        self.blind_num = config.params.blind_num
        self.beam_width = config.params.beam_width
        self.max_depth = config.params.max_depth
        self.max_states_workers = config.params.max_states_workers
        self.max_theorems_workers = config.params.max_theorems_workers
        self.model_use = config.params.model_use
        self.max_attempts = config.params.max_attempts
        self.max_retries = config.params.max_retries

        self.prompt_generator = self.init_prompt_generator()
        if self.if_explanation:
            if self.use_api:
                pass
                # self.state_explanation = StateExplanation(self.state_explanation_model_path)
            else:
                self.state_explanation = StateExplanation(model_name=self.state_explanation_model_path)

    def resume_from_log(self):
        processed_theorems = {}
        with open(self.detailed_log_path, 'r') as f:
            for line in tqdm(f, desc='Resuming from log'):
                log_entry = json.loads(line)
                if log_entry['theorem_name'] not in processed_theorems:
                    assert log_entry['position'].split('/')[0] == '1'
                    processed_theorems[log_entry['theorem_name']] = (log_entry, 1) 
                else:
                    _, count = processed_theorems[log_entry['theorem_name']]
                    assert log_entry['position'].split('/')[0] == str(count + 1)
                    processed_theorems[log_entry['theorem_name']] = (log_entry, count + 1)
        print(f"Resumed from log: {len(processed_theorems)} theorems")
        return processed_theorems

    def get_all_theorem_tuples(self) -> List[Tuple[str, str, PSItem]]:
        theorem_list = []
        count = 0
        for item in tqdm(self.def_table, desc='Loading theorems', total=len(self.def_table)):
            if item['kind'] == 'Proof':
                try:
                    ps_item = self.ps_table_dict[item['name']]
                    theorem_list.append((item['name'], item['file_path'], PSItem.from_dict(ps_item), 0, {}))
                except KeyError as e:
                    # print(f"Error: {item['name']} not found in ps_table")
                    count += 1
                except Exception as e:
                    raise e
        print(f"Error: {count} theorems not found in ps_table")
        if len(theorem_list) == 0:
            raise ValueError("No theorems found")
        print(f"Total: {len(theorem_list)} theorems found need to be processed")
        return theorem_list    
    
    def get_theorems_to_process(self, theorem_list: List[Tuple[str, str, PSItem, int, Dict]], processed_theorems: Dict[str, Tuple[Dict, int]]):
        theorems_to_process = []
        for theorem in tqdm(theorem_list, desc='Getting theorems to process'):
            if theorem[0] not in processed_theorems:
                theorems_to_process.append(theorem)
            else:
                previous_result, count = processed_theorems[theorem[0]]
                total_steps = theorem[2].get_proof_steps()
                
                assert count <= total_steps
                
                if count == total_steps:
                    continue
                else:
                    theorems_to_process.append((theorem[0], theorem[1], theorem[2], count, previous_result))
        print(f"Theorems to process: {len(theorems_to_process)}")
        return theorems_to_process

    async def generate_data(self):
        parallel_theorems = 500
        
        theorem_list = self.get_all_theorem_tuples()

        if os.path.exists(self.detailed_log_path):
            processed_theorems = self.resume_from_log()
            if processed_theorems:
                theorem_list = self.get_theorems_to_process(theorem_list, processed_theorems)
        
        pbar = tqdm(total=len(theorem_list), desc='Generating data')
        
        semaphore = asyncio.Semaphore(parallel_theorems)
        
        async def process_theorem_with_semaphore(theorem_tuple):
            name, path, ps_item, processed_steps, previous_result = theorem_tuple
            async with semaphore:
                await self.generate_data_single_theorem(name, path, ps_item, processed_steps, previous_result)
                pbar.update(1)
        
        tasks = [process_theorem_with_semaphore(theorem) for theorem in theorem_list]
        await asyncio.gather(*tasks)
        
        pbar.close()

    async def generate_data_single_theorem(self, theorem_name: str, theorem_path: str, ps_item: PSItem, processed_steps: int, previous_result: Dict):
        assert len(ps_item.Tactic_sequence) == len(ps_item.Content.ProofStates)
        
        if processed_steps != 0 and not previous_result:
            raise ValueError(f"Previous result is not found for theorem {theorem_name}, processed_steps: {processed_steps}")
        
        if processed_steps > 0:
            proof_summary_with_tactic = (previous_result['proof_summary'], previous_result['state']['states'][0]['tactic']['name'])
            public_notes = previous_result['public_notes']
            proof_traces = previous_result['proof_traces']
        else:
            proof_summary_with_tactic = ({'proof_trace': 'Initial state', 'steps': 0, 'score': 0}, ['Initial state'])
            public_notes = []
            proof_traces = []

        if len(ps_item.Content.ProofStates) == 0:
            print(f"Theorem {theorem_name} has no proof states")

        for state in ps_item.Content.ProofStates[processed_steps:]:
            proof_summary_with_tactic, public_notes, proof_traces = await self.generate_data_single_state(theorem_name, theorem_path, state, proof_summary_with_tactic, public_notes, proof_traces)
    
    async def generate_data_single_state(self, theorem_name: str, theorem_path: str, state: ps_object_single, proof_summary_with_tactic: List[Tuple[str, str]], public_notes: List[Tuple[str, str]], proof_traces: List[Tuple[str, str, str]]):
        info = await self.prompt_generator.generate(
                state.Before_state,
                ifInit=False,
                if_single_after_state=True,
                ft_mode=True,
                actual_tactic=state.Tactic.Name,
                if_reorganize_concisely=True,
                current_def=theorem_name,
                proof_summary_with_tactic=proof_summary_with_tactic,
                public_notes=public_notes,
                if_background=self.if_background,
                simplify_ps=self.simplify_ps,
                plain_prompt=self.plain_prompt,
                state_encode=self.state_encode,
                if_use_intuition=self.if_use_intuition,
                concept_num=self.concept_num,
                blind_num=self.blind_num,
                use_origin=self.use_origin
            )
        
        prompt_method, response_method, (prompt_tactic, prompt_tactic_reorganize), brief_strategy = info
        
        tactic = state.Tactic.Name
        position = state.position

        explanation, explain_prompt = await self.generate_explanation(state)
        proof_traces.append((brief_strategy, explanation, tactic))
        proof_summary, proof_summary_prompt = await self.generate_proof_summary(proof_traces)
        history_tactic = [tactic for _, _, tactic in proof_traces]
        proof_summary_with_tactic = (proof_summary, history_tactic + [tactic])
        public_notes.append(self.generate_note_book(explanation, brief_strategy, tactic))
        log_entries = self.get_log_entry(theorem_name,
                            theorem_path, 
                            position, 
                            tactic,
                            state, 
                            prompt_method,
                            response_method, 
                            prompt_tactic, 
                            prompt_tactic_reorganize,
                            explain_prompt, 
                            explanation, 
                            proof_summary_prompt, 
                            proof_summary, 
                            public_notes, 
                            proof_traces)
        self.write_to_log_dict(log_entries)

        return proof_summary_with_tactic, public_notes, proof_traces

    def generate_note_book(self, explanation: Dict, brief_strategy: str, tactic: str):
        explanation_str = '\n'.join([f"{key}: {value['en']}" for key, value in explanation.items()])
        method_str = f"## Strategy: {brief_strategy}\n## Tactic: {tactic}\n## Intuition:\n{explanation_str}\n"
        method_info_str = f"strategy_method_info: tactic:{tactic}"
        return method_str, method_info_str
    
    async def generate_explanation(self, state: ps_object_single):
        explain_prompt = [self.prompt_generator.generate_state_explanation(state, if_use_intuition=self.if_use_intuition, use_origin=self.use_origin)]
        if self.use_api:
            explanation = await llm_state_explanation(explain_prompt, mode='state')
        else:
            explanation = await self.state_explanation.generate(explain_prompt, mode='state')
        return explanation[0],explain_prompt
    
    async def generate_proof_summary(self, proof_traces: List[Tuple[str, str, str]]):
        proof_summary_prompt = self.prompt_generator.generate_proof_trace(proof_traces)
        if self.use_api:
            proof_summary = await llm_state_explanation(proof_summary_prompt, mode='strategy')
        else:
            proof_summary = await self.state_explanation.generate(proof_summary_prompt, mode='strategy')
        return proof_summary[0], proof_summary_prompt

    def init_prompt_generator(self):
        retrieval = Retrieval(
            emb_file=self.emb_data_path,
            model_name=self.emb_model_path
        )
        return PromptGenerator(
            def_path=self.def_table_path,
            tokenizer_path=self.tokenizer_path,
            retrieval=retrieval
        )
    
    def get_log_entry(self, theorem_name: str, 
                      theorem_path: str, 
                      position: int,
                      tactic: str,
                      state: ps_object_single,
                      prompt_method: str,
                      response_method: str,
                      prompt_tactic: str,
                      prompt_tactic_reorganize: str,
                      explain_prompt: str,
                      explanation: Dict,
                      proof_summary_prompt: str,
                      proof_summary: str,
                      public_notes: List[Tuple[str, str]], 
                      proof_traces: List[Tuple[str, str, str]]):
        
        detailed_log_entry = {
            'theorem_name': theorem_name,
            'theorem_path': theorem_path,
            'position': position,
            'state': state.to_dict(),
            'prompt_method': prompt_method,
            'response_method': response_method,
            'prompt_tactic': prompt_tactic,
            'prompt_tactic_reorganize': prompt_tactic_reorganize,
            'explain_prompt': explain_prompt,
            'explanation': explanation,
            'proof_summary_prompt': proof_summary_prompt,
            'proof_summary': proof_summary,
            'public_notes': public_notes,
            'proof_traces': proof_traces
        }
        prompt_tactic_pair = {
            'theorem_name': theorem_name,
            'theorem_path': theorem_path,
            'position': position,
            'prompt': prompt_tactic,
            'prompt_reorganize': prompt_tactic_reorganize,
            'tactic': tactic
        }
        return detailed_log_entry, prompt_tactic_pair
    
    def write_to_log_dict(self, log_entries):
        detailed_log_entry, prompt_tactic_pair = log_entries
        with open(self.detailed_log_path, 'a') as f:
            json.dump(detailed_log_entry, f, ensure_ascii=False)
            f.write('\n')
        with open(self.prompt_tactic_path, 'a') as f:
            json.dump(prompt_tactic_pair, f, ensure_ascii=False)
            f.write('\n')