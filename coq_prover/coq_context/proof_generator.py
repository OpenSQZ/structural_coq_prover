import os
import asyncio
import re
from utils import get_config, read_jsonl_file
import json
import copy
import time
import random
from coq_prover.coq_context.proof_data_class import *
from coq_prover.coq_context.prompt_gen import PromptGenerator
from coq_prover.coq_context.retrieval import Retrieval
from data_extraction.coq_tokenize.tokenizer import Tokenizer
from coq_prover.coq_context.run_tactic import TacticRunner
from typing import Dict, List, Tuple, Any, Union, Optional
from coqc import Coqc
from data_extraction.coq_data.Parser import Parser
from coq_prover.coq_context.run_tactic import get_base_file_content
from coq_prover.coq_context.llm_method import llm_state_explanation
import coq_prover.coq_context.llm_method as llm_method
from data_extraction.coq_data.Ps_class import PSItem, ps_object_single, State
import warnings

class ProofGenerator:
    def __init__(self, config_path, generate_mode=False):
        random.seed(42)
        self._load_config(config_path)
        self.generate_mode = generate_mode
        self._init_components()
    
    def _load_config(self, config_path):
        config = get_config(config_path)
        self.config_path = config_path
        self.coqc_path = config.paths.coqc_path
        self.proof_log_dir = config.paths.proof_log_dir
        if not os.path.exists(self.proof_log_dir):
            os.makedirs(self.proof_log_dir)
        self.system_log_dir = config.paths.system_log_dir
        if not os.path.exists(self.system_log_dir):
            os.makedirs(self.system_log_dir)
        self.def_table_path = config.paths.def_table_path

        self.tokenizer_path = config.paths.tokenizer_path
        self.emb_model_path = config.paths.emb_model_path
        self.emb_data_path = config.paths.emb_data_path
        self.path_prefix = config.paths.data_dir
        self.state_explanation_model_path = config.paths.state_explanation_model_path

        self.ft_model_path = config.paths.ft_model_path
        if self.ft_model_path is not None:
            self.ft_model = self.ft_model_path.split('/')[-1]
        else:
            self.ft_model = None

        self.if_background = config.flags.if_background
        self.if_use_intuition = config.flags.if_use_intuition
        self.simplify_ps = config.flags.simplify_ps
        self.plain_prompt = config.flags.plain_prompt
        self.state_encode = config.flags.state_encode
        self.use_origin = config.flags.use_origin
        self.if_explanation = config.flags.if_explanation
        self.reconsider_mode = config.flags.reconsider_mode
        self.use_api = config.flags.use_api
        if not (self.reconsider_mode == "hierarchical" or self.reconsider_mode == "normal" or self.reconsider_mode == "disabled"):
            raise ValueError(f"reconsider_mode must be 'hierarchical' or 'normal' or 'disabled', but got {self.reconsider_mode}")
        self.use_ft_model = config.flags.use_ft_model
        if self.use_ft_model:
            warnings.warn('now ft_model normal reconsider mode will give another method to generate, hierarchical mode will refine by the ft model itself')
            # warnings.warn('use ft model can not support reconsider_mode so far, setting reconsider_mode to "disabled"')
            # self.reconsider_mode = "disabled"
        self.if_reorganize_concisely = config.flags.if_reorganize_concisely
        self.resume_mode = config.flags.resume_mode
        self.new_theorem_mode = config.flags.new_theorem_mode
        self.ablation_mode = config.flags.ablation_mode
        
        if self.ablation_mode:
            self.if_strategy = config.flags.if_strategy
            self.if_def = config.flags.if_def
            self.if_retrieve = config.flags.if_retrieve
            self.if_proof_trace = config.flags.if_proof_trace
            self.if_public_notes = config.flags.if_public_notes
            self.ablation_proof_mode = config.flags.ablation_proof_mode
            self.ablation_scope = config.flags.ablation_scope
            ablation_params = self.init_ablation_params()
            self.ablation_config = (self.if_def, self.if_retrieve, self.if_proof_trace, self.if_public_notes, self.if_strategy, ablation_params)

        if self.ft_model_path is not None:
            warnings.warn('if ft model is used, if_reorganize_concisely will be set by the model name')
            if 'reorganize' in self.ft_model_path:
                self.if_reorganize_concisely = True
            else:
                self.if_reorganize_concisely = False

        self.concept_num = config.params.concept_num
        self.blind_num = config.params.blind_num
        self.beam_width = config.params.beam_width
        self.max_depth = config.params.max_depth
        self.max_states_workers = config.params.max_states_workers
        self.max_theorems_workers = config.params.max_theorems_workers
        self.model_use = config.params.model_use

        self.max_attempts = config.params.max_attempts
        self.max_retries = config.params.max_retries
        self.theorem_parallel_num = config.params.theorem_parallel_num
        self.max_coqc_workers = config.params.max_coqc_workers
        
    def _init_components(self):
        self.def_table = read_jsonl_file(self.def_table_path)
        self.def_table_dict = {item['name']: item for item in self.def_table}

        if self.use_ft_model:
            self.init_ft_llm()
            
        self.coqc = Coqc(config_path=self.config_path, mode="proof", new_theorem_mode=self.new_theorem_mode, max_coqc_workers=self.max_coqc_workers)
        self.parser = Parser()
        self.tokenizer = Tokenizer(self.tokenizer_path)
        self.prompt_generator = self.init_prompt_generator()
        if self.if_explanation:
            if self.use_api:
                pass
                # self.state_explanation = StateExplanation(self.state_explanation_model_path)
            else:
                from coq_prover.coq_context.state_explanation import StateExplanation
                self.state_explanation = StateExplanation(model_name=self.state_explanation_model_path)
        
        if self.generate_mode:
            if not self.plain_prompt:
                self.public_notes = []
            self.tactic_runner = TacticRunner(self.coqc, self.config_path, self.tokenizer, self.new_theorem_mode, self.def_table_dict, log_prefix=self.proof_history_dir)

    async def generate_step(self, theorem_name: str, theorem_file_path: str, tactic_sequence: List[str], 
                      proof_traces: List[Tuple] = None, proof_summary: Dict = None, 
                      public_notes: List[Tuple[str,str]] = None, max_retries: Optional[int] = None):
        if max_retries is not None:
            original_max_retries = self.max_retries
            self.max_retries = max_retries
        
        if not theorem_file_path.startswith(self.path_prefix):
            theorem_file_path = os.path.join(self.path_prefix, theorem_file_path)

        if public_notes is None:
            public_notes = []
        
        prefix = self.parser.get_file_prefix(theorem_file_path)
        if '.' not in theorem_name:
            theorem_name = prefix + '.' + theorem_name
        
        self.tactic_runner.init_base_file_content(theorem_file_path, theorem_name)
        
        base_file_content = self.tactic_runner.base_file_content
        if not base_file_content:
            raise ValueError(f"Failed to get base file content for theorem {theorem_name}")
        
        if len(tactic_sequence) == 0:
            content = base_file_content
            init_ps = True
        else:
            tactic_str = '.\n'.join(tactic_sequence) + '.'
            content = base_file_content + '\n' + tactic_str
            init_ps = False
        
        output, error, temp_file = await self.coqc.run(theorem_file_path, content, init_ps=init_ps)
        
        if "Timeout error" in error:
            output, error, temp_file = await self.coqc.run(theorem_file_path, content, init_ps=init_ps, timeout=1500)
            if "Timeout error" in error:
                raise ValueError(f"Timeout error when running Coq for theorem {theorem_name}")
            
        if not init_ps and not ('Attempt to save an incomplete proof' in error or 'There are pending proofs' in error):
            raise ValueError(f"Unexpected error when running Coq for theorem {theorem_name}: {error}")
        
        if init_ps:
            ps, actual_name, _, type_dict = self.parser.parse_proof(output, theorem_file_path, theorem_name, use_tqdm=False)
            ps_item = self.tokenizer.process_ps_proof(ps, def_table=self.def_table_dict, type_dict=type_dict, 
                                                     actual_name=actual_name, txt_file_path=temp_file, if_refined_ps=False)
        else:
            ps, actual_name, _, type_dict = self.parser.parse_proof(output, theorem_file_path, theorem_name, use_tqdm=False)
            ps_item = self.tokenizer.process_ps_proof(ps, def_table=self.def_table_dict, type_dict=type_dict,
                                                        actual_name=actual_name, txt_file_path=temp_file, if_refined_ps=False)
        
        ps_init = copy.deepcopy(ps_item)
        ps_init.Content.ProofStates = ps_init.Content.ProofStates[0]
        proof_context = ProofContext(
            theorem_name=theorem_name,
            theorem_path=theorem_file_path,
            ps_init=ps_init,
            public_notes=public_notes,
            depth=len(tactic_sequence),
            log_file=self.proof_log_dir + f'/generation_mode/{theorem_name}_depth_{len(tactic_sequence)}.jsonl'
        )

        if proof_traces is None:
            warnings.warn(f"proof_traces is None, generate dummy proof_traces and proof_summary")
            if len(tactic_sequence) == 0:
                proof_summary_with_tactic = []
            else:
                proof_traces = []
                for i, tactic in enumerate(tactic_sequence):
                    dummy_dict = {
                                'before': {'en': ''},
                                'after': {'en': ''},
                                'tactic': {'en': ''}
                    }
                    proof_traces.append(ProofTrace(brief_strategy='intermediate_state', state_explanation=dummy_dict, tactic=tactic))
                
                if proof_summary is None:
                    prompt_proof_summary = self.prompt_generator.generate_proof_trace(proof_traces)
                    if self.use_api:
                        proof_summary_result = await llm_state_explanation(prompt_proof_summary, mode='strategy')
                    else:
                        proof_summary_result = await self.state_explanation.generate(prompt_proof_summary, mode='strategy')
                    assert len(proof_summary_result) == 1
                    proof_summary = proof_summary_result[0]
                
                proof_summary_with_tactic = ProofSummaryWithTactic(proof_summary=proof_summary, tactic=tactic_sequence)
        else:
            if proof_summary is not None:
                proof_summary_with_tactic = ProofSummaryWithTactic(proof_summary=proof_summary, tactic=tactic_sequence)
            else:
                prompt_proof_summary = self.prompt_generator.generate_proof_trace(proof_traces)
                if self.use_api:
                    proof_summary_result = await llm_state_explanation(prompt_proof_summary, mode='strategy')
                else:
                    proof_summary_result = await self.state_explanation.generate(prompt_proof_summary, mode='strategy')
                assert len(proof_summary_result) == 1
                proof_summary = proof_summary_result[0]
                proof_summary_with_tactic = ProofSummaryWithTactic(proof_summary=proof_summary, tactic=tactic_sequence)
        
        current_state, idx_list = ps_item.get_proof_states_normal()
        curr_path = '-'.join([f'{layer[0]}/{layer[1]}' for layer in idx_list[2]])
        proof_info = ProofInfo(curr_ps=current_state if len(tactic_sequence) != 0 else ps_item, 
                               prev_result=PreviousResult(ps_item=ps_item, tactic_sequence=tactic_sequence), 
                               curr_path=curr_path, 
                               proof_traces=proof_traces, 
                               proof_summary_with_tactic=proof_summary_with_tactic)
        
        proof_info_layer = ProofInfoLayer(
            proof_context=proof_context,
            proof_infos=[proof_info]
        )
        
        result = await self.proof_generate(
            proof_info_layer,
            tactic_runner=self.tactic_runner,
            is_init=(len(tactic_sequence) == 0),
            if_single_after_state=(len(tactic_sequence) != 0)
        )
        
        if max_retries is not None:
            self.max_retries = original_max_retries
        
        return result

    async def run_proof_generation(self, package_name: str = None, theorem_name: str = None, theorem_file_path: str = None, ratio: float = 0.1, total_shards: int = None, shard: int = None):
        start_proof_time = time.strftime("%Y-%m-%d-%H", time.localtime())
        self.init_log_file(package_name=package_name, theorem_name=theorem_name)
        
        log_file_prefix = f'{self.proof_log_dir}/{package_name}/{start_proof_time}/' if not theorem_name else f'{self.proof_log_dir}/theorems/{start_proof_time}/'
        if not os.path.exists(log_file_prefix):
            os.makedirs(log_file_prefix)

        theorem_list = self.collect_theorem_func(package_name, theorem_name, theorem_file_path, ratio)
        if self.ablation_mode:
            if len(theorem_list) > 100:
                theorem_list = random.sample(theorem_list, 100)

        if self.resume_mode:
            print('enter resume mode')
            theorem_list = self.resume_theorem_list(theorem_list)
            if self.resume_mode == 'failed':
                self.proof_history_log = f'{self.proof_history_dir}/failed_proof.log'
        
        if total_shards is not None and shard is not None:
            if (total_shards is not None and shard is None) or (total_shards is None and shard is not None):
                raise ValueError('total_shards and shard must be provided at the same time')
            random.shuffle(theorem_list)
            chunk_size = len(theorem_list) // total_shards
            start_idx = shard * chunk_size
            end_idx = start_idx + chunk_size if shard < total_shards - 1 else len(theorem_list)
            theorem_list = theorem_list[start_idx:end_idx]
        
        print(f'theorems need to be proved: {len(theorem_list)}')

        # theorem_list = theorem_list[::-1]
        
        semaphore = asyncio.Semaphore(self.theorem_parallel_num)

        async def prove_with_semaphore(theorem):
            async with semaphore:
                return await self.prove_single_theorem(theorem, log_file_prefix)

        await asyncio.gather(*[prove_with_semaphore(theorem) for theorem in theorem_list])
    
    async def prove_single_theorem(self, theorem: Dict, log_file_prefix: str):
        init_ps, theorem_path, theorem_name = await self.init_ps_func(theorem)
        if init_ps is None:
            return
        proof_info_layer, tactic_runner = self.proof_initial(init_ps, theorem_path, theorem_name)

        for attempt in range(1, self.max_attempts + 1):
            start_time = time.time()
            print(f"Proofing theorem {theorem_name} at attempt {attempt}")
            proof_info_layer.proof_context.log_file = f'{log_file_prefix}/{theorem_name}_attempt_{attempt}.jsonl'

            result = await self.proof_generate(
                proof_info_layer,
                tactic_runner=tactic_runner,
                is_init=True,
                if_single_after_state=False
            )
        
            if isinstance(result,ProofContext):
                if result.depth >= self.max_depth:
                    print(f'FAILED: {theorem_name} was not proved successfully reached max depth')
                    result_message = f"""FAILED: {theorem_name} was not proved successfully reached max depth\nlog_file: {proof_info_layer.proof_context.log_file}\nproof_time: {time.time() - start_time:.3f} seconds\ndepth: {result.depth}\n\n"""
                else:
                    print(f'FAILED: {theorem_name} was not proved successfully no states available')
                    result_message = f"""FAILED: {theorem_name} was not proved successfully no states available in {result.depth} layer\nlog_file: {proof_info_layer.proof_context.log_file}\nproof_time: {time.time() - start_time:.3f} seconds\ndepth: {result.depth}\n\n"""
            else:
                result_message = f"""SUCCEEDED: {theorem_name} was proved successfully\nlog_file: {proof_info_layer.proof_context.log_file}\nproof_time: {time.time() - start_time:.3f} seconds\ndepth: {proof_info_layer.proof_context.depth}\n\n"""
                print(f'SUCCEEDED: {theorem_name} was proved successfully')
                with open(self.proof_history_log, 'a') as f:
                    f.write(result_message)
                return True
            
            with open(self.proof_history_log, 'a') as f:
                f.write(result_message)
        return False
                    
    
    async def proof_generate(self, proof_info_layer: ProofInfoLayer, tactic_runner: TacticRunner, is_init: bool = False, if_single_after_state: bool = True):
        if proof_info_layer.proof_context.depth > self.max_depth:
            return proof_info_layer.proof_context

        tasks = [
            self.process_single_proof_info(proof_info, proof_info_layer.proof_context, tactic_runner, is_init, if_single_after_state)
            for proof_info in proof_info_layer.proof_infos
        ]

        next_layer = []
        pending_tasks = {asyncio.create_task(task) for task in tasks}

        for future in asyncio.as_completed(pending_tasks):
            proof_infos = await future
            if proof_infos is None:
                continue
            for proof_info in proof_infos:
                if proof_info.is_finished:
                    for task in pending_tasks:
                        if not task.done():
                            task.cancel()
                    return proof_info
                
                next_layer.append(proof_info)

        if self.generate_mode:
            return next_layer
        
        if not next_layer:
            return proof_info_layer.proof_context
        
        ## now retrieval info is not used for update public notes, set None. 
        ## if need, get from generate_info
        if self.plain_prompt == True:
            pass
        elif self.ablation_mode and not getattr(self, 'if_public_notes', True):
            pass
        else:
            retrieval_info = None
            next_layer_public_notes = await self.update_public_notes(proof_info_layer.proof_context, next_layer, retrieval_info)
            proof_info_layer.proof_context.public_notes = next_layer_public_notes
        proof_info_layer.proof_context.depth += 1
        
        if not self.plain_prompt:
            selected_next_layer = await self.proof_info_selection(proof_info_layer.proof_context, next_layer)
            proof_info_layer.proof_infos = selected_next_layer
        else:
            proof_info_layer.proof_infos = random.sample(next_layer, min(len(next_layer), self.beam_width))
        return await self.proof_generate(proof_info_layer, tactic_runner)

    async def process_single_proof_info(
        self,
        proof_info: ProofInfo,
        proof_context: ProofContext,
        tactic_runner: TacticRunner,
        is_init: bool = False,
        if_single_after_state: bool = True,
    ):
        theorem_name = tactic_runner.current_name
        generate_info = await self.prompt_generator.generate(proof_info, 
                                                    is_init=is_init,
                                                    if_single_after_state=if_single_after_state,
                                                    public_notes=proof_context.public_notes,
                                                    current_def=theorem_name,
                                                    if_use_ft_model=self.use_ft_model,
                                                    if_reorganize_concisely=self.if_reorganize_concisely,
                                                    if_background=self.if_background,
                                                    simplify_ps=self.simplify_ps,
                                                    plain_prompt=self.plain_prompt,
                                                    state_encode=self.state_encode,
                                                    if_use_intuition=self.if_use_intuition,
                                                    concept_num=self.concept_num,
                                                    blind_num=self.blind_num,
                                                    use_origin=self.use_origin,
                                                    **({"ablation_config": self.ablation_config} if self.ablation_mode else {})
                                                    )
        
        tactic_result_group = await tactic_runner.run(generate_info.response_tactic, previous_result=proof_info.prev_result)
        return await self.handle_tactic_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)

    async def handle_tactic_result(self, proof_info: ProofInfo, proof_context: ProofContext, tactic_runner: TacticRunner, generate_info: GenerateInfo, tactic_result_group: TacticResultGroup):
        if tactic_result_group.status in (TacticStatus.COMPLETED, TacticStatus.SUBCOM) or tactic_result_group.all_success or self.plain_prompt:
            return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)
        else:
            return await self.tactic_reconsider(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)

    async def process_result(self, proof_info: ProofInfo, proof_context: ProofContext, tactic_runner: TacticRunner, generate_info: GenerateInfo, tactic_result_group: TacticResultGroup):
        if tactic_result_group.status == TacticStatus.COMPLETED:
            _, log_entry = await self.extra_info_generate_log(proof_info, proof_context, generate_info, [tactic_result_group.completed_results])
            self.non_extra_info_log(proof_info, proof_context, tactic_result_group.tactic_results, log_entry, index_start=1)
            self.write_to_log_dict(log_entry, proof_info.curr_path, proof_context.log_file)
            return [ProofInfo(
                    curr_ps=tactic_result_group.completed_results.ps_item,
                    prev_result=PreviousResult(
                        ps_item=tactic_result_group.completed_results.ps_item,
                        tactic_sequence=tactic_result_group.completed_results.tactic_sequence
                    ),
                    curr_path=f"{proof_info.curr_path}@1-1/1",
                    is_finished=True
                )]

        elif tactic_result_group.status == TacticStatus.SUBCOM:
            extra_info, log_entry = await self.extra_info_generate_log(proof_info, proof_context, generate_info, [tactic_result_group.subcompleted_results])
            self.non_extra_info_log(proof_info, proof_context, tactic_result_group.tactic_results, log_entry, index_start=1)
            self.write_to_log_dict(log_entry, proof_info.curr_path, proof_context.log_file)
            return await self.handle_subcompleted_result(proof_info, proof_context, tactic_runner, tactic_result_group.subcompleted_results, extra_info)
        else:
            success_results = []
            failed_results = []
            for tactic_result in tactic_result_group.tactic_results:
                if tactic_result.status == TacticStatus.SUCCESS:
                    success_results.append(tactic_result)
                else:
                    failed_results.append(tactic_result)
            extra_info, log_entry = await self.extra_info_generate_log(proof_info, proof_context, generate_info, success_results)
            self.non_extra_info_log(proof_info, proof_context, failed_results, log_entry, index_start=len(success_results))
            self.write_to_log_dict(log_entry, proof_info.curr_path, proof_context.log_file)
            return self.handle_normal_result(proof_info, success_results, extra_info)
    
    async def handle_subcompleted_result(self, proof_info: ProofInfo, proof_context: ProofContext, tactic_runner: TacticRunner, subgoal_completed_result: TacticResult, extra_info: Tuple[List[Tuple[str,Dict,str]], Dict]):
        # here is only a assert, we use idtac to refresh the current goal
        # assert len(proof_info.remaining_list) > 0

        if len(proof_info.remaining_list) == 0:
            print('some error occurs, remaining list is empty')
            with open('error_log.txt', 'a') as f:
                f.write(f'some error occurs, remaining list is empty: name: {proof_context.theorem_name} path: {proof_context.theorem_path}\n')
                f.write(f'proof info: {subgoal_completed_result.tactic_sequence}\n')
            ##TODO need to fake a proofinfo to return

        proof_traces, proof_summarys = extra_info
        # we do not give the refreshed goal a new tactic list, return the current directly
        refreshed_ps_after_state = await self.refresh_current_goal(subgoal_completed_result, tactic_runner)
        if refreshed_ps_after_state is None:
            return None
        
        last_remaining = proof_info.remaining_list[-1]
        if len(last_remaining.states) > 1:
            updated_remaining_list = copy.deepcopy(proof_info.remaining_list)
            updated_remaining_list[-1].states.pop(0)
        else:
            updated_remaining_list = copy.deepcopy(proof_info.remaining_list)[:-1]

        return [ProofInfo(
                curr_ps=refreshed_ps_after_state[0],
                prev_result=PreviousResult(
                    ps_item=subgoal_completed_result.ps_item,
                    tactic_sequence=subgoal_completed_result.tactic_sequence
                ),
                curr_path=proof_info.remaining_list[-1].path,
                remaining_list=updated_remaining_list,
                proof_traces=proof_traces[0],
                proof_summary_with_tactic=ProofSummaryWithTactic(
                    proof_summary=proof_summarys[0],
                    tactic=subgoal_completed_result.tactic_sequence)
                )]
    
    async def refresh_current_goal(self, subgoal_completed_result: TacticResult, tactic_runner: TacticRunner):
        return await tactic_runner.run('idtac.', PreviousResult(ps_item=subgoal_completed_result.ps_item, tactic_sequence=subgoal_completed_result.tactic_sequence), refresh_mode=True)
        
    def handle_normal_result(self, proof_info: ProofInfo, success_results: List[TacticResult], extra_info: Tuple[List[Tuple[str,Dict,str]], Dict]):
        proof_infos = []
        proof_traces, proof_summarys = extra_info
        for i, (success_result, proof_trace, proof_summary) in enumerate(zip(success_results, proof_traces, proof_summarys)):
            current_after_states = success_result.ps_item.get_proof_state().After_state
            if len(current_after_states) > 1:
                path = f"{proof_info.curr_path}@{i+1}-1/{len(current_after_states)}"
                
                new_remaining = RemainingState(
                    states=current_after_states[1:],
                    tactic_result=success_result,
                    path=path
                )
                current_remaining_list = copy.deepcopy(proof_info.remaining_list)
                current_remaining_list.append(new_remaining)

                proof_infos.append(ProofInfo(
                    curr_ps=current_after_states[0],
                    prev_result=PreviousResult(
                        ps_item=success_result.ps_item,
                        tactic_sequence=success_result.tactic_sequence
                    ),
                    curr_path=path,
                    remaining_list=current_remaining_list,
                    proof_traces=proof_trace,
                    proof_summary_with_tactic=ProofSummaryWithTactic(
                        proof_summary=proof_summary,
                        tactic=success_result.tactic_sequence
                    )
                ))
            else:
                proof_infos.append(ProofInfo(
                    curr_ps=current_after_states[0],
                    prev_result=PreviousResult(
                        ps_item=success_result.ps_item,
                        tactic_sequence=success_result.tactic_sequence
                    ),
                    curr_path=f"{proof_info.curr_path}@{i+1}",
                    remaining_list=copy.deepcopy(proof_info.remaining_list),
                    proof_traces=proof_trace,
                    proof_summary_with_tactic=ProofSummaryWithTactic(
                        proof_summary=proof_summary,
                        tactic=success_result.tactic_sequence
                    )
                ))

        return proof_infos
    
    async def extra_info_generate_log(self, proof_info: ProofInfo, proof_context: ProofContext, generate_info: GenerateInfo, tactic_results: List[TacticResult]):
        log_entry = self.get_log_entry(generate_info)
        
        if (self.plain_prompt == True) or (self.ablation_mode and not getattr(self, 'if_public_notes', True) and not getattr(self, 'if_proof_trace', True)):
            # Skip explanation and proof summary generation for plain prompt mode
            proof_traces = []
            proof_summarys = []
            for i, tactic_result in enumerate(tactic_results):
                proof_trace_updated = proof_info.proof_traces + [ProofTrace(brief_strategy="", state_explanation={}, tactic=tactic_result.tactic_sequence[-1])]
                proof_traces.append(proof_trace_updated)
                proof_summarys.append("")
                
                log_entry.items_info.append(SingleItemInfo(
                    path = f"{proof_info.curr_path}@{i+1}",
                    ps_item = tactic_result.ps_item,
                    tactics = tactic_result.tactic_sequence,
                    depth = proof_context.depth,
                    status = tactic_result.status,
                    tactic_traces = tactic_result.tactic_trace,
                ))
            
            return (proof_traces, proof_summarys), log_entry
        
        # Original complex logic for non-plain prompt mode
        explain_prompts = [self.prompt_generator.generate_state_explanation(tactic_result.ps_item.get_proof_state(),if_use_intuition=self.if_use_intuition, use_origin=self.use_origin) for tactic_result in tactic_results]
        if self.use_api:
            explanations = await llm_state_explanation(explain_prompts, mode='state')
        else:
            explanations = await self.state_explanation.generate(explain_prompts, mode='state')
        
        proof_summary_prompts = []
        proof_traces = []
        for tactic_result,explanation in zip(tactic_results, explanations):
            proof_trace_updated = proof_info.proof_traces + [ProofTrace(brief_strategy=generate_info.brief_strategy,state_explanation=explanation, tactic=tactic_result.tactic_sequence[-1])]
            proof_traces.append(proof_trace_updated)
            proof_summary_prompts.append(self.prompt_generator.generate_proof_trace(proof_trace_updated))
            ## proof trace should be the same length as the tactic sequence as the current info has been added
            if not self.generate_mode:
                assert len(proof_trace_updated) == len(tactic_result.tactic_sequence)
        
        if self.use_api:
            proof_summarys = await llm_state_explanation(proof_summary_prompts, mode='strategy')
        else:
            proof_summarys = await self.state_explanation.generate(proof_summary_prompts, mode='strategy')
            
        if not self.generate_mode:
            assert len(proof_summarys) == len(tactic_results)

        for i, (explain_prompt, explanation, proof_summary_prompt, proof_summary, tactic_result) in enumerate(zip(explain_prompts,explanations,proof_summary_prompts,proof_summarys,tactic_results)):
            extra_info = {
                'prompt_explain': explain_prompt,
                'response_explain': explanation,
                'prompt_proof_summary': proof_summary_prompt,
                'response_proof_summary': proof_summary
            }
            log_entry.items_info.append(SingleItemInfo(
                path = f"{proof_info.curr_path}@{i+1}",
                ps_item = tactic_result.ps_item,
                tactics = tactic_result.tactic_sequence,
                depth = proof_context.depth,
                status = tactic_result.status,
                tactic_traces = tactic_result.tactic_trace,
                proof_summary = proof_summary,
                explanation = explanation,
                extra_info = extra_info,
            ))
        
        return (proof_traces,proof_summarys), log_entry

    def non_extra_info_log(self, proof_info: ProofInfo, proof_context: ProofContext, tactic_results: List[TacticResult], log_entry: LogInfo, index_start: int = 0):
        for i, tactic_result in enumerate(tactic_results):
            log_entry.items_info.append(SingleItemInfo(
                path = f"{proof_info.curr_path}@{i+index_start+1}",
                ps_item = tactic_result.ps_item,
                tactics = tactic_result.tactic_sequence,
                depth = proof_context.depth,
                status = tactic_result.status,
                tactic_traces = tactic_result.tactic_trace,
            ))

    def get_log_entry(self, generate_info: GenerateInfo):
        log_entry = {}
        for attr_name, attr_value in generate_info.__dict__.items():
            if 'response' in attr_name or 'prompt' in attr_name:
                log_entry[attr_name] = attr_value
        return LogInfo(prompt_response_info=log_entry)
    
    def write_to_log_dict(self, log_entry: Union[LogInfo, Dict], current_path: str, log_file: str):
        with open(log_file, 'a', encoding='utf-8') as f:
            if not isinstance(log_entry, Dict):
                json.dump({current_path: log_entry.to_dict()}, f, ensure_ascii=False)
            else:
                json.dump({current_path: log_entry}, f, ensure_ascii=False)
            f.write('\n')

    async def tactic_reconsider(self, proof_info: ProofInfo, proof_context: ProofContext, tactic_runner: TacticRunner, generate_info: GenerateInfo, tactic_result_group: TacticResultGroup):
        if self.reconsider_mode == "hierarchical":
            return await self.hierarchical_reconsider(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)
        elif self.reconsider_mode == "normal":
            return await self.normal_reconsider(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)
        else:
            raise ValueError(f"Invalid reconsider mode: {self.reconsider_mode}")
    
    async def hierarchical_reconsider(self, proof_info: ProofInfo, proof_context: ProofContext, tactic_runner: TacticRunner, generate_info: GenerateInfo, tactic_result_group: TacticResultGroup):
        failed_results = []
        success_results = []
        for tactic_result in tactic_result_group.tactic_results:
            if tactic_result.status == TacticStatus.FAIL:
                failed_results.append(tactic_result)
            else:
                success_results.append(tactic_result)

        if self.use_ft_model:
            tasks = [self.prompt_generator.generate_re_consider(failed, generate_info.prompt_tactic, re_con_mode="hierarchical", ft_mode=self.use_ft_model) 
                        for failed in failed_results]
            refined_tactic_list = await asyncio.gather(*tasks)
            com_refined_tactics = []

            for refined_tactics, _ in refined_tactic_list:
                if not isinstance(refined_tactics, list):
                    raise Exception('refined_tactics is not a list in ft mode')
                for tactic in refined_tactics:
                    if tactic not in com_refined_tactics and tactic not in generate_info.response_tactic:
                        com_refined_tactics.append(tactic)
            generate_info.prompt_reconsider_tactic = refined_tactic_list[0][1]
            generate_info.response_reconsider_tactic = com_refined_tactics
            
            print(f"com_refined_tactics: {len(com_refined_tactics)}, prodeuced by {len(failed_results)} failed results in hierarchical mode")
            if len(com_refined_tactics) > 10:
                random.shuffle(com_refined_tactics)
                warnings.warn(f"com_refined_tactics: {len(com_refined_tactics)}, prodeuced by {len(failed_results)} failed results in hierarchical mode, random select 10 tactics")
                com_refined_tactics = com_refined_tactics[:10]
            tactic_result_group_reconsider = await tactic_runner.run(com_refined_tactics, previous_result=proof_info.prev_result)
            tactic_result_group_reconsider.merge(tactic_result_group)
            return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group_reconsider)

        for _ in range(self.max_retries):
            if not failed_results:
                break
            refined_failed_results = []
            give_up_results = []

            tasks = [self.prompt_generator.generate_re_consider(failed, generate_info.prompt_tactic, re_con_mode="hierarchical") 
                    for failed in failed_results]
            refined_tactics = await asyncio.gather(*tasks)

            assert len(refined_tactics) == len(failed_results)

            async def refine_single_failed(refined_tactic:str, failed:TacticResult):
                if refined_tactic['refined_tactic']:
                    result = await tactic_runner.run(refined_tactic, previous_result=proof_info.prev_result, tactic_trace=failed.tactic_trace, if_gathered=False)
                    if result:
                        reason = refined_tactic['reason']
                        return result
                    else:
                        reason = 'tactic failed'
                else:
                    reason = refined_tactic['reason']
    
                give_up_trace = TacticTrace_Single(tactic='', error_message='failed', reason=reason)
                failed.tactic_trace.append(give_up_trace)
                failed.status = TacticStatus.GIVEUP
                return [failed]

            pending_tasks = [asyncio.create_task(refine_single_failed(refined_tactic,failed)) 
                    for refined_tactic,failed in zip(refined_tactics,failed_results)]
            
            for future in asyncio.as_completed(pending_tasks):
                result = await future
                assert len(result) == 1
                result = result[0]
                if result.status == TacticStatus.COMPLETED:
                    for task in pending_tasks:
                        if not task.done():
                            task.cancel()
                    tactic_result_group.completed_results = result
                    tactic_result_group.status = TacticStatus.COMPLETED
                    ## use failed_results may casue duplicate, not that strict
                    tactic_result_group.tactic_results = success_results + failed_results + give_up_results
                    return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)
                elif result.status == TacticStatus.SUBCOM:
                    for task in pending_tasks:
                        if not task.done():
                            task.cancel()
                    tactic_result_group.subcompleted_results = result
                    tactic_result_group.status = TacticStatus.SUBCOM
                    ## use failed_results may casue duplicate, not that strict
                    tactic_result_group.tactic_results = success_results + failed_results + give_up_results
                    return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)
                elif result.status == TacticStatus.GIVEUP:
                    give_up_results.append(result)
                elif result.status == TacticStatus.SUCCESS:
                    success_results.append(result)
                else:
                    refined_failed_results.append(result)

            failed_results = refined_failed_results

        tactic_result_group.tactic_results = success_results + failed_results + give_up_results
        return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)

    async def normal_reconsider(self, proof_info: ProofInfo, proof_context: ProofContext, tactic_runner: TacticRunner, generate_info: GenerateInfo, tactic_result_group: TacticResultGroup):
        failed_results = []
        success_results = []
        for tactic_result in tactic_result_group.tactic_results:
            if tactic_result.status == TacticStatus.SUCCESS:
                success_results.append(tactic_result)
            else:
                failed_results.append(tactic_result)
        
        ## ft_model is not sensitive to the error, so we generate new method
        if self.use_ft_model:
            success_results = None
            response_reconsider_method, response_reconsider_tactic, prompt_reconsider_tactic, prompt_reconsider_method = await self.prompt_generator.generate_re_consider(failed_results, generate_info.prompt_tactic, success_results, generate_info.prompt_method, generate_info.response_method, ft_mode=self.use_ft_model)
            generate_info.response_reconsider_tactic = response_reconsider_tactic
            generate_info.response_reconsider_method = response_reconsider_method
            generate_info.prompt_reconsider_tactic = prompt_reconsider_tactic
            generate_info.prompt_reconsider_method = prompt_reconsider_method

            tactic_result_group_reconsider = await tactic_runner.run(response_reconsider_tactic, previous_result=proof_info.prev_result)
            tactic_result_group_reconsider.merge(tactic_result_group)
            return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group_reconsider)

        prompt_reconsider_tactic, tactic_list = await self.prompt_generator.generate_re_consider(failed_results, generate_info.prompt_tactic, success_results, generate_info.prompt_method, generate_info.response_method)
        generate_info.prompt_reconsider_tactic = prompt_reconsider_tactic
        generate_info.response_reconsider_tactic = tactic_list
        
        if len(tactic_list) == 0:
            return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group)
        
        tactic_result_group_reconsider = await tactic_runner.run(tactic_list, previous_result=proof_info.prev_result)
        tactic_result_group_reconsider.merge(tactic_result_group)
        return await self.process_result(proof_info, proof_context, tactic_runner, generate_info, tactic_result_group_reconsider)

    async def update_public_notes(self,proof_context: ProofContext, next_layer: List[ProofInfo], retrieval_info: Tuple[List[str], List[str]]):
        current_candidates = []
        for proof_info in next_layer:
            explanation_str = '\n'.join([f"{key}: {value['en']}" for key, value in proof_info.proof_traces[-1].state_explanation.items()])
            method_str = f"## Strategy: {proof_info.proof_traces[-1].brief_strategy}\n## Tactic: {proof_info.proof_traces[-1].tactic}\n## Intuition:\n{explanation_str}\n"
            method_info_str = f"strategy_method_info: tactic:{proof_info.proof_traces[-1].tactic}"
            current_candidates.append((method_str, method_info_str))
        
        prompt_public_notes, response_public_notes = await self.prompt_generator.generate_note(proof_context.ps_init, proof_context, proof_context.public_notes, current_candidates, if_use_intuition=self.if_use_intuition, use_origin=self.use_origin)
        next_layer_public_notes = self._update(proof_context.public_notes, current_candidates, response_public_notes)
        
        public_notes_log_dict = {
            'prompt_public_notes': prompt_public_notes,
            'response_public_notes': response_public_notes,
            'previous_public_notes': proof_context.public_notes,
            'updated_public_notes': next_layer_public_notes,
            'depth': proof_context.depth
        }

        self.write_to_log_dict(public_notes_log_dict, 'public_notes', proof_context.log_file)

        # premises, tactics = retrieval_info
        # premises_candidates = []
        # premises_names = []
        # for premise in premises:
        #     name = self.extract_name(premise, mode='premise')
        #     if name not in premises_names and name not in public_notes_names:
        #         premises_names.append(name)
        #         premises_candidates.append((premise,f"premise_info: {name}"))
        
        # tactics_candidates = []
        # tactics_names = []
        # for tactic in tactics:
        #     name = self.extract_name(tactic, mode='tactic')
        #     if name not in tactics_names and name not in public_notes_names:
        #         tactics_names.append(name)
        #         tactics_candidates.append((tactic,f"tactic_info: {name}"))


        return next_layer_public_notes

    def _update(self, public_notes: List[Tuple[str,str]], current_candidates: List[Tuple[str,str]], update_dict: Dict):
        if update_dict['remove']:
            index_list = sorted([int(i) for i in update_dict['remove']], reverse=True)
            if public_notes:
                for index in index_list:
                    try:
                        public_notes.pop(index)
                    except Exception as e:
                        print("update_public_notes error")
                        # print('update_dict', update_dict)
                        print('public_notes', len(public_notes))
                        print('current_candidates', len(current_candidates))
                        # raise e
                        # llm response may not properly give the index
                        print(e)
                        continue
                
        if update_dict['add']:
            for index in update_dict['add']:
                try:
                    public_notes.append(current_candidates[int(index)])
                except Exception as e:
                    print("update_public_notes error")
                    # print('update_dict', update_dict)
                    print('public_notes', len(public_notes))
                    print('current_candidates', len(current_candidates))
                    print(e)
                    # raise e
                    # llm response may not properly give the index
                    continue

        if len(public_notes) > 15:
            public_notes = random.sample(public_notes, 15)
            warnings.warn(f"public_notes is too long, random sample 15 notes")
        
        return public_notes

    async def proof_info_selection(self, proof_context: ProofContext, next_layer: List[ProofInfo]):
        if len(next_layer) > self.beam_width:
            prompt_ps_selection, response_ps_selection, selected_idx = await self.prompt_generator.generate_ps_selection(proof_context, next_layer, use_origin=self.use_origin, if_use_intuition=self.if_use_intuition)

            ps_selection_log_dict = {
                'prompt_ps_selection': prompt_ps_selection,
                'response_ps_selection': response_ps_selection,
                'depth': proof_context.depth - 1
            }
            self.write_to_log_dict(ps_selection_log_dict, 'ps_selection', proof_context.log_file)
            next_layer = [next_layer[i] for i in selected_idx]

        return next_layer

    def resume_theorem_list(self, theorem_list: List[Dict], resume_mode: Union[bool, str] = True):
        processed_theorem_list = []
        failed_theorem_list = []
        if not os.path.exists(self.proof_history_log):
            return theorem_list
        with open(self.proof_history_log, 'r') as f:
            for line in f:
                if 'SUCCEEDED' in line or 'FAILED' in line:
                    theorem_name = line.split(' ')[1].strip()
                    processed_theorem_list.append(theorem_name)
                    if self.resume_mode == 'failed':
                        if 'FAILED' in line:
                            failed_theorem_list.append(theorem_name)

        if self.resume_mode == 'failed':
            return [theorem for theorem in theorem_list if theorem['theorem_name'] in failed_theorem_list]
        
        need_proof_theorem_list = []
        for theorem in theorem_list:
            if theorem['theorem_name'] not in processed_theorem_list:
                need_proof_theorem_list.append(theorem)
        return need_proof_theorem_list

    def proof_initial(self, init_ps: PSItem, theorem_path: str, theorem_name: str):
        ## TODO: for different theorem, get different init info
        proof_info_layer = ProofInfoLayer(proof_context=ProofContext(theorem_name=theorem_name,
                                                                     theorem_path=theorem_path,
                                                                     ps_init=init_ps,
                                                                     depth=0),
                                           proof_infos=[ProofInfo(curr_ps=init_ps)])
        
        tactic_runner = TacticRunner(self.coqc, self.config_path,
                                     self.tokenizer,self.new_theorem_mode,
                                     self.def_table_dict,
                                     current_file=theorem_path,
                                     current_name=theorem_name,
                                     log_prefix=self.proof_history_dir)
        
        return proof_info_layer, tactic_runner
    
    def collect_theorem_func(self, package_name: str = None, theorem_name: str = None, theorem_file_path: str = None, ratio: float = 0.1):
        if package_name and (theorem_name or theorem_file_path):
            warnings.warn('theorem_name or theorem_file_path will be ignored when package_name is provided')
        if not (theorem_name or theorem_file_path) and not package_name:
            raise ValueError('theorem_name or theorem_file_path must be provided together')
        
        theorem_list = []
        if package_name:
            if package_name == 'all':
                from data.coq_test_package import coq_test
                package_name = coq_test
                theorem_list = [
                {
                    'theorem_name': item['name'],
                    'file_path': item['file_path']
                }
                for item in self.def_table 
                if any(pkg in item['file_path'] for pkg in package_name) and item['kind'] == 'Proof'
                ]

                sample_size = max(1, int(len(theorem_list) * ratio))
                theorem_list = random.sample(theorem_list, sample_size)
                print(f"collect {len(theorem_list)} theorems from {package_name} (sampled from {len(theorem_list)} theorems with ratio {ratio})")
            else:
                theorem_list = [
                    {
                        'theorem_name': item['name'],
                        'file_path': item['file_path']
                    }
                    for item in self.def_table if package_name in item['file_path'] and item['kind'] == 'Proof'
                ]
                print(f"collect {len(theorem_list)} theorems from {package_name}")
        else:
            if not theorem_file_path.startswith(self.path_prefix):
                theorem_file_path = os.path.join(self.path_prefix, theorem_file_path)
                warnings.warn(f"theorem_file_path {theorem_file_path} is not in the path_prefix {self.path_prefix}, automatically add the path_prefix, new theorem_file_path: {theorem_file_path}")

            if '.' not in theorem_name:
                prefix = self.parser.get_file_prefix(theorem_file_path)
                theorem_name = prefix + '.' + theorem_name
                warnings.warn(f"theorem_name {theorem_name} looks like a simple name, automatically add the file_prefix, new theorem_name: {theorem_name}")
                
            theorem_list = [
                {
                    'theorem_name': theorem_name,
                    'file_path': theorem_file_path
                }
            ]
            print(f"Proof theorem {theorem_name}, {theorem_file_path}")
        
        return theorem_list

    async def init_ps_func(self, theorem: Dict):
        theorem_path = theorem['file_path'] if self.path_prefix in theorem['file_path'] else os.path.join(self.path_prefix, theorem['file_path'])
        theorem_name = theorem['theorem_name']
        try:
            base_file_content = get_base_file_content(theorem_path, theorem_name)
            if not base_file_content:
                raise ValueError(f"Failed to get base file content for theorem {theorem_name}")
        except Exception as e:
            raise ValueError(f"Failed to get base file content for theorem {theorem_name}: {e}")
        
        output, error, temp_file = await self.coqc.run(theorem_path, base_file_content, init_ps=True)
        if "Timeout error" in error:
            output, error, temp_file = await self.coqc.run(theorem_path, base_file_content, init_ps=True, timeout=1500)
            with open(f'{self.proof_history_dir}/init_log.txt', 'a') as f:
                f.write(f'{theorem["file_path"]} {theorem["theorem_name"]} init ps is None\n')
                f.write('Timeout error\n\n')
            return None, None, None
        if not ('Attempt to save an incomplete proof' in error or 'There are pending proofs' in error):
            with open(f'{self.proof_history_dir}/init_log.txt', 'a') as f:
                f.write(f'{theorem["file_path"]} {theorem["theorem_name"]} init ps is None\n')
                f.write(f'{error}\n\n')
            return None, None, None
            # raise ValueError(f"error when init ps for theorem {theorem_name}: {error}")
        
        ps, actual_name, _ , type_dict = self.parser.parse_proof(output,theorem_path,theorem_name,use_tqdm=False)
        ps_item = self.tokenizer.process_ps_proof(ps, def_table=self.def_table_dict, type_dict=type_dict, actual_name=actual_name, txt_file_path=temp_file, if_refined_ps=False)
        
        assert len(ps_item.Content.ProofStates) == 1
        return ps_item, theorem_path, theorem_name
    
    def init_prompt_generator(self):
        if self.plain_prompt:
            # Skip retrieval initialization for plain prompt mode
            return PromptGenerator(
                def_path=self.def_table_path,
                tokenizer=self.tokenizer,
                retrieval=None
            )
        else:
            # Original initialization with retrieval for complex mode
            retrieval = Retrieval(
                emb_file=self.emb_data_path,
                model_name=self.emb_model_path
            )
            return PromptGenerator(
                def_path=self.def_table_path,
                tokenizer=self.tokenizer,
                retrieval=retrieval
            )

    def init_log_file(self, package_name: str = None, theorem_name: str = None):
        log_prefix = f'{self.system_log_dir}/{self.ft_model}_{self.reconsider_mode}' if self.use_ft_model else f'{self.system_log_dir}/{self.model_use}_{self.reconsider_mode}'
        if self.plain_prompt:
            log_prefix = f'{self.system_log_dir}/{self.model_use}_plain'
        if self.ablation_mode:
            ablation_mode = 'def_' if self.if_def else '' + 'retrieve_' if self.if_retrieve else '' + 'proof_trace_' if self.if_proof_trace else '' + 'public_notes_' if self.if_public_notes else '' + 'strategy_' if self.if_strategy else ''
            if ablation_mode.endswith('_'):
                ablation_mode = ablation_mode[:-1]
            if self.ablation_proof_mode:
                ablation_mode = self.ablation_proof_mode
            log_prefix = f'{self.system_log_dir}/{self.model_use}_ablation_{ablation_mode}'

        if package_name:
            self.proof_history_dir = f'{log_prefix}/{package_name}/'
        else:
            theorem_name = theorem_name.split('.')[-1]
            self.proof_history_dir = f'{log_prefix}/{theorem_name}/'

        os.makedirs(self.proof_history_dir, exist_ok=True)
        self.proof_history_log = f'{self.proof_history_dir}/proof.log'
        print(f'proof_history_log: {self.proof_history_log}')
        # TODO: coqc log 
    
    def init_ft_llm(self):
        ft_model_dict = {
            'qwen-ins-2.5-7b-reorganize': '',
            'qwen-base-2.5-7b-reorganize': '',
        }
        from openai import AsyncOpenAI

        if self.ft_model is not None:
            if self.ft_model in ft_model_dict:
                llm_host = ft_model_dict[self.ft_model]
                llm_method.ft_host = llm_host
                llm_method.ft_name = self.ft_model
                llm_method.client_ft = AsyncOpenAI(api_key="", base_url=llm_host, timeout=300, max_retries=2)
                llm_method.is_instruct = True if '-ins' in self.ft_model else False
                from transformers import AutoTokenizer
                llm_method.tokenizer = AutoTokenizer.from_pretrained(self.ft_model_path)
            else:
                raise Exception(f"ft_model {self.ft_model} not supported, only {ft_model_dict.keys()} are supported")

    def init_ablation_params(self):
        if self.ablation_scope not in ['def_only', 'all']:
            raise ValueError(f"ablation_scope {self.ablation_scope} not supported, only def_only, all are supported")
        if self.ablation_proof_mode not in ['no_ref_ps_all', 'no_ref_ps_origin', 'origin_only', 'internal_only', 'intuition_only', 'origin_internal', 'origin_intuition', 'internal_intuition', 'origin_internal_intuition', 'zh_explanation', 'base']:
            raise ValueError(f"proof_mode {self.proof_mode} not supported, only no_ref_ps_all, no_ref_ps_origin, origin_only, internal_only, intuition_only, origin_internal, origin_intuition, internal_intuition, origin_internal_intuition, zh_explanation, base are supported")
        
        if not self.ablation_proof_mode:
            return None
        
        ablation_params = {
            "ablation_scope": self.ablation_scope,
            "use_origin": self.use_origin,
            "if_use_intuition": self.if_use_intuition,
        }
        
        match self.ablation_proof_mode:
            case 'no_ref_ps_all':
                self.if_def = False
                ablation_params['use_origin'] = 'mixed'
            case 'no_ref_ps_origin':
                self.if_def = False
                ablation_params['use_origin'] = 'origin'
            case 'origin_only':
                ablation_params['use_origin'] = 'origin'
                ablation_params['if_use_intuition'] = False
            case 'internal_only':
                ablation_params['use_origin'] = 'internal'
                ablation_params['if_use_intuition'] = False
            case 'intuition_only':
                ablation_params['use_origin'] = 'empty'
                ablation_params['if_use_intuition'] = True
            case 'origin_internal':
                ablation_params['use_origin'] = 'origin'
                ablation_params['if_use_intuition'] = True
            case 'origin_intuition':
                ablation_params['use_origin'] = 'origin'
                ablation_params['if_use_intuition'] = True
            case 'internal_intuition':
                ablation_params['use_origin'] = 'internal'
                ablation_params['if_use_intuition'] = True
            case 'origin_internal_intuition':
                ablation_params['use_origin'] = 'mixed'
                ablation_params['if_use_intuition'] = True
            case 'zh_explanation':
                pass
            case 'base':
                self.plain_prompt = True
        return ablation_params
