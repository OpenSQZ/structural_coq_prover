import re
import warnings
import asyncio
from coq_prover.coq_context.prompt import *
from utils import read_jsonl_file, read_json_file
from typing import List, Union, Dict, Tuple
from data_extraction.coq_tokenize.tokenizer import Tokenizer
from coq_prover.coq_context.llm_method import llm_simplify_response, llm_selection_response, llm_note_response, llm_reorganize_response
import numpy as np
import random
from coq_prover.coq_context.retrieval import Retrieval
from coq_prover.coq_context.utils import format_def
from coq_prover.coq_context.llm_method import llm_response, llm_normal
from data_extraction.coq_data.Ps_class import ps_object_single, PSItem, State
from data_extraction.coq_data.Def_class import def_object
from data_extraction.coq_tokenize.tokenizer import new_theorem_start_id
from dataclasses import dataclass
from coq_prover.coq_context.proof_data_class import *
    

class PromptGenerator:
    def __init__(self, def_path: str, ps_with_id_path: str = None, concept_path: str = None, tokenizer: Tokenizer = None, retrieval: Retrieval = None):
        random.seed(42)
        self.def_table = self._load_definitions(def_path)
        self.tokenizer = tokenizer
        # we do not give the tactic example in the prompt, so we do not need to load the concept2tactic
        if concept_path:
            self.concept2tactic = self._load_concepts(concept_path)
        else:
            self.concept2tactic = {}
        
        self.filtered_tactics = {
            'intro', 'clear', 'elim', 'with_uniform_flags', 
            'exact', 'flatten_contravariant_disj', 
            'flatten_contravariant_conj', 'with typeclass_instances', 'autoapply'
        }

        ## work for now, however if the whole dataset's ps_table is too large, we need to use another method
        ## we do not give the tactic example in the prompt, so we do not need to load the ps_table
        if ps_with_id_path:
            self.ps_table = self._load_proof_states(ps_with_id_path)
        else:
            self.ps_table = {}

        self.retrieval = retrieval
        random.seed(42)
        np.random.seed(42)
    
    def update_def_table(self, new_def_dict: Dict):
        self.new_theorem_mode = True
        self.def_table[new_def_dict['def_id']] = new_def_dict

    def _load_definitions(self, path: str) -> Dict:
        def_table = read_jsonl_file(path)
        self.glob_start_id = def_table[0]['def_id']
        return {item['def_id']: item for item in def_table}
    
    def _load_proof_states(self, path: str) -> Dict:
        ps_table = read_jsonl_file(path)
        return {
            ps['step_id']: ps 
            for item in ps_table 
            for ps in item['content']['proofstates']
        }

    def _load_concepts(self, path: str) -> Dict:
        concept2tactic = read_json_file(path)
        return {
            self.tokenizer.encode(def_): content 
            for def_, content in concept2tactic.items()
        }
    
    def get_single_state(slef, ps:State, use_origin='mixed'):
        if isinstance(ps, State):
            ps = ps.to_dict()
        hyps = []
        token_ids = []
        if 'hyps' in ps.get('hyps',{}):
            for hyp_name, hyp_content in ps['hyps']['hyps'].items():
                origin_content = hyp_content['text']
                internal_content = hyp_content['processed']['origin']
                if use_origin == 'origin':
                    content = origin_content
                elif use_origin == 'internal':
                    content = internal_content
                elif use_origin == 'mixed':
                    content = INTERNAL_ORIGIN_MIXED_FORMAT.format(internal=internal_content, origin=origin_content)
                else:
                    raise ValueError(f"Invalid use_origin: {use_origin}")
                hyps.append(hyp_name + ":" + '\n' + content + "\n")
                token_ids.extend(map(int, hyp_content['processed']['token_ids'].split(',')))
        
        goal = ps['goal']
        origin_goal_str = goal['text']
        internal_goal_str = goal['processed']['origin']
        if use_origin == 'origin':
            goal_str = origin_goal_str
        elif use_origin == 'internal':
            goal_str = internal_goal_str
        elif use_origin == 'mixed':
            goal_str = INTERNAL_ORIGIN_MIXED_FORMAT.format(internal=internal_goal_str, origin=origin_goal_str)
        else:
            raise ValueError(f"Invalid use_origin: {use_origin}")
        token_ids.extend(map(int, goal['processed']['token_ids'].split(',')))
        return hyps, goal_str, token_ids
    
    def extract_proof_state(self, data: 'ps_object_single', prefix='before', use_origin='mixed', extra_tokens: List=None):
        tokens_list = []
        hyps_list = []
        goal_list = []

        if prefix == 'before':
            ps = data[f'{prefix}_state']
            hyps, goal, token_list = self.get_single_state(ps, use_origin=use_origin)
            hyps_list.extend(hyps)
            goal_list.append(goal)
            tokens_list.extend(token_list)
            if extra_tokens:
                tokens_list.extend(extra_tokens)
            tokens_list = list(set(tokens_list))
        elif prefix == 'after':
            for ps in data[f'{prefix}_state']:
                hyps, goal, token_list = self.get_single_state(ps, use_origin=use_origin)
                hyps_list.append(hyps)
                goal_list.append(goal)
                token_list = list(set(token_list))
                tokens_list.append(token_list)
        
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

        # if goal['processed']['origin'] == 'goalcompleted':
        #     return None
        # else:
        #     goal = goal['processed']['origin']

        return hyps_list, goal_list, tokens_list

    def generate_state(self, data: Union['ps_object_single', 'PSItem'], mode='state', use_origin='mixed', extra_tokens: List=None, if_use_intuition: bool = True, plain_prompt: bool = False, ablation_params: Dict=None):
        if isinstance(data, ps_object_single):
            # now ps_obj to dict will return a list for nested states, we only need the top level
            data = data.to_dict()['states'][0]
        elif isinstance(data, PSItem):
            data = data.to_dict()
        
        if 'content' in data and 'proofstates' in data.get('content',{}):
            data = data['content']['proofstates'][0]
        elif 'before_state' in data:
            data = data
        else:
            print(data)
            print("Invalid Data for init state, must PSItem or ps_object_single")
            raise ValueError("Invalid Data for init state, must PSItem or ps_object_single")
        if mode == 'state':
            before_state_hyps_list, goal_list, token_list = self.extract_proof_state(data, prefix='before', use_origin=use_origin, extra_tokens=extra_tokens)
            if plain_prompt:
                return STATE_NODEF_FORMAT.format(hyps='\n'.join(before_state_hyps_list), goal='\n'.join(goal_list))
            try:
                def_content, glob_tokens = self.get_def(token_list, use_origin=use_origin if not ablation_params else ablation_params['use_origin'], if_use_intuition=if_use_intuition if not ablation_params else ablation_params['if_use_intuition'])
            except Exception as e:
                print(data)
                print('get_def error, please check the data, in state mode')
                raise e
            before_state_str = STATE_FORMAT.format(hyps='\n'.join(before_state_hyps_list), goal='\n'.join(goal_list), glob_def=def_content)
            return before_state_str, glob_tokens
        elif plain_prompt:
            raise ValueError("Plain prompt is only allowed for state mode")
        elif mode == 'proof':
            after_state_list = []
            glob_token_list = []
            after_state_hyps_list, after_goal_list, token_list = self.extract_proof_state(data, prefix='after', use_origin=use_origin)
            for after_state_hyp, after_goal, glob_token in zip(after_state_hyps_list, after_goal_list, token_list):
                glob_tokens = self.get_def(glob_token, use_origin=use_origin, token_only=True)
                after_state_list.append(TACTIC_STATE_FORMAT.format(hyps='\n'.join(after_state_hyp), goal=after_goal))
                glob_token_list.append(glob_tokens)
            return after_state_list, glob_token_list
        elif mode == 'tactic':
            before_state_hyps_list, before_goal_list, _ = self.extract_proof_state(data, prefix='before', use_origin=use_origin)
            after_state_hyps_list, after_goal_list, _= self.extract_proof_state(data, prefix='after', use_origin=use_origin)
            before_state_str = TACTIC_STATE_FORMAT.format(hyps='\n'.join('      ' + line for line in before_state_hyps_list), goal='\n'.join(before_goal_list))
            
            after_state_str = ''
            for hps, goal in zip(after_state_hyps_list, after_goal_list):
                after_state_str += TACTIC_STATE_FORMAT.format(hyps='\n'.join('      ' + line for line in hps), goal=goal) + '\n'

            tactic_state_str = TACTIC_FORMAT.format(name=data['tactic']['name'], before_state=before_state_str, after_states=after_state_str)
            return tactic_state_str
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def extract_definition(self, def_item: def_object, mode='understanding', use_origin='mixed', if_use_intuition=False, if_give_def=True):
        if isinstance(def_item, def_object):
            def_item = def_item.to_dict()

        if mode=='normal':
            def_ = format_def(def_item, use_origin=use_origin if if_give_def else 'origin')
            current_def_id = self.tokenizer.encode(def_item['name'])
            glob_tokens = []
            glob_tokens.extend(map(int, def_item['internal_context']['content']['processed']['token_ids'].split(',')))
            if def_item['internal_context']['body']:
                glob_tokens.extend(map(int, def_item['internal_context']['body']['processed']['token_ids'].split(',')))
            for item in def_item['local_vars']:
                if item['type']:
                    glob_tokens.extend(map(int, item['type'].split(',')))
            glob_tokens = list(set([token for token in glob_tokens if token >= self.glob_start_id and token != current_def_id]))
            def_content, glob_tokens_items = self.get_def(glob_tokens, use_origin=use_origin, if_use_intuition=if_use_intuition)
            if not if_give_def:
                return DEFINITION_WITHOUT_DEFS_FORMAT.format(name=def_item['name'], definition=def_)
            return DEFINITION_WITH_DEFS_FORMAT.format(name=def_item['name'], definition=def_, related_references=def_content)
        
        if not if_give_def:
            def_ = format_def(def_item, use_origin='origin')
            if mode == 'relation':
                current_def_id = self.tokenizer.encode(def_item['name'])
                glob_tokens = []
                glob_tokens.extend(map(int, def_item['internal_context']['content']['processed']['token_ids'].split(',')))
                if def_item['internal_context']['body']:
                    glob_tokens.extend(map(int, def_item['internal_context']['body']['processed']['token_ids'].split(',')))
                for item in def_item['local_vars']:
                    if item['type']:
                        glob_tokens.extend(map(int, item['type'].split(',')))
                glob_tokens = list(set([token for token in glob_tokens if token >= self.glob_start_id and token != current_def_id]))
                glob_tokens_items = self.get_def(glob_tokens, token_only=True)
                if len(glob_tokens_items) <= 2:
                    return None, None
                return glob_tokens_items, DEFINITION_WITHOUT_DEFS_FORMAT.format(name=def_item['name'], definition=def_)
            return DEFINITION_WITHOUT_DEFS_FORMAT.format(name=def_item['name'], definition=def_)
        
        
        def_ = format_def(def_item, use_origin=use_origin)
        current_def_id = self.tokenizer.encode(def_item['name'])
        glob_tokens = []
        glob_tokens.extend(map(int, def_item['internal_context']['content']['processed']['token_ids'].split(',')))
        if def_item['internal_context']['body']:
            glob_tokens.extend(map(int, def_item['internal_context']['body']['processed']['token_ids'].split(',')))
        for item in def_item['local_vars']:
            if item['type']:
                glob_tokens.extend(map(int, item['type'].split(',')))
        glob_tokens = list(set([token for token in glob_tokens if token >= self.glob_start_id and token != current_def_id]))
        def_content, glob_tokens_items = self.get_def(glob_tokens, use_origin=use_origin, if_use_intuition=False)
        
        if len(glob_tokens_items) <= 2:
            if mode == 'relation':
                return None, None
        
        if mode == 'relation':
            return glob_tokens_items, DEFINITION_WITH_DEFS_FORMAT.format(name=def_item['name'], definition=def_, related_references=def_content)
        return DEFINITION_WITH_DEFS_FORMAT.format(name=def_item['name'], definition=def_, related_references=def_content)
                
    def get_new_theorem_def_ids(self, token_list: List[int]):
        if not hasattr(self, 'new_theorem_mode'):
            raise ValueError("New theorem mode is not set, but found new theorem token")
        new_theorem_def_ids = []
        for token in token_list:
            def_token_ids = []
            def_ = self.def_table[token]
            def_token_ids.extend(map(int, def_['internal_context']['content']['processed']['token_ids'].split(',')))
            if def_['internal_context']['body']:
                def_token_ids.extend(map(int, def_['internal_context']['body']['processed']['token_ids'].split(',')))
            for item in def_['local_vars']:
                if item['type']:
                    def_token_ids.extend(map(int, item['type'].split(',')))

            if any(token >= new_theorem_start_id for token in def_token_ids):
                recurion_ids = self.get_new_theorem_def_ids([token_ for token_ in def_token_ids if (token_ >= new_theorem_start_id and token_ != token)])
                def_token_ids.extend(recurion_ids)

            new_theorem_def_ids.extend([token for token in def_token_ids if token >= self.glob_start_id])
        return list(set(new_theorem_def_ids))
    
    def get_def(self, token_list: List[int], use_origin='mixed', token_only=False, if_use_intuition=True):
        token_list = list(set(token_list))
        if not any(token >= self.glob_start_id for token in token_list):
            warnings.warn('no global token found, please check the token list')
            # raise ValueError("No global token found")
        glob_tokens = [token for token in token_list if token >= self.glob_start_id]

        new_theorem_token_tokens = [token for token in token_list if token >= new_theorem_start_id]
        
        if new_theorem_token_tokens:
            new_theorem_def_ids = self.get_new_theorem_def_ids(new_theorem_token_tokens)
            glob_tokens.extend(new_theorem_def_ids)
            glob_tokens = list(set(glob_tokens))

        if token_only:
            return glob_tokens
        def_list = []
        for token in glob_tokens:
            if token not in self.def_table:
                raise ValueError(f"Token {token} not found in def_table")
            item = self.def_table[token]
            if item['kind'] == 'Primitive':
                continue
            def_content = format_def(item, use_origin=use_origin)
            if if_use_intuition:
                try:
                    intuition = self.retrieval.get_intuition(item['name'])
                except Exception as e:
                    print(item)
                    print('get_intuition error, please check the data, in get def')
                    raise e
                def_list.append(DEFINITION_FORMAT.format(name=item['name'], content=def_content, intuition=intuition))
            else:
                def_list.append(DEFINITION_FORMAT_NO_INTUITION.format(name=item['name'], content=def_content))
        return '\n'.join(def_list), glob_tokens

    def sample_concept_with_weight(self, concept_list, blind_num=3):
        blind_list = []
        for token in concept_list:
            tactic_weights = []
            selected_tactics = []
            for tactic, tactic_info in self.concept2tactic[token].items():
                if any(filtered in tactic for filtered in self.filtered_tactics):
                    continue
                count = tactic_info['count']
                if count > 0:
                    tactic_weights.append((tactic, count))
            
            if tactic_weights:
                tactics, weights = zip(*tactic_weights)
                k = min(blind_num, len(tactic_weights))
                probs = np.array(weights) / sum(weights)
                selected_indices = np.random.choice(len(tactics), size=k, replace=False, p=probs)
                selected_tactics = [tactics[i] for i in selected_indices]
            
            if selected_tactics:
                blind_list.append((token, selected_tactics))
            # else:
                # for new defs, may be empty
                # may consider to add the ps have been proved in test
                # dynamic add the ps
                # raise ValueError(f"No tactics found for token {token}"
        return blind_list
    
    def sample_ps_ids(self, concept_with_tactics: List[Tuple[int, List[str]]], current_ps_id: int = None):
        selected_ps_ids = []
        for token, tactics in concept_with_tactics:
            ps_id_list = []
            for tactic in tactics:
                ps_ids = self.concept2tactic[token][tactic]['ps_ids']
                ps_id = random.choice(ps_ids)

            ## avoid give the real info to the model
                if current_ps_id is not None:
                    if ps_id == current_ps_id:
                        ps_ids = [ps_id for ps_id in ps_ids if ps_id != current_ps_id]
                        if not ps_ids:
                            continue
                    ps_id = random.choice(ps_ids)
                ps_id_list.append(ps_id)
            selected_ps_ids.append((token, ps_id_list))
        return selected_ps_ids

    def state_selection_generate(self, proof_context: ProofContext, next_layer_info: List[ProofInfo], use_origin='mixed', if_use_intuition=True):
        def_id = self.tokenizer.encode(proof_context.theorem_name)

        if def_id is None:
            raise ValueError(f"Def id not found for {proof_context.Name}, if ps_name is not provided, ps_data must be a PSItem")
        ## TODO: give a v file, an init state and the proof's def can be directly generated
        ## maybe idtac? or some show command?
        
        def flatten(lst):
            result = []
            for item in lst:
                if isinstance(item, list):
                    result.extend(flatten(item))
                else:
                    result.append(item)
            return result
        
        state_str_list = []
        glob_tokens_lists = []
        remaining_states = []

        # if index starts from 0, offset will not be considered
        state_num = 0
        for proof_info in next_layer_info:
            if proof_info.remaining_list:
                for remaining in proof_info.remaining_list:
                    for state in remaining.states:
                        hyps_list, goal_str, token_ids = self.get_single_state(state, use_origin=use_origin)
                        glob_tokens_lists.append(token_ids)
                        remaining_states.append(STATE_NODEF_FORMAT.format(hyps='\n'.join(hyps_list), goal=goal_str))

            current_state = proof_info.curr_ps # single_after_state
            assert isinstance(current_state, State)
            
            try:
                hyps_list, goal_str, token_ids = self.get_single_state(current_state, use_origin=use_origin)
                glob_tokens_list = self.get_def(token_ids, token_only=True)
                after_state_str = TACTIC_STATE_FORMAT.format(hyps='\n'.join(hyps_list), goal=goal_str)
            except Exception as e:
                print(e)
                print(proof_info.curr_ps)
                print(current_state)
                print('generate_state error, please check the data, in state selection')
            if remaining_states:
                state_str = SELECT_STATE_FORMAT.format(number=state_num, tactic='. '.join(proof_info.prev_result.tactic_sequence)+'.', state=after_state_str, remaining_goals='\n'.join(remaining_states), proof_trace=proof_info.proof_summary_with_tactic.proof_summary['proof_trace'])
            else:
                state_str = SELECT_STATE_FORMAT.format(number=state_num, tactic='. '.join(proof_info.prev_result.tactic_sequence)+'.', state=after_state_str, remaining_goals='No more remaining goals\n', proof_trace=proof_info.proof_summary_with_tactic.proof_summary['proof_trace'])
            state_str_list.append(state_str)
            glob_tokens_lists.append(glob_tokens_list)
            state_num += 1
        
        glob_tokens = list(set(flatten(glob_tokens_lists)))

        def_data = self.def_table[def_id]
        try:
            content = def_data['origin_context']['content']
        except:
            content = ''
        
        before_state_str, _ = self.generate_state(proof_context.ps_init, mode='state', use_origin=use_origin, extra_tokens=glob_tokens, if_use_intuition=if_use_intuition)
        theorem_str = THEOREM_FORMAT.format(name=proof_context.theorem_name, content=content, state=before_state_str)

        return PS_SELECTION_PROMPT.format(theorem=theorem_str, states='\n'.join(state_str_list))

    async def generate_ps_selection(self, proof_context: ProofContext, next_layer_info: List[ProofInfo], use_origin='mixed', if_use_intuition=True):
        prompt = self.state_selection_generate(proof_context, next_layer_info, use_origin=use_origin, if_use_intuition=if_use_intuition)
        response = await llm_selection_response(prompt)
        if response is None:
            print('selection error, random select 3 states')
            total_states = len(next_layer_info)
            state_list = random.sample(range(total_states), 3)
        else:
            state_list = []
            for i in response:
                if isinstance(i, int):
                    state_list.append(i)
                elif isinstance(i, str):
                    if i.isdigit():
                        state_list.append(int(i))
                    else:
                        match = re.search(r'\d+', i)
                        if match:
                            state_list.append(int(match.group()))
        for i in state_list:
            if i < 0 or i >= len(next_layer_info):
                state_list.remove(i)
        return prompt, response, state_list
    
    def generate_state_explanation(self, state: ps_object_single, if_use_intuition=False, use_origin='mixed'):
        if not isinstance(state, ps_object_single):
            raise ValueError("Invalid state type, must be ps_object_single")
        state_str = self.generate_state(state, mode='tactic', use_origin=use_origin, if_use_intuition=if_use_intuition)
        prompt = EXPLANATION_FORMAT.format(state=state_str)
        return prompt

    def generate_single_strategy(self, brief_strategy: str, state_intuition: Dict, tactic: str):
        state_intuition_str = STATE_INTUITION_FORMAT.format(before=state_intuition['before']['en'], after=state_intuition['after']['en'],tactic=state_intuition['tactic']['en'])
        single_strategy_str = SINGLE_STRATEGY_FORMAT.format(tactic=tactic, brief_strategy=brief_strategy, state_intuition=state_intuition_str)
        return single_strategy_str

    def generate_proof_trace(self, proof_traces: List[ProofTrace]):
        strategy_list = []
        for proof_trace in proof_traces:
            strategy_list.append(self.generate_single_strategy(proof_trace.brief_strategy, proof_trace.state_explanation, proof_trace.tactic))
        proof_trace_str = '\n'.join(strategy_list) if strategy_list else 'Initial state'
        return STRATEGY_SUMMARY_FORMAT.format(proof_trace=proof_trace_str)
    
    async def generate_re_consider(self, fail_result: Union[List[TacticResult],TacticResult], previous_tactic_prompt:str, success_result:List[TacticResult]=None, previous_method_prompt=None, previous_method_response=None, re_con_mode="normal", ft_mode=False):
        if ft_mode:
            if re_con_mode == "normal" and (previous_method_prompt == None or previous_method_response == None):
                raise ValueError("previous_method_prompt and previous_method_response are required for normal mode")
            if re_con_mode == 'normal':
                assert 'Describe your proof strategy' in previous_method_prompt
                context_part = previous_method_prompt.split('Describe your proof strategy',1)[0]
                status_list = []
                for failed in fail_result:
                    status_list.append(TACTIC_STATUS_FORMAT.format(tactic=failed.tactic_sequence[-1], status=failed.error_message))
                method_prompt = GENERATE_METHOD4NEXT_STEP_PROMPT_RECONSIDER.format(previous_method_prompt=context_part, previous_method=previous_method_response, previous_tactic='\n'.join(status_list))
                method_response, _ = await llm_response(method_prompt, ifGen=False)
                flag1 = 'Some hints may help you to understand the proof:'
                flag2 = '=========================='
                context_part = previous_tactic_prompt.split(flag1,1)[0]
                action_part = previous_tactic_prompt.split(flag2,1)[1]
                tactic_prompt = context_part + flag1 + '\n' + method_response + '\n' + flag2 + '\n' + action_part
                tactic_response = await llm_response(tactic_prompt, use_ft_model=True)
                return method_response, tactic_response, tactic_prompt, method_prompt
            if re_con_mode == 'hierarchical':
                error_str = TACTIC_ERROR_FORMAT_NOREASON.format(tactic=fail_result.tactic_sequence[-1], error=fail_result.error_message)
                flag = '=========================='
                context_part = previous_tactic_prompt.split(flag,1)[0]
                action_part = previous_tactic_prompt.split(flag,1)[1]
                tactic_prompt = context_part + '\n' + error_str + '\n' + flag + '\n' + action_part
                tactic_response = await llm_response(tactic_prompt, use_ft_model=True)
                return tactic_response, tactic_prompt
                
        assert '=== Available Actions ===' in previous_tactic_prompt 
        prefix_prompt = previous_tactic_prompt.split('=== Available Actions ===',1)[0]
        if re_con_mode == "hierarchical":
            if len(fail_result.tactic_trace) > 1:
                previous_refine = []
                for trace_single in fail_result.tactic_trace[:-1]:
                    previous_refine.append(TACTIC_ERROR_FORMAT.format(tactic=trace_single.tactic, error=trace_single.error_message, reason=trace_single.reason))
                previous_refinement_attempts = '\n'.join(previous_refine)
            else:
                previous_refinement_attempts = ''
            
            assert fail_result.tactic_trace[-1].tactic == fail_result.tactic_sequence[-1]
            
            prompt = RECONSIDER_HIERARCHICAL_FORMAT.format(before_prompt_tactic=prefix_prompt, current_tactic=fail_result.tactic_sequence[-1], error_message=fail_result.tactic_trace[-1].error_message, previous_refinement_attempts=previous_refinement_attempts, reason=fail_result.tactic_trace[-1].reason)
            response_tactic = await llm_response(prompt, ifGen=True, refine_mode=True)
            return response_tactic
        
        status_list = []
        for success in success_result:
            status_list.append(TACTIC_STATUS_FORMAT.format(tactic=success.tactic_sequence[-1], status='Success'))
        for failed in fail_result:
            status_list.append(TACTIC_STATUS_FORMAT.format(tactic=failed.tactic_sequence[-1], status=failed.error_message))
        prompt = RECONSIDER_TACTIC_PROMPT.format(before_prompt_tactic=prefix_prompt, previous_tactics_and_errors='\n'.join(status_list))
        response_tactic = await llm_response(prompt, ifGen=True, force_tactics=True)
        return prompt, response_tactic["tactics"]
    
    async def generate_note(self, init_states: Union['ps_object_single', 'PSItem'], proof_context: ProofContext, public_notes: List[Tuple[str,str]], current_candidates: List[Tuple[str,str]], if_use_intuition=True, use_origin='mixed'):
        def_id = self.tokenizer.encode(proof_context.theorem_name)
        def_data = self.def_table[def_id]
        try:
            content = def_data['origin_context']['content']
        except:
            content = ''
        
        before_state_str, _ = self.generate_state(init_states, mode='state', use_origin=use_origin, if_use_intuition=if_use_intuition)
        theorem_str = THEOREM_FORMAT.format(name=proof_context.theorem_name, content=content, state=before_state_str)

        public_notes_list = []
        current_candidates_list = []
        public_notes_info_list = []
        current_candidates_info_list = []
        for note_idx, (note, note_info) in enumerate(public_notes):
            public_notes_list.append(NOTE_ITEM_FORMAT.format(id=note_idx, content=note))
            public_notes_info_list.append(f"ID {note_idx}: {note_info}")
        for candidate_idx, (candidate, candidate_info) in enumerate(current_candidates):
            current_candidates_list.append(NOTE_ITEM_FORMAT.format(id=candidate_idx, content=candidate))
            current_candidates_info_list.append(f"ID {candidate_idx}: {candidate_info}")
        
        public_notes_str = '\n'.join(public_notes_list) if public_notes_list else 'Initial state note is empty, do not need to remove any note'
        current_candidates_str = '\n'.join(current_candidates_list)
        available_items_note = '\n'.join(public_notes_info_list) if public_notes_info_list else 'Initial state note is empty, do not need to remove any note'
        available_items_candidate = '\n'.join(current_candidates_info_list)
        
        note_prompt = PUBLIC_NOTE_FORMAT.format(current_state=theorem_str, current_notebook=public_notes_str, new_candidates=current_candidates_str,available_items_note=available_items_note,available_items_candidate=available_items_candidate)
        note_response = await llm_note_response(note_prompt)
        return note_prompt,note_response

    async def reorganize_tactic(self, prompt_tactic: str):
        sign = '======================'
        trancated_prompt_tactic, action_part = prompt_tactic.split(sign,1)
        prompt = REORGANIZE_PROMPT_FORMAT.format(prompt_tactic=trancated_prompt_tactic)
        response = await llm_reorganize_response(prompt)
        prompt_com = response + '\n' + sign + '\n' + action_part
        return prompt_com

    async def generate_def_relation(self, def_item: def_object, logprobs: bool = False, response_num: int = 10, mode='understanding', use_origin='mixed', if_use_intuition=True, if_give_def=False):
        if mode == 'understanding':
            def_format = self.extract_definition(def_item, mode=mode, use_origin=use_origin, if_use_intuition=if_use_intuition, if_give_def=if_give_def)
            if def_format is None:
                return None, None
            prompt = DEFINITION_UNDERSTANDING_PROMPT.format(definition=def_format)
        elif mode == 'relation':
            glob_tokens_items, def_format = self.extract_definition(def_item, mode=mode, use_origin=use_origin, if_use_intuition=if_use_intuition, if_give_def=if_give_def)
            if glob_tokens_items is None:
                return None, None
            sample_glob_ids = random.sample(glob_tokens_items, 2)
            sample_defs_str = '\n'.join([f"ID {i}: {self.def_table[idx]['name']}" for i, idx in enumerate(sample_glob_ids)])
            prompt = DEFINITION_RELATIONSHIP_PROMPT.format(definition=def_format, concepts=sample_defs_str)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        response = await llm_normal(prompt, sample_size=response_num, logprobs=logprobs)
        return prompt, response

    def generate_premise_selection(self, ps_object: ps_object_single, theorem: def_object, use_origin: str = 'mixed', if_use_intuition=True, if_give_def: bool = True):
        if isinstance(ps_object, ps_object_single):
            ps_object = ps_object.to_dict()
        if isinstance(theorem, def_object):
            theorem = theorem.to_dict()
        
        tactic_ids = map(int, ps_object['tactic']['token_ids'].split(','))
        glob_premise_idx = [token for token in tactic_ids if token >= self.glob_start_id and '.' in self.tokenizer.decode(token)]
        current_premise_name = self.tokenizer.decode(glob_premise_idx[0])

        def_str = self.extract_definition(theorem, mode='normal', use_origin=use_origin, if_use_intuition=if_use_intuition, if_give_def=if_give_def).replace('DEFINITION','THEOREM')
        
        if if_give_def:
            plain_prompt = False
        else:
            plain_prompt = True
        
        ps_info = self.generate_state(ps_object, mode='state', use_origin=use_origin, if_use_intuition=if_use_intuition, plain_prompt=plain_prompt)
        if plain_prompt:
            ps_str = ps_info
        else:
            ps_str,_ = ps_info

        premises = []

        premise_list = self.retrieval.retrieve_internal(current_premise_name)
        if premise_list is None:
            return None, None
        
        random.shuffle(premise_list)
        
        for i, premise in enumerate(premise_list):
            if premise['doc_name'] == current_premise_name:
                right_index = i
                break

        for premise_idx, premise in enumerate(premise_list):
            if not if_give_def:
                premises.append(f"ID {premise_idx}: {premise['doc_name']}")
                continue

            def_id = self.tokenizer.encode(premise['doc_name'])
            def_ = self.def_table[def_id]
            if def_['kind'] == 'Primitive' or def_['kind'] == 'Ltac':
                continue

            premises.append(f"ID {premise_idx}: " + 
               (DEFINITION_FORMAT.format(
                    name=def_['name'],
                    content=format_def(def_, use_origin=use_origin),
                    intuition=premise['intuition']) 
                
                if if_use_intuition else 
                
                DEFINITION_FORMAT_NO_INTUITION.format(
                    name=def_['name'],
                    content=format_def(def_, use_origin=use_origin)
                )
            ))
        premise_str = '\n'.join(premises)

        prompt = PREMISE_SELECTION_PROMPT.format(theorem=def_str, state=ps_str, premises=premise_str)
        return prompt, right_index

    async def generate(self, 
                 data,
                 is_init=False, 
                 if_single_after_state=False,   
                 if_use_ft_model=False,
                 plain_prompt=False,
                 ft_mode=False,
                 if_reorganize_concisely=False,
                 **kwargs):
        if 'ablation_config' in kwargs:
            _, _, _, _, if_strategy, _ = kwargs['ablation_config']
        else:
            if_strategy = True

        if not if_strategy:
            prompt_tactic, reteieval_info = await self.generate_com(data, is_init=is_init, ft_mode=ft_mode,
                 if_single_after_state=if_single_after_state, ifGen=True,
                 plain_prompt=plain_prompt, **kwargs)
            response_tactic = await llm_response(prompt_tactic, ifGen=True)
            return GenerateInfo(prompt_tactic=prompt_tactic, response_tactic=response_tactic['tactics'], retrieval_info=reteieval_info)
        
        if plain_prompt:
            prompt_tactic = await self.generate_com(data, is_init=is_init, ft_mode=ft_mode,
                 if_single_after_state=if_single_after_state, 
                 plain_prompt=True, **kwargs)
            
            ### for short name test
            prompt_tactic_list = prompt_tactic.split('\n')
            processed_lines = []
            for l in prompt_tactic_list:
                l_list = l.split(' ')
                l = ' '.join([s.split('.')[-1] if ('.' in s and not s.endswith('.')) else s for s in l_list])
                processed_lines.append(l)
            prompt_tactic = '\n'.join(processed_lines)

            if ft_mode:
                return (None, None, None, prompt_tactic)
            if if_use_ft_model:
                response_tactic = await llm_response(prompt_tactic, ifGen=True, use_ft_model=True)
                return GenerateInfo(prompt_tactic=prompt_tactic, response_tactic=response_tactic)
            else:
                response_tactic = await llm_response(prompt_tactic, ifGen=True, if_reason=False)
                return GenerateInfo(prompt_tactic=prompt_tactic, response_tactic=response_tactic['tactics'])
        
        prompt_method = await self.generate_com(data, is_init=is_init, ifGen=False, ft_mode=ft_mode, if_single_after_state=if_single_after_state, **kwargs)
        response_method, brief_strategy = await llm_response(prompt_method, ifGen=False)
        
        if if_use_ft_model:
            # we use ft-template to generate tactics so ft_mode should be True
            prompt_tactic, reteieval_info = await self.generate_com(data, is_init=is_init, ifGen=True, ft_mode=True, query=response_method, if_single_after_state=if_single_after_state, **kwargs)
            prompt_tactic_reorganize = None
            if if_reorganize_concisely:
                prompt_tactic_reorganize = await self.reorganize_tactic(prompt_tactic)
            response_tactic = await llm_response(prompt_tactic, ifGen=True, use_ft_model=True)
            if prompt_tactic_reorganize:
                return GenerateInfo(prompt_method=prompt_method, 
                                    response_method=response_method, 
                                    prompt_tactic=prompt_tactic, 
                                    prompt_tactic_reorganize=prompt_tactic_reorganize, 
                                    response_tactic=response_tactic, 
                                    brief_strategy=brief_strategy, 
                                    retrieval_info=reteieval_info)
            else:
                return GenerateInfo(prompt_method=prompt_method, 
                                    response_method=response_method, 
                                    prompt_tactic=prompt_tactic, 
                                    response_tactic=response_tactic, 
                                    brief_strategy=brief_strategy, 
                                    retrieval_info=reteieval_info)
        
        prompt_tactic, reteieval_info = await self.generate_com(data, is_init=is_init, ifGen=True, ft_mode=ft_mode, query=response_method, if_single_after_state=if_single_after_state, **kwargs)
        
        if ft_mode:
            prompt_tactic_reorganize = None
            if if_reorganize_concisely:
                prompt_tactic_reorganize = await self.reorganize_tactic(prompt_tactic)
            if prompt_tactic_reorganize:
                return (prompt_method, response_method, (prompt_tactic, prompt_tactic_reorganize), brief_strategy)
            else:
                return (prompt_method, response_method, prompt_tactic, brief_strategy)
        
        if if_reorganize_concisely:
            warnings.warn("Reorganize concisely is only supported for ft-model or ft-data, setting to False")
            if_reorganize_concisely = False
            
        response_tactic = await llm_response(prompt_tactic, ifGen=True)
        if 'tactics' in response_tactic:
            return GenerateInfo(prompt_method=prompt_method, 
                                response_method=response_method, 
                                prompt_tactic=prompt_tactic,
                                response_tactic=response_tactic['tactics'],
                                brief_strategy=brief_strategy,
                                retrieval_info=reteieval_info)
        elif 'info' in response_tactic:
            # raise ValueError("TODO") 
            warnings.warn("No implementation for info, concept or tactics")
            response_tactic = await llm_response(prompt_tactic, ifGen=True, force_tactics=True)
            if 'tactics' in response_tactic:
                return GenerateInfo(prompt_method=prompt_method, 
                                    response_method=response_method, 
                                    prompt_tactic=prompt_tactic,
                                    response_tactic=response_tactic['tactics'],
                                    brief_strategy=brief_strategy,
                                    retrieval_info=reteieval_info)
            else:
                raise ValueError("No tactics found")

    async def generate_com(self, 
                     data: ProofInfo, 
                     use_origin='mixed', 
                     is_init=False,
                     if_single_after_state=False,
                     if_use_intuition=True,
                     plain_prompt=False,
                     ft_mode=False,
                     **kwargs
                     ):
        if 'ablation_config' in kwargs:
            if_def, _, _, _, _, ablation_params = kwargs['ablation_config']
            if ablation_params:
                def_origin = ablation_params['use_origin']
                def_intuition = ablation_params['if_use_intuition']
            else:
                def_origin = use_origin
                def_intuition = if_use_intuition
        else:
            if_def = True
            ablation_params = None
            def_origin = use_origin
            def_intuition = if_use_intuition
        
        # print('==================')
        # print(ablation_params)
        # print(def_origin)
        # print(def_intuition)
        
        if use_origin != 'mixed' and use_origin != 'origin' and use_origin != 'internal':
            warnings.warn("use_origin must be 'mixed', 'origin' or 'internal', invalid value, force mixed now")
            use_origin = 'mixed'

        if isinstance(data.curr_ps, PSItem):
            curr_ps = data.curr_ps.to_dict()
        elif isinstance(data.curr_ps, ps_object_single):
            curr_ps = data.curr_ps.to_dict()['states'][0]
        elif isinstance(data.curr_ps, State):
            curr_ps = data.curr_ps
        
        if is_init:
            if if_single_after_state:
                raise ValueError("if_single_after_state is not allowed for init state")
            if 'content' in curr_ps and 'proofstates' in curr_ps.get('content',{}):
                curr_ps = curr_ps['content']['proofstates'][0]
            elif 'before_state' in curr_ps:
                curr_ps = curr_ps
            else:
                print(curr_ps)
                print('invalid data for init state, must PSItem or ps_object_single')
                raise ValueError("Invalid Data for init state, must PSItem or ps_object_single")
            if plain_prompt:
                return self._plain_prompt(self.generate_state(curr_ps, mode='state', use_origin='origin', plain_prompt=True), ft_mode=ft_mode)
            state_item, glob_tokens_item = self.generate_state(curr_ps, mode='state', use_origin=use_origin, if_use_intuition=if_use_intuition, ablation_params=ablation_params)
        elif if_single_after_state:
            if plain_prompt:
                hyps_list, goal_str, token_ids = self.get_single_state(curr_ps, use_origin='origin')
                return self._plain_prompt(STATE_NODEF_FORMAT.format(hyps='\n'.join(hyps_list), goal=goal_str), ft_mode=ft_mode)
            hyps_list, goal_str, token_ids = self.get_single_state(curr_ps, use_origin=use_origin)
            token_ids = list(set(token_ids))
            try:
                def_content, glob_tokens_item = self.get_def(token_ids, use_origin=def_origin, if_use_intuition=def_intuition)
            except Exception as e:
                print(curr_ps)
                print('get_def error, please check the data, in single after state')
                raise e
            state_item = STATE_FORMAT.format(hyps='\n'.join(hyps_list), goal=goal_str, glob_def=def_content)
        else:
            raise ValueError("TODO")
            # for all after states combined to generate a prompt, here we only need the first.
            # TODO:
            state_item, glob_tokens_item = self.generate_state(curr_ps, mode='proof', use_origin=use_origin)
        
        if not if_def:
            flag = 'Global definitions referenced:'
            state_item = state_item.split(flag)[0]

        if isinstance(state_item, str):
            if glob_tokens_item:
                if isinstance(glob_tokens_item[0], int):
                    return await self.generate_step(state_str=state_item, 
                                            glob_tokens=glob_tokens_item,
                                            use_origin=use_origin,
                                            ft_mode=ft_mode,
                                            plain_prompt=plain_prompt, 
                                            if_use_intuition=if_use_intuition,
                                            proof_summary_with_tactic=data.proof_summary_with_tactic,
                                            **kwargs)
            else:
                return await self.generate_step(state_str=state_item, 
                                            glob_tokens=glob_tokens_item,
                                            use_origin=use_origin,
                                            ft_mode=ft_mode,
                                            plain_prompt=plain_prompt, 
                                            if_use_intuition=if_use_intuition, 
                                            proof_summary_with_tactic=data.proof_summary_with_tactic,
                                            **kwargs)
            
        elif isinstance(state_item, list) and isinstance(glob_tokens_item[0], list):
            raise ValueError("TODO")
            # for all after states combined to generate a prompt, here we only need the first.
            # TODO:
            prompt_list = []
            for state, glob_tokens in zip(state_item, glob_tokens_item):
                prompt_list.append(await self.generate_step(state_str=state, 
                                                      glob_tokens=glob_tokens,
                                                      use_origin=use_origin,
                                                      ft_mode=ft_mode,
                                                      plain_prompt=plain_prompt, 
                                                      if_use_intuition=if_use_intuition, 
                                                      proof_summary_with_tactic=data.proof_summary_with_tactic,
                                                      **kwargs))
            return prompt_list
        else:
            raise ValueError("Invalid state item")

    async def generate_step(self, 
                 state_str: str,
                 glob_tokens: List[int],
                 proof_summary_with_tactic: Tuple[str, List[str]] = None,
                 public_notes: List[Tuple[str,str]] = None,
                 ifGen: bool = True, 
                 ft_mode: bool = False,
                 actual_tactic: str = None,
                 state_encode: bool = False, 
                 current_ps_id: int = None, 
                 query: Union[str, List[str], Dict] = None,
                 current_def: str = None,
                 use_origin: str = 'mixed',
                 plain_prompt: bool = False,
                 simplify_ps: bool = False,
                 if_background: bool = True,
                 if_use_intuition: bool = True,
                 concept_num: int = 3,
                 blind_num: int = 3,
                 **kwargs
                 ) -> str:
        """

        Args:
            ifGen: whether to generate tactics, if True, need to provide retrieval and query
            state_encode: whether to encode the state instead of using llm response to retrieve context
            current_ps_id: current proof state id, avoid to use current state in examples
            public_notes: public notes to be used in the prompt
            query: llm response to retrieve context or state
            current_def: current definition name, avoid to use current def in examples
            use_origin: whether to use origin text or internal text or mixed text, 'origin', 'internal' or 'mixed'
            plain_prompt: whether to generate a simple prompt (without concept examples and retrieval content)
            concept_num: number of concepts to sample
            blind_num: number of tactics to sample for each concept
        """
        if 'ablation_config' in kwargs:
            _, if_retrieve, if_proof_trace, if_public_notes, if_strategy, ablation_params = kwargs['ablation_config']
            if ablation_params:
                all_origin = use_origin
                all_intuition = if_use_intuition
                if ablation_params['ablation_scope'] == 'all':
                    all_origin = ablation_params['use_origin']
                    all_intuition = ablation_params['if_use_intuition']
            else:
                all_origin = use_origin
                all_intuition = if_use_intuition
        else:
            if_retrieve = True
            if_proof_trace = True
            if_public_notes = True
            if_strategy = True
            ablation_params = None
            all_origin = use_origin
            all_intuition = if_use_intuition

        if plain_prompt:
            return self._plain_prompt(state_str, ft_mode=ft_mode)
        
        if if_background:
            if len(glob_tokens) <= concept_num:
                concept_list = glob_tokens
            else:
                flag = True
                concept_list = random.sample(glob_tokens, concept_num)

            concept_with_tactics = self.sample_concept_with_weight(concept_list, blind_num)

            if not concept_with_tactics and flag:
                refined_glob_tokens = [glob_token for glob_token in glob_tokens if glob_token not in concept_list]
                concept_list = random.sample(refined_glob_tokens, concept_num)
                concept_with_tactics = self.sample_concept_with_weight(concept_list, blind_num)
                flag = False
            
            if not concept_with_tactics and not flag:
                concept_examples = []
            else:
                concept_examples = []
                selected_ps_ids = self.sample_ps_ids(concept_with_tactics, current_ps_id)
                for token, ps_id_list in selected_ps_ids:
                    tactic_state_str = ''
                    for ps_id in ps_id_list:
                        ps = self.ps_table[ps_id]
                        tactic_state_str += self.generate_state(ps, mode='tactic', use_origin=use_origin, if_use_intuition=if_use_intuition)
                    concept_example = CONCEPT_FORMAT.format(concept=self.tokenizer.decode(token), tactics=tactic_state_str)
                    concept_examples.append(concept_example)

            concept_examples = ''.join(concept_examples)

            if simplify_ps:
                concept_examples = await self._simplify_ps(concept_examples)
        
        if not if_background and simplify_ps:
            warnings.warn("Simplify ps flag is ignored when background knowledge is not used")
        
        if not if_background:
            top_k = 10
        else:
            top_k = 5

        if if_proof_trace:
            if proof_summary_with_tactic:
                tactic = proof_summary_with_tactic.tactic
                proof_summary_str = proof_summary_with_tactic.proof_summary['proof_trace']
            else:
                proof_summary_str = 'Initial state'
                tactic = ''
            
            if isinstance(tactic, list):
                tactic_seq = '. '.join(tactic) + '.'
            else:
                tactic_seq = tactic

            proof_summary = PROOF_TRACING_FORMAT.format(tactic_seq=tactic_seq, proof_summary=proof_summary_str)
        else:
            proof_summary = ''
        
        if if_public_notes:
            if public_notes:
                public_notes_str = '\n'.join([note for note, _ in public_notes])
            else:
                public_notes_str = ''
            public_note = PUBLIC_NOTES_FORMAT.format(public_notes=public_notes_str)
        else:
            public_note = ''

        if ifGen:
            if self.retrieval is None:
                raise ValueError("Retrieval is None")
            if query is None and (if_strategy == True):
                raise ValueError("Query is None")
            
            if if_retrieve:
                if if_strategy:
                    premises, tactics = self._retrieve_context(query, current_def, state_encode, all_origin, all_intuition, top_k=top_k)
                else:
                    state_encode = True
                    flag = 'Global definitions referenced:'
                    query = state_str.replace(flag, '')
                    premises, tactics = self._retrieve_context(query, current_def, state_encode, all_origin, all_intuition, top_k=top_k)
                premises_str = '\n'.join(premises)
                tactics_str = '\n'.join(tactics)
                retrieve_info = RETRIEVE_INFO_FORMAT.format(premises=premises_str, tactics=tactics_str)
            else:
                retrieve_info = ''
            
            if not if_strategy:
                prompt = GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND_TEST.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info)
                return prompt, (premises, tactics)

            if state_encode:
                if if_background:
                    if ft_mode:
                        prompt = GENERATE_NEXT_ACTION_PROMPT_FT_DATA.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info, concepts=concept_examples, public_notes=public_note, strategy='')
                    else:
                        prompt = GENERATE_NEXT_ACTION_PROMPT.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info, concepts=concept_examples, public_notes=public_note, strategy='')
                else:
                    if ft_mode:
                        prompt = GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND_FT_DATA.format(states=state_str,proof_tracing=proof_summary, retrieve_info=retrieve_info, public_notes=public_note, strategy='')
                    else:
                        prompt = GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info, public_notes=public_note, strategy='')
            else:
                if if_background:
                    if ft_mode:
                        prompt = GENERATE_NEXT_ACTION_PROMPT_FT_DATA.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info, concepts=concept_examples, public_notes=public_note, strategy=STRATEGY_FORMAT.format(hint=query))
                    else:
                        prompt = GENERATE_NEXT_ACTION_PROMPT.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info, concepts=concept_examples, public_notes=public_note, strategy=STRATEGY_FORMAT.format(hint=query))
                else:
                    if ft_mode:
                        prompt = GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND_FT_DATA.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info, public_notes=public_note, strategy=STRATEGY_FORMAT.format(hint=query))
                    else:
                        prompt = GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND.format(states=state_str, proof_tracing=proof_summary, retrieve_info=retrieve_info, public_notes=public_note, strategy=STRATEGY_FORMAT.format(hint=query))
            return prompt, (premises,tactics)
        else:
            if if_background:
                if ft_mode:
                    if not actual_tactic:
                        raise ValueError("Actual tactic is None")
                    return GENERATE_METHOD4NEXT_STEP_PROMPT_FT_DATA.format(states=state_str, proof_tracing=proof_summary, concepts=concept_examples, public_notes=public_note, answer=actual_tactic)
                else:
                    return GENERATE_METHOD4NEXT_STEP_PROMPT.format(states=state_str, proof_tracing=proof_summary, concepts=concept_examples, public_notes=public_note)
            else:
                if ft_mode:
                    if not actual_tactic:
                        raise ValueError("Actual tactic is None")
                    return GENERATE_METHOD4NEXT_STEP_PROMPT_NO_BACKGROUND_FT_DATA.format(states=state_str, proof_tracing=proof_summary, public_notes=public_note, answer=actual_tactic)
                else:
                    return GENERATE_METHOD4NEXT_STEP_PROMPT_NO_BACKGROUND.format(states=state_str, proof_tracing=proof_summary, public_notes=public_note)
    
    async def _simplify_ps(self, concept_examples: str):
        prompt = PS_SIMPLIFY_PROMPT.format(input=concept_examples)
        return await llm_simplify_response(prompt)

    def _plain_prompt(self, state_str: str, ft_mode: bool = False):
        if ft_mode:
            return PLAIN_PROMPT_FT_DATA.format(state=state_str)
        else:
            return PLAIN_PROMPT.format(state=state_str)

    def _retrieve_context(self, query, current_def: str, state_encode: bool, use_origin: str, if_use_intuition: bool, top_k: int = 5):
        premises_dict, tactics_dict = self.retrieval.retrieve(query, top_k=top_k, state_encode=state_encode)
        
        premises = []
        for premise in premises_dict[0]:
            if current_def:
                if premise['doc_name'] in {current_def, current_def.rsplit('_', 1)[0]}:
                    continue
                
            def_id = self.tokenizer.encode(premise['doc_name'])
            def_ = self.def_table[def_id]
            if def_['kind'] == 'Primitive':
                continue
            if if_use_intuition:
                premises.append(DEFINITION_FORMAT.format(
                    name=def_['name'],
                    content=format_def(def_, use_origin=use_origin),
                    intuition=premise['intuition']
                ))
            else:
                premises.append(DEFINITION_FORMAT_NO_INTUITION.format(
                    name=def_['name'],
                    content=format_def(def_, use_origin=use_origin),
                ))

        tactics = []
        for tactic in tactics_dict[0]:
            def_id = self.tokenizer.encode(tactic['doc_name'])
            def_ = self.def_table[def_id]
            intuition = self.retrieval.get_intuition(tactic['doc_name'])
            if if_use_intuition:
                tactics.append(TACTIC_STR_FORMAT.format(
                    name=def_['name'],
                    context=format_def(def_, use_origin=use_origin),
                    intuition=intuition
                ))
            else:
                tactics.append(TACTIC_STR_FORMAT_NO_INTUITION.format(
                    name=def_['name'],
                    context=format_def(def_, use_origin=use_origin),
                ))

        return premises, tactics
    