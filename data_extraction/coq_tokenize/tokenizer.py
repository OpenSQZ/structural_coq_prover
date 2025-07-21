import json
from typing import List, Union, Dict, Any
from .glob import internal_glob, internal_tactic, signs
from utils import read_json_file
from data_extraction.coq_data.Def_class import def_object, TypeItem
from data_extraction.coq_data.Ps_class import *
import re

new_theorem_start_id = 1000000
global_token_id = 1000

class Tokenizer:
    def __init__(self, tokenizer_path: str = None):
        self.total_tokens = 0
        self.fallback_tokens = 0
        if tokenizer_path is not None:
            self.tokenizer = read_json_file(tokenizer_path)
            self.build_tokenizer_sup()
            self.id_to_token = {v: k for k, v in self.tokenizer.items()}
            self.identifier = None
            self.temp_name = None
            self.internal_tokenizer = {}
        else:
            self.max_local_var = 100
            self.tokenizer = {}
            self.internal_tokenizer = {}
            self.identifier = None
            self.temp_name = None

    def local_var(self):
        tokenizer = {}
        for i in range(1, self.max_local_var + 1):
            tokenizer[f"local_var_{i}"] = i
        return tokenizer
    
    def internal_glob(self, tokenizer):
        for token in internal_glob:
            # some token duplicate in the tokenizer
            if token not in tokenizer:
                tokenizer[token] = len(tokenizer) + 1
        return tokenizer

    def encode(self, text: str, return_global: bool = False):
        if not return_global:
            return self.tokenizer.get(text)
            
        def process_token_single(token: str):
            token = token.strip()
            token_id = self.tokenizer.get(token)
            if token_id is not None:
                return token_id
            
            if "." in token:
                if token.split(".")[-1] == token.split(".")[-2]:
                    token_id = self.tokenizer.get(token)
                    if token_id is not None:
                        return token_id

                    modified_token = ".".join(token.split(".")[:-1])
                    token_id = self.tokenizer.get(modified_token)
                    if token_id is not None:
                        return token_id
                    token_id = self.tokenizer_sup.get(token)
                    if token_id is not None:
                        return token_id
                else:
                    token_id = self.tokenizer.get(token)
                    if token_id is not None:
                        return token_id

                    token_id = self.tokenizer_sup.get(token)
                    if token_id is not None:
                        return token_id
                    
            token_id = self.tokenizer.get(token)
            if token_id is None:
                token_id = self.tokenizer_sup.get(token)

            return token_id
    
        if isinstance(text, str):
            text_list = text.split()
        else:
            raise ValueError("text must be a string")
        
        token_list_origin = [process_token_single(text) for text in text_list]

        token_list = [token for token in token_list_origin if token is not None]
        if return_global:
            token_list = [token for token in token_list if token > global_token_id]
        
        return token_list

    def decode(self, tokens: Union[List[int], List[str], str, int], local_vars: Dict[str, Any] = None):
        if isinstance(tokens, str) and ',' in tokens:
            return self.decode([int(t.strip()) for t in tokens.split(',')], local_vars)
        
        if isinstance(tokens, list):
            has_local_token = any((int(t) if isinstance(t, str) else t) < 100 for t in tokens if t != "<pp>")
        else:
            token_id = int(tokens) if isinstance(tokens, str) else tokens
            has_local_token = token_id < 100
        
        if has_local_token and local_vars is None:
            raise ValueError("Found local token but local_vars dict not provided")

        if isinstance(tokens, (str, int)):
            token_id = int(tokens) if isinstance(tokens, str) else tokens
            if token_id < 100:
                for name, var_info in local_vars.items():
                    if var_info['token_ids'] == str(token_id):
                        return name
                raise ValueError(f"Local variable {token_id} not found in local_vars")
            return self.id_to_token.get(token_id)
        
        return " ".join(
            token if token == "<pp>" else (
                next(name for name, var_info in local_vars.items() 
                    if var_info['token_ids'] == str(int(token) if isinstance(token, str) else token))
                if (int(token) if isinstance(token, str) else token) < 100
                else self.id_to_token.get(int(token) if isinstance(token, str) else token))
            for token in tokens
        )

    def build_tokenizer_sup(self):
        self.tokenizer_sup = {}
        for key, value in self.tokenizer.items():
            if '.' in key:
                parts = key.split('.')
                if len(parts) > 2:
                    first, last = parts[0], parts[-1]
                    middle_parts = parts[1:-1]
                    
                    while middle_parts:
                        middle_parts.pop()
                        new_key = '.'.join([first] + middle_parts + [last])
                        self.tokenizer_sup[new_key] = value

        # with open('./data/tokenizer_sup.json', 'w') as f:
        #     json.dump(self.tokenizer_sup, f, indent=2)

    def add_global_tokens(self, input_data: Union[str, def_object, Dict[str, Any]], new_theorem_mode: bool = False):
        ## like some arguments will override defs or some tactis
        ## fortunately, def will always defined a little bit before
        ## so it will not casue some problem (Need to be checked)
        tokens_to_remove = set()
        new_tokens = {}

        if isinstance(input_data, str):
            with open(input_data, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    name = data.get('name')
                    type_info = data.get('kind')

                    if name:
                        if type_info == 'Ltac':
                            name = name.split('.')[-1]
                            
                         # for some nonstandard tactic def, like tlc defined "=>" as a tactic, we special handle it
                        if name in internal_tactic or name in signs:
                            continue

                        if name in self.internal_tokenizer:
                            tokens_to_remove.add(name)

                        new_tokens[name] = None
        
        elif isinstance(input_data, def_object):
            name = input_data.Name
            if name:
                if input_data.Kind == 'Ltac':
                    name = name.split('.')[-1]

                    # for some nonstandard tactic def, like tlc defined "=>" as a tactic, we special handle it
                if name in internal_tactic or name in signs:
                    pass

                elif name in self.internal_tokenizer:
                    tokens_to_remove.add(name)

                new_tokens[name] = None
        
        elif isinstance(input_data, dict):
            name = input_data.get('name')
            type_info = input_data.get('kind')
            if name:
                if type_info == 'Ltac':
                    name = name.split('.')[-1]
                
                # for some nonstandard tactic def, like tlc defined "=>" as a tactic, we special handle it
                if name in internal_tactic or name in signs:
                    pass

                elif name in self.internal_tokenizer:
                    tokens_to_remove.add(name)

                new_tokens[name] = None
        
        else:
            raise ValueError("Input must be either a jsonl file path, def_object, or dictionary")
        
        rebuilt_tokenizer = {}
        current_id = 1

        for token in self.internal_tokenizer:
            if token not in tokens_to_remove and token not in rebuilt_tokenizer:
                rebuilt_tokenizer[token] = current_id
                current_id += 1
        self.internal_tokenizer = rebuilt_tokenizer

        if len(self.tokenizer) == 0:
            self.tokenizer = {**self.internal_tokenizer}
        else:
            self.tokenizer = {k: v for k, v in self.tokenizer.items() 
                         if k not in self.internal_tokenizer}
            self.tokenizer = {**self.internal_tokenizer, **self.tokenizer}

        current_max_id = max(self.tokenizer.values(), default=-1)
        
        if new_theorem_mode:
            if current_max_id >= new_theorem_start_id:
                start_id = current_max_id+1
            else:
                start_id = max(current_max_id, new_theorem_start_id)
        else:
            start_id = max(current_max_id, global_token_id)
        # internal tokenizer will not up to 1000

        for token in new_tokens:
            if token not in self.tokenizer:
                self.tokenizer[token] = start_id
                start_id += 1
    
        self.build_tokenizer_sup()

    def init_tokenizer(self, input_data: Union[str, def_object, Dict[str, Any]]):
        tokenizer = self.local_var()
        self.internal_tokenizer = self.internal_glob(tokenizer)
        self.add_global_tokens(input_data)

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.tokenizer, f, indent=2)

    def process_token(self, token: str, name: str, local_vars: Dict[str, int]):
        if "." in token:
            if token.split(".")[-1] == token.split(".")[-2]:
                token_id = self.tokenizer.get(token)
                if token_id is not None:
                    return token_id

                modified_token = ".".join(token.split(".")[:-1])
                token_id = self.tokenizer.get(modified_token)
                if token_id is not None:
                    return token_id
                token_id = self.tokenizer_sup.get(token)
                if token_id is not None:
                    return token_id
            else:
                token_id = self.tokenizer.get(token)
                if token_id is not None:
                    return token_id

                token_id = self.tokenizer_sup.get(token)
                if token_id is not None:
                    return token_id

                # pattern = token.rsplit('.', 1)[0] + '.*.' + token.rsplit('.', 1)[1]
                
        token_id = self.tokenizer.get(token)

        if token == name.split(".")[-1]:
            token_id = self.tokenizer.get(name)
            return token_id

        if token_id is not None:
            return token_id
        
        if token in local_vars:
            return local_vars[token]['token_ids']
        
        # for some glob defined Variable, Parameter, etc. fixpoint with its corresponding fixpoint are also global
        if token not in local_vars and self.tokenizer.get(name.rsplit(".", 1)[0] + "." + token):
            return self.tokenizer[name.rsplit(".", 1)[0] + "." + token]
        
        return None
        

    def process_origin(self, origin: Union[str, List[str]], local_vars: Dict[str, int], type_items: Dict[str, TypeItem], name: str, if_tactic: bool = False) -> str:
        if isinstance(origin, str):
            tokens = origin.split()
        elif isinstance(origin, list):
            tokens = origin
        else:
            raise ValueError("Origin must be either a string or a list of strings")

        result_tokens = []

        for token in tokens:
            self.total_tokens += 1
            if "<ker>" in token:
                token1, token2 = token.split('<ker>')
                if self.identifier and self.temp_name and self.temp_name in token1:
                    token1 = self.identifier + token1.split(self.temp_name)[-1]
                if self.identifier and self.temp_name and self.temp_name in token2:
                    token2 = self.identifier + token2.split(self.temp_name)[-1]
                token_id = self.process_token(token1, name, local_vars)
                if token_id is not None:
                    result_tokens.append(token_id)
                    continue
                token_id = self.process_token(token2, name, local_vars)
                if token_id is not None:
                    result_tokens.append(token_id)
                    continue
                token = token1
            else:
                if self.identifier and self.temp_name and self.temp_name in token:
                    token = self.identifier + token.split(self.temp_name)[-1]
                token_id = self.process_token(token, name, local_vars)
                if token_id is not None:
                    result_tokens.append(token_id)
                    continue

            if token in type_items:
                local_vars[token] = {
                    'token_ids': str(len(local_vars) + 1),
                    'type': ''
                }
                result_tokens.append(len(local_vars))
                continue
            
            ## for some nested module, we do string match
            ## it is not a good idea, but include/import module or class or some type would produce some internal new def/constant need more time to fix
            # here is a simple solution
            if '.' in token:
                if name.split('.')[0] not in token:
                    # for some N.pos/ x.xx module name instead of full name
                    parts = name.split('.')[:-1] + token.split('.')
                else:
                    parts = token.split('.')
            else:
                # single token
                if if_tactic:
                    ## for tactic, now Ltac will include some internal tokenizer, so we do not need to process it
                    parts = []
                parts = name.split('.')[:-1] + [token]

            if len(parts) > 2:
                first, last = parts[0], parts[-1]
                middle_parts = parts[1:-1].copy()
                token_id = None

                while middle_parts:
                    middle_parts.pop()
                    new_key = '.'.join([first] + middle_parts + [last])
                    token_id = self.tokenizer_sup.get(new_key)
                    if token_id is not None:
                        result_tokens.append(token_id)
                        break
            
            if token_id:
                continue

            # TODO: This is an issue - theoretically, all tokens should have been processed above
            # since they should either be typed locals or globals. However, we still handle
            # unprocessed tokens here as a fallback
            if token not in local_vars:
                # print(f"Name: {name}")
                # print(f"Token: {token}")
                self.fallback_tokens += 1
                local_vars[token] = {
                    'token_ids': str(len(local_vars) + 1),
                    'type': ''
                }
                result_tokens.append(len(local_vars))
                continue
            # if token not in local_vars:
            #     print(f"Name: {name}")
            #     print(f"Token: {token}")
            #     # print(origin)
            #     continue
            raise ValueError(f"Token {token} not found")

        return ','.join(str(token) for token in result_tokens)

    def process_context_token_pair(self, context_token_pair, local_vars: Dict[str, int], type_items: Dict[str, TypeItem], name: str):
        if context_token_pair and context_token_pair.Origin:
            token_id= self.process_origin(context_token_pair.Origin, local_vars, type_items, name)
            context_token_pair.Token_ids = token_id
    
    def local_tokenizer_def(self, local_vars: Dict[str, Dict[str, Union[int, str]]], type_items: Dict[str, TypeItem]):
        for name, type_item in type_items.items():
            if name in local_vars and type_item.content and type_item.content.processed and not local_vars[name]['type']:
                local_vars[name]['type'] = type_item.content.processed.Token_ids
    
    def local_tokenizer_ps(self, local_vars: Dict[str, Dict[str, Union[int, str]]], item: ps_object_single, type_items: Dict[str, TypeItem]):
        for hyp_name, hyp in item.Before_state.hyps.hyps.items():
            if hyp_name in local_vars and hyp.processed and not local_vars[hyp_name]['type']:
                local_vars[hyp_name]['type'] = hyp.processed.Token_ids
        
        for after_state in item.After_state:
            for hyp_name, hyp in after_state.hyps.hyps.items():
                if hyp_name in local_vars and hyp.processed and not local_vars[hyp_name]['type']:
                    local_vars[hyp_name]['type'] = hyp.processed.Token_ids
        
        for name, type_item in type_items.items():
            if name in local_vars and type_item.content and type_item.content.processed and not local_vars[name]['type']:
                local_vars[name]['type'] = type_item.content.processed.Token_ids

    def process_def(self, item: Union[def_object, Dict[str, Any]]):
        if isinstance(item, def_object):
            def_obj = item
        elif isinstance(item, dict):
            def_obj = def_object.from_dict(item)
        
        if def_obj.local_vars is None:
            def_obj.local_vars = {}
        
        if def_obj.Internal_context:
            if def_obj.Internal_context.content:
                self.process_context_token_pair(def_obj.Internal_context.content.processed, def_obj.local_vars, def_obj.Type , def_obj.Name)
                
            if def_obj.Internal_context.body:
                self.process_context_token_pair(def_obj.Internal_context.body.processed, def_obj.local_vars, def_obj.Type , def_obj.Name)
        
        for name, type_item in def_obj.Type.items():
            if name in def_obj.local_vars:
                if type_item.content:
                    self.process_context_token_pair(type_item.content.processed, def_obj.local_vars, def_obj.Type, def_obj.Name)
        
        self.local_tokenizer_def(def_obj.local_vars, def_obj.Type)
        return def_obj

    def process_state(self, state: State, local_vars: Dict[str, int], type_items: Dict[str, TypeItem], def_name: str):
        for hyp_name, hyp in state.hyps.hyps.items():
            self.process_context_token_pair(hyp.processed, local_vars, type_items, def_name)
        
        if state.goal:
            self.process_context_token_pair(state.goal.processed, local_vars, type_items, def_name)

    def process_single_ps(self, item: ps_object_single, type_items: Dict[str, TypeItem], def_name: str):
        def splitTextToTokens(text):
            parts = re.split(r'(:=|=>|<=|<-|/\\|\\/|\|-|\+|-\s|\s|\(|\)|\[|\]|\{|\}|\||,|~|%|;|:|@|\?|/)', text)
            filtered_parts = [part for part in parts if part != ' ' and part != '']
            
            result = []
            for part in filtered_parts:
                if part and len(part) > 1 and '!' in part and part[0].isdigit():
                    excl_pos = part.find('!')
                    if excl_pos == len(part) - 1:
                        result.extend([part[:excl_pos], "!"])
                    else:
                        result.extend([part[:excl_pos], part[excl_pos+1:]])
                elif part and len(part) > 1 and part[0] == '!':
                    result.extend(['!', part[1:]])
                else:
                    result.append(part)
            return result    

        if item.local_vars is None:
            item.local_vars = {}

        self.process_state(item.Before_state, item.local_vars, type_items, def_name)

        for after_state in item.After_state:
            self.process_state(after_state, item.local_vars, type_items, def_name)
        
        tactic_str = item.Tactic.Name
        tactic = splitTextToTokens(tactic_str)
        token_str = self.process_origin(tactic, item.local_vars, type_items, def_name)
        
        # for some Ml tactic use _ to connect two tokens, we need to split them
        if all(int(t) < 100 for t in token_str.split(",")):
            token_str = self.process_origin([item for x in tactic for item in x.split('_')], item.local_vars, type_items, def_name)

        item.Tactic.token_ids = token_str

        self.local_tokenizer_ps(item.local_vars, item, type_items)

    def process_ps(self, item: Union[PSItem, Dict[str, Any]], def_table: Dict[str, def_object], actual_name: str = None):
        if isinstance(item, dict):
            ps_item = PSItem.from_dict(item)

        else:
            ps_item = item
        # name as the key? may some duplicate name in the def_table??? 
        if not actual_name:
            name = ps_item.Name
        else:
            name = actual_name

        def_obj = def_table.get(name)

        if def_obj is None:
            def_obj = def_table.get(name.split('_')[0])
            # if def_obj is None:
                # print(f"Definition {name} not found in def_table")
                # 305 cannot find
                # raise ValueError(f"Definition {name} not found in def_table")
        if def_obj:
            type_items = {name: TypeItem.from_dict(type_data) for name, type_data in def_obj.get('type').items()}
        else:
            type_items = {}
        
        # if ps_item.Content.ProofStates:
        #     if hasattr(ps_item.Content.ProofStates[0], 'nested_states'):
        #         ps_item.flatten()

        for ps_single in ps_item.Content.ProofStates:
            self.process_single_ps(ps_single, type_items, name)

        return ps_item

    def refined_ps(self, ps_item: PSItem, actual_name: str):
        ## for some manually defined proof, origin part need to be refined
        ## the origin part will be blinded to the temp_file_name
        ps_item.Name = actual_name
        for ps_single in ps_item.Content.ProofStates:
            for hyp_name, hyp in ps_single.Before_state.hyps.hyps.items():
                hyp.processed.Origin = self.decode(hyp.processed.Token_ids, local_vars=ps_single.local_vars)
            ps_single.Before_state.goal.processed.Origin = self.decode(ps_single.Before_state.goal.processed.Token_ids, local_vars=ps_single.local_vars)
            for after_state in ps_single.After_state:
                for hyp_name, hyp in after_state.hyps.hyps.items():
                    hyp.processed.Origin = self.decode(hyp.processed.Token_ids, local_vars=ps_single.local_vars)
                after_state.goal.processed.Origin = self.decode(after_state.goal.processed.Token_ids, local_vars=ps_single.local_vars)

    def process_ps_proof(self, 
                            item: Union[PSItem, Dict[str, Any]] = None, 
                            def_table: Dict[str, def_object] = None, 
                            type_dict: Dict[str, TypeItem] = None, 
                            actual_name: str = None, 
                            txt_file_path: str = None,
                            if_refined_ps: bool = False
                            ):
        if def_table is None and type_dict is None:
            raise ValueError("def_table and type_item cannot be None at the same time")
        
        if isinstance(item, dict):
            ps_item = PSItem.from_dict(item)
        else:
            ps_item = item

        # name as the key? may some duplicate name in the def_table??? 
        if not actual_name:
            name = ps_item.Name
        else:
            name = actual_name

        def_name = None
        def_obj = None

        if not type_dict and def_table:
            def_obj = def_table.get(name)
            if def_obj is None:
                def_id = self.tokenizer_sup.get(name)
                def_name = self.id_to_token.get(def_id)
                def_obj = def_table.get(def_name)
            # if def_obj is None:
                # print(f"Definition {name} not found in def_table")
                # 305 cannot find
                # raise ValueError(f"Definition {name} not found in def_table")
        if def_name:
            identifier = def_name.rsplit('.', 1)[0]
        else:
            identifier = actual_name.rsplit('.', 1)[0]

        self.identifier = identifier
        self.temp_name = txt_file_path.replace('.txt', '').replace('.v', '').rsplit('/', 1)[-1].rsplit('_', 1)[0]

        if type_dict:
            type_items = type_dict
        elif def_obj:
            type_items = {name: TypeItem.from_dict(type_data) for name, type_data in def_obj.get('type').items()}
        else:
            type_items = {}
            
        # if ps_item.Content.ProofStates:
        #     if hasattr(ps_item.Content.ProofStates[0], 'nested_states'):
        #         ps_item.flatten()

        for ps_single in ps_item.Content.ProofStates:
            self.process_single_ps(ps_single, type_items, name)
        
        self.identifier = None
        self.temp_name = None
        if if_refined_ps:
            self.refined_ps(ps_item, actual_name)
        return ps_item