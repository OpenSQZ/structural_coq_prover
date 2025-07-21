from .Def_class import coq_constr
from typing import Optional, List, Tuple, Dict, Any
from enum import Enum
from dataclasses import dataclass

# for trace_tactic_tree, we combine these three types into one
# and we will not classify them in the future
# class TacticType(Enum):
#     Atomic = "[Atomic]"
#     ML = "[ML]" # for plugins
#     Alias = "[Alias]" # for Ltac

@dataclass
class Hyps:
    hyps: Dict[str, coq_constr]
    hyps_name: Dict[str,str]

    @classmethod
    def from_string(cls, lines: List[str]) -> 'Hyps':
        hyps = []
        hyps_names = []
        current_hyps_lines = []
        in_hyps = False
        for line in lines:
            if line.startswith("PS Begin Context"):
                in_hyps = True
                current_hyps_lines.append(line)
            elif line.startswith("PS End Context"):
                in_hyps = False
                current_hyps_lines.append(line)
                hyps_name, (_,hyp) = cls.get_hyp_single(current_hyps_lines)
                hyps_names.append(hyps_name)
                hyps.append(hyp)
                current_hyps_lines = []
            elif in_hyps:
                current_hyps_lines.append(line)
                
        hyps_name_dict = {name: None for name in hyps_names}
        hyps_dict = {name: hyp for name, hyp in zip(hyps_names, hyps)}
        return cls(hyps_name=hyps_name_dict, hyps=hyps_dict)

    @classmethod
    def get_hyp_single(cls, lines: List[str]) -> Tuple[str, coq_constr]:
        hpy_lins = [line.strip() for line in lines[1:-1]]
        if hpy_lins[0].startswith("Name:"):
            name = hpy_lins[0].replace("Name: ", "").strip()
            hpy_lins = hpy_lins[1:]
        else:
            raise ValueError(f"Invalid Hypotheses format: {hpy_lins[0]}")
        return name, coq_constr.from_string(hpy_lins)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hyps": {name: hyp.to_dict() for name, hyp in self.hyps.items()},
            "hyps_name": self.hyps_name
        }
    
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'Hyps':
        return cls(
            hyps={name: coq_constr.from_dict(hyp) for name, hyp in dict_data["hyps"].items()},
            hyps_name=dict_data["hyps_name"]
        )

@dataclass
class State:
    hyps: Hyps
    goal: coq_constr
    Type: Optional[Dict[str, coq_constr]] = None

    @classmethod
    def from_string(cls, lines: List[str]) -> 'State':
        hyps_lines = []
        goal_lines = []
        in_hyps = False
        in_goal = False

        if 'PS Current subgoal Completed' in lines:
            empty_hyps = Hyps(hyps={}, hyps_name={})
            completed_goal = coq_constr.create_completed_goal()
            return cls(hyps=empty_hyps, goal=completed_goal)

        for line in lines:
            if line.startswith("PS Begin Hyps"):
                in_hyps = True
                hyps_lines.append(line)
            elif line.startswith("PS End Hyps"):
                in_hyps = False
                hyps_lines.append(line)
                hyps = Hyps.from_string(hyps_lines)
                hyps_lines = []
            elif line.startswith("PS Begin Goal"):
                in_goal = True
                goal_lines.append(line)
            elif line == "PS End Goal":
                in_goal = False
                goal_lines.append(line)
                goal_lines = [line.strip() for line in goal_lines[1:-1]]
                _, goal = coq_constr.from_string(goal_lines)
                goal_lines = []
            elif in_hyps or in_goal:
                if in_hyps:
                    hyps_lines.append(line)
                else:
                    goal_lines.append(line)

        return cls(hyps=hyps, goal=goal)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hyps": self.hyps.to_dict(),
            "goal": self.goal.to_dict(),
            "type": self.Type.to_dict() if self.Type else None
        }
    
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'State':
        return cls(
            hyps=Hyps.from_dict(dict_data["hyps"]),
            goal=coq_constr.from_dict(dict_data["goal"]),
            Type={name: coq_constr.from_dict(type_data) 
                  for name, type_data in dict_data["Type"].items()} if dict_data.get("Type") else None
        )

@dataclass
class Tactic:
    Name: str
    Parameters: Optional[List[str]] = None
    origin_tactic: Optional[bool] = False # for Ml tactic
    token_ids: str = None

    @classmethod
    def get_origin_tactic(cls, line: str) -> str:
        start = line.find('::') + 2
        end = line.find('@')
        if start == -1 or end == -1 or start > end:
            raise ValueError(f"Invalid tactic name: {line}")
        
        return line[start:end]
    
    @classmethod
    def get_tactic_ps(cls, line: str) -> str:
        start = line.find("<T>") + 3
        end = line.find("</T>")
        if start == -1 or end == -1 or start > end:
            raise ValueError(f"Invalid tactic ps: {line}")
        
        tactic = line[start:end]
        if "PS Begin of ML" in line:
            return cls.get_origin_tactic(tactic)

        return tactic

    # tactic has been combined into one type
    # obsolete

    # ## TODO: 
    # @classmethod
    # def get_tactic(cls, lines: List[str]) -> str:
    #     """
    #     From multiple executing tactics, returns the one that actually changed the proof state.
    #     For ML and Alias tactics that do the same thing, prefer Alias.
    #     Priority order: Atomic > Alias > ML

    #     example:
    #     [ML] Executing ML tactic: <ltac_plugin::auto@0> $1 $2 $3  # this is ignored as the state is the same
    #     [ALIAS] Executing Ltac tactic: right with y
    #     [ML] Executing ML tactic: <ltac_plugin::right_with@0> $1
    #     """
    #     effective_tactics = []
    #     for line in lines:
    #         if line.startswith("[ML]"):
    #             tactic_type = TacticType.ML
    #             tactic = cls.get_origin_tactic(tactic)
    #             effective_tactics.append((tactic_type, tactic))
    #         elif line.startswith("[ALIAS]"):
    #             tactic_type = TacticType.Alias
    #             tactic = line.split("atomic tactic:", 1)[1].strip()
    #             effective_tactics.append((tactic_type, tactic))
    #         elif line.startswith("[ATOMIC]"):
    #             tactic_type = TacticType.Atomic
    #             tactic = line.split("Ltac tactic:", 1)[1].strip()
    #             effective_tactics.append((tactic_type, tactic))
        
    #     return effective_tactics

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.Name,
            "parameters": self.Parameters if self.Parameters else None,
            "token_ids": self.token_ids if self.token_ids else None,
            "origin_tactic": self.origin_tactic
        }
        return result

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'Tactic':
        return cls(
            Name=dict_data["name"],
            Parameters=dict_data.get("parameters"),
            token_ids=dict_data.get("token_ids"),
            origin_tactic=dict_data.get("origin_tactic")
        )

@dataclass
class ps_object_single:
    Before_state: State
    Tactic: Tactic
    After_state: List[State]
    nested_states: List['ps_object_single'] = None
    local_vars: Optional[Dict[str, Any]] = None
    position: str = None
    step_id: Optional[int] = None

    @classmethod
    def from_string(cls, lines: List[str]) -> 'ps_object_single':
        tactic_str = Tactic.get_tactic_ps(lines[0])
        ps_lines = [line.strip() for line in lines[1:-1]]
    
        tactic = Tactic(Name=tactic_str)

        states = []
        current_state_lines = []
        ps_flag = False
        for line in ps_lines:
            if line.startswith("PS Before") or line.startswith("PS After"):
                ps_flag = True
                if current_state_lines:
                    states.append(State.from_string(current_state_lines))
                    current_state_lines = []
                current_state_lines = [line]
            elif ps_flag:
                current_state_lines.append(line)

        if current_state_lines:
            states.append(State.from_string(current_state_lines))

        before_state = states[0]
        after_states = states[1:]

        local_vars = {}
        current_idx = 1
        
        # we know that hyps must be local, so we manually assign the local index to hyps
        # while some hyps may be global, like keyword Variable, Parameter in coq, it is essential a local variable
        for state in states:
            for name in state.hyps.hyps_name:
                if name not in local_vars:
                    local_vars[name] = {
                        'token_ids': str(current_idx), 
                        'type': ''
                    }
                    current_idx += 1
                state.hyps.hyps_name[name] = local_vars[name]['token_ids']

        return cls(
            Before_state=before_state,
            After_state=after_states,
            Tactic=tactic,
            local_vars=local_vars
        )
        
    def flatten(self) -> List['ps_object_single']:
        flattened_states = [self] 

        if self.nested_states:
            for nested in self.nested_states:
                flattened_states.extend(nested.flatten())
        return flattened_states

    def to_dict(self) -> Dict[str, Any]:
        state_dict = {
            "before_state": self.Before_state.to_dict(),
            "tactic": self.Tactic.to_dict(),
            "after_state": [state.to_dict() for state in self.After_state],
            "position": self.position,
            "local_vars": [{
                'name': name,
                'token_ids': data['token_ids'],
                'type': data['type']
            } for name, data in self.local_vars.items()]
        }
        if hasattr(self, 'step_id'):
            state_dict['step_id'] = self.step_id

        all_states = [state_dict]
        if self.nested_states:
            for nested in self.nested_states:
                all_states.extend(nested.to_dict()["states"])
        return {"states": all_states}

    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'ps_object_single':
        local_vars = {}
        if dict_data.get("local_vars"):
            for item in dict_data["local_vars"]:
                local_vars[item['name']] = {
                    'token_ids': item['token_ids'],
                    'type': item['type']
                }
        instance = cls(
                Before_state=State.from_dict(dict_data["before_state"]),
                Tactic=Tactic.from_dict(dict_data["tactic"]),
                After_state=[State.from_dict(state) for state in dict_data["after_state"]],
                Origin_used=dict_data.get("origin_used"),
                local_vars=local_vars
            )
        
        if 'step_id' in dict_data:
            instance.step_id = dict_data['step_id']
            
        return instance

@dataclass
class ps_object:
    ProofStates: List[ps_object_single]

    @classmethod
    def format_ps(cls, lines: List[str], max_depth: float = float('inf')) -> 'ps_object':
        ## TODO: max_depth is designed for the depth of nested tactics
        ## actually, when nested some bugs may occur(e.g. not all nested patterns can be successfully parsed)
        ## now we only use top level tactics, but for trace the tactic tree, we need to use nested tactics
        ## so we need to fix this bug

        def extract_tactic_name(line: str) -> str:
            try:
                if '<T>' in line and '</T>' in line:
                    return line.split('<T>')[1].split('</T>')[0]
                raise ValueError(f"Invalid PS format: {line}")
            except:
                raise ValueError(f"Invalid PS format: {line}")
            
        def find_matching_end(lines: List[str], start_idx: int, tactic_name: str) -> Tuple[int, bool]:
            count = 1
            i = start_idx + 1
            while i < len(lines):
                line = lines[i]
                if f"PS Begin of" in line and f"<T>{tactic_name}</T>" in line:
                    count += 1
                elif f"PS End of" in line and f"<T>{tactic_name}</T>" in line:
                    count -= 1
                    if count == 0:
                        return i, True
                elif f"PS Notice" in line and f"<T>{tactic_name}</T>" in line:
                    count -= 1
                    if count == 0:
                        return i, False
                i += 1
            ## here is only a fallback, if the end is not found, we return the last index
            ## theoretically, the end should be found, the bug needs to be fixed
            ## some nested_proof has been fixed, suppose the current bug is some unknown goals
            # print(lines[-1])
            # print(tactic_name)
            return i, False
            # raise ValueError(f"PS End not found for {tactic_name}")
        
        def find_ps_before_backwards(lines: List[str], start_idx: int, end_idx: int, tactic_name: str) -> int:
            for i in range(end_idx - 1, start_idx, -1):
                if lines[i].startswith("PS Before") and f"<T>{tactic_name}</T>" in lines[i]:
                    return i
            raise ValueError(f"PS Before not found for {tactic_name}")
        
        def process_tactic_block(block_lines: List[str], current_depth: int = 1, current_path: List[Tuple[int, int]] = None) -> ps_object_single:
            # print(block_lines)
            # print('============')
            if not block_lines:
                return None
            if current_path is None:
                current_path = []

            top_level_blocks = []
            i = 0
            while i < len(block_lines):
                if block_lines[i].startswith("PS Begin of"):
                    tactic_name = extract_tactic_name(block_lines[i])
                    if tactic_name:
                        end_idx, is_normal_end = find_matching_end(block_lines, i, tactic_name)
                        if is_normal_end:
                            top_level_blocks.append((i, end_idx, tactic_name))
                        i = end_idx + 1
                        continue
                i += 1

            top_level_tactics = []
            for block_idx, (start_idx, end_idx, tactic_name) in enumerate(top_level_blocks, 1):
                current_level_path = current_path + [(block_idx, len(top_level_blocks))]
                position = "-".join(f"{pos}/{total}" for pos, total in current_level_path)
                # print(position)
                # print(tactic_name)

                before_idx = find_ps_before_backwards(block_lines, start_idx, end_idx, tactic_name)
                
                outer_tactic_lines = []
                outer_tactic_lines.append(block_lines[start_idx])  # Begin
                
                outer_tactic_lines.extend(block_lines[before_idx:end_idx+1])

                outer_state = ps_object_single.from_string(outer_tactic_lines)
                outer_state.position = position

                ## for a linear proof state, when max depth is 1, we should also allow nested tactics when ; is found
                should_process_nested = (
                    current_depth < max_depth or
                    (max_depth == 1 and current_depth == 1 and ';' in tactic_name)
                )
                
                if should_process_nested:
                    nested_content = block_lines[start_idx+1:before_idx]
                    if nested_content:
                        nested_state = process_tactic_block(nested_content, current_depth + 1, current_level_path)
                        if nested_state:
                            outer_state.nested_states = nested_state.ProofStates
                
                top_level_tactics.append(outer_state)

            return cls(ProofStates=top_level_tactics)

        return process_tactic_block(lines)
    
    def flatten(self) -> 'ps_object':
        flattened_states = []
        for ps in self.ProofStates:
            flattened_states.extend(ps.flatten()) 
        
        return ps_object(ProofStates=flattened_states)

    def to_dict(self) -> Dict[str, Any]:
        all_states = []
        for ps in self.ProofStates:
            all_states.extend(ps.to_dict()["states"])
        return {
            "proofstates": all_states
        }
    
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'ps_object':
        states = dict_data["proofstates"]
        states.sort(key=lambda x: len(x["position"].split("-") if x.get("position") else ""))
        
        position_map = {}
        top_level_states = []
        
        for state_data in states:
            local_vars = {}
            if state_data.get("local_vars"):
                for item in state_data["local_vars"]:
                    local_vars[item['name']] = {
                        'token_ids': item['token_ids'],
                        'type': item['type']
                    }
                    
            instance = ps_object_single(
                Before_state=State.from_dict(state_data["before_state"]),
                Tactic=Tactic.from_dict(state_data["tactic"]),
                After_state=[State.from_dict(state) for state in state_data["after_state"]],
                position=state_data.get("position"),
                local_vars=local_vars,
                nested_states=[]
            )
            
            if 'step_id' in state_data:
                instance.step_id = state_data['step_id']
            
            position = state_data.get("position")
            if position:
                position_map[position] = instance
                
                parent_position = "-".join(position.split("-")[:-1])
                if parent_position:
                    parent = position_map[parent_position]
                    parent.nested_states.append(instance)
                else:
                    top_level_states.append(instance)
            else:
                top_level_states.append(instance)
                
        return cls(ProofStates=top_level_states)

@dataclass
class PSItem:
    Name: str
    Content: ps_object
    Vernac_tactics: List[str]
    Tactic_sequence: List[str]=None
    Linear_tactics: List[str]=None

    def get_tactic_list(self, if_decompose: bool = True) -> List[str]:
        # TODO: now only support if_decompose = False
        ## need refine source code to support if_decompose = True
        ## PSItem will have a linear tactic sequence and a vernac tactic sequence
        tactic_list = []
        for state in self.Content.ProofStates:
            if not '-' in state.position:
                if not if_decompose:
                    tactic_list.append(state.Tactic.Name)
                else:
                    if ';' in state.Tactic.Name:
                        if not state.nested_states:
                            raise ValueError("Tactic has been decomposed but no nested states")
                        else:
                            for nested in state.nested_states:
                                tactic_list.extend(nested.Tactic.Name.split(';'))
                    else:
                        tactic_list.append(state.Tactic.Name)
        return tactic_list

    def get_proof_steps(self) -> int:
        for state in self.Content.ProofStates:
            if not state.position or '-' in state.position:
                continue
            _, total = map(int, state.position.split('/'))
            return total
            
    
    def get_proof_state(self) -> ps_object_single:
        ## when use this function, subgoal compeleted has been processed, so easily find the last state
        for state in self.Content.ProofStates:
            if not state.position or '-' in state.position:
                continue
                
            num, total = map(int, state.position.split('/'))
            if num == total:
                return state
        raise ValueError("No current proof state found")
    
    def get_proof_states_normal(self) -> ps_object_single:
        # when use this function, subgoal compeleted has not been processed
        # need to find the remaining goal if the last state is subgoal completed
        last_state = self.get_proof_state()
        if last_state.After_state[0].goal.processed.Origin != 'goalcompleted':
            return last_state.After_state[0], self.backward_arithmetic(find_pos=True)
        else:
            if len(last_state.After_state) > 1:
                raise ValueError("Subgoal completed but more than one state")
            else:
                return self.backward_arithmetic()
    
    def backward_arithmetic(self, find_pos: bool = False) -> Tuple[ps_object_single, Tuple[int, int]]:
        """NEEDTOCHECK"""
        ## return the remaining goal and the number of steps
        ## 1-3-2-1-1-1 the last three 1 are goalcompleted, for this example the answer is (1, 3)
        ## however, when tracing, the last 1 will be regarded as the compelete of the third layer's first goal
        ## it actually finished the second layer's second goal
        ## fortunately, the counter will not be affected. Whether the answer will always be correct?
        state_num = []
        for state in self.Content.ProofStates:
            if '-' in state.position:
                raise ValueError("ps_item has been flattened, error")
            if len(state.After_state) > 1:
                if state.After_state[0].goal.processed.Origin == 'goalcompleted':
                    raise ValueError("Subgoal completed but more than one state")
                else:
                    state_num.append((1, len(state.After_state)))
            else:
                if state.After_state[0].goal.processed.Origin == 'goalcompleted':
                    for i in range(len(state_num) - 1, -1, -1):
                        current, total = state_num[i]
                        if current < total:
                            state_num[i] = (current + 1, total)
                            state_num = state_num[:i+1]
                            break
                    else:
                        raise ValueError("Subgoal completed but no state to back, proof completed?")
                else:
                    state_num.append((1, 1))
        
        if find_pos:
            return (None, None, [(1,1)] + state_num)
        ## here is somehow correct
        ## as it will truncate the list when the branch is finished so index like 3/3 2/2 will not a problem
        for i in range(len(state_num) - 1, -1, -1):
            current, total = state_num[i]
            if current < total or (current == total and current != 1):
                return self.Content.ProofStates[i].After_state[current-1], (i, current-1, [(1,1)] + state_num)
        raise ValueError("No remaining goal found")

    @classmethod
    def from_string(cls, lines: List[str], max_depth: float = float('inf')) -> 'PSItem':
        if not lines:
            raise ValueError("Empty PS block")
            
        first_line = lines[0].strip()
        if not first_line.startswith("ProofState"):
            raise ValueError(f"Invalid PS format: {first_line}")
            
        if "proof of" in first_line:
            name = first_line.split("proof of")[-1].strip()

        vernac_tactics, modified_lines = cls.get_vernac_tactic(lines[1:])
        content = ps_object.format_ps(modified_lines, max_depth)

        PSItem = cls(Name=name, 
                   Content=content, 
                   Vernac_tactics=' '.join(vernac_tactics))
        
        # for those first tactic is wrong, the content is empty
        if not PSItem.Content:
            PSItem.Tactic_sequence=[]
            PSItem.Linear_tactics=[]
            return PSItem
        
        PSItem.Tactic_sequence = PSItem.get_tactic_list(if_decompose=False)
        PSItem.Linear_tactics = []
        # PSItem.Linear_tactics = PSItem.get_tactic_list(if_decompose=True)
        return PSItem

    # TODO: tactic can not be handled properly

    # @classmethod
    # def split_tactic(cls,tactic: str) -> List[str]:
    #     def remove_unnecessary_parentheses(tactic: str) -> str:
    #         return tactic.replace("[", "").replace("]", "").replace("|", "").strip()
    #     if tactic.startswith("(") and tactic.endswith(")"):
    #         return [t.strip() for t in remove_unnecessary_parentheses(tactic[1:-1]).split(";")]
    #     elif ';' not in tactic:
    #         return [remove_unnecessary_parentheses(tactic)]
    #     elif tactic.startswith("(") and not tactic.endswith(")"):
    #         end = tactic.rfind(")")
    #         tactic = tactic[:end+1]
    #         return [t.strip() for t in remove_unnecessary_parentheses(tactic[1:-1]).split(";")]
    #     elif not tactic.startswith("(") and tactic.endswith(")"):
    #         start = tactic.find("(")
    #         tactic = tactic[start:]
    #         return [t.strip() for t in remove_unnecessary_parentheses(tactic[1:-1]).split(";")]
    #     elif '(' not in tactic:
    #         return [t.strip() for t in remove_unnecessary_parentheses(tactic).split(";")]
    #     elif not tactic.startswith("(") and not tactic.endswith(")") and ';' in tactic:
    #         return [t.strip() for t in remove_unnecessary_parentheses(tactic).split(";")]
    #     else:
    #         raise ValueError(f"Invalid tactic format: {tactic}")
    
    @classmethod
    def get_vernac_tactic(cls,content: List[str]) -> Tuple[str, List[str]]:
        tactics = []
        cleaned_content = []
        in_tactic = False
        current_tactic = ""

        for line in content:
            if line.startswith("TACTIC Begin"):
                in_tactic = True
                current_tactic = ""
                continue

            elif line.startswith("TACTIC End"):
                if in_tactic:
                    in_tactic = False
                    tactics.append(current_tactic.strip())
                    current_tactic = ""
                continue
            
            elif in_tactic:
                current_tactic += ' ' + line.strip()

            if not in_tactic:
                cleaned_content.append(line)
            
        return tactics, cleaned_content
    

    def flatten(self) -> 'PSItem':
        return PSItem(
            Name=self.Name,
            Content=self.Content.flatten(),
            Vernac_tactics=self.Vernac_tactics,
            Tactic_sequence=self.Tactic_sequence,
            Linear_tactics=self.Linear_tactics
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.Name,
            "content": self.Content.to_dict(),
            "vernac_tactics": self.Vernac_tactics,
            "tactic_sequence": self.Tactic_sequence if self.Tactic_sequence else [],
            "linear_tactics": self.Linear_tactics if self.Linear_tactics else []
        }
        
    @classmethod
    def from_dict(cls, dict_data: Dict[str, Any]) -> 'PSItem':
        if not dict_data.get("tactic_sequence"):
            PSItem = cls(
                Name=dict_data["name"],
                Content=ps_object.from_dict(dict_data["content"]),
                Vernac_tactics=dict_data["vernac_tactics"],
            )
            PSItem.Tactic_sequence = PSItem.get_tactic_list(if_decompose=False)
            PSItem.Linear_tactics = []
            return PSItem
        else:
            return cls(
                Name=dict_data["name"],
                Content=ps_object.from_dict(dict_data["content"]),
                Vernac_tactics=dict_data["vernac_tactics"],
                Tactic_sequence=dict_data["tactic_sequence"],
                Linear_tactics=dict_data["linear_tactics"]
        )

@dataclass
class ps_table:
    items: Dict[str, PSItem]
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        return {name: item.to_dict() for name, item in self.items.items()}

