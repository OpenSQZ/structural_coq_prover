from typing import Dict, Optional
from .prompt_format import PromptFormatStrategy

class CompletePromptGenerator:
    def __init__(self, item: Dict, type_case: str):
        self.item = item
        self.type_case = type_case
    
    def _transform_def_list(self, item_list, format_type: str = '') -> str:
        if self.type_case.startswith('no_ref'):
            return ''
        def_item_format_list = []
        for def_item in item_list:
            content = PromptFormatStrategy.format_by_type(def_item, format_type, self.type_case)
            def_item_format_list.append(content)
        return '\n'.join(def_item_format_list)
    
    def _process_before_prompt(self, before_format: str) -> str:
        if self.type_case == 'no_ref_ps_origin':
            item_before_prompt_list = before_format.split('\n')
            return '\n'.join([part for part in item_before_prompt_list if '## Internal' not in part])
        return before_format
    
    def _build_all_prompt(self) -> str:
        # format various information
        global_def = self._transform_def_list(self.item['global_def_extracted_info'])
        premises = self._transform_def_list(self.item['related_premises_extracted_info'])
        tactic = self._transform_def_list(self.item['related_tactic_extracted_info'], format_type='tactic_format')
        # build prompt
        parts = [
            self._process_before_prompt(self.item['before_prompt']),
            'Global definitions referenced:'
        ]
        if global_def: parts.append(global_def)
        parts.append(self.item['middle_prompt'])
        if premises: parts.extend(['=== Related Premises ===', premises])
        if tactic: parts.extend(['=== Related Tactic ===', tactic])
        parts.append(self.item['after_prompt'])
        return '\n'.join(parts)
    
    def _process_base_case(self, before_format):
        item_before_prompt_list = before_format.split('\n')
        item_before_list = []
        for part in item_before_prompt_list:
            if '## Internal' in part:
                continue
            elif '## Origin' in part:
                item_before_list.append(' '.join(p.split('.')[-1] for p in part.split(' ')))
            else:
                item_before_list.append(part)
        return '\n'.join(item_before_list)
    
    def _build_global_def_prompt(self):
        before_format = self.item['before_prompt']
        after_format = self.item['after_prompt']
        if self.type_case.startswith('base'):
            return self._process_base_case(before_format)
        before_format = self._process_before_prompt(before_format)
        def_item_format = self._transform_def_list(self.item['global_def_extracted_info'])
        return before_format + '\n' + 'Global definitions referenced:' + '\n' + def_item_format + '\n' + after_format
    
    def _transform_complete_prompt(self, zh_explanations: Optional[str] = None, case_target: str = 'global_def_only'):
        if zh_explanations:
            # use simple concatenation
            def_item_format = zh_explanations
            before_format = self.item['before_prompt']
            after_format = self.item['after_prompt']
            return before_format + '\n' + 'Global definitions referenced:' + '\n' + def_item_format + '\n' + after_format
        if case_target == 'global_def_only':
            return self._build_global_def_prompt()
        else:
            return self._build_all_prompt()
    
    def _get_concept_name(self) -> str:
        if self.type_case == 'base_simple':
            return self.item['name'].split('.')[-1]
        else:
            return self.item['name']
    
    def _process_def_origin(self, def_origin: str) -> str:
        if self.type_case == 'base_simple':
            def_origin_list = def_origin.split('\n')
            def_origin_list = [' '.join(p.split('.')[-1] for p in def_origin_item.split(' ')) for def_origin_item in def_origin_list]
            return '\n'.join(def_origin_list)
        else:
            return def_origin
    
    def generate_zh_def_prompt(self) -> str:
        concept_parts = PromptFormatStrategy.format_concept_parts(self.item)
        concept_content = '\n'.join(concept_parts)
        return PromptFormatStrategy.get_zh_def_prompt(concept_content=concept_content)
    
    def generate_def_prompt(self, zh_explanations: Optional[str] = None, case_target: str = 'global_def_only') -> str:
        item_info = self._transform_complete_prompt(zh_explanations, case_target)
        concept_name = self._get_concept_name()
        return PromptFormatStrategy.get_coq_complete_prompt(item_info=item_info, concept_name=concept_name)
    
    def generate_equivalence_check_prompt(self, llm_definition: Optional[str] = None) -> str:
        processed_def_origin = self._process_def_origin(self.item['origin'])
        return PromptFormatStrategy.get_coq_equivalence_check_prompt(
            definition1=processed_def_origin, 
            definition2=llm_definition
        )
