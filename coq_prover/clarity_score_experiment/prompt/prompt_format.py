from typing import Dict, List
from .prompt import *

class PromptFormatStrategy:

    @staticmethod
    def get_zh_def_prompt(concept_content):
        return ZH_DEF_REQUEST_TEMPLATE.format(concept_content=concept_content)

    @staticmethod
    def get_coq_complete_prompt(item_info, concept_name):
        return COMPLETE_TEMPLATE.format(item_info=item_info, concept_name=concept_name) + FINAL_REQUEST.format(concept_name=concept_name)

    @staticmethod
    def get_coq_equivalence_check_prompt(definition1, definition2):
        return EQUIVALENCE_CHECK_TEMPLATE.format(definition1=definition1, definition2=definition2)
    
    @staticmethod
    def get_field_combination(type_case: str) -> List[str]:
        base_fields = ['name']
        if 'origin' in type_case:
            base_fields.append('origin')
        if 'internal' in type_case:
            base_fields.append('internal')
        if 'intuition' in type_case:
            base_fields.append('intuition')
        return base_fields
    
    @staticmethod
    def format_def_item(def_item: Dict, format_type: str = '') -> Dict[str, str]:
        if format_type == 'tactic_format':
            origin_format = TACTIC_ORIGIN_FORMAT.format(origin=def_item.get('origin', ''))
            name_format = TACTIC_NAME_FORMAT.format(name=def_item['name'])
        else:
            origin_format = ORIGIN_FORMAT.format(origin=def_item.get('origin', ''))
            name_format = NAME_FORMAT.format(name=def_item['name'])
        
        internal_format = INTERNAL_FORMAT.format(internal=def_item.get('internal', '')) if def_item.get('internal') else ''
        intuition_format = INTUITION_FORMAT.format(intuition=def_item.get('intuition', ''))
        
        return {
            'name': name_format,
            'origin': origin_format,
            'internal': internal_format,
            'intuition': intuition_format
        }
    
    @staticmethod
    def format_by_type(def_item: Dict, format_type: str, type_case: str) -> str:
        fields = PromptFormatStrategy.format_def_item(def_item, format_type)
        field_names = PromptFormatStrategy.get_field_combination(type_case)
        content_parts = []
        for field_name in field_names:
            if field_name in fields and fields[field_name]:
                content_parts.append(fields[field_name])
        return '\n'.join(content_parts)
    
    @staticmethod
    def format_concept_parts(item: Dict) -> List[str]:
        return [
            NAME_FORMAT.format(name=item['name']),
            ORIGIN_FORMAT.format(origin=item['origin']),
            INTERNAL_FORMAT.format(internal=item['internal']),
            INTUITION_FORMAT.format(intuition=item['intuition'])
        ]
