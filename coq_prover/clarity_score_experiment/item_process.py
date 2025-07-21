import json
import random

class ItemInfoProcess:

    # ===== Section delimiters =====
    SEPARATOR = '=========================='
    GLOBAL_DEF_REF_START = 'Global definitions referenced:'
    GLOBAL_DEF_REF_END = '=== Proof Tracing ==='
    RELATED_PREMISES_START = '=== Related Premises ==='
    RELATED_PREMISES_END = '=== Related Tactic ==='
    RELATED_TACTIC_START = '=== Related Tactic ==='
    RELATED_TACTIC_END = '=== Public Notes ==='

    # ===== Field parsing rules =====
    FIELD_MAPPINGS = {
        '## Name:': 'name',
        '## Tactic Name:': 'name',
        '## Internal:': 'internal',
        '## Origin:': 'origin',
        '## Context:': 'origin',
        '## Intuition:': 'intuition'
    }

    REQUIRED_FIELDS = {'name', 'origin', 'intuition'}

    def __init__(self, complete_path: str, extract_mode: str = 'global_def_only', random_seed: int = 42):
        self.complete_path = complete_path
        self.random_seed = random_seed
        self.item_info = self._get_item_info_complete(complete_path)
        self.item_info_list = self._extract_item_info(extract_mode=extract_mode)

    def _get_item_info_complete(self, path: str):
        item_info = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item_info_item = json.loads(line)
                if isinstance(item_info_item, dict) and item_info_item:
                    item_info.append({'theorem_name': item_info_item['theorem_name'], 'prompt': item_info_item['prompt']})
                elif not isinstance(item_info_item, dict):
                    item_info.append({'theorem_name': '', 'prompt': item_info_item})
        return item_info
    
    def _is_item_complete(self, item: dict) -> bool:
        if not item or not self.REQUIRED_FIELDS.issubset(item.keys()):
            return False
        expected_count = 4 if 'internal' in item else 3
        return len(item) == expected_count

    def _extract_section_info(self, section_content: str):
        extracted_info = []
        section_lines = section_content.split('\n')
        current_item = {}
        for line in section_lines:
            line = line.strip()
            for prefix, field_name in self.FIELD_MAPPINGS.items():
                if line.startswith(prefix):
                    if field_name == 'name':
                        if self._is_item_complete(current_item):
                            extracted_info.append(current_item)
                        field_value = line.split(prefix)[1].split('##')[0].strip()
                        current_item = {field_name: field_value}
                    else:
                        field_value = line.split(prefix)[1].split('##')[0].strip()
                        current_item[field_name] = field_value
                    break
        if self._is_item_complete(current_item):
            extracted_info.append(current_item)
        return extracted_info
        
    def _extract_item_info(self, extract_mode: str = 'global_def_only'):
        item_info_list = []
        for item in self.item_info:
            prompt: str = item['prompt']

            before_prompt = prompt.split(self.GLOBAL_DEF_REF_START)[0]
            global_def_section = prompt.split(self.GLOBAL_DEF_REF_START)[1].split(self.GLOBAL_DEF_REF_END)[0]
            global_def_extracted_info = self._extract_section_info(global_def_section)

            result_item = {
                'theorem_name': item['theorem_name'],
                'before_prompt': before_prompt,
                'global_def_extracted_info': global_def_extracted_info,
            }
            
            if extract_mode == 'global_def_only':
                # Global_def extraction mode
                after_prompt = self.GLOBAL_DEF_REF_END + prompt.split(self.GLOBAL_DEF_REF_END)[1].split(self.SEPARATOR)[0]
                result_item['after_prompt'] = after_prompt
            else:
                # Complete extraction mode
                middle_prompt = self.GLOBAL_DEF_REF_END + prompt.split(self.GLOBAL_DEF_REF_END)[1].split(self.RELATED_PREMISES_START)[0]
                related_premises_section = prompt.split(self.RELATED_PREMISES_START)[1].split(self.RELATED_PREMISES_END)[0]
                related_tactic_section = prompt.split(self.RELATED_TACTIC_START)[1].split(self.RELATED_TACTIC_END)[0]
                after_prompt = self.RELATED_TACTIC_END + prompt.split(self.RELATED_TACTIC_END)[1].split(self.SEPARATOR)[0]
                
                result_item.update({
                    'middle_prompt': middle_prompt,
                    'after_prompt': after_prompt,
                    'related_premises_extracted_info': self._extract_section_info(related_premises_section),
                    'related_tactic_extracted_info': self._extract_section_info(related_tactic_section)
                })
            item_info_list.append(result_item)
        return item_info_list

    def process_items(self, start_idx: int, end_idx: int, def_num_per_item: int, case_target: str = 'global_def_only'):
        all_items = []
        item_info_list = self.item_info_list[start_idx:end_idx]
        random.seed(self.random_seed)
        for item_info in item_info_list:
            sample_size = min(len(item_info['global_def_extracted_info']), def_num_per_item)
            sampled_info = random.sample(item_info['global_def_extracted_info'], sample_size)
            
            base_item = {
                'before_prompt': item_info['before_prompt'],
                'after_prompt': item_info['after_prompt'],
                'global_def_extracted_info': item_info['global_def_extracted_info'],
            }
            
            if case_target == 'all':
                base_item.update({
                    'middle_prompt': item_info['middle_prompt'],
                    'related_premises_extracted_info': item_info['related_premises_extracted_info'],
                    'related_tactic_extracted_info': item_info['related_tactic_extracted_info'],
                })
            
            for extracted_info in sampled_info:
                all_items.append({**base_item, **extracted_info})
        
        return all_items
