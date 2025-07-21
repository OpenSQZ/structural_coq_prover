import os
from typing import Dict, List, Optional
from tqdm import tqdm
import argparse
import json
import asyncio
import yaml
from item_process import ItemInfoProcess
from prompt.prompt_gen import CompletePromptGenerator
from llm_service import LLMService
from score import calculate_scores

class ClarityScorePipeline:
    def __init__(self, complete_path: str, save_path: str, random_seed: int = 42, extract_mode: str = 'global_def_only'):
        self.complete_path = complete_path
        self.save_path = save_path
        self.random_seed = random_seed
        self.global_semaphore = asyncio.Semaphore(10)

        self.item_processor = ItemInfoProcess(complete_path, extract_mode=extract_mode, random_seed=random_seed)
        self.llm_service = LLMService()

    async def _get_zh_def_with_name(self, item):
        try:
            prompt_gen = CompletePromptGenerator(item, 'zh_explanation')
            zh_def = await self.llm_service.get_zh_def(prompt_gen.generate_zh_def_prompt())
            return item['name'], zh_def
        except Exception as e:
            print(f"Error getting zh_def: {str(e)}")
            return item['name'], None
        
    async def process_single_content(self, item, case: str, zh_defs_str: Optional[str] = None, case_target: str = 'global_def_only'):
        try:
            prompt_gen = CompletePromptGenerator(item, case)
            def_prompt = prompt_gen.generate_def_prompt(zh_defs_str, case_target)
            llm_def = await self.llm_service.get_llm_definition(def_prompt)
            check_prompt = prompt_gen.generate_equivalence_check_prompt(llm_def)
            result, logprobs = await self.llm_service.check_equivalence_with_logprobs(check_prompt)
            scores = calculate_scores(logprobs) if logprobs else None
            if scores:
                return {
                    'def_prompt': def_prompt,
                    'llm_def': llm_def,
                    'check_prompt': check_prompt,
                    'result': result,
                    'scores': scores,
                }
            return None
        except Exception as e:
            print(f"Error processing {case}: {e}")
            return None
            

    async def process_type_cases(self, item: Dict, case: str, case_target: str = 'global_def_only'):
        try:
            if case == 'zh_explanation':
                zh_def_tasks = [
                    self._get_zh_def_with_name(item)
                    for item in item['global_def_extracted_info']
                ]
                try:
                    zh_def_results = await asyncio.gather(*zh_def_tasks)
                    valid_results = [(name, zh_def) for name, zh_def in zh_def_results if zh_def is not None]
                    zh_defs_str = '\n'.join([f"{name}: {zh_def}" for name, zh_def in valid_results])
                    result = await self.process_single_content(item=item,case=case, zh_defs_str=zh_defs_str, case_target=case_target)
                except Exception as e:
                    print(f"Error in zh_def gathering: {str(e)}")
                    result = None
            else:
                result = await self.process_single_content(
                    item=item,
                    case=case,
                    case_target=case_target
                )
            return result if result else None
        except Exception as e:
            print(f"Error in process_type_cases for {case}: {str(e)}")
            return None

    
    async def process_single_item(self, item: Dict, case: str, case_target: str = 'global_def_only'):
        async with self.global_semaphore:
            result = await self.process_type_cases(item, case, case_target)
            if result:                
                return {
                    'item_name': item['name'],
                    'item_origin': item['origin'],
                    case: result
                }
            return None
        
    async def _save_result(self, save_file_name: str, all_tasks: List[asyncio.Task]):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(os.path.join(self.save_path, save_file_name), 'a', encoding='utf-8') as f:
            for task in tqdm(asyncio.as_completed(all_tasks), total=len(all_tasks), desc='Processing items and types'):
                result = await task
                if result:
                    async with asyncio.Lock():
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f.flush()

    async def run_different_cases(
        self, 
        start_idx: int,
        end_idx: int,
        concurrent_limit: int,
        def_num_per_item: int,
        save_file_name: str,
        cases: List[str],
        case_target: str
    ):
        self.global_semaphore = asyncio.Semaphore(concurrent_limit)
        all_items = self.item_processor.process_items(start_idx, end_idx, def_num_per_item, case_target)
        print(f"Processing {len(all_items)} items from index {start_idx} to {end_idx if end_idx else 'end'}")
        
        all_tasks = [
            asyncio.create_task(self.process_single_item(item, case, case_target))
            for item in all_items
            for case in cases
        ]
        await self._save_result(save_file_name, all_tasks)

if __name__ == "__main__":
    # set arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='clarity_score_experiment/clarity_score_config.yaml', help='set config file')
    parser.add_argument('--start_idx', type=int, help='set start index')
    parser.add_argument('--end_idx', type=int, help='set end index')
    parser.add_argument('--def_num_per_item', type=int, help='set number of definitions per item')
    parser.add_argument('--concurrent_limit', type=int, help='set concurrent limit')
    parser.add_argument('--save_file_name', type=str, help='set save file name')
    parser.add_argument('--cases', type=str, nargs='*', 
                   choices=['no_ref_ps_all', 'no_ref_ps_origin', 'origin_only', 'internal_only', 'intuition_only',
                           'origin_internal', 'origin_intuition', 'internal_intuition', 
                           'origin_internal_intuition', 'zh_explanation', 'base', 'base_simple'],
                   help='set cases list, will override config file')
    parser.add_argument('--case_target', type=str, choices=['global_def_only', 'all'], help='set case target')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)['default']
    
    # process cases and case target
    cases = args.cases or config['cases']
    case_target = args.case_target or config['case_target']
    if case_target == 'all':
        cases = [case for case in cases if case not in config['except_cases']]
        invalid_cases = set(config['except_cases']) & set(cases)
        if invalid_cases:
            raise ValueError(f"Cases {invalid_cases} cannot be used when case_target is 'all'")
    
    pipeline = ClarityScorePipeline(
        complete_path=config['complete_path'], 
        save_path=config['save_path'], 
        random_seed=config['random_seed'], 
        extract_mode=case_target
    )
    
    asyncio.run(pipeline.run_different_cases(
        start_idx=args.start_idx if args.start_idx is not None else config['start_idx'],
        end_idx=args.end_idx or config['end_idx'],
        concurrent_limit=args.concurrent_limit or config['concurrent_limit'],
        def_num_per_item=args.def_num_per_item or config['def_num_per_item'],
        save_file_name=args.save_file_name or config['save_file_name'],
        cases=cases,
        case_target=case_target
    ))