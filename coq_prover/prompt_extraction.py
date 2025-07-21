"""
Prompt Extraction Module

This module provides functionality for:
1. Extracting all prompts from proof log files 
2. Extracting prompts filtered by specific depth levels
"""

import random
import json5
import json
import os
from dataclasses import dataclass
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

random.seed(42)

# Constants
NORMAL_GENERATE_PROMPT = """Suggest a list of 10 tactics to try - prefer single atomic tactics over compound ones unless the combination is highly confident. I will provide the compiler's response for each
Your response must be in this json format:
{
    tactics: [tactic1, tactic2, ...]
}
Ensure your response is a valid JSON without any other text.
"""

CURRENT_PROMPT = """Now please respond tactics:\n\n[TACTIC]: """
GLOBAL_DEF_FLAG = '\n\nGlobal definitions referenced:'


@dataclass
class PromptInfo:
    """Data class for storing prompt information"""
    prompt: str
    plain_prompt: str
    theorem_name: str
    tactic_list: list
    depth: int
    status: str

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'prompt': self.prompt,
            'plain_prompt': self.plain_prompt,
            'theorem_name': self.theorem_name,
            'tactic_list': self.tactic_list,
            'depth': self.depth,
            'status': self.status
        }


class LogFileParser:
    """Parser for proof log files"""
    
    @staticmethod
    def safe_json_load(line: str) -> Optional[dict]:
        """Safely load JSON from line with fallback to json5"""
        try:
            return json.loads(line)
        except:
            try:
                return json5.loads(line)
            except:
                return None

    def find_depth_prompt(self, file_path: str, depth: int, all_prompt_mode: bool = False) -> Tuple[any, List]:
        """
        Extract prompts from log file based on mode
        
        Args:
            file_path: Path to log file
            depth: Target depth for filtering (ignored in all_prompt_mode)
            all_prompt_mode: If True, extract all prompts; if False, extract by depth
            
        Returns:
            Tuple of (prompts, tactic_list)
        """
        prompt_list = []
        print(file_path)
        
        with open(file_path, 'r') as f:
            for line in f:
                if 'items_info' not in line:
                    continue
                    
                data = self.safe_json_load(line)
                if not data:
                    continue
                    
                key = list(data.keys())[0]
                items_info = data[key]['items_info']
                
                # Handle different data structures
                if not isinstance(items_info, list):
                    if all_prompt_mode:
                        prompt_list.append(data[key]['prompt_response_info']['prompt_tactic'])
                    else:
                        # Search for specific depth in dictionary format
                        for item_key in items_info:
                            if items_info[item_key]['depth'] == depth:
                                tactic_list = items_info[item_key]['tactics']
                                if len(tactic_list) > 0:
                                    tactic_list = tactic_list[:-1]
                                else:
                                    print(items_info[item_key])
                                    exit()
                                return data[key]['prompt_response_info']['prompt_tactic'], tactic_list
                else:
                    # Handle list format
                    for item in items_info:
                        if all_prompt_mode:
                            prompt_list.append(data[key]['prompt_response_info']['prompt_tactic'])
                        elif item.get('depth') == depth:
                            tactic_list = item.get('tactics', [])
                            if len(tactic_list) > 0:
                                tactic_list = tactic_list[:-1]
                            return data[key]['prompt_response_info']['prompt_tactic'], tactic_list
        
        if all_prompt_mode:
            return prompt_list, []
        
        raise ValueError(f'depth {depth} not found in {file_path}')

    def parse_log_file(self, log_file_path: str) -> Tuple[Dict, Dict]:
        """
        Parse main log file to extract succeed and fail information
        
        Returns:
            Tuple of (succeed_dict, fail_dict)
        """
        fail_dict = {}
        succeed_dict = {}
        
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            i = 0
            
            while i < len(lines):
                line = lines[i]
                
                if 'FAILED' in line or 'SUCCEEDED' in line:
                    theorem_name = line.split(' ')[1].strip()
                    
                    # Extract log file and depth from next lines
                    if i + 2 < len(lines):
                        log_file = lines[i + 1].split('log_file: ')[1].strip()
                        depth = int(lines[i + 2].split('depth: ')[1].strip())
                        
                        entry = {'log_file': log_file, 'theorem_name': theorem_name}
                        
                        if 'FAILED' in line:
                            if depth not in fail_dict:
                                fail_dict[depth] = []
                            fail_dict[depth].append(entry)
                        else:  # SUCCEEDED
                            if depth not in succeed_dict:
                                succeed_dict[depth] = []
                            succeed_dict[depth].append(entry)
                    
                    i += 3
                else:
                    i += 1
        
        return succeed_dict, fail_dict


class PromptProcessor:
    """Processor for prompt formatting and extraction"""
    
    @staticmethod
    def get_plain_prompt(prompt: str) -> str:
        """Convert prompt to plain format by removing internal information"""
        truncated_prompt = prompt.split(GLOBAL_DEF_FLAG)[0]
        lines = truncated_prompt.split('\n')
        filtered_lines = [line for line in lines if "## Internal" not in line]
        return '\n'.join(filtered_lines) + '\n\n' + NORMAL_GENERATE_PROMPT

    def process_depth_based_extraction(self, succeed_dict: Dict, fail_dict: Dict, 
                                     max_depth: int = 10, target_per_depth: int = 10) -> List[PromptInfo]:
        """
        Process prompts based on depth filtering
        
        Args:
            succeed_dict: Dictionary of successful cases
            fail_dict: Dictionary of failed cases
            max_depth: Maximum depth to process
            target_per_depth: Target number of samples per depth
            
        Returns:
            List of PromptInfo objects
        """
        prompt_info_list = []
        parser = LogFileParser()
        
        # Process succeeded cases
        for depth, cases in tqdm(succeed_dict.items(), desc="Processing succeeded cases"):
            if depth > max_depth:
                continue
                
            processed_cases = self._process_cases_for_depth(
                cases, succeed_dict, depth, target_per_depth, parser, 'SUCCEEDED'
            )
            prompt_info_list.extend(processed_cases)
        
        # Process failed cases
        for depth, cases in tqdm(fail_dict.items(), desc="Processing failed cases"):
            if depth > max_depth:
                continue
                
            processed_cases = self._process_cases_for_depth(
                cases, fail_dict, depth, target_per_depth, parser, 'FAILED'
            )
            prompt_info_list.extend(processed_cases)
        
        return prompt_info_list

    def _process_cases_for_depth(self, cases: List, all_dict: Dict, depth: int, 
                               target_len: int, parser: LogFileParser, status: str) -> List[PromptInfo]:
        """Helper method to process cases for a specific depth"""
        random.shuffle(cases)
        
        # Ensure we have enough cases
        if len(cases) < target_len and depth + 2 in all_dict:
            need_len = target_len - len(cases)
            cases.extend(all_dict[depth + 2][:need_len])
        else:
            cases = cases[:target_len]
        
        processed_cases = []
        for case in cases:
            try:
                # Determine actual depth to search for
                search_depth = depth - 2 if depth > 4 else depth
                prompt, tactic_list = parser.find_depth_prompt(case['log_file'], search_depth)
                
                processed_prompt = prompt.replace(CURRENT_PROMPT, NORMAL_GENERATE_PROMPT)
                plain_prompt = self.get_plain_prompt(prompt)
                
                prompt_info = PromptInfo(
                    prompt=processed_prompt,
                    plain_prompt=plain_prompt,
                    theorem_name=case['theorem_name'],
                    tactic_list=tactic_list,
                    depth=depth,
                    status=status
                )
                processed_cases.append(prompt_info)
                
            except Exception as e:
                print(f"Error processing case {case['theorem_name']}: {e}")
                continue
        
        return processed_cases


class AllPromptExtractor:
    """Extractor for all prompts mode"""
    
    def __init__(self, output_dir: str = './data'):
        self.output_dir = output_dir
        self.all_prompt_file = os.path.join(output_dir, 'all_prompt.jsonl')
        self.selected_prompt_file = os.path.join(output_dir, 'selected_prompt.jsonl')
    
    def extract_all_prompts(self, succeed_dict: Dict, fail_dict: Dict) -> None:
        """Extract all prompts from succeed and fail dictionaries"""
        all_prompts = []
        parser = LogFileParser()
        
        # Process succeeded cases
        for _, cases in tqdm(succeed_dict.items(), desc="Extracting from succeeded cases"):
            for case in cases:
                try:
                    prompts, _ = parser.find_depth_prompt(case['log_file'], 0, all_prompt_mode=True)
                    all_prompts.extend(prompts)
                except Exception as e:
                    print(f"Error extracting from {case['log_file']}: {e}")
        
        # Process failed cases
        for _, cases in tqdm(fail_dict.items(), desc="Extracting from failed cases"):
            for case in cases:
                try:
                    prompts, _ = parser.find_depth_prompt(case['log_file'], 0, all_prompt_mode=True)
                    all_prompts.extend(prompts)
                except Exception as e:
                    print(f"Error extracting from {case['log_file']}: {e}")
        
        print(f"Total prompts extracted: {len(all_prompts)}")
        
        # Save all prompts
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.all_prompt_file, 'w') as f:
            for prompt in all_prompts:
                f.write(json.dumps(prompt) + '\n')
    
    def sample_prompts(self, sample_size: int = 1000) -> None:
        """Sample a subset of prompts from the all_prompt file"""
        all_prompts = []
        
        with open(self.all_prompt_file, 'r') as f:
            for line in f:
                all_prompts.append(json.loads(line))
        
        selected_prompts = random.sample(all_prompts, min(sample_size, len(all_prompts)))
        
        with open(self.selected_prompt_file, 'w') as f:
            for prompt in selected_prompts:
                f.write(json.dumps(prompt) + '\n')
        
        print(f"Sampled {len(selected_prompts)} prompts from {len(all_prompts)} total")


def main():
    # Configuration
    LOG_FILE = './test/proof_test.log'
    OUTPUT_FILE = './data/prompt_info_list.jsonl'
    
    parser = LogFileParser()
    succeed_dict, fail_dict = parser.parse_log_file(LOG_FILE)
    
    # Option 1: Extract all prompts (uncomment to use)
    # all_extractor = AllPromptExtractor()
    # all_extractor.extract_all_prompts(succeed_dict, fail_dict)
    # all_extractor.sample_prompts(sample_size=1000)
    
    # Option 2: Extract prompts by depth (current default)
    processor = PromptProcessor()
    prompt_info_list = processor.process_depth_based_extraction(
        succeed_dict, fail_dict, max_depth=10, target_per_depth=10
    )
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        for prompt_info in prompt_info_list:
            f.write(json.dumps(prompt_info.to_dict()) + '\n')
    
    print(f"Processed {len(prompt_info_list)} prompts and saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()