import json
from utils import get_config, read_jsonl_file
from tenacity import retry, stop_after_attempt, wait_fixed
from coq_prover.coq_context.llm_method import truncate_prompt, refine_response, client_reasoning
import asyncio
import json5
import numpy as np
import os
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
import warnings

# Import shared utilities
from shared_utils import (
    EntropyCalculator, SimilarityCalculator, DataProcessor, 
    StatisticsAggregator, FileHandler, process_tactic_similarities
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class UncertaintyMetricExp:
    def __init__(self, config_path, resume=False):
        self.config = get_config(config_path)
        self.prompt_info_path = './data/prompt_info_list.jsonl'
        self.basic_dir = os.path.join(self.config.paths.extra_log_dir, 'uncertainty_metric/logits_entropy')
        os.makedirs(self.basic_dir, exist_ok=True)
        self.output_path = os.path.join(self.basic_dir, 'output.jsonl')
        self.output_plain_path = os.path.join(self.basic_dir, 'output_plain.jsonl')
        self.prompt_info_list = read_jsonl_file(self.prompt_info_path)
        self.max_retry = 3
        self.response_num = 100
        
        # Initialize shared utilities
        self.entropy_calc = EntropyCalculator()
        self.sim_calc = SimilarityCalculator()
        self.data_processor = DataProcessor()
        self.stats_agg = StatisticsAggregator()
        self.file_handler = FileHandler()
        
        if resume:
            self._resume_from_output()

    def _resume_from_output(self):
        with open(self.output_path, 'r') as f:
            lines = f.readlines()
            last_dict = json.loads(lines[-1])
            processed_num = len(lines)
            assert last_dict['theorem_name'] == self.prompt_info_list[processed_num-1]['theorem_name']
            self.prompt_info_list = self.prompt_info_list[processed_num:]

    async def generate_tactic_list(self):
        for prompt_info in tqdm(self.prompt_info_list, total=len(self.prompt_info_list)):
            tasks = [self.mini_llm_generate(prompt_info['prompt']) for _ in range(self.response_num)]
            results = await asyncio.gather(*tasks)
 
            tasks = [self.mini_llm_generate(prompt_info['plain_prompt']) for _ in range(self.response_num)]
            results_plain = await asyncio.gather(*tasks)
            self.process_results(results, results_plain, prompt_info, self.output_path)
    
    def statistic_entropy(self):
        all_info = []   
        with open(self.output_path, 'r') as f:
            for line in tqdm(f, total=len(self.prompt_info_list)):
                item = json.loads(line)
                depth = item['depth']
                status = item['status']
                tactics = self.data_processor.flatten(item['info']['tactic_responses'])
                tactics_plain = self.data_processor.flatten(item['plain_info']['tactic_responses'])
                
                if not isinstance(tactics[0], str) or not isinstance(tactics_plain[0], str):
                    print(tactics[0])
                    print(tactics_plain[0])
                    print('tactic_responses is not a list of strings')
                    continue

                # Use shared utilities for entropy calculations
                mean_entropy = self.data_processor.transform_response_info_to_mean(item['info']['entropy'])
                mean_std_entropy = self.data_processor.transform_response_info_to_mean(item['info']['std'])
                mean_entropy_meaningful = self.data_processor.transform_response_info_to_mean(item['info']['entropy_meaningful'])
                mean_std_entropy_meaningful = self.data_processor.transform_response_info_to_mean(item['info']['std_meaningful'])
                
                mean_entropy_plain = self.data_processor.transform_response_info_to_mean(item['plain_info']['entropy'])
                mean_std_entropy_plain = self.data_processor.transform_response_info_to_mean(item['plain_info']['std'])
                mean_entropy_meaningful_plain = self.data_processor.transform_response_info_to_mean(item['plain_info']['entropy_meaningful'])
                mean_std_entropy_meaningful_plain = self.data_processor.transform_response_info_to_mean(item['plain_info']['std_meaningful'])
                
                # Use shared utilities for similarity calculations
                mean_similarity, std_similarity, n_clusters = process_tactic_similarities(tactics)
                if mean_similarity > 80:
                    print(tactics)
                    print(item['theorem_name'])
                mean_similarity_plain, std_similarity_plain, n_clusters_plain = process_tactic_similarities(tactics_plain)

                all_info.append({
                    'depth': depth,
                    'status': status,
                    'n_clusters': n_clusters,
                    'n_clusters_plain': n_clusters_plain,
                    'mean_entropy': mean_entropy,
                    'mean_std_entropy': mean_std_entropy,
                    'mean_entropy_meaningful': mean_entropy_meaningful,
                    'mean_std_entropy_meaningful': mean_std_entropy_meaningful,
                    'mean_entropy_plain': mean_entropy_plain,
                    'mean_std_entropy_plain': mean_std_entropy_plain,
                    'mean_entropy_meaningful_plain': mean_entropy_meaningful_plain,
                    'mean_std_entropy_meaningful_plain': mean_std_entropy_meaningful_plain,
                    'mean_similarity': mean_similarity,
                    'std_similarity': std_similarity,
                    'mean_similarity_plain': mean_similarity_plain,
                    'std_similarity_plain': std_similarity_plain,
                })
        
        # Write results using shared utility
        for item in all_info:
            self.file_handler.write_jsonl_entry(
                self.output_path.replace('.jsonl', '_statistic.jsonl'), 
                item, 
                mode='w' if item == all_info[0] else 'a'
            )
        
        self.log_info(all_info)
    
    def log_info(self, info, mode=''):
        if mode == 'resume':
            info = self.file_handler.read_jsonl_with_progress(
                self.output_path.replace('.jsonl', '_statistic.jsonl'),
                "Loading statistics"
            )

        # Use shared utilities for grouping
        success_info, fail_info = self.stats_agg.group_by_status(info)
        depth_groups = self.stats_agg.group_by_depth(info)

        # Define metrics to aggregate
        metric_names = [
            'n_clusters', 'n_clusters_plain', 'mean_entropy', 'mean_entropy_plain',
            'mean_std_entropy', 'mean_std_entropy_plain', 'mean_entropy_meaningful',
            'mean_entropy_meaningful_plain', 'mean_std_entropy_meaningful',
            'mean_std_entropy_meaningful_plain', 'mean_similarity', 'std_similarity',
            'mean_similarity_plain', 'std_similarity_plain'
        ]

        # Overall statistics
        overall_stats = self.stats_agg.aggregate_metrics(info, metric_names)
        self._print_metrics("Overall", overall_stats)

        # Success/Fail statistics
        if success_info:
            success_stats = self.stats_agg.aggregate_metrics(success_info, metric_names)
            self._print_metrics("Success", success_stats)

        if fail_info:
            fail_stats = self.stats_agg.aggregate_metrics(fail_info, metric_names)
            self._print_metrics("Fail", fail_stats)

        # Depth-based statistics
        for depth in sorted(depth_groups.keys()):
            print(f"\n=== Depth {depth} ===")
            depth_info = depth_groups[depth]
            depth_stats = self.stats_agg.aggregate_metrics(depth_info, metric_names)
            print(f"Total samples: {len(depth_info)}")
            self._print_metrics(f"Depth {depth}", depth_stats)

            # Success/Fail within depth
            depth_success, depth_fail = self.stats_agg.group_by_status(depth_info)
            if depth_success:
                print(f"\nSuccess samples: {len(depth_success)}")
                depth_success_stats = self.stats_agg.aggregate_metrics(depth_success, metric_names)
                self._print_key_metrics("Success", depth_success_stats)

            if depth_fail:
                print(f"\nFail samples: {len(depth_fail)}")
                depth_fail_stats = self.stats_agg.aggregate_metrics(depth_fail, metric_names)
                self._print_key_metrics("Fail", depth_fail_stats)
    
    def _print_metrics(self, label, stats):
        """Print metrics in organized format"""
        print(f"{label} Statistics:")
        print(f"n_clusters: {stats.get('mean_n_clusters', 0):.4f}, n_clusters_plain: {stats.get('mean_n_clusters_plain', 0):.4f}")
        print(f"mean_entropy: {stats.get('mean_mean_entropy', 0):.4f}, mean_entropy_plain: {stats.get('mean_mean_entropy_plain', 0):.4f}")
        print(f"mean_std_entropy: {stats.get('mean_mean_std_entropy', 0):.4f}, mean_std_entropy_plain: {stats.get('mean_mean_std_entropy_plain', 0):.4f}")
        print(f"mean_entropy_meaningful: {stats.get('mean_mean_entropy_meaningful', 0):.4f}, mean_entropy_meaningful_plain: {stats.get('mean_mean_entropy_meaningful_plain', 0):.4f}")
        print(f"mean_std_entropy_meaningful: {stats.get('mean_mean_std_entropy_meaningful', 0):.4f}, mean_std_entropy_meaningful_plain: {stats.get('mean_mean_std_entropy_meaningful_plain', 0):.4f}")
        print(f"mean_similarity: {stats.get('mean_mean_similarity', 0):.4f}, mean_similarity_plain: {stats.get('mean_mean_similarity_plain', 0):.4f}")
        print(f"std_similarity: {stats.get('mean_std_similarity', 0):.4f}, std_similarity_plain: {stats.get('mean_std_similarity_plain', 0):.4f}")

    def _print_key_metrics(self, label, stats):
        """Print key metrics for success/fail analysis"""
        print(f"{label}_n_clusters: {stats.get('mean_n_clusters', 0):.4f}, {label}_n_clusters_plain: {stats.get('mean_n_clusters_plain', 0):.4f}")
        print(f"{label}_entropy: {stats.get('mean_mean_entropy', 0):.4f}, {label}_entropy_plain: {stats.get('mean_mean_entropy_plain', 0):.4f}")
        print(f"{label}_entropy_meaningful: {stats.get('mean_mean_entropy_meaningful', 0):.4f}, {label}_entropy_meaningful_plain: {stats.get('mean_mean_entropy_meaningful_plain', 0):.4f}")
        print(f"{label}_similarity: {stats.get('mean_mean_similarity', 0):.4f}, {label}_similarity_plain: {stats.get('mean_mean_similarity_plain', 0):.4f}")
    
    def process_results(self, results, results_plain, prompt_info, output_path):
        tactic_batch = []
        logprobs_batch = []
        entropy_batch = []
        std_batch = []
        entropy_meaningful_batch = []
        std_meaningful_batch = []

        tactic_batch_plain = []
        logprobs_batch_plain = []
        entropy_batch_plain = []
        std_batch_plain = []
        entropy_meaningful_batch_plain = []
        std_meaningful_batch_plain = []
        
        for (tactic_list, response_logprobs), (tactic_list_plain, response_logprobs_plain) in zip(results, results_plain):
            if tactic_list is None or tactic_list_plain is None:
                continue
            
            # Use shared utilities for entropy calculations
            entropy_meaningful, std_meaningful = self.logprobs_to_entropy_meaningful_tokens(response_logprobs)
            info = [self.entropy_calc.logprobs_to_entropy(token_logprobs.top_logprobs) for token_logprobs in response_logprobs]
            entropy = [item[0] for item in info]
            std = [item[1] for item in info]
            
            tactic_batch.append(tactic_list)
            logprobs_batch.append(self.data_processor.transform_logprobs_to_dict(response_logprobs))
            entropy_batch.append(entropy)
            std_batch.append(std)
            entropy_meaningful_batch.append(entropy_meaningful)
            std_meaningful_batch.append(std_meaningful)

            entropy_meaningful_plain, std_meaningful_plain = self.logprobs_to_entropy_meaningful_tokens(response_logprobs_plain)
            info_plain = [self.entropy_calc.logprobs_to_entropy(token_logprobs.top_logprobs) for token_logprobs in response_logprobs_plain]
            entropy_plain = [item[0] for item in info_plain]
            std_plain = [item[1] for item in info_plain]
            
            tactic_batch_plain.append(tactic_list_plain)
            logprobs_batch_plain.append(self.data_processor.transform_logprobs_to_dict(response_logprobs_plain))
            entropy_batch_plain.append(entropy_plain)
            std_batch_plain.append(std_plain)
            entropy_meaningful_batch_plain.append(entropy_meaningful_plain)
            std_meaningful_batch_plain.append(std_meaningful_plain)

        # Prepare output data
        info_dict = {
            'tactic_responses': tactic_batch,
            'logprobs': logprobs_batch,
            'entropy': entropy_batch,
            'std': std_batch,
            'entropy_meaningful': entropy_meaningful_batch,
            'std_meaningful': std_meaningful_batch,
        }
        plain_info_dict = {
            'tactic_responses': tactic_batch_plain,
            'logprobs': logprobs_batch_plain,
            'entropy': entropy_batch_plain,
            'std': std_batch_plain,
            'entropy_meaningful': entropy_meaningful_batch_plain,
            'std_meaningful': std_meaningful_batch_plain,
        }
        
        prompt_info['info'] = info_dict
        prompt_info['plain_info'] = plain_info_dict
        
        # Use shared utility for file writing
        self.file_handler.write_jsonl_entry(output_path, prompt_info)

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    async def mini_llm_call_entropy(self, prompt):
        long_context, prompt = truncate_prompt(prompt)
        response = await asyncio.wait_for(
            client_reasoning.chat.completions.create(
                    model="",
                    logprobs=True,
                    top_logprobs=20,
                    messages=[
                {"role": "system", "content": "You are an expert in Coq formal proof system."},
                {"role": "user", "content": prompt},
            ],
                stream=False
            ),
            timeout=300
        )
        response_content = response.choices[0].message.content
        response_logprobs = response.choices[0].logprobs.content

        return response_content, response_logprobs

    async def mini_llm_generate(self, prompt):
        response_content, response_logprobs = await self.mini_llm_call_entropy(prompt)
        
        for _ in range(self.max_retry):
            response_content = refine_response(response_content)
            try:
                json_response = json.loads(response_content)
                if 'tactics' in json_response and isinstance(json_response['tactics'][0], str):
                    return json_response['tactics'], response_logprobs
            except:
                print(response_content)
                try:
                    json_response = json5.loads(response_content)
                    if 'tactics' in json_response and isinstance(json_response['tactics'][0], str):
                        return json_response['tactics'], response_logprobs
                except:
                    response_content, response_logprobs = await self.mini_llm_call_entropy(prompt + '\nEnsure your response contains the "tactics" field.')
        
        print('--------------------------------')
        print('this prompt can not generate a valid json response')
        print(prompt)
        print(response_content)
        return None, None
    
    def logprobs_to_entropy_meaningful_tokens(self, logprobs):
        """Calculate entropy for meaningful tokens using shared utilities"""
        meaningful_tokens = self.data_processor.get_meaningful_token(logprobs)
        info = [self.entropy_calc.logprobs_to_entropy(token_logprobs.top_logprobs) for token_logprobs in meaningful_tokens]
        entropy_list = [item[0] for item in info]
        std_list = [item[1] for item in info]
        return entropy_list, std_list