from coq_prover.coq_context.proof_generator import *
from coq_prover.coq_context.prompt_gen import PromptGenerator
from coq_prover.coq_context.emb_model import LinqEmbedding
from utils import get_config, read_jsonl_file
import coq_prover.coq_context.llm_method as llm_method
import random
import numpy as np
import os
from tqdm import tqdm
from sklearn.exceptions import ConvergenceWarning
import warnings

# Import shared utilities
from shared_utils import (
    EntropyCalculator, SimilarityCalculator, DataProcessor,
    StatisticsAggregator, FileHandler, process_embedding_analysis,
    process_entropy_metrics
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class SemanticEntropy:
    def __init__(self, config_path, sample_size=100):
        random.seed(42)
        self.logprobs = True
        self.sample_size = sample_size
        self.config = get_config(config_path)
        
        # Initialize shared utilities
        self.entropy_calc = EntropyCalculator()
        self.sim_calc = SimilarityCalculator()
        self.data_processor = DataProcessor()
        self.stats_agg = StatisticsAggregator()
        self.file_handler = FileHandler()
        
        self._init_components()
    
    def _init_components(self):
        self.if_use_intuition = self.config.flags.if_use_intuition
        self.use_origin = self.config.flags.use_origin
        
        
        self.basic_dir = os.path.join(self.config.paths.extra_log_dir, 'uncertainty_metric/semantic_entropy_100')
        self.understanding_log_file = os.path.join(self.basic_dir, 'understanding_log.jsonl')
        self.relation_log_file = os.path.join(self.basic_dir, 'relation_log.jsonl')
        self.understanding_no_def_log_file = os.path.join(self.basic_dir, 'understanding_no_def_log.jsonl')
        self.relation_no_def_log_file = os.path.join(self.basic_dir, 'relation_no_def_log.jsonl')
        
        for log_file in [self.understanding_log_file, self.relation_log_file, 
                        self.understanding_no_def_log_file, self.relation_no_def_log_file]:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.tokenizer = Tokenizer(self.config.paths.tokenizer_path)
        self.def_table = read_jsonl_file(self.config.paths.def_table_path)
        self.def_table = [d for d in self.def_table if d['kind'] in ('Proof', 'Definition','Fixpoint','Inductive')]
        self.random_def_table = random.sample(self.def_table, self.sample_size)
        self.def_table_dict = {item['name']: item for item in self.def_table}

        self.prompt_generator = PromptGenerator(self.config.paths.def_table_path, tokenizer=self.tokenizer)
        self.retrieval_model = LinqEmbedding()

    async def gather_in_batches(self, tasks, batch_size=10):
        results = []
        for i in tqdm(range(0, len(tasks), batch_size)):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
        return results
    
    async def generate_def_understanding(self):
        tasks = [self.prompt_generator.generate_def_relation(ps_item, logprobs=self.logprobs, response_num=50, mode='understanding', use_origin=self.use_origin, if_use_intuition=self.if_use_intuition, if_give_def=True) for ps_item in self.random_def_table]
        results = await self.gather_in_batches(tasks, batch_size=10)
        self.process_result(results, self.understanding_log_file)

    async def generate_def_understanding_no_def(self):
        tasks = [self.prompt_generator.generate_def_relation(ps_item, logprobs=self.logprobs, response_num=50, mode='understanding', use_origin=self.use_origin, if_use_intuition=self.if_use_intuition, if_give_def=False) for ps_item in self.random_def_table]
        results = await self.gather_in_batches(tasks, batch_size=10)
        self.process_result(results, self.understanding_no_def_log_file)
    
    async def generate_def_relation(self):
        tasks = [self.prompt_generator.generate_def_relation(ps_item, logprobs=self.logprobs, response_num=50, mode='relation', use_origin=self.use_origin, if_use_intuition=self.if_use_intuition, if_give_def=True) for ps_item in self.random_def_table]
        results = await self.gather_in_batches(tasks, batch_size=10)
        self.process_result(results, self.relation_log_file)
                    
    async def generate_def_relation_no_def(self):
        tasks = [self.prompt_generator.generate_def_relation(ps_item, logprobs=self.logprobs, response_num=50, mode='relation', use_origin=self.use_origin, if_use_intuition=self.if_use_intuition, if_give_def=False) for ps_item in self.random_def_table]
        results = await self.gather_in_batches(tasks, batch_size=10)
        self.process_result(results, self.relation_no_def_log_file)

    def process_result(self, results, log_file):
        for prompt, result in results:
            if result is not None:
                if self.logprobs:
                    responses = []
                    logprobs = []
                    for response, logprob in result:
                        responses.append(response)
                        logprobs.append(logprob)
                else:
                    responses = result
                
                entry = {
                    'prompt': prompt,
                    'responses': responses
                }

                if self.logprobs:
                    # Use shared utility for entropy calculations
                    entropy_stats = process_entropy_metrics(logprobs)
                    entry.update(entropy_stats)
                
                # Use shared utility for file writing
                self.file_handler.write_jsonl_entry(log_file, entry)

    def statistic_entropy(self):
        self._process_files_with_stats(process_existing=True)
        
    def get_semantic_entropy(self):
        self._process_files_with_stats(process_existing=False)

    def _process_files_with_stats(self, process_existing=False):
        """Process files and calculate statistics using shared utilities"""
        entropy_means = {}
        
        for file in os.listdir(self.basic_dir):
            if process_existing and not file.endswith('_emb.jsonl'):
                continue
            if not process_existing and file.endswith('.jsonl') and not file.endswith('_emb.jsonl'):
                # Process new files
                self._process_single_file(file)
            
            # Calculate statistics for processed files
            if file.endswith('_emb.jsonl'):
                stats = self._calculate_file_statistics(file)
                entropy_means[file] = stats
                self._print_file_statistics(file, stats)

    def _process_single_file(self, file):
        """Process a single file to generate embeddings and entropy metrics"""
        input_path = os.path.join(self.basic_dir, file)
        output_path = os.path.join(self.basic_dir, file.replace('.jsonl', '_emb.jsonl'))
        
        with open(input_path, 'r') as f:
            for line in tqdm(f, desc=f"Processing {file}"):
                entry = json.loads(line)
                responses = entry['responses']

                # Generate embeddings and calculate metrics
                emb = self.retrieval_model.encode(responses)
                embeddings = np.array(emb)
                
                # Use shared utilities for embedding analysis
                embedding_stats = process_embedding_analysis(embeddings)
                
                # Convert embeddings to string format for storage
                emb_str = [','.join(str(item) for item in emb[i]) for i in range(len(emb))]
                entry['emb'] = emb_str
                entry.update(embedding_stats)
                
                # Write processed entry
                self.file_handler.write_jsonl_entry(output_path, entry)

    def _calculate_file_statistics(self, file):
        """Calculate statistics for a processed file using shared utilities"""
        file_path = os.path.join(self.basic_dir, file)
        
        # Collect metrics
        cluster_entropies = []
        semantic_entropies = []
        sim_infos = []
        sim_stds = []
        entropy_metrics = {
            'all_entropies': [],
            'all_stds': [],
            'first_token_entropies': [],
            'first_token_stds': [],
            'five_token_entropies': [],
            'five_token_stds': []
        }
        
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=f"Calculating stats for {file}"):
                entry = json.loads(line)
                
                if self.logprobs:
                    # Aggregate entropy metrics
                    for key in entropy_metrics:
                        if key in entry:
                            avg_value = sum(entry[key]) / len(entry[key]) if entry[key] else 0
                            entropy_metrics[key].append(avg_value)

                cluster_entropies.append(entry['cluster_entropy'])
                semantic_entropies.append(entry['semantic_entropy'])
                sim_stds.append(entry['sim_std'])
                
                # Reconstruct embeddings for similarity calculation
                embs = entry['emb']
                embs = np.array([list(map(float, emb_str.split(','))) for emb_str in embs])
                mean_sim, pairwise_sims = self.sim_calc.basic_info(embs)
                sim_infos.append((mean_sim, pairwise_sims.max(), pairwise_sims.min(), np.median(pairwise_sims)))
        
        # Calculate aggregate statistics
        stats = {
            'mean_cluster_entropy': np.mean(cluster_entropies),
            'std_cluster_entropy': np.std(cluster_entropies),
            'mean_semantic_entropy': np.mean(semantic_entropies),
            'std_semantic_entropy': np.std(semantic_entropies),
            'mean_sim': np.mean([s[0] for s in sim_infos]),
            'max_sim': np.mean([s[1] for s in sim_infos]),
            'min_sim': np.mean([s[2] for s in sim_infos]),
            'median_sim': np.mean([s[3] for s in sim_infos]),
            'mean_sim_std': np.mean(sim_stds)
        }
        
        if self.logprobs:
            for key, values in entropy_metrics.items():
                if values:
                    stats[f'mean_{key}'] = np.mean(values)
        
        return stats

    def _print_file_statistics(self, file, stats):
        """Print statistics for a file using shared formatting"""
        if self.logprobs:
            entropy_info = f", mean all token entropy: {stats.get('mean_all_entropies', 0):.4f}, mean all token std: {stats.get('mean_all_stds', 0):.4f}, mean first token entropy: {stats.get('mean_first_token_entropies', 0):.4f}, mean first token std: {stats.get('mean_first_token_stds', 0):.4f}, mean five token entropy: {stats.get('mean_five_token_entropies', 0):.4f}, mean five token std: {stats.get('mean_five_token_stds', 0):.4f}"
            print(f"{file}{entropy_info}")
        
        print(f"{file} mean cluster entropy: {stats['mean_cluster_entropy']:.4f}, mean semantic entropy: {stats['mean_semantic_entropy']:.4f}, cluster entropy std: {stats['std_cluster_entropy']:.4f}, semantic entropy std: {stats['std_semantic_entropy']:.4f}, mean sim: {stats['mean_sim']:.4f}, max sim: {stats['max_sim']:.4f}, min sim: {stats['min_sim']:.4f}, median sim: {stats['median_sim']:.4f}, mean sim std: {stats['mean_sim_std']:.4f}")