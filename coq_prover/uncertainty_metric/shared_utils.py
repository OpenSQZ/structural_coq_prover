"""
Shared utilities for uncertainty metric calculations.

This module contains common functionality used by both exp.py and semantic_entropy.py
to eliminate code duplication while preserving the original logic.
"""

import json
import numpy as np
import warnings
from rapidfuzz import fuzz, distance
from rapidfuzz.distance import Levenshtein
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.exceptions import ConvergenceWarning
from scipy.special import softmax
from tqdm import tqdm

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class EntropyCalculator:
    """Shared entropy calculation utilities"""
    
    @staticmethod
    def logprobs_to_entropy(top_logprobs):
        """Calculate entropy from log probabilities"""
        logps = np.array([item.logprob for item in top_logprobs])
        ps = np.exp(logps - np.max(logps))  # Prevent overflow
        ps = ps / ps.sum()
        std = np.std(logps)
        entropy = -np.sum(ps * np.log(ps + 1e-8))
        return float(entropy), float(std)
    
    @staticmethod
    def semantic_entropy(embeddings):
        """Calculate semantic entropy from embeddings"""
        # embeddings: numpy array, shape (n, d)
        # 1. Calculate mean vector
        mean_vec = np.mean(embeddings, axis=0)
        # 2. Calculate cosine similarity with mean
        sims = embeddings @ mean_vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(mean_vec) + 1e-8)
        # 3. Softmax to get probability distribution
        probs = softmax(sims)
        # 4. Calculate entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy
    
    @staticmethod
    def cluster_entropy(embeddings, k_min=2, k_max=10):
        """Calculate entropy based on optimal clustering"""
        best_score = -1
        best_k = k_min
        best_labels = None
        
        for k in range(k_min, min(k_max, len(embeddings)) + 1):
            kmeans = KMeans(n_clusters=k).fit(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = kmeans.labels_
        
        # Calculate entropy
        labels, counts = np.unique(best_labels, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return entropy, best_k


class SimilarityCalculator:
    """Shared similarity calculation utilities"""
    
    @staticmethod
    def tactic_list_similarity(tactic_list):
        """Calculate mean and std of pairwise similarities in tactic list"""
        n = len(tactic_list)
        upper_similarity = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                similarity = fuzz.ratio(tactic_list[i], tactic_list[j])
                upper_similarity[i][j] = similarity

        upper_triangle_no_diag = upper_similarity[np.triu_indices(n, k=1)]
        return np.mean(upper_triangle_no_diag), np.std(upper_triangle_no_diag)
    
    @staticmethod
    def tactic_list_cluster(tactic_list):
        """Find optimal number of clusters for tactic list"""
        n = len(tactic_list)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    distance_matrix[i][j] = 0
                else:
                    dist = Levenshtein.distance(tactic_list[i], tactic_list[j])
                    distance_matrix[i][j] = dist
                    distance_matrix[j][i] = dist

        best_n_clusters = 1
        best_silhouette = -1

        for n_clusters in range(2, 30):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_n_clusters = n_clusters

        return best_n_clusters
    
    @staticmethod
    def basic_info(embs):
        """Calculate basic similarity information for embeddings"""
        norm_embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
        sim_matrix = np.dot(norm_embs, norm_embs.T)
        n = sim_matrix.shape[0]
        mean_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))
       
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]

        return mean_sim, pairwise_sims


class DataProcessor:
    """Shared data processing utilities"""
    
    @staticmethod
    def flatten(data_list):
        """Recursively flatten nested lists"""
        result = []
        for item in data_list:
            if isinstance(item, list):
                result.extend(DataProcessor.flatten(item))
            else:
                result.append(item)
        return result
    
    @staticmethod
    def transform_response_info_to_mean(info_list):
        """Transform nested response info to mean values"""
        if not isinstance(info_list, list):
            return info_list
        
        means = []
        for item in info_list:
            if isinstance(item, list):
                means.append(DataProcessor.transform_response_info_to_mean(item))
            else:
                means.append(item)
        return np.mean(means)
    
    @staticmethod
    def transform_logprobs_to_dict(logprobs):
        """Transform logprobs to dictionary format"""
        return [item.model_dump() for item in logprobs]
    
    @staticmethod
    def get_meaningful_token(content):
        """Extract meaningful tokens from content"""
        tactic_begin_idx = 0
        tactic_end_idx = len(content)
        
        for i, item in enumerate(content):
            if '[' in item.token:
                tactic_begin_idx = i + 1
                break
        
        for i, item in enumerate(content[::-1]):
            if ']' in item.token:
                tactic_end_idx = len(content) - (i + 1)
                break
        
        meaningful_tokens = []
        for item in content[tactic_begin_idx:tactic_end_idx]:
            if not item.token.strip() or item.token.strip() in ['\'', '\"', ':"', '",', ':', ',']:
                continue
            meaningful_tokens.append(item)
        return meaningful_tokens


class StatisticsAggregator:
    """Shared statistics aggregation utilities"""
    
    @staticmethod
    def aggregate_metrics(data_list, metric_names):
        """Aggregate multiple metrics from data list"""
        aggregated = {}
        
        for metric in metric_names:
            values = [item.get(metric, 0) for item in data_list if metric in item]
            if values:
                aggregated[f'mean_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
        
        return aggregated
    
    @staticmethod
    def group_by_status(data_list):
        """Group data by success/failure status"""
        success_items = [item for item in data_list if item.get('status') == 'SUCCEEDED']
        fail_items = [item for item in data_list if item.get('status') != 'SUCCEEDED']
        return success_items, fail_items
    
    @staticmethod
    def group_by_depth(data_list):
        """Group data by depth"""
        depth_groups = {}
        for item in data_list:
            depth = item.get('depth', 0)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(item)
        return depth_groups
    
    @staticmethod
    def calculate_entropy_stats(logprobs_list, entropy_calculator):
        """Calculate entropy statistics from logprobs list"""
        all_entropies = []
        all_stds = []
        all_first_token_entropies = []
        all_first_token_stds = []
        five_token_entropies = []
        five_token_stds = []
        
        for single_resp_logprobs in logprobs_list:
            info_list = [entropy_calculator.logprobs_to_entropy(token_logprobs.top_logprobs) 
                        for token_logprobs in single_resp_logprobs]
            entropy_list = [item[0] for item in info_list]
            std_list = [item[1] for item in info_list]
            
            mean_entropy = sum(entropy_list) / len(entropy_list)
            mean_std = sum(std_list) / len(std_list)
            all_entropies.append(mean_entropy)
            all_stds.append(mean_std)
            all_first_token_entropies.append(info_list[0][0])
            all_first_token_stds.append(info_list[0][1])
            five_token_entropies.append(sum(entropy_list[:5]) / len(entropy_list[:5]))
            five_token_stds.append(sum(std_list[:5]) / len(std_list[:5]))
        
        return {
            'all_entropies': all_entropies,
            'all_stds': all_stds,
            'all_first_token_entropies': all_first_token_entropies,
            'all_first_token_stds': all_first_token_stds,
            'five_token_entropies': five_token_entropies,
            'five_token_stds': five_token_stds
        }


class FileHandler:
    """Shared file handling utilities"""
    
    @staticmethod
    def write_jsonl_entry(file_path, entry, mode='a'):
        """Write a single entry to JSONL file"""
        with open(file_path, mode) as f:
            json.dump(entry, f)
            f.write('\n')
    
    @staticmethod
    def read_jsonl_with_progress(file_path, progress_desc="Processing"):
        """Read JSONL file with progress bar"""
        entries = []
        with open(file_path, 'r') as f:
            for line in tqdm(f, desc=progress_desc):
                entries.append(json.loads(line))
        return entries


# Convenience functions that combine multiple utilities
def process_entropy_metrics(logprobs_list):
    """Process entropy metrics using shared utilities"""
    entropy_calc = EntropyCalculator()
    stats_agg = StatisticsAggregator()
    return stats_agg.calculate_entropy_stats(logprobs_list, entropy_calc)


def process_tactic_similarities(tactic_list):
    """Process tactic similarities using shared utilities"""
    sim_calc = SimilarityCalculator()
    mean_similarity, std_similarity = sim_calc.tactic_list_similarity(tactic_list)
    n_clusters = sim_calc.tactic_list_cluster(tactic_list)
    return mean_similarity, std_similarity, n_clusters


def process_embedding_analysis(embeddings):
    """Process embedding analysis using shared utilities"""
    entropy_calc = EntropyCalculator()
    sim_calc = SimilarityCalculator()
    
    semantic_entropy = entropy_calc.semantic_entropy(embeddings)
    cluster_entropy, best_k = entropy_calc.cluster_entropy(embeddings)
    mean_sim, pairwise_sims = sim_calc.basic_info(embeddings)
    
    return {
        'semantic_entropy': semantic_entropy,
        'cluster_entropy': cluster_entropy,
        'best_k': best_k,
        'mean_sim': mean_sim,
        'pairwise_sims': pairwise_sims,
        'sim_std': np.std(pairwise_sims)
    }