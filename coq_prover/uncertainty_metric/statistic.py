import json
import numpy as np
from scipy.stats import entropy
import scipy
from tabulate import tabulate

def analyze_distribution(logprobs):
    digit_probs = np.zeros(10)
    
    for item in logprobs['top_logprobs']:
        if item['token'] in '0123456789':
            try:
                digit_probs[int(item['token'])] = np.exp(item['logprob'])
            except:
                print(item)
    
    digit_probs = digit_probs / np.sum(digit_probs)
    return entropy(digit_probs), scipy.stats.kurtosis(digit_probs), digit_probs

def analyze_top3(digit_probs, ground_truth):
    non_zero_probs = [(i, p) for i, p in enumerate(digit_probs) if p > 0]
    non_zero_probs.sort(key=lambda x: x[1], reverse=True)
    top3_probs = non_zero_probs[:3] if len(non_zero_probs) >= 3 else non_zero_probs
    return int(any(idx == ground_truth for idx, _ in top3_probs))

def process_entry(entry):
    entropy_, kurtosis, digit_probs = analyze_distribution(entry['logprobs'][0])
    is_correct = int(entry['answer']) == int(entry['ground_truth'])
    is_in_top3 = analyze_top3(digit_probs, int(entry['ground_truth']))
    
    return {
        'entropy': entropy_,
        'kurtosis': kurtosis,
        'is_correct': is_correct,
        'is_in_top3': is_in_top3
    }

def main():
    file = './data/extra_logs/uncertainty_metric/premise_selection/output.jsonl'
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    methods = ['no_def', 'def_origin_no_intuition', 'def_mixed_no_intuition', 'def_mixed_intuition']
    results = {method: {
        'entropy': [],
        'kurtosis': [],
        'success': 0,
        'top3_success': 0
    } for method in methods}
    
    total = len(data)
    for item in data:
        for method in methods:
            entry = item[f'{method}_entry']
            result = process_entry(entry)
            
            results[method]['entropy'].append(result['entropy'])
            results[method]['kurtosis'].append(result['kurtosis'])
            results[method]['success'] += result['is_correct']
            results[method]['top3_success'] += result['is_in_top3']
    
    table_data = []
    headers = ['Method', 'Accuracy', 'Top3 Accuracy', 'Avg Entropy', 'Avg Kurtosis']
    
    for method in methods:
        row = [
            method,
            f"{results[method]['success'] / total:.4f}",
            f"{results[method]['top3_success'] / total:.4f}",
            f"{np.mean(results[method]['entropy']):.4f}",
            f"{np.mean(results[method]['kurtosis']):.4f}"
        ]
        table_data.append(row)
    
    print("\nResults Summary:")
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    print("\nDetailed Statistics:")
    for method in methods:
        print(f"\n{method}:")
        print(f"  Total samples: {total}")
        print(f"  Correct predictions: {results[method]['success']}")
        print(f"  Top3 predictions: {results[method]['top3_success']}")
        print(f"  Entropy stats: mean={np.mean(results[method]['entropy']):.4f}, std={np.std(results[method]['entropy']):.4f}")
        print(f"  Kurtosis stats: mean={np.mean(results[method]['kurtosis']):.4f}, std={np.std(results[method]['kurtosis']):.4f}")

if __name__ == "__main__":
    main()