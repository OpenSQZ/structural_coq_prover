"""
Calculate average clarity score for different types in result
"""
import json
import argparse
from typing import Dict, List
from pathlib import Path
from tabulate import tabulate


def calculate_average_scores(file_path: str, basic_fields: List[str]) -> tuple[Dict[str, float], Dict[str, int]]:
    # Initialize score dictionary and count dictionary
    score_sums = {field: 0.0 for field in basic_fields}
    count_dict = {field: 0 for field in basic_fields}

    with Path(file_path).open('r') as f:
        for line in f:
            data = json.loads(line)
            # Calculate scores for basic fields
            for field in basic_fields:
                if (field in data and 
                    data[field] is not None and 
                    isinstance(data[field], dict) and 
                    'scores' in data[field] and 
                    data[field]['scores'] is not None and
                    'lm_score' in data[field]['scores']):
                    score_sums[field] += data[field]['scores']['lm_score']
                    count_dict[field] += 1

    # Calculate average for each field
    averages = {}
    for field in score_sums:
        if count_dict[field] > 0:
            averages[field] = score_sums[field] / count_dict[field]
        else:
            averages[field] = 0.0

    return averages, count_dict

def main():
    # Default basic fields
    default_basic_fields = ['no_ref_ps_all', 'no_ref_ps_origin', 'origin_only', 'internal_only', 
                           'intuition_only', 'origin_internal', 'origin_intuition', 'internal_intuition', 
                           'origin_internal_intuition', 'zh_explanation', 'base', 'base_simple']
    default_data_path = 'clarity_score_experiment/data/clarity_score_result/result_global_def.jsonl'

    parser = argparse.ArgumentParser(description='Calculate average lm_score for different types in result.jsonl')
    parser.add_argument('--file_path', '-f', type=str, default=default_data_path,
                        help='Path to input jsonl file')
    parser.add_argument('--basic_fields', '-b', type=str, nargs='+', default=default_basic_fields,
                        help='List of basic fields to calculate (default: all predefined fields)')
    
    args = parser.parse_args()
    
    averages, count_dict = calculate_average_scores(args.file_path, args.basic_fields)
    
    # Prepare table data, including sample count for each field
    table_data = [[key, f"{value:.4f}", count_dict[key]] for key, value in averages.items()]

    print("\nResults:")
    print(tabulate(table_data, 
                  headers=['Method', 'Clarity Score', 'Sample Count'],
                  tablefmt='grid',
                  floatfmt='.4f'))

if __name__ == '__main__':
    main()
