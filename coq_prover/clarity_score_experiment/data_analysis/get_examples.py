import json
import os
import argparse


def get_examples(data_path, output_path, case_type_list, n=10, result=None):
    """
    Extract data from jsonl file and save to data_examples.jsonl file
    
    Args:
        data_path (str): Path to input jsonl file
        output_path (str): Path to output jsonl file
        case_type_list (list or None): List of case types to filter, if None then no case_type filtering
        n (int): Number of data entries to get for each case_type (if case_type_list is None, get n total entries), default is 10
        result (str, optional): Result value to filter, can be 'YES' or 'NO', if None then no result filtering
    """
    try:
        all_examples = []
        
        if case_type_list is None:
            # No case_type filtering, directly get n data entries
            count = 0
            with open(data_path, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    if count >= n:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        
                        # If result parameter is specified, need to find matching result in all fields
                        if result is not None:
                            result_found = False
                            # Traverse all fields in data to find fields containing result
                            for key, value in data.items():
                                if isinstance(value, dict) and 'result' in value and value['result'] == result:
                                    result_found = True
                                    break
                            
                            if result_found:
                                all_examples.append(data)
                                count += 1
                        else:
                            # If no result specified, accept all data
                            all_examples.append(data)
                            count += 1
                            
                    except json.JSONDecodeError as e:
                        continue
        else:
            # Original logic: get n data entries for each case_type separately
            case_type_counts = {ct: 0 for ct in case_type_list}
            
            with open(data_path, 'r', encoding='utf-8') as input_file:
                for line in input_file:
                    # Check if all case_types have collected enough data
                    if all(count >= n for count in case_type_counts.values()):
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        
                        # Check for each case_type
                        for case_type in case_type_list:
                            # If this case_type hasn't collected enough data
                            if case_type_counts[case_type] < n and case_type in data:
                                # If result parameter is specified, perform result filtering
                                if result is not None:
                                    if 'result' in data[case_type] and data[case_type]['result'] == result:
                                        all_examples.append(data)
                                        case_type_counts[case_type] += 1
                                        break  # This data has been used by a case_type, don't reuse
                                else:
                                    # If no result specified, accept all result values
                                    all_examples.append(data)
                                    case_type_counts[case_type] += 1
                                    break  # This data has been used by a case_type, don't reuse
                                    
                    except json.JSONDecodeError as e:
                        continue
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write data to new jsonl file
        with open(output_path, 'w', encoding='utf-8') as output_file:
            for example in all_examples:
                json.dump(example, output_file, ensure_ascii=False)
                output_file.write('\n')
        
        # Output statistics
        total_examples = len(all_examples)
        result_filter_msg = f", result filter: '{result}'" if result is not None else ", no result filter"
        
        if case_type_list is None:
            print(f"Successfully extracted {total_examples} data entries from {data_path}")
            print(f"Filter conditions: no case_type filter{result_filter_msg}")
        else:
            print(f"Successfully extracted {total_examples} data entries from {data_path}")
            print(f"Filter conditions: case_type_list={case_type_list}{result_filter_msg}")
            print("Data count for each case_type:")
            for case_type, count in case_type_counts.items():
                print(f"  {case_type}: {count} entries")
        
        print(f"Data saved to: {output_path}")
        
    except FileNotFoundError:
        print(f"Error: File not found {data_path}")
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    default_data_path = 'clarity_score_experiment/data/clarity_score_result/result_global_def.jsonl'
    default_output_path = 'clarity_score_experiment/data/examples/data_examples.jsonl'
    default_case_types = ['no_ref_ps_all', 'no_ref_ps_origin', 'origin_only', 'internal_only', 
                         'intuition_only', 'origin_internal', 'origin_intuition', 'internal_intuition', 
                         'origin_internal_intuition', 'zh_explanation', 'base', 'base_simple']
    
    parser = argparse.ArgumentParser(description='Extract examples from jsonl file with filtering options')
    parser.add_argument('--data_path', '-d', type=str, default=default_data_path,
                        help=f'Path to input jsonl file (default: {default_data_path})')
    parser.add_argument('--output_path', '-o', type=str, default=default_output_path,
                        help=f'Path to output jsonl file (default: {default_output_path})')
    parser.add_argument('--case_types', '-c', type=str, nargs='*', default=default_case_types,
                        help='List of case types to filter (default: all predefined types). Use --case_types to specify none.')
    parser.add_argument('--n', '-n', type=int, default=10,
                        help='Number of examples per case type (default: 10)')
    parser.add_argument('--result', '-r', type=str, choices=['YES', 'NO'], default=None,
                        help='Filter by result value: YES or NO (default: no filtering)')
    parser.add_argument('--no_case_filter', action='store_true',
                        help='Disable case type filtering and get n total examples')
    
    args = parser.parse_args()
    
    # Handle case type filtering
    case_type_list = None if args.no_case_filter else (args.case_types if args.case_types else None)
    
    get_examples(args.data_path, args.output_path, case_type_list, args.n, args.result)

if __name__ == "__main__":
    main()

