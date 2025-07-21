from data_extraction.coq_tokenize.tokenizer import Tokenizer
from utils import read_jsonl_file, write_jsonl_file
from data_extraction.coq_data.Def_class import def_object
from tqdm import tqdm
from utils import get_config
import os
import argparse

def run_tokenizer(mode='std'):
    config_path = './config.json'
    config = get_config(config_path)
    base_path = config.paths.output_data
    
    def_input_path = f'{base_path}/Def_{mode}.jsonl'
    ps_input_path = f'{base_path}/PS_{mode}.jsonl'
    tokenizer_path = f'{base_path}/tokenizer_{mode}.json'
    def_output_path = f'{base_path}/def_table_{mode}.jsonl'
    ps_output_path = f'{base_path}/ps_table_{mode}.jsonl'

    tokenizer = Tokenizer()
    tokenizer.init_tokenizer(def_input_path)
    def_data = read_jsonl_file(def_input_path)
    tokenizer.save(tokenizer_path)


    def_objs = []
    for item in tqdm(def_data, desc="Processing defs", total=len(def_data)):
        def_obj = tokenizer.process_def(item)
        item_dict = def_obj.to_dict()
        def_objs.append(item_dict)

    fallback_ratio = tokenizer.fallback_tokens / tokenizer.total_tokens if tokenizer.total_tokens > 0 else 0
    print("\nDef Processing Statistics:")
    print(f"Total tokens: {tokenizer.total_tokens}")
    print(f"Fallback tokens: {tokenizer.fallback_tokens}")
    print(f"Fallback ratio: {fallback_ratio:.2%}")

    tokenizer.total_tokens = 0
    tokenizer.fallback_tokens = 0

    processed_id = []
    filtered_def_objs = []
    for item in tqdm(def_objs, desc="Processing defs", total=len(def_objs)):
        try:
            token_id = tokenizer.tokenizer[item['name']]
            if token_id not in processed_id and token_id >= 1000:
                processed_id.append(token_id)
                item['def_id'] = token_id
                filtered_def_objs.append(item)
        except:
            print(f"Warning: {item['name']} not found in tokenizer vocabulary")
    
    print(f"Processed {len(processed_id)} defs")
    assert len(filtered_def_objs) == len(processed_id)
    write_jsonl_file(filtered_def_objs, def_output_path)

    def_table = {item['name']: item for item in filtered_def_objs}

    ps_objs = []
    ps_data = read_jsonl_file(ps_input_path)

    for item in tqdm(ps_data, desc="Processing ps", total=len(ps_data)):
        ps_obj = tokenizer.process_ps(item, def_table)
        item_dict = ps_obj.to_dict()
        ps_objs.append(item_dict)
    
    fallback_ratio = tokenizer.fallback_tokens / tokenizer.total_tokens if tokenizer.total_tokens > 0 else 0
    print("\nPS Processing Statistics:")
    print(f"Total tokens: {tokenizer.total_tokens}")
    print(f"Fallback tokens: {tokenizer.fallback_tokens}")
    print(f"Fallback ratio: {fallback_ratio:.2%}")

    write_jsonl_file(ps_objs, ps_output_path)

    return def_output_path, ps_output_path, tokenizer_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Coq tokenizer')
    parser.add_argument('--mode', type=str, default='std', 
                       help='Mode for tokenizer (default: std)')
    
    args = parser.parse_args()
    run_tokenizer(args.mode)

