from coq_prover.coq_context.emb_model import LinqEmbedding
from utils import get_config
import os
from concurrent.futures import ProcessPoolExecutor
import argparse

def split_data(input_file, total_shards=8):
    with open(input_file, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    total_samples = len(all_lines)
    samples_per_shard = total_samples // total_shards
    
    for shard_id in range(1, total_shards + 1):
        shard_file = input_file.replace('.jsonl', f'_shard_{shard_id}.jsonl')
        start_idx = (shard_id - 1) * samples_per_shard
        end_idx = start_idx + samples_per_shard if shard_id < total_shards else total_samples
        
        with open(shard_file, 'w', encoding='utf-8') as f:
            for line in all_lines[start_idx:end_idx]:
                f.write(line)

def process_shard(args):
    shard_file, gpu_id, model_path = args
    
    print(f"Processing shard {shard_file} on GPU {gpu_id}")
    model = LinqEmbedding(model_path, gpu_id)
    try:
        model.generate(shard_file)
    except Exception as e:
        print(f"Error processing {shard_file} on GPU {gpu_id}: {str(e)}")

def process_all_shards(input_file,model_path, total_shards=8):
    process_args = []
    for i in range(total_shards):
        shard_file = input_file.replace('.jsonl', f'_shard_{i+1}.jsonl')
        if os.path.exists(shard_file):
            process_args.append((shard_file, i, model_path))
    
    print(f"Starting parallel processing with {total_shards} shards...")
    with ProcessPoolExecutor(max_workers=total_shards) as executor:
        executor.map(process_shard, process_args)

def process_whole_file(input_file, model_path, gpu_id=0):
    print(f"Processing whole file {input_file} on GPU {gpu_id}")
    model = LinqEmbedding(model_path, gpu_id)
    try:
        model.generate(input_file)
    except Exception as e:
        print(f"Error processing {input_file} on GPU {gpu_id}: {str(e)}")
    return input_file.replace('.jsonl', '_emb.jsonl')

if __name__ == "__main__":
    config = get_config('./config.json')
    model_path = config.paths.emb_model_path
    parser = argparse.ArgumentParser(description='Process embeddings with distributed or single processing')
    parser.add_argument('--input_file', type=str, default="path/to/def_table_train.jsonl",
                       help='Input JSONL file path (default: path/to/def_table_train.jsonl)')
    parser.add_argument('--mode', type=str, choices=['distributed', 'single'], default='single',
                       help='Processing mode: distributed (parallel shards) or single (whole file)')
    parser.add_argument('--shards', type=int, default=8,
                       help='Number of shards for distributed processing (default: 8)')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID for single processing mode (default: 0)')
    parser.add_argument('--split_only', action='store_true',
                       help='Only split the data into shards without processing')
    
    args = parser.parse_args()
    
    if args.split_only:
        print(f"Splitting {args.input_file} into {args.shards} shards...")
        split_data(args.input_file, args.shards)
        print("Data splitting completed.")
    elif args.mode == 'distributed':
        # First split the data if shard files don't exist
        shard_1_file = args.input_file.replace('.jsonl', '_shard_1.jsonl')
        if not os.path.exists(shard_1_file):
            print(f"Splitting {args.input_file} into {args.shards} shards...")
            split_data(args.input_file, args.shards)
        
        process_all_shards(args.input_file, model_path, args.shards)
    else:  # single mode
        process_whole_file(args.input_file, model_path, args.gpu_id)