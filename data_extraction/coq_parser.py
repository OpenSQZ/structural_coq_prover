from data_extraction.coq_data import Parser
from utils import get_config, read_json_file, read_txt_file, split_coqc_file, read_jsonl_file
import json
from coqc import *
from tqdm import tqdm
from data.coq_test_package import coq_test
import asyncio

def main():
    config_path = "./config.json"
    subset_file = "./data/subset.json"
    subset_test_file = "./data/subset_test.json"
    
    if args.subset or args.both:
        subset_train = read_json_file(subset_file)
        subset_test = read_json_file(subset_test_file)
        all_subset = subset_train + subset_test

    config = get_config(config_path)
    parser = Parser()

    file_path = 'path/to/coq_train.txt'
    content = read_txt_file(file_path)

    def process_and_write(file_dict, subset_test=None, is_subset=False):
        def_table = []
        ps_table = []
        
        for file, content in file_dict.items():
            if subset_test and file in subset_test:
                def_table_obj, _ = parser.parse(file=content, file_path=file, max_depth=1)
                def_table.extend(def_table_obj)
            else:
                def_table_obj, ps_table_obj = parser.parse(file=content, file_path=file, max_depth=1)
                def_table.extend(def_table_obj)
                ps_table.extend(ps_table_obj)

        suffix = "_subset" if is_subset else "_std"
        output_def_path = f"{config.paths.output_data}_Def{suffix}.jsonl"
        output_ps_path = f"{config.paths.output_data}_PS{suffix}.jsonl"

        with open(output_def_path, 'w', encoding='utf-8') as f:
            for item in def_table:
                f.write(json.dumps(item.to_dict()) + '\n')
                
        with open(output_ps_path, 'w', encoding='utf-8') as f:
            for item in ps_table:
                f.write(json.dumps(item.to_dict()) + '\n')

    if args.both:
        subset_train = read_json_file(subset_file)
        subset_test = read_json_file(subset_test_file)
        all_subset = subset_train + subset_test
        std_dict, subset_dict = split_coqc_file(content, subset=all_subset, both=True)
        process_and_write(std_dict, is_subset=False)
        process_and_write(subset_dict, subset_test=subset_test, is_subset=True)
    elif args.subset:
        subset_train = read_json_file(subset_file)
        subset_test = read_json_file(subset_test_file)
        all_subset = subset_train + subset_test
        
        file_dict = split_coqc_file(content, if_subset=True, subset=all_subset)
        process_and_write(file_dict, subset_test=subset_test, is_subset=True)
    elif args.std:
        file_dict = split_coqc_file(content)
        process_and_write(file_dict, is_subset=False)      
    else:
        raise ValueError("Invalid argument, both, subset, std must be provided")     

def test():
    parser = Parser()
    def_table_obj, ps_table_obj = parser.parse(file = "/root/CMorphisms.txt", file_path = "dummy")
    
    with open('/path/to/def_table.jsonl', 'w', encoding='utf-8') as f:
        for item in def_table_obj:
            f.write(json.dumps(item.to_dict()) + '\n')
    
    with open('/path/to/ps_table.jsonl', 'w', encoding='utf-8') as f:
        for item in ps_table_obj:
            f.write(json.dumps(item.to_dict()) + '\n')

def std():
    config_path = "./config.json"
    config = get_config(config_path)
    parser = Parser()

    file_path = './data/coq_std.txt'
    content = read_txt_file(file_path)
    def_table = []
    ps_table = []
    
    def_table_obj, ps_table_obj = parser.parse(file = content,file_path = "std")
    def_table.extend(def_table_obj)
    ps_table.extend(ps_table_obj)

    output_def_path = f"{config.paths.output_data}_Def_std.jsonl"
    output_ps_path = f"{config.paths.output_data}_PS_std.jsonl"

    with open(output_def_path, 'w', encoding='utf-8') as f:
        for item in def_table:
            f.write(json.dumps(item.to_dict()) + '\n')
            
    with open(output_ps_path, 'w', encoding='utf-8') as f:
        for item in ps_table:
            f.write(json.dumps(item.to_dict()) + '\n')

def subset():
    config_path = "./config.json"
    subset_file = "./data/subset.json"
    subset_test_file = "./data/subset_test.json"

    subset_train = read_json_file(subset_file)
    subset_test = read_json_file(subset_test_file)
    all_subset = subset_train + subset_test

    config = get_config(config_path)
    parser = Parser()

    file_path = 'path/to/coq_std.txt'
    content = read_txt_file(file_path)
    file_dict = split_coqc_file(content, if_subset = True, subset = all_subset)
    print(len(file_dict))
    
    def_table = []
    ps_table = []

    for file, content in file_dict.items():
        if file in subset_test:
            def_table_obj, _ = parser.parse(file = content, file_path = file)
            def_table.extend(def_table_obj)
        else:
            def_table_obj, ps_table_obj = parser.parse(file = content, file_path = file)
            def_table.extend(def_table_obj)
            ps_table.extend(ps_table_obj)

    output_def_path = f"{config.paths.output_data}_Def_subset.jsonl"
    output_ps_path = f"{config.paths.output_data}_PS_subset.jsonl"

    with open(output_def_path, 'w', encoding='utf-8') as f:
        for item in def_table:
            f.write(json.dumps(item.to_dict()) + '\n')
            
    with open(output_ps_path, 'w', encoding='utf-8') as f:
        for item in ps_table:
            f.write(json.dumps(item.to_dict()) + '\n')

def resume_from_error_log(error_log_path):
    error_files = []
    with open(error_log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("Error in file "):
                path = line[len("Error in file "):].strip()
                if path.endswith(":"):
                    path = path[:-1]
                error_files.append(path)
    return error_files

async def customized(patch_mode=False):
    config_path = "./config.json"
    config = get_config(config_path)
    parser = Parser()
    coqc = Coqc(config_path = config_path)
    file_path = config.paths.ordered_data_file
    file_list = read_json_file(file_path)
    if 'path/to/' in config.paths.data_dir:
        raise ValueError("Data directory is placeholder, please set data_dir in config.json")
    file_list = [file for file in file_list if 'tactician' not in file]
    def_table = []
    ps_table = []
    # file_list = resume_from_error_log('./data/coqc_error_log_2025-05-06-00.txt')

    for file in tqdm(file_list,total=len(file_list)):
        # if 'coq_train/coq-coqtail.8.14/Arith/MillerRabin.v' in file:      
            if 'tactician' in file:
                continue
            # for test
            if not ('ceres' in file or 'high-school-geometry' in file):
                continue
            result = await coqc.run(os.path.join(config.paths.data_dir, file), patch_mode=patch_mode)
            if result is not None:
                # now ps for test data will not add to our dataset, however as now tactic will not give some example ps, may need to add them?
                if any(test_file in file for test_file in coq_test):
                    def_table_obj, _ = parser.parse(file = result, file_path = file, max_depth = 1, use_tqdm = False)
                    def_table.extend(def_table_obj)
                else:
                    def_table_obj, ps_table_obj = parser.parse(file = result, file_path = file, max_depth = 1, use_tqdm = False)
                    def_table.extend(def_table_obj)
                    ps_table.extend(ps_table_obj)

    current_time = datetime.now().strftime("%Y-%m-%d-%H")
    output_def_path = f"{config.paths.output_data}/Def_{current_time}.jsonl"
    output_ps_path = f"{config.paths.output_data}/PS_{current_time}.jsonl"

    with open(output_def_path, 'w', encoding='utf-8') as f:
        for item in def_table:
            f.write(json.dumps(item.to_dict()) + '\n')
    with open(output_ps_path, 'w', encoding='utf-8') as f:
        for item in ps_table:
            f.write(json.dumps(item.to_dict()) + '\n')
    return current_time

if __name__ == "__main__":
    ## STD must be created first, then all the dataset can be processed
    ## coq_std is the log file when make the coqc
    ## as the STD is created when coqc is run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--both', action='store_true', help='Process both std and subset')
    parser.add_argument('--subset', action='store_true', help='Process subset only')
    parser.add_argument('--std', action='store_true', help='Process std only')
    args = parser.parse_args()

    if args.both or args.subset or args.std:
        main()
    else:
        asyncio.run(customized())

