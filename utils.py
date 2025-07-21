import json
from tqdm import tqdm
import os

def read_normal_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def read_jsonl_file(file_path, partial_num = None):
    with open(file_path, 'r') as f:
        if partial_num:
            result = []
            if not isinstance(partial_num, int):
                raise Exception("Partial num must be an integer")
            for _ in tqdm(range(partial_num), desc="Reading jsonl file", total=partial_num):
                line = f.readline().strip()
                if not line:
                    raise Exception("Partial num is larger than the file size")
                result.append(json.loads(line))
            return result
        else:
            output = os.popen("wc -l " + file_path).read()
            total = int(output.strip().split(" ")[0])
            return [json.loads(line) for line in tqdm(f, desc="Reading jsonl file", total=total)]

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_jsonl_file(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            try:
                f.write(json.dumps(item) + '\n')
            except:
                print(item)
                raise Exception("Error writing to file")

def write_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)

def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DotDict(value))
            else:
                setattr(self, key, value)

def get_config(file_path):
    with open(file_path, 'r') as f:
        config_dict = json.load(f)
    return DotDict(config_dict)

def coqc_subset_file(command,subset):
    coqc_file = command.split(" ")[-1].strip('\n').strip(' ')
    for item in subset:
        if coqc_file in item:
            return item.split('coq_train')[-1]
    return None

def split_coqc_file(content, if_subset = False, subset = None, both = False):
    if (if_subset or both) and not subset:
        raise Exception("Subset file is not provided")
    if both:
        if_subset = True

    current_file = None
    current_block_lines = []
    file_dict = {}
    subset_dict = {}
    subset_file = False
    std_prefix = "coq-stdlib/"

    for line in content:
        if not line.strip():    
            continue
        if line.startswith("COQC") and not current_file:
            if if_subset:
                file_path = coqc_subset_file(line,subset)
                if file_path:
                    subset_file = True
                    current_file = file_path
                else:
                    current_file = None
                    if both:
                        subset_file = False
                        current_file = line.split(" ")[-1].split('coq_train')[-1]
            else:
                current_file = line.split(" ")[-1].split('coq_train')[-1]
        elif line.startswith("COQC") and current_file:
            if both:
                if subset_file:
                    subset_dict[std_prefix + current_file] = current_block_lines
                file_dict[std_prefix + current_file] = current_block_lines
            else:
                file_dict[std_prefix + current_file] = current_block_lines

            if if_subset:
                file_path = coqc_subset_file(line,subset)
                if file_path:
                    subset_file = True
                    current_file = file_path
                else:
                    current_file = None
                    if both:
                        subset_file = False
                        current_file = line.split(" ")[-1].split('coq_train')[-1]
            else:
                current_file = line.split(" ")[-1].split('coq_train')[-1]
            current_block_lines = []
        else:
            if current_file:
                current_block_lines.append(line)
            else:
                continue

    if current_file and current_block_lines:
        if both:
            if subset_file:
                subset_dict[std_prefix + current_file] = current_block_lines
            else:
                file_dict[std_prefix + current_file] = current_block_lines
        else:
            file_dict[std_prefix + current_file] = current_block_lines

    if both:
        return file_dict, subset_dict

    return file_dict