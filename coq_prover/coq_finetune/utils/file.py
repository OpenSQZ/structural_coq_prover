import glob
import io
import itertools
import json
import os


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def find_json_files(directory):
    for root, dirs, files in os.walk(directory):
        for pattern in ["*.json", "*.jsonl"]:
            for file in glob.glob(os.path.join(root, pattern)):
                print(file)
                yield file


def load_jsonl_file(path: str, limit: int = -1):
    data_list = []
    with open(path, "r") as f:
        if limit > 0:
            for _ in range(limit):
                try:
                    data_list.append(json.loads(f.readline().strip()))
                except BaseException:
                    break
        else:
            lines = f.readlines()
            data_list = [json.loads(line.strip()) for line in lines]
    return data_list


def load_json_file(path: str, limit: int = -1):
    try:
        list_data_dict = jload(path)
        if limit > 0:
            list_data_dict = list_data_dict[:limit]
    except BaseException:
        with open(path, "r") as f:
            if limit > 0:
                lines = list(itertools.islice(f, limit))
            else:
                lines = f.readlines()
        list_data_dict = [json.loads(line.strip()) for line in lines]
    return list_data_dict


def dump_jsonl_file(data_list, output_file):
    with open(output_file, "w") as file:
        for item in data_list:
            # 将字典转换为JSON字符串，并写入文件
            json_string = json.dumps(item)
            file.write(json_string + "\n")  # 确保每个JSON对象后面都有一个换行符


def load_json_dir(path: str):
    list_data_dict = []
    for file in find_json_files(path):
        list_data_dict += load_json_file(file)
    return list_data_dict


def dump_json_file(path, data):
    with open(path, "w") as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)
