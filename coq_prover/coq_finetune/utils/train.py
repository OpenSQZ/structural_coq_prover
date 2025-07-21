import gc
import glob
import io
import json
import os
import random
from typing import List, TypeVar

import deepspeed
import numpy as np
import torch
import torch.distributed
from deepspeed.runtime.zero import ZeroParamStatus
from transformers import set_seed


def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)

def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except Exception as e:
            output[k] = v
    return output


def is_rank_0():
    """Check whether it is rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True
    
def parse_remaining_args_to_dict(remaining_args):
    """将剩余参数列表转换为字典。将成对出现的 '--key value' 映射到相应的键值对，单独的键映射为 True。"""
    it = iter(remaining_args)
    args_dict = {}
    for key in it:
        clean_key = key.lstrip('-')
        value = next(it, True)  # 默认为 True，如果没有下一个值，表示这是一个标志位
        if isinstance(value, str) and value.startswith('--'):
            # 如果下一个值实际是另一个键，将当前键映射为 True，并将迭代器回退一步
            args_dict[clean_key] = True
            remaining_args.insert(remaining_args.index(value), key)  # 将迭代器回退到当前位置
        else:
            args_dict[clean_key] = value
    return args_dict

def print_rank_0(msg, rank=None, wrap=False):
    if wrap:
        msg = f"===================={msg}===================="
    if rank is not None and rank <= 0:
        print(msg)
    elif is_rank_0():
        print(msg)

def clean_dict(d) -> None:
    """
    清理字典，删除所有项目。

    参数:
    d (Dict[str, Any]): 需要清理的字典

    返回:
    None
    """
    for key in list(d.keys()):
        del d[key]
    del d
    

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()