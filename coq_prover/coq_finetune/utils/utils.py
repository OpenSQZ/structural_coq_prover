# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0
import gc
import math
# DeepSpeed Team
import random
from typing import Any, Dict, Tuple, Type, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from deepspeed.accelerator import get_accelerator
from torch.types import Device
from transformers import set_seed, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, AutoConfig

from coq_prover.coq_finetune.utils.logger import print_rank_0



def configure_dropout(model_config, dropout):
    if dropout is not None:
        for key in ('dropout', 'attention_dropout', 'hidden_dropout',
                    'activation_dropout'):
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)




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


def to_cuda(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device, non_blocking=True)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        get_accelerator().manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor



def add_special_tokens_to_to_tokenizer(tokenizer: PreTrainedTokenizer, special_token_dict: dict):
    return add_special_tokens_to_tokenizer(tokenizer, special_token_dict)


def model_embedding_resize(model, tokenizer, num_new_tokens):
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_special_tokens_to_tokenizer(tokenizer: PreTrainedTokenizer, special_token_dict: dict):
    for key in special_token_dict.keys():
        if key not in tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            tokenizer.SPECIAL_TOKENS_ATTRIBUTES.append(key)

    num_new_tokens = tokenizer.add_special_tokens(special_token_dict)

    print_rank_0("num_new_tokens =" + str(num_new_tokens))

    if num_new_tokens:
        for val in special_token_dict.values():
            print_rank_0(f"token = {val}, token_id = {str(tokenizer.convert_tokens_to_ids(val))}")

    return num_new_tokens


def pad_sequences(seqs, pad_value, padding='right', pad_to: int = None):
    """
    Padding sequence to the same length
    """
    max_len = max(len(seq) for seq in seqs) if pad_to is None else pad_to
    if padding == 'right':
        padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in seqs]
    elif padding == 'left':
        padded_seqs = [[pad_value] * (max_len - len(seq)) + seq for seq in seqs]
    else:
        padded_seqs = []
        assert ValueError
    return padded_seqs


@torch.no_grad()
def get_global_statistics(xs: torch.Tensor, mask=None, device='cpu') -> Tuple[float, float, int]:
    """
    Computes element-wise mean and variance of the tensor across processes
    https://github.com/microsoft/LMOps/blob/cde1fb1ef4608a7ac5bf00675fa3e94b1d960abb/minillm/minillm/utils.py#L108
    """
    xs = xs.to(device)
    sum_and_count = torch.tensor([xs.sum(), (xs.numel() if mask is None else mask.sum())], device=xs.device)
    dist.all_reduce(sum_and_count, op=dist.ReduceOp.SUM)
    global_sum, count = sum_and_count
    global_mean = global_sum / count

    sum_var = torch.sum(((xs - global_mean) ** 2).mul(1 if mask is None else mask))
    dist.all_reduce(sum_var, op=dist.ReduceOp.SUM)
    global_var = sum_var / count
    
    return global_mean.to(device), global_var.to(device), count.to(device)


@torch.no_grad()
def whiten(xs: torch.Tensor, mask: torch.BoolTensor, device: Device, shift_mean=True) -> torch.Tensor:
    """
    Whitens values
    """
    mean, var, count = get_global_statistics(xs, mask=mask, device=device)

    whitened = (xs - mean) * torch.rsqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    logpy = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy


def get_category_distribution_entropy(bsz, logits):
    """
    Compute category distribution entropy
    """
    logits_distribution = torch.distributions.categorical.Categorical(logits=logits.reshape(-1, logits.size(-1)))
    ent = logits_distribution.entropy().reshape(bsz, -1)
    return ent


def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    https://github.com/OpenLMLab/MOSS/blob/e088f438d1a95d424c6dffef0d73134ebe62cb72/models_jittor/generation.py#L146
    """
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))
        
    return cum_logits


def clean_dict(d: Dict[str, Any]) -> None:
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
