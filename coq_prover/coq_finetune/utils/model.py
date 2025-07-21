import os.path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from coq_prover.coq_finetune.utils.train import print_rank_0

def init_tokenizer(tokenizer_name: str, model_max_length: int, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, model_max_length=model_max_length, use_fast=True,**kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def init_from_pretrained(
    pretrained_dir: str,
    attn_implementation: Optional[str] = "flash_attention_2",
):

    config = AutoConfig.from_pretrained(
        pretrained_dir, attn_implementation=attn_implementation
    )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_dir, 
        config=config, 
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer, config

def init_llm_and_tokenizer(
    base_model_name: str,
    pretrained_dir: str = None,
    attn_implementation: str = "flash_attention_2",
    **kwargs,
):
    if pretrained_dir:
        return init_from_pretrained(pretrained_dir, attn_implementation)
    
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {(total_params / 1024 / 1024 / 1024):.2f} B")

def save_model(
    model,
    config,
    tokenizer,
    output_dir,
):
    model.save_fp16_model(f"{output_dir}/fp16")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    config.save_pretrained(f"{output_dir}/config")