import random
from dataclasses import dataclass, field, asdict
from typing import Optional, Sequence

import transformers

@dataclass
class SFTArguments(transformers.TrainingArguments):
    # Model arguments
    model_name_or_path: Optional[str] = field(default="/hf_cache/hub/Qwen2.5-7B")
    dropout: float = field(default=0, metadata={"help": "model dropout"})
    zero_stage: int = field(default=2, metadata={"help": "zero stage"})
    offload_adam: bool = field(default=False, metadata={"help": "offload adam parameters to cpu"})
    offload_params: bool = field(default=False, metadata={"help": "offload model parameters to cpu"})
    
    # LoRA arguments
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA for fine-tuning"})
    lora_r: int = field(default=32, metadata={"help": "LoRA attention dimension"})
    lora_alpha: int = field(default=64, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of module names to apply LoRA to"}
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "LoRA bias type: none, all, or lora_only"}
    )

    # Data arguments
    data_path: str = field(default="ft_data",
                           metadata={"help": "Path to the training data."})

    eval_path: str = field(default="ft_data",
                           metadata={"help": "Path to the evaluating data."})

    # Training arguments
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Path to the huggingface hub."})
    model_max_length: int = field(default=20480, metadata={
        "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    num_train_epochs: int = field(default=5, metadata={"help": "number of training epochs"})
    overwrite_output_dir: bool = field(default=True)
    model_parallel_size: int = field(default=1, metadata={"help": "tensor parallel size"})
    learning_rate: float = field(default=2e-5, metadata={"help": "init learning rate"})
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "attention implementation"})
    fp32_loss: bool = field(default=False, metadata={"help": "whether calculate loss in fp32"})
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The scheduler type to use, default to consine"})
    min_lr_rate: float =  field(default=0.01, metadata={"help": "The minimum learning rate as a ratio of the initial learning rate."})
    deepspeed_file: str = field(default=None, metadata={"help": "deepspeed config file"})
    gradient_checkpointing: bool = field(default=True, metadata={"help": "gradient checkpointing"})
    warmup_steps: int = field(default=1000, metadata={"help": "warmup steps"})
    gradient_log_freq: int = field(default=50, metadata={"help": "gradient log frequency"})

    # wandb
    wandb_enabled: bool = field(default=False, metadata={"help": "whether use wandb"})
    wandb_project_name: str = field(default="pretrain", metadata={"help": "wandb project name"})
    

    def __str__(self):
        # 使用dataclasses.asdict()将所有字段转换为字典
        params_dict = asdict(self)
        # 创建表示参数的字符串
        params_str = '\n'.join(f"{k}: {v}" for k, v in params_dict.items())
        return params_str
