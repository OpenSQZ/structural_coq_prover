from unsloth import FastLanguageModel
import logging
import os
import time
import math
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import deepspeed
import torch
import torch.distributed
import torch.optim
import torch.utils.data
from tqdm import tqdm
import transformers
from transformers import get_scheduler

from coq_prover.coq_finetune.utils.ds_utils import get_train_ds_config
from deepspeed import DeepSpeedConfig, get_accelerator
from deepspeed.monitor.monitor import MonitorMaster
from coq_prover.coq_finetune.sft_dataloader import SFTDataset, SFTDataCollectFunctor
from coq_prover.coq_finetune.arguments import SFTArguments
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from coq_prover.coq_finetune.utils.model import print_model_parameters, save_model, init_from_pretrained
from coq_prover.coq_finetune.utils.file import dump_json_file, load_json_file
from coq_prover.coq_finetune.utils.train import get_all_reduce_mean, print_rank_0, set_random_seed, to_device, is_rank_0, check_model_precision
from coq_prover.coq_finetune.utils.train import clean_dict, clear_memory
from coq_prover.coq_finetune.utils.utils import parse_remaining_args_to_dict

set_random_seed(42)

class Trainer:
    def __init__(self):
        self.training_args = None

        self.ds_config = None
        self.device = None

        self.model = None
        self.tokenizer = None
        self.config = None
        self.ds_engine = None

        self.data_collator = None
        self.train_dataloader = None
        self.total_steps = None
        self.eval_dataloader = None
        self.monitor = None

        self.init()


    def init(self):
        # 初始化参数
        self.parse_arguments()
        # 初始化分布式
        self.init_distributed()
        # 初始化deepspeed配置
        self.init_ds_config()

        torch.distributed.barrier()
        print_rank_0("torch distributed barrier", rank=self.training_args.global_rank, wrap=True)

        # 初始化模型和tokenizer
        self.init_model_and_tokenizer()

        # 初始化数据集,dataloader
        self.build_dataloader()

        self.total_steps = len(self.train_dataloader) * self.training_args.num_train_epochs

        self.init_deepspeed()

        # self.init_monitor()

    def parse_arguments(self):
        parser = transformers.HfArgumentParser(SFTArguments)

        (self.training_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
        remaining_args_dict = parse_remaining_args_to_dict(remaining_args)

        print_rank_0(self.training_args)
        print_rank_0(remaining_args_dict)

    
    # def init_monitor(self):
    #    self.monitor = MonitorMaster(DeepSpeedConfig(self.training_args.deepspeed).monitor_config)

    def init_distributed(self):
        """Initialize distributed training setup."""
        accelerator = get_accelerator()
        if self.training_args.local_rank == -1:
            self.device = torch.device(accelerator.device_name())
        else:
            accelerator.set_device(self.training_args.local_rank)
            self.device = torch.device(accelerator.device_name(), self.training_args.local_rank)
            deepspeed.init_distributed(dist_backend="nccl", timeout=timedelta(minutes=10))
        self.training_args.global_rank = torch.distributed.get_rank()

    def build_ds_config(self):
        
        ds_config = get_train_ds_config(offload=self.training_args.offload_params,
                                        adam_offload=self.training_args.offload_adam,
                                        stage=self.training_args.zero_stage)
        
        ds_config["train_micro_batch_size_per_gpu"] = self.training_args.per_device_train_batch_size
        ds_config["train_batch_size"] = (
                self.training_args.per_device_train_batch_size
                * torch.distributed.get_world_size()
                * self.training_args.gradient_accumulation_steps
        )
        
        ds_config['wandb'] = {
            "enabled": self.training_args.wandb_enabled,
            "project": self.training_args.wandb_project_name
        }
        
        return ds_config
    
    def init_ds_config(self):
        if self.training_args.deepspeed_file:
            ds_config = load_json_file(self.training_args.deepspeed_file)
        
            ds_config["train_micro_batch_size_per_gpu"] = self.training_args.per_device_train_batch_size
            ds_config["train_batch_size"] = (
                self.training_args.per_device_train_batch_size
                * torch.distributed.get_world_size()
                * self.training_args.gradient_accumulation_steps
            )
        else:
            ds_config = self.build_ds_config()

        self.ds_config = ds_config
    
    def init_optimizer(self):
        no_decay = ["bias", "gamma", "beta", "layer_norm.weight", "layer_norm_1.weight", "layer_norm_2.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": self.training_args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad], "weight_decay": 0.0}
        ]

        if self.training_args.deepspeed:
            zero_stage = self.ds_config["zero_optimization"]["stage"]
            if zero_stage in (1, 2):
                optimizer_class = deepspeed.ops.adam.FusedAdam
            elif zero_stage == 3:
                optimizer_class = deepspeed.ops.adam.DeepSpeedCPUAdam
        else:
            if self.ds_config['zero_optimization']['offload_optimizer']['device']:
                offload_device = self.ds_config['zero_optimization']['offload_optimizer']['device']
                if offload_device in ('none', None):
                    optimizer_class = torch.optim.AdamW
                else:
                    optimizer_class = deepspeed.ops.adam.DeepSpeedCPUAdam
            else:
                raise ValueError(f"No optimizer class specified")
        optimizer = optimizer_class(optimizer_grouped_parameters, 
                                    lr=self.training_args.learning_rate, 
                                    betas=(0.9, 0.999))
        
        return optimizer
    
    def build_scheduler(self,optimizer):
        max_steps = self.training_args.num_train_epochs * len(self.train_dataloader)
        num_update_steps = math.ceil(max_steps / self.training_args.gradient_accumulation_steps)
        
        num_warmup_steps = math.ceil(self.training_args.warmup_steps / self.training_args.gradient_accumulation_steps)

        scheduler = get_scheduler(
            name=self.training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_update_steps,
            # scheduler_specific_kwargs={"min_lr_rate": self.training_args.min_lr_rate}
        )

        return scheduler

    def init_model_and_tokenizer(self):
        print_rank_0("start load model", rank=self.training_args.global_rank, wrap=True)
        print_rank_0("Applying LoRA adaptation to model", rank=self.training_args.global_rank, wrap=True)
        
        target_modules = self.training_args.lora_target_modules.split(",")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.training_args.model_name_or_path,
            max_seq_length = self.training_args.model_max_length,
            attn_implementation="flash_attention_2",
            dtype = torch.bfloat16,
            use_gradient_checkpointing = "unsloth",
            load_in_4bit=False,
            load_in_8bit=False,
            gpu_memory_utilization = 0.9,
            device_map = {"": self.device.index if hasattr(self.device, 'index') else 0}
        )

        model.config.use_cache = False
        model.config.attn_implementation = "flash_attention_2"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = FastLanguageModel.get_peft_model(
            model,
            r = self.training_args.lora_r,
            target_modules = target_modules, 
            lora_alpha = self.training_args.lora_alpha, 
            lora_dropout = self.training_args.lora_dropout, 
            bias = self.training_args.lora_bias,
            )
        
        if is_rank_0():
            model.print_trainable_parameters()
            check_model_precision(model)
        
        self.model = model
        self.tokenizer = tokenizer

        if is_rank_0():
            print_model_parameters(model)

        print_rank_0("end load model", rank=self.training_args.global_rank, wrap=True)
    
    def build_dataloader(self):
        train_dataset = SFTDataset(self.training_args.data_path)
        
        if self.training_args.eval_path:
            eval_dataset = SFTDataset(self.training_args.eval_path)
        else:
            eval_dataset = None

        if self.training_args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            if eval_dataset:
                eval_sampler = RandomSampler(eval_dataset)
            else:
                eval_sampler = None
        else:
            train_sampler = DistributedSampler(train_dataset, shuffle=False)
            if eval_dataset:
                eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
            else:
                eval_sampler = None

        if '-it' in self.training_args.model_name_or_path.lower() or 'instruct' in self.training_args.model_name_or_path.lower():
            print_rank_0("NOTICE: INSTURCT MODELS USE CHAT TEMPLATE", wrap=True)
            data_collect_functor = SFTDataCollectFunctor(self.tokenizer, self.training_args.model_max_length, use_chat_template=True)
        else:
            print_rank_0("NOTICE: BASE MODELS USED", wrap=True)
            data_collect_functor = SFTDataCollectFunctor(self.tokenizer, self.training_args.model_max_length)
        
        self.train_dataloader = DataLoader(
            train_dataset,
            num_workers=4,
            prefetch_factor=4,
            sampler=train_sampler,
            batch_size=self.training_args.per_device_train_batch_size,
            pin_memory=True,
            collate_fn=data_collect_functor
        )

        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                num_workers=4,
                prefetch_factor=4,
                sampler=eval_sampler,
                batch_size=self.training_args.per_device_eval_batch_size,
                pin_memory=True,
                collate_fn=data_collect_functor
            )
        else:
            self.eval_dataloader = None

    def init_deepspeed(self):
        print_rank_0("start deepspeed init", rank=self.training_args.global_rank, wrap=True)
        optimizer = self.init_optimizer()
        lr_scheduler = self.build_scheduler(optimizer)

        self.ds_engine, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            collate_fn=self.data_collator,
            args=self.training_args,
            config=self.ds_config,
            dist_init_required=True,
        )
        
        print_rank_0("end deepspeed init", rank=self.training_args.global_rank, wrap=True)

    def train(self):
        print_rank_0("start training", rank=self.training_args.global_rank, wrap=True)
        start_epoch = 0
        start_step = 0

        for epoch in range(start_epoch, int(self.training_args.num_train_epochs)):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{self.training_args.num_train_epochs}, "
                f"Total Micro Batches {len(self.train_dataloader)}",
                rank=self.training_args.global_rank,
            )
            self.train_epoch(epoch, start_step)
            start_step = 0
        
        self.save()
        if self.training_args.output_dir is not None:
            print_rank_0(
                "Saving the final model ...",
                rank=self.training_args.global_rank,
            )
            output_dir = self.training_args.output_dir
            final_model_path = f"{output_dir}/final"
            if hasattr(self.training_args, 'use_lora') and self.training_args.use_lora:
                if is_rank_0():
                    self.ds_engine.save_pretrained(f"{output_dir}/lora_adapter")
                    self.ds_engine.save_pretrained_merged(f"{output_dir}/lora_adapter_merged")
            else:
                self.ds_engine.save_checkpoint(final_model_path)
                self.ds_engine.save_fp16_model(f"{final_model_path}/fp16", "pytorch_model.bin") 
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        start_time = time.time()

        print_rank_0(f'Start evaluation on test data.')
        self.ds_engine.eval()

        losses = 0
        step = 0

        for batch in tqdm(self.eval_dataloader, desc="evaluating...", total=len(self.eval_dataloader), disable=not is_rank_0()):
            if batch is None:
                continue
            batch = to_device(batch, self.device)
            outputs = self.ds_engine(**batch, use_cache=False)
            loss = outputs.loss
            losses += loss.float()
            step += 1
            
            clean_dict(batch)
            del outputs
            del loss

        losses = losses / (step + 1)

        losses = get_all_reduce_mean(losses.clone().detach())

        perplexity = torch.exp(losses).item()

        print_rank_0(
            f'Evaluation completed in {(time.time() - start_time):.2f} seconds, loss = {losses.item()}, perplexity= {perplexity}')

        return losses.item(), perplexity

    def train_step(self, batch, step, epoch):
        assert self.ds_engine.training
        batch = to_device(batch, self.device)
        outputs = self.ds_engine(**batch, use_cache=False)
        loss = outputs.loss

        self.ds_engine.backward(loss)
        
        if step % self.training_args.gradient_log_freq == 0:
            self.log_gradient_info(step, epoch)
        
        self.ds_engine.step()

        self.log_step(epoch, step, loss)
        
        clean_dict(batch)

    def train_epoch(self, epoch, start_step=0):
        self.ds_engine.train()

        step_times = []
        start_epoch_time = time.time()

        for step, batch in tqdm(
            enumerate(self.train_dataloader),
            total=len(self.train_dataloader),
            desc="Train",
        ):
            if batch is None:
                continue

            try:
                start_time = time.time()
                self.train_step(batch,step,epoch)
                cost_time = time.time() - start_time
                self.log_time_info(epoch, step, start_epoch_time, start_time, step_times)

            except torch.cuda.OutOfMemoryError as e:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"{k}: {v.shape}")
                print(f"由于内存溢出错误，跳过此批次: {e}")
                torch.cuda.empty_cache()
                continue
            
            self.ds_engine.monitor.write_events([("ups", 1. / cost_time, self.ds_engine.global_samples)])

            if (step + 1) % self.training_args.save_steps == 0 and step != start_step:
                self.evaluate_and_save(epoch, step)
                self.ds_engine.train()
                # clear_memory()

        self.evaluate_and_save(epoch, step)

    def log_time_info(self, epoch: int, step: int, start_epoch_time: float, start_time: float, step_times: list):
        elapsed_time = time.time() - start_epoch_time
        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)
        steps_remaining = len(self.train_dataloader) - (step + 1)
        est_time_remaining = steps_remaining * avg_step_time
        samples_per_second = self.training_args.per_device_train_batch_size / avg_step_time
        
        est_time_remaining = time.strftime("%H:%M:%S", time.gmtime(est_time_remaining))
        
        print_rank_0(
            f"\nEpoch: {epoch}, Step: {step}, Global Rank: {self.training_args.global_rank}, current step time: {step_time:.2f}s"
            + f"\tElapsed: {elapsed_time:.2f}s, Est. Remaining: {est_time_remaining}s,"
            + f"\tSamples/sec: {samples_per_second:.2f}"
        )

    def log_step(self, epoch: int, step: int, loss):
        print_rank_0(
            f"\nEpoch: {epoch}, Step: {step}, Global Rank: {self.training_args.global_rank}, Loss: {loss} \t"
            + f"Lr: {self.ds_engine.optimizer.param_groups[0]['lr']}"
        )

        self.ds_engine.monitor.write_events(
            [('epoch', epoch, self.ds_engine.global_samples),
             ("step", step, self.ds_engine.global_samples),
             ("loss", loss, self.ds_engine.global_samples),
             ("lr", self.ds_engine.optimizer.param_groups[0]["lr"], self.ds_engine.global_samples)])

    def log_gradient_info(self, step, epoch):
        total_norm = 0.0
        param_count = 0
        param_with_grad_count = 0
        max_norm = 0.0
        
        global_norm = self.ds_engine.get_global_grad_norm()

        for name, param in self.ds_engine.named_parameters():
            grad = deepspeed.utils.safe_get_full_grad(param)
            param_count += 1
            if grad is not None:
                param_with_grad_count += 1
                param_norm = grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
        
        total_norm = total_norm ** 0.5
        
        global_norm = global_norm if global_norm else -1
        total_norm = total_norm if total_norm else -1
        max_norm = max_norm if max_norm else -1

        self.ds_engine.monitor.write_events([
            ("global_grad_norm", global_norm, self.ds_engine.global_samples),
            ("grad_norm", total_norm, self.ds_engine.global_samples),
            ("max_grad_norm", max_norm, self.ds_engine.global_samples),
            ("params_with_grad_ratio", param_with_grad_count / max(1, param_count), self.ds_engine.global_samples)
        ])
        
        print_rank_0(
            f"Epoch: {epoch}, Step: {step}, Global Gradient Norm: {global_norm:.4f}, "
            f"Gradient Norm: {total_norm:.4f}, "
            f"Max Gradient: {max_norm:.4f}, Params with grad: {param_with_grad_count}/{param_count}",
            rank=self.training_args.global_rank
        )

    def evaluate_and_save(self, epoch, step):
        loss,perplexity  = self.evaluate()
        self.ds_engine.monitor.write_events(
            [("eval_loss", loss, self.ds_engine.global_samples),
             ("eval_perplexity", perplexity, self.ds_engine.global_samples)])
        print_rank_0(
            f"Epoch: {epoch}, Step: {step}, Loss: {loss}, Perplexity: {perplexity}",
            rank=self.training_args.global_rank,
        )
        self.save(epoch,step=step)

    def save(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ):
        output_dir = self.training_args.output_dir

        if epoch is not None:
            model_save_dir = f"{output_dir}/epoch_{epoch}"
            if step:
                model_save_dir += f"_step_{step}"
        else:
            model_save_dir = f"{output_dir}/final"

        print_rank_0(
            f"Saving the model, epoch {epoch}, step {step}...",
            rank=self.training_args.global_rank,
        )

        # client_state = {
        #     "epoch": epoch,
        #     "step": step,
        #     "global_samples": self.ds_engine.global_samples,
        #     "scheduler_state": self.lr_scheduler.state_dict()
        # }
        # self.ds_engine.save_checkpoint(output_dir, client_state=client_state)

        try:
            if hasattr(self.training_args, 'use_lora') and self.training_args.use_lora:
                if is_rank_0():
                    os.makedirs(model_save_dir, exist_ok=True)
                    self.ds_engine.save_pretrained(f"{model_save_dir}/lora_adapter")
                    self.ds_engine.save_pretrained_merged(f"{model_save_dir}/lora_adapter_merged")
            else:
                save_model(self.ds_engine, self.config, self.tokenizer, model_save_dir)
        except:
            print_rank_0(
                f"Failed to save the model, epoch {epoch}, step {step}",
                rank=self.training_args.global_rank,
            )
            self._save_checkpoint(step)
    
    def _save_checkpoint(self, total_steps: int):
        steps_model_path = os.path.join(self.training_args.output_dir, '{}_steps'.format(total_steps))

        if self.ds_config["zero_optimization"]["stage"] == 3:
            state_dict = self.ds_engine._zero3_consolidated_16bit_state_dict()
        else:
            state_dict = self.ds_engine.state_dict()
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        if is_rank_0():
            if hasattr(self.training_args, 'use_lora') and self.training_args.use_lora:
                lora_path = os.path.join(steps_model_path, 'lora_adapter')
                os.makedirs(lora_path, exist_ok=True)
                self.ds_engine.save_pretrained(lora_path)
                lora_path = os.path.join(steps_model_path, 'lora_adapter_merged')
                os.makedirs(lora_path, exist_ok=True)
                self.ds_engine.save_pretrained_merged(lora_path)
            else:
                self.ds_engine.save_pretrained(steps_model_path, state_dict=state_dict)
                self.tokenizer.save_pretrained(steps_model_path)


            logging.info(f'Saved model of {total_steps} steps to {steps_model_path}')

        del state_dict
        torch.distributed.barrier()
