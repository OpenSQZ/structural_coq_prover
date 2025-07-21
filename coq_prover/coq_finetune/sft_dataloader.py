import json
import mmap
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from coq_prover.coq_finetune.utils.logger import is_rank_0, print_rank_0
from coq_prover.coq_finetune.utils.utils import pad_sequences
  
IGNORE_INDEX = -100 

def generate_prompt_func(e: json) -> Tuple[str, str]:
    # for base
    # prompt = f"[INS] {e['prompt']} [/INS]"
    # prompt = f"{e['prompt_reorganize']}"
    prompt = e['prompt']
    output = e['tactic']
    return prompt, output

class MemoryMappedDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.line_offsets = []
        
        print_rank_0(f"Indexing file: {file_path}")
        with open(file_path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 0
            while True:
                line = mm.readline()
                if not line:
                    break
                self.line_offsets.append(offset)
                offset = mm.tell()
            mm.close()
        
        print_rank_0(f"Indexed {len(self.line_offsets)} lines")
        
        self.file = open(file_path, 'rb')
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, idx):
        self.mm.seek(self.line_offsets[idx])
        
        line = self.mm.readline().decode('utf-8')
        item = json.loads(line)
        
        return generate_prompt_func(item)
    
    def __del__(self):
        if hasattr(self, 'mm') and self.mm:
            self.mm.close()
        if hasattr(self, 'file') and self.file:
            self.file.close()

class SFTDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        
        # 读取 JSONL 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="loading data", disable=not is_rank_0()):
                item = json.loads(line)
                # prompt = apply_chat_template(item)
                # if len(tokenizer.encode(prompt, add_special_tokens=False)) < max_length:
                #     self.data.append(item)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return generate_prompt_func(item)


class SFTDataCollectFunctor:
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int, use_chat_template: bool = False):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.ignore_tactic_num = 0
        self.total_tactic_num = 0
        self.use_chat_template = use_chat_template

    def __call__(self, batch: List[Tuple[str, str]]):
        encoded_data = []
        for input_text, output_text in batch:

            self.total_tactic_num += 1

            if self.use_chat_template:
                user_messages = [{"role": "user", "content": input_text}]
                input_encoding = self.tokenizer.apply_chat_template(user_messages, 
                                                                add_generation_prompt=True,
                                                                return_tensors='pt')[0]
            else:
                input_encoding = self.tokenizer.encode(
                                                    input_text, 
                                                    add_special_tokens=False,
                                                    padding=False,
                                                    return_tensors='pt')[0]

            output_encoding = self.tokenizer.encode(
                output_text, 
                add_special_tokens=False,
                padding=False,
                return_tensors='pt'
            )[0]
            
            ## tactic is so complex, so we filter it
            if len(output_encoding) > 250:
                print(f"tactic is too complex, length: {len(output_encoding)}")
                # print(f"output_text: {output_text}")
                self.ignore_tactic_num += 1
                ## for bs=1 none will cause asyn error ?? 
                output_encoding = output_encoding[:250]
                print(f"ignore_tactic_num: {self.ignore_tactic_num}, total_tactic_num: {self.total_tactic_num}, ratio: {self.ignore_tactic_num / self.total_tactic_num}")
                # continue

            if len(input_encoding) + len(output_encoding) + 2 > self.max_length:  # +2 for BOS and EOS tokens
                available_length = self.max_length - 2 - len(output_encoding)
                half_length = available_length // 2
                first_part = input_encoding[:half_length]
                second_part = input_encoding[-half_length:]
                input_encoding = torch.cat([first_part, second_part])
            
            encoded_data.append((input_encoding, output_encoding))
        
        if encoded_data:
            batch_max_length = min(
                self.max_length,
                max([len(input_enc) + len(output_enc) + 2 for input_enc, output_enc in encoded_data])
            )
        else:
            return None

        input_ids = []
        attention_masks = []
        labels = []
        
        eos_token_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        try:
            bos_token_id = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        except:
            bos_token_id = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        
        for input_encoding, output_encoding in encoded_data:
            full_input_ids = torch.cat([bos_token_id, input_encoding, output_encoding, eos_token_id])
            
            if len(full_input_ids) > self.max_length:
                print(self.tokenizer.decode(full_input_ids))
                raise ValueError(f"Input length exceeds max length: {len(full_input_ids)}")
            
            attention_mask = torch.ones_like(full_input_ids)

            output_start_pos = 1 + len(input_encoding)
            labels_ids = torch.full_like(full_input_ids, -100)
            labels_ids[output_start_pos:] = full_input_ids[output_start_pos:]
            
            if len(full_input_ids) < batch_max_length:
                padding_length = batch_max_length - len(full_input_ids)
                full_input_ids = torch.cat([full_input_ids, torch.full((padding_length,), self.tokenizer.pad_token_id, dtype=torch.long)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
                labels_ids = torch.cat([labels_ids, torch.full((padding_length,), -100, dtype=torch.long)])
                
            input_ids.append(full_input_ids)
            attention_masks.append(attention_mask)
            labels.append(labels_ids)

        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }


