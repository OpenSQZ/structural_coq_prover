from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoModel
import torch
import re
import math
import json
from tqdm import tqdm
import torch.nn.functional as F
import os
import glob
from coq_prover.coq_context.utils import format_def

class LinqEmbedding:
    def __init__(self, model_path, gpu_id=0):
        self.max_length = 2048
        self.batch_size = 2
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.device = f'cuda:{gpu_id}'
        self.model.to(self.device)

    def last_token_pool(self, last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def get_detailed_instruct(self, premise, mode='premise') -> str:
        task_description = 'You are a master in mathematics and formal proofs. Please provide a deep interpretation of the following context:'
        if mode == 'premise':
            def_text = format_def(premise)
            query = f"Mathematical premise: {def_text} + Additional information: {json.dumps(premise['additional_info'])}"
        elif mode == 'string':
            query = premise
        elif mode == 'state':
            query = premise
        return f'Instruction: {task_description}\nQuery: {query}'

    def get_hidden_states_emb(self, texts, return_vec=False, mode='premise'):
        input_texts = [self.get_detailed_instruct(text, mode) for text in texts]
        batch_dict = self.tokenizer(input_texts, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt", truncation_strategy='longest_first')

        outputs = self.model(**batch_dict.to(self.device))
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Normalize embeddings  
        embeddings = F.normalize(embeddings, p=2, dim=1)
        if return_vec:
            return [vec.float().cpu().detach().numpy().flatten() for vec in embeddings]
        result = [','.join(map(str, vec.float().cpu().detach().numpy().flatten())) for vec in embeddings]
        return result

    def encode(self, texts):
        bs = 6
        batch_emb = []
        for i in range(0, len(texts), bs):
            batch_texts = texts[i:i+bs]
            batch_emb.extend(self.get_hidden_states_emb(batch_texts, return_vec=True, mode='string'))
        return batch_emb

    def generate(self, data_path):
        if isinstance(data_path, list):
            data = data_path
            output_file = None
            output_data = []
        else:
            output_file = data_path.replace('.jsonl','_emb.jsonl')
            with open(data_path, 'r', encoding='utf-8') as input_:
                all_lines = input_.readlines()
            data = [json.loads(line) for line in all_lines]

        if output_file is None:
            for batch_start_index in tqdm(range(0, len(data), self.batch_size), desc="Processing data_list"):
                data_batch = data[batch_start_index : batch_start_index + self.batch_size]
                emb_list = self.get_hidden_states_emb(data_batch)
                for premise,emb in zip(data_batch, emb_list):
                    premise['emb'] = emb
                    output_data.append(premise)
            return output_data
        
        with open(output_file, 'a', encoding='utf-8') as output:
            for batch_start_index in tqdm(range(0, len(data), self.batch_size), desc="Processing data_list"):
                data_batch = data[batch_start_index : batch_start_index + self.batch_size]
                emb_list = self.get_hidden_states_emb(data_batch)
                for premise,emb in zip(data_batch, emb_list):
                    premise['emb'] = emb
                for premise in data_batch:
                    output.write(json.dumps(premise,ensure_ascii=False) + '\n')