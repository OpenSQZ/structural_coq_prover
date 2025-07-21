from coq_prover.coq_context.proof_generator import *
from coq_prover.coq_context.prompt_gen import PromptGenerator
from coq_prover.coq_context.emb_model import LinqEmbedding
from utils import get_config, read_jsonl_file
from tenacity import retry, stop_after_attempt, wait_fixed
import random
import numpy as np
from tqdm import tqdm
import json
from coq_prover.coq_context.llm_method import truncate_prompt, client_huoshan, refine_response
import json5
import asyncio
from coq_prover.coq_context.retrieval import Retrieval
from data_extraction.coq_data.Def_class import def_object
import re
import random
import os
from tqdm import tqdm
import scipy
from scipy.stats import entropy

class PremiseSelection:
    def __init__(self, config_path, sample_size=1000):
        random.seed(42)
        self.max_retry = 3
        self.sample_size = sample_size
        self.config = get_config(config_path)
        self.retrieval = Retrieval(self.config.paths.emb_data_path, model_name=self.config.paths.emb_model_path, mode='internal')
        self._init_components()
    
    def _init_components(self):
        self.ps_table = read_jsonl_file(self.config.paths.ps_table_path)
        self.ps_name = [item['name'] for item in self.ps_table]
        self.basic_dir = os.path.join(self.config.paths.extra_log_dir, 'uncertainty_metric/premise_selection')
        os.makedirs(self.basic_dir, exist_ok=True)
        self.output_dir = os.path.join(self.basic_dir, 'output.jsonl')

        ### tokenizer/def_table/ps_table now use old version
        self.tokenizer = Tokenizer(self.config.paths.tokenizer_path)
        self.def_table = read_jsonl_file(self.config.paths.def_table_path)
        self.theorems = {}
        for d in self.def_table:
            if d['kind'] == 'Proof' and d['name'] in self.ps_name:
                self.theorems[d['name']] = d
        print('valid def num:', len(self.theorems))
        self.retrieval = Retrieval(self.config.paths.emb_data_path, model_name=self.config.paths.emb_model_path, mode='internal')
        self.prompt_generator = PromptGenerator(self.config.paths.def_table_path, tokenizer=self.tokenizer, retrieval=self.retrieval)
        self.premise_ps = self.find_premise_ps_step()
        self.premise_ps = random.sample(self.premise_ps, self.sample_size)
        self.id_list = ['0','1','2','3','4','5','6','7','8','9']
    
    async def generate(self):
        batch_size = 50
        for i in tqdm(range(0, len(self.premise_ps), batch_size), desc='Generating',total=len(self.premise_ps)//batch_size):
            batch = self.premise_ps[i:i + batch_size]
            batch_tasks = []
            
            # Create tasks for each entry in the batch
            for ps_object, theorem in batch:
                # Run all four generate functions concurrently for each entry
                tasks = [
                    self.generate_no_def(ps_object, theorem),
                    self.generate_def_origin_no_intuition(ps_object, theorem),
                    self.generate_def_mixed_no_intuition(ps_object, theorem),
                    self.generate_def_mixed_intuition(ps_object, theorem)
                ]
                batch_tasks.append(tasks)
            
            batch_results = await asyncio.gather(*[asyncio.gather(*tasks) for tasks in batch_tasks])
            
            with open(self.output_dir, 'a') as f:
                for (ps_object, theorem), results in tqdm(zip(batch, batch_results), total=len(batch), desc='Generating'):
                    no_def_entry, def_origin_no_intuition_entry, def_mixed_no_intuition_entry, def_mixed_intuition_entry = results
                    if no_def_entry == None or def_origin_no_intuition_entry == None or def_mixed_no_intuition_entry == None or def_mixed_intuition_entry == None:
                        continue
                    final_entry = {
                        'ps_object': ps_object,
                        'theorem': theorem,
                        'no_def_entry': no_def_entry,
                        'def_origin_no_intuition_entry': def_origin_no_intuition_entry,
                        'def_mixed_no_intuition_entry': def_mixed_no_intuition_entry,
                        'def_mixed_intuition_entry': def_mixed_intuition_entry
                    }
                    f.write(json.dumps(final_entry) + '\n')
                
    async def process_prompt(self, prompt, right_index):
        if prompt == None:
            return None
        response, logprobs = await self.mini_llm_generate(prompt)
        return self.log_entry(prompt, response, logprobs, right_index)

    async def generate_no_def(self, ps_object: ps_object_single, theorem: def_object):
        prompt, right_index = self.prompt_generator.generate_premise_selection(ps_object, theorem, use_origin='origin', if_use_intuition=False, if_give_def=False)
        return await self.process_prompt(prompt, right_index)

    async def generate_def_origin_no_intuition(self, ps_object: ps_object_single, theorem: def_object):
        prompt, right_index = self.prompt_generator.generate_premise_selection(ps_object, theorem, use_origin='origin', if_use_intuition=False, if_give_def=True)
        return await self.process_prompt(prompt, right_index)
    
    async def generate_def_mixed_no_intuition(self, ps_object: ps_object_single, theorem: def_object):
        prompt, right_index = self.prompt_generator.generate_premise_selection(ps_object, theorem, use_origin='mixed', if_use_intuition=False, if_give_def=True)
        return await self.process_prompt(prompt, right_index)
    
    async def generate_def_mixed_intuition(self, ps_object: ps_object_single, theorem: def_object):
        prompt, right_index = self.prompt_generator.generate_premise_selection(ps_object, theorem, use_origin='mixed', if_use_intuition=True, if_give_def=True)
        return await self.process_prompt(prompt, right_index)
    
    def splitTextToTokens(self,text):
        parts = re.split(r'(:=|=>|<=|<-|/\\|\\/|\|-|\+|-\s|\s|\(|\)|\[|\]|\{|\}|\||,|~|%|;|:|@|\?|/)', text)
        filtered_parts = [part for part in parts if part != ' ' and part != '']
        
        result = []
        for part in filtered_parts:
            if part and len(part) > 1 and '!' in part and part[0].isdigit():
                excl_pos = part.find('!')
                if excl_pos == len(part) - 1:
                    result.extend([part[:excl_pos], "!"])
                else:
                    result.extend([part[:excl_pos], part[excl_pos+1:]])
            elif part and len(part) > 1 and part[0] == '!':
                result.extend(['!', part[1:]])
            else:
                result.append(part)
        return result 
    
    def find_premise_ps_step(self):
        premise_ps = []
        for item in tqdm(self.ps_table, desc='Finding premise ps',total=len(self.ps_table)):
            for proofstate in item['content']['proofstates']:
                tactic_str = proofstate['tactic']['name']
                tactic_str = self.splitTextToTokens(tactic_str)
                try:
                    tactic_idx = map(int, proofstate['tactic']['token_ids'].split(','))
                except:
                    continue
                # glob_premise_idx = [token for token in tactic_idx if token >= 1000 and '.' in self.tokenizer.decode(token)]
                try:
                    glob_premise_idx = [token for i,token in enumerate(tactic_idx) if token >= self.prompt_generator.glob_start_id and '.' in self.tokenizer.decode(token) and '.' in tactic_str[i]]
                except:
                    continue
                if len(glob_premise_idx) > 0 and item['name'] in self.theorems:
                    premise_ps.append((proofstate, self.theorems[item['name']]))
        print('premise ps num:', len(premise_ps))
        return premise_ps
    
    def transform_logprobs_to_dict(self, logprobs):
        return [item.model_dump() for item in logprobs]
    
    def log_entry(self, prompt, response_content, response_logprobs, target_answer):
        if response_content == None:
            return None
        
        if len(response_content) > 1:
            response_content = response_content[0]
            response_logprobs = response_logprobs[0]
        
        return {
            'answer': response_content,
            'logprobs': self.transform_logprobs_to_dict(response_logprobs),
            'ground_truth': target_answer,
            'prompt': prompt
        }

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
    async def mini_llm_call_entropy(self, prompt):
        model = 'ep-20250331194048-m97xb'
        _, prompt = truncate_prompt(prompt)
        response = await asyncio.wait_for(
            client_huoshan.chat.completions.create(
                    model=model,
                    logprobs=True,
                    top_logprobs=20,
                    max_tokens=3,
                    messages=[
                {"role": "system", "content": "You are an expert in Coq formal proof system."},
                {"role": "user", "content": prompt},
            ],
                stream=False
            ),
            timeout=300
        )
        response_content = response.choices[0].message.content
        response_logprobs = response.choices[0].logprobs.content

        return response_content, response_logprobs

    async def mini_llm_generate(self, prompt):
        response_content, response_logprobs = await self.mini_llm_call_entropy(prompt)
        
        for _ in range(self.max_retry):
            response_content, response_logprobs = await self.mini_llm_call_entropy(prompt)
            if response_content[0] in self.id_list:
                return response_content, response_logprobs
            else:
                response_content, response_logprobs = await self.mini_llm_call_entropy(prompt)
        print(prompt)
        print(response_content)
        return None,None