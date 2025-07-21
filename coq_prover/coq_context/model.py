import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from typing import Dict, List, Union
import torch.nn.functional as F
from torch import Tensor

class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.init_model()

    def init_model(self):
        if "DeepSeek" in self.model_name:
            max_memory = {i: "65GB" for i in range(8)}
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, device_map="sequential", max_memory=max_memory, torch_dtype=torch.float16, attn_implementation='flash_attention_2')
        elif 'NV-Embed-v2' in self.model_name:
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch.float16)
            self.model.to('cuda:0')
        elif 'Linq-Embed-Mistral' in self.model_name:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to('cuda:0')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode(self, input: Union[str, List[str], Dict], state_encode = False):
        if "DeepSeek" in self.model_name:
            return self.deepseek_encode(input, state_encode=state_encode)
        elif 'NV-Embed-v2' in self.model_name:
            return self.emb_encode(input, state_encode=state_encode)
        elif 'Linq-Embed-Mistral' in self.model_name:
            return self.emb_generate(input, state_encode=state_encode)
    
    def state_input(self, text: Dict):
        before_hyps = []
        for hyp_name, hyp_content in text['before_state']['hyps']['hyps'].items():
            before_hyps.append(hyp_name + ' : ' + hyp_content.get('processed', {}).get('origin', ''))
        before_hyps = '\n'.join(before_hyps)
        before_goal = text['before_state']['goal']['processed']['origin']
        return 'Hypotheses : \n' + before_hyps + '\n' + 'Goal : \n' + before_goal

    def deepseek_encode(self, texts: Union[str, List[str]], state_encode = False):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list):
            texts = [text for text in texts]
        elif isinstance(texts, dict):
            texts = [texts]
        else:
            raise ValueError("Invalid input type. Expected a string or a list of strings.")
        
        if state_encode:
            texts = [self.state_input(text) for text in texts]

        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors="pt")
        device = self.model.device
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  
            query_embeddings = hidden_states[:, -20:].mean(dim=1) 
        return query_embeddings
    
    def emb_encode(self, input: List[str], query_prefix: str = '', max_length: int = 4096, state_encode = False):
        if isinstance(input, str):
            input = [input]
        elif isinstance(input, list):
            input = [text for text in input]
        elif isinstance(input, dict):
            input = [input]
        else:
            raise ValueError("Invalid input type. Expected a string or a list of strings.")
        
        if state_encode:
            input = [self.state_input(text) for text in input]

        queries = [query_prefix + query for query in input]
        query_embeddings = self.model.encode(queries, instruction=query_prefix, max_length=max_length)
        embeddings = F.normalize(query_embeddings, p=2, dim=1)
        return embeddings

    def get_detailed_instruct(self, query: str) -> str:
        task_description = 'Given a problem-solving strategy or hint text, retrieve the most relevant Coq premises or tactics that should be applied.'
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def get_state_instruct(self, state: str) -> str:
        task_description = 'Given a Coq proof state with hypotheses and goal, retrieve the most relevant Coq premises or tactics that should be applied to progress the proof.'
        return f'Instruct: {task_description}\nQuery: {state}'

    def last_token_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def emb_generate(self, input: List[str], max_length: int = 2048, state_encode = False):
        if isinstance(input, str):
            input = [input]
        elif isinstance(input, list):
            input = [text for text in input]
        elif isinstance(input, dict):
            input = [input]
        else:
            raise ValueError("Invalid input type. Expected a string or a list of strings.")
       
        if state_encode:
            input_texts = [self.get_state_instruct(text) for text in input]
        else:
            input_texts = [self.get_detailed_instruct(text) for text in input]
        batch_dict = self.tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt", truncation_strategy='longest_first')
        
        device = self.model.device
        outputs = self.model(**batch_dict.to(device))
        embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # Normalize embeddings  
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings