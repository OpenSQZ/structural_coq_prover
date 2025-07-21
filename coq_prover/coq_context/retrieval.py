import json
from utils import read_jsonl_file
from .model import Model
from typing import List, Union, Dict
import torch
import warnings

class Retrieval:
    def __init__(self, emb_file, model_name, mode='external'):
        self.mode = mode
        (self.premises_names, self.premises_vectors), (self.tactics_names, self.tactics_vectors) = self.load_vector_store(emb_file, model_name)
        
        if mode == 'internal':
            warnings.warn('No model name provided, can not use external retrieval')
        else:
            self.model = Model(model_name)
            

        self.premises_vectors = torch.stack(self.premises_vectors) if isinstance(self.premises_vectors, list) else self.premises_vectors
        self.tactics_vectors = torch.stack(self.tactics_vectors) if isinstance(self.tactics_vectors, list) else self.tactics_vectors
        
        assert len(self.premises_intuition) == len(self.premises_names)
        assert len(self.tactics_intuition) == len(self.tactics_names)
        self.premises_intuition_dict = dict(zip(self.premises_names, self.premises_intuition))
        self.tactics_intuition_dict = dict(zip(self.tactics_names, self.tactics_intuition))

    def retrieve_internal(self, premise_str:str, top_k=10):
        try:
            idx = self.premises_names.index(premise_str)
        except:
            return None
        
        premise_vector = self.premises_vectors[idx]
        device = 'cuda:0'
        scores = (premise_vector.to(device) @ self.premises_vectors.T.to(device)) * 100
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k)
        
        results = []
        for score, doc_idx in zip(top_k_scores, top_k_indices):
            results.append({
                'doc_name': self.premises_names[doc_idx],
                'score': score.item(),
                'intuition': self.premises_intuition[doc_idx]
            })
        return results

    def load_vector_store(self, emb_file, model_name):
        if 'DeepSeek' in model_name:
            use_ds_emb = True
        else:
            use_ds_emb = False

        data = read_jsonl_file(emb_file)
        premises_names = []
        premises_embs = []
        premises_intuition = []
        tactics_names = []
        tactics_embs = []
        tactics_intuition = []
        if use_ds_emb:
            emb_name = 'ds_emb'
        else:
            emb_name = 'emb'
        for item in data:
            emb = [float(x) for x in item[emb_name].split(',')]
            if item['kind'] == "Ltac":
                tactics_names.append(item['name'])
                tactics_embs.append(torch.tensor(emb))
                tactics_intuition.append(item['additional_info']['intuitive_explanation'])
            else:
                premises_names.append(item['name'])
                premises_embs.append(torch.tensor(emb))
                premises_intuition.append(item['additional_info']['intuitive_explanation'])
        self.premises_intuition = premises_intuition
        self.tactics_intuition = tactics_intuition
        return (premises_names,torch.stack(premises_embs)), (tactics_names,torch.stack(tactics_embs))

    def retrieve(self, query: Union[str, List[str], Dict], top_k=5, state_encode = False):
        if self.mode == 'internal':
            raise ValueError('Internal retrieval mode is not supported this method')
        
        device = 'cuda:0'
        query_emb = self.model.encode(query, state_encode = state_encode)

        premises_scores = (query_emb.to(device) @ self.premises_vectors.T.to(device)) * 100
        tactics_scores = (query_emb.to(device) @ self.tactics_vectors.T.to(device)) * 100
        top_k_premises_scores, top_k_premises_indices = torch.topk(premises_scores, k=top_k)
        top_k_tactic = top_k//2
        top_k_tactics_scores, top_k_tactics_indices = torch.topk(tactics_scores, k=top_k_tactic)
        premises_results = []
        for query_idx, (scores_per_query, indices_per_query) in enumerate(zip(top_k_premises_scores, top_k_premises_indices)):
            query_results = []
            for score, doc_idx in zip(scores_per_query, indices_per_query):
                query_results.append({
                    'doc_name': self.premises_names[doc_idx],
                    'score': score.item(),
                    'intuition': self.premises_intuition[doc_idx]
                })
            premises_results.append(query_results)

        tactics_results = []
        for query_idx, (scores_per_query, indices_per_query) in enumerate(zip(top_k_tactics_scores, top_k_tactics_indices)):
            query_results = []
            for score, doc_idx in zip(scores_per_query, indices_per_query):
                query_results.append({
                    'doc_name': self.tactics_names[doc_idx],
                    'score': score.item(),
                    'intuition': self.tactics_intuition[doc_idx]
                })
            tactics_results.append(query_results)
        return premises_results, tactics_results

    def get_intuition(self, doc_name):
        return self.premises_intuition_dict.get(doc_name, 
               self.tactics_intuition_dict.get(doc_name, ''))
        # if doc_name in self.premises_names:
        #     return self.premises_intuition[self.premises_names.index(doc_name)]
        # elif doc_name in self.tactics_names:
        #     return self.tactics_intuition[self.tactics_names.index(doc_name)]
        # else:
        #     ## bug return '' first
        #     return ''
        #     # raise ValueError(f"Document name {doc_name} not found in vector store")
