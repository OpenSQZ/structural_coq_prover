import random
import re
from transformers import AutoTokenizer,set_seed
from vllm import LLM, SamplingParams
import torch
import json
from tqdm import tqdm
from .utils import format_def
from .prompt import EXTRACT_COQ_ESSENCE_PROMPT_JSON
import numpy as np

def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class DeepSeekCoder:
    def __init__(self, model_name, max_model_len=8192, tp_size=8):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.tp_size = tp_size
        self.model = None
        self.tokenizer = None
        self._create_model()

    def _create_model(self):
        self.model = LLM(
            model=self.model_name,
            tensor_parallel_size=self.tp_size,
            dtype=torch.bfloat16,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            enforce_eager=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, enforce_eager=True)

    def generate(self, input_tensors,sampling_params):
        outputs = self.model.generate(prompt_token_ids=input_tensors,sampling_params=sampling_params,use_tqdm=True)
        processed_outputs = []
        for output in outputs:
            raw_text = output.outputs[0].text.strip()
            if '```json' in raw_text:
                raw_text = raw_text.split('```json',1)[1]
            if '```' in raw_text:
                raw_text = raw_text.replace('```', '').strip()
            processed_outputs.append(self.process_output(raw_text))
        return processed_outputs

    def get_tokenizer(self):
        return self.tokenizer

    def process_output(self, raw_text):
        required_keys = [
            "mathematical_domains",
            "key_concepts",
            "concept_relations",
            "intuitive_explanation",
            "dependent_premises",
            "potential_applications"
        ]
        try:
            json_output = json.loads(raw_text)
            if all(key in json_output for key in required_keys):
                return json_output
        except json.JSONDecodeError:
            pass
        extracted_info = self.extract_info_from_text(raw_text)
        return extracted_info
    
    def extract_info_from_text(self, text):
        extracted_info = {
            "mathematical_domains": [],
            "key_concepts": [],
            "concept_relations": "",
            "intuitive_explanation": "",
            "dependent_premises": [],
            "potential_applications": []
        }
        pattern = r"\"([\w\s]+)\":\s*((?:(?!\"[\w\s]+\":)[\s\S])*)"
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for match in matches:
            key = match.group(1).lower().replace(" ", "_")
            value = match.group(2).strip().rstrip(',')

            if key in ["mathematical_domains", "key_concepts", "dependent_premises", "potential_applications"]:
                extracted_info[key] = [item.strip().strip('"') for item in value.strip('[]').split(',')]
            elif key in ["concept_relations", "intuitive_explanation"]:
                extracted_info[key] = value.strip('"')           
        return extracted_info
    
class DataProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def get_inputs(self, data):
        prompts = []
        for premise in data:
            def_text = format_def(premise)
            if def_text == '':
                continue
            if premise['kind'] == 'Primitive':
                continue
            if premise['kind'] == 'Ltac':
                formatted_prompt = EXTRACT_COQ_ESSENCE_PROMPT_JSON.format(
                    file_name=premise['file_path'].split('coq_train/')[1],
                    premise_text=premise['name'] + ': ' + def_text
                )
            else:
                formatted_prompt = EXTRACT_COQ_ESSENCE_PROMPT_JSON.format(
                    file_name=premise['file_path'].split('coq_train/')[1],
                    premise_text=premise['name'] + ': ' + def_text
                )
            prompts.append([{"role": "user", "content": formatted_prompt}])
        inputs = [self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, truncation=True, max_length=4096) for prompt in prompts]
        return inputs


def data_argumentation(model_name, input_file):
    output_file = input_file.replace('.jsonl','_arged.jsonl')
    set_random_seed(42)
    if "DeepSeek-Coder-V2-Instruct" in model_name:
        model = DeepSeekCoder(model_name)
    else:
        # for test
        model = DeepSeekCoder(model_name, tp_size=1)

    processor = DataProcessor(model.get_tokenizer())
    sampling_params = SamplingParams(temperature=0.3, max_tokens=1024, stop_token_ids=[model.get_tokenizer().eos_token_id])

    with open(input_file, 'r', encoding='utf-8') as input_:
        all_lines = input_.readlines()
        data = [json.loads(line) for line in all_lines]
            
    with open (output_file, 'w', encoding='utf-8') as output:
        inputs = processor.get_inputs(data)
        answers = model.generate(inputs,sampling_params)

        assert len(data) == len(answers)

        for premise,answer in zip(data,answers):
            premise['additional_info'] = answer
        
        for premise in data:
            output.write(json.dumps(premise,ensure_ascii=False) + '\n')
        
    model.model.release()
    model.tokenizer.release()
