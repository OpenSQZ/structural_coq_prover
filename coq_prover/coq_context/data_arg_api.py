import re
import json
import os
import asyncio
from openai import AsyncOpenAI
from tqdm import tqdm
from .prompt import EXTRACT_COQ_ESSENCE_PROMPT_JSON
from .utils import format_def
from .llm_method import get_env_variable
import json5
from json_repair import repair_json

try:
    reasoning_api_key = get_env_variable("API_KEY_REASONING")
    reasoning_base_url = get_env_variable("BASE_URL_REASONING")
    reasoning_model = get_env_variable("MODEL_REASONING")
except ValueError as e:
    print(f"Error: {e}")
    print("Please set the following environment variables:")
    print("- API_KEY_REASONING: API key for reasoning client")
    print("- BASE_URL_REASONING: Base URL for reasoning client")
    print("- MODEL_REASONING: Model for reasoning client")

client_reasoning = AsyncOpenAI(api_key=reasoning_api_key, base_url=reasoning_base_url)

def fix_json_escapes(json_str: str) -> str:
    try:
        pattern = r'\\(?![\\/"bfnrts]|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})'
        fixed_str = re.sub(pattern, r'\\\\', json_str)
        return fixed_str
    except Exception as e:
        print("err in _fix_json_escapes : %s", e)
        return json_str

def process_output(raw_text):
    required_keys = [
        "mathematical_domains",
        "key_concepts",
        "concept_relations",
        "intuitive_explanation",
        "dependent_premises",
        "potential_applications"
    ]
    content = raw_text
    if content.startswith('```'):
        content = content.split('\n', 1)[1]
        content = content.rsplit('\n', 1)[0]
    
    try:
        json_output = json.loads(content)
        if all(key in json_output for key in required_keys):
            return json_output
    except json.JSONDecodeError:
        content = repair_json(content)
        content = fix_json_escapes(content)
        try:
            json_output = json5.loads(content)
            if all(key in json_output for key in required_keys):
                return json_output
        except:
            pass

    return None

def resume_from_log(input_data,output_file):
    input_dict = {item['def_id']: item for item in input_data}
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        data = [json.loads(line) for line in all_lines]
        data_dict = {item['def_id']: item for item in data}
        need_process_items = [item for def_id, item in input_dict.items() if def_id not in data_dict]
        return need_process_items
    return input_data

error_log = 'data_arg_error.txt'

async def llm_call_async(prompts, premises, error_file_handle):
    async def process_single_prompt(prompt):
        try:
            response = await client_reasoning.chat.completions.create(
                model=reasoning_model,
                messages=[
                    {"role": "system", "content": "You are an expert in Coq formal proof system."},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                response_format={ "type": "json_object" }
            )
            return response.choices[0].message.content
        except Exception as e:
            return (prompt,e)

    tasks = [process_single_prompt(prompt) for prompt in prompts]
    responses = await asyncio.gather(*tasks)

    for i, r in enumerate(responses):
        if isinstance(r, tuple):
            prompt, e = r
            error_file_handle.write(f"Error processing premise: {premises[i]['name']}\n")
            error_file_handle.write(f"Premise: {premises[i]['name']}\n")
            error_file_handle.write(f"Prompt: {prompt}\n")
            error_file_handle.write(f"Error: {e}\n")
    
    format_responses = []
    for i, r in enumerate(responses):
        if not isinstance(r, tuple):
            format_response = process_output(r)
            if format_response is not None:
                format_responses.append(format_response)
            else:
                format_responses.append(None)
                error_file_handle.write(f"Error format json: {premises[i]['name']}\n")
                error_file_handle.write(f"Raw text: {r}\n")
        else:
            format_responses.append(None)
    return format_responses

async def data_argumentation_async(input_file, batch_size=150):
    with open(error_log, 'w', encoding='utf-8') as f:
        f.write('')

    with open(error_log, 'w', encoding='utf-8') as error_file_handle:
        if isinstance(input_file, list):
            data = input_file
            output_file = None
            output_data = []
        else:
            output_file = input_file.replace('.jsonl','_arged.jsonl')
        
            with open(input_file, 'r', encoding='utf-8') as input_:
                all_lines = input_.readlines()
                data = [json.loads(line) for line in all_lines]

        data = resume_from_log(data,output_file)

        total_processed = 0
        success_count = 0
        failed_count = 0

        for i in tqdm(range(0, len(data), batch_size), desc="Processing data:"):
            batch_data = data[i:i+batch_size]
            prompts = []
            vaild_data = []

            for premise in batch_data:
                def_text = format_def(premise)
                if def_text == '':
                    continue
                if premise['kind'] == 'Primitive':
                    continue
                vaild_data.append(premise)
                file_name=premise['file_path'] if not 'coq_train' in premise['file_path'] else premise['file_path'].split('coq_train/')[1]
                prompt = EXTRACT_COQ_ESSENCE_PROMPT_JSON.format(
                    file_name=file_name,
                    premise_text=def_text
                )

                if len(prompt) > 50000:
                    prompt = prompt[:20000] + "...[truncated]..." + prompt[-20000:]
                prompts.append(prompt)

            assert len(prompts) == len(vaild_data)
            answers = await llm_call_async(prompts, vaild_data, error_file_handle)
            
            for premise, answer in zip(vaild_data, answers):
                total_processed += 1
                if answer is not None:
                    premise['additional_info'] = answer
                    success_count += 1
                else:
                    premise['additional_info'] = ''
                    failed_count += 1
            
            if output_file is None:
                for data in batch_data:
                    if data['additional_info'] != '':
                        output_data.append(data)
            else:
                with open(output_file, 'a', encoding='utf-8') as output:
                    for premise in batch_data:
                        if premise['additional_info'] != '':
                            output.write(json.dumps(premise, ensure_ascii=False) + '\n')
        
    if failed_count > 0:
        print(f"Failed to process {failed_count} premises out of {total_processed}, please check the {error_log} for more details")

    if output_file is None:
        return output_data
    return output_file



