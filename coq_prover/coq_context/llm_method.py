import asyncio
import os
from openai import AsyncOpenAI
import json
import time
import json5
import random
import warnings
import requests
import re
from json_repair import repair_json
from tenacity import retry, stop_after_attempt, wait_fixed

max_sample_n = 1

def get_env_variable(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Environment variable {var_name} is not set or empty")
    return value

try:
    reasoning_api_key = get_env_variable("API_KEY_REASONING")
    reasoning_base_url = get_env_variable("BASE_URL_REASONING")
    reasoning_model = get_env_variable("MODEL_REASONING")
    explanation_api_key = get_env_variable("API_KEY_EXPLANATION")
    explanation_base_url = get_env_variable("BASE_URL_EXPLANATION")
    explanation_model = get_env_variable("MODEL_EXPLANATION")

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the following environment variables:")
    print("- API_KEY_REASONING: API key for reasoning client")
    print("- BASE_URL_REASONING: Base URL for reasoning client")
    print("- MODEL_REASONING: Model for reasoning client")
    print("- API_KEY_EXPLANATION: API key for explanation client")
    print("- BASE_URL_EXPLANATION: Base URL for explanation client")
    print("- MODEL_EXPLANATION: Model for explanation client")
    raise

client_reasoning = AsyncOpenAI(api_key=reasoning_api_key, base_url=reasoning_base_url, timeout=120, max_retries=2)
client_explanation = AsyncOpenAI(api_key=explanation_api_key, base_url=explanation_base_url, timeout=120, max_retries=2)

client_use = client_reasoning

ft_host = None
ft_name = None
client_ft = None
is_instruct = None
tokenizer = None

def refine_response(response):
    if '</think>' in response:
        response = response.split('</think>')[1]
    if '```json' in response:
        response = response.split('```json',1)[1]
    if '```' in response:
        response = response.replace('```', '')
    # if response.strip().endswith('```'):
    #     return response.rsplit('```',1)[0]
    return response

def truncate_prompt(prompt: str, max_length: int = 100000, keep_length: int = 45000) -> tuple[bool, str]:
    if len(prompt) <= max_length:
        if len(prompt) > 60000:
            return True, prompt
        else:
            return False, prompt
    else:
        return False, prompt[:keep_length] + "\n...[TRUNCATED]...\n" + prompt[-keep_length:]

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
async def llm_call_logprobs(prompt, client):
    long_context, prompt = truncate_prompt(prompt)
    if client == client_reasoning:
        model = reasoning_model
    elif client == client_explanation:
        model = explanation_model
    else:
        raise ValueError("Invalid client", client)

    response = await asyncio.wait_for(
        client.chat.completions.create(
            logprobs=True,
            top_logprobs=20,
            model=model,
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

@retry(stop=stop_after_attempt(5), wait=wait_fixed(10))
async def llm_call(prompt, client, sample_size=1):
    long_context, prompt = truncate_prompt(prompt)
    if client == client_reasoning:
        model = reasoning_model
    elif client == client_ft:
        model = ft_name
    elif client == client_explanation:
        model = explanation_model
    else:
        raise ValueError("Invalid client", client)
    
    if client == client_ft:
        if is_instruct:
            resposne = await asyncio.wait_for(
                client.chat.completions.create(
                    n=10,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                    extra_body={'use_beam_search': True, 'length_penalty':1.0}
                ),
                timeout=300
            )
        else:
            resposne = await asyncio.wait_for(
                client.completions.create(
                    n=10,
                    model=model,
                    prompt=[prompt],
                    temperature=0.0,
                    max_tokens=100,
                    extra_body={'use_beam_search': True, 'length_penalty':1.0}
                ),
                timeout=300
            )
            
        result_list = []
        for choice in resposne.choices:
            if is_instruct:
                result_list.append(choice.message.content)
            else:
                result_list.append(choice.text)
        return result_list
    
    response = await asyncio.wait_for(
        client.chat.completions.create(
            # n=sample_size,
            model=model,
            messages=[
        {"role": "system", "content": "You are an expert in Coq formal proof system."},
        {"role": "user", "content": prompt},
    ],
        stream=False
    ),
    timeout=300
)
    if sample_size == 1:
        return response.choices[0].message.content
    else:
        return [choice.message.content for choice in response.choices]

async def format_simplify_respose(response, prompt):
    client = client_use
    attempts = 0
    max_attempts = 3

    while attempts < max_attempts:
        try:
            response = refine_response(response)
            json_response = json.loads(response)
            valid = True
            for concept_name, concept_data in json_response.items():
                if not all(k in concept_data for k in ['core_meaning', 'key_properties', 'tactics']):
                    valid = False
                    break
                    
                for tactic in concept_data['tactics']:
                    if not all(k in tactic for k in ['tactic_name', 'purpose', 'before_state', 'after_states']):
                        valid = False
                        break
                    if not all(k in tactic['before_state'] for k in ['hypotheses', 'goal']):
                        valid = False
                        break
                    for state in tactic['after_states']:
                        if not all(k in state for k in ['case', 'hypotheses', 'goal']):
                            valid = False
                            break

            if valid:
                return json_response
            else:
                print('simplify json key error: \n', response)
                attempts += 1
                if attempts >= max_attempts:
                    return response
                response = await llm_call(
                    prompt + "\nPlease ensure the response follows the exact format specified in the example.", 
                    client
                )
        except json.JSONDecodeError:
            print('simplify json format error: \n', response)
            attempts += 1
            if attempts >= max_attempts:
                return response
            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object.", client)
    
    return None

async def format_gen_response(response, prompt, force_tactics=False, refine_mode=False, if_reason=True):
    client = client_use
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        response = refine_response(response)
        try:
            json_response = json.loads(response)
            json_response = {k.lower(): v for k, v in json_response.items()}
            if force_tactics:
                if any(key in json_response.keys() for key in ['tactics']):
                    tactics_valid = True
                    if not if_reason:
                        if isinstance(json_response['tactics'], list):
                            return json_response
                    for tactic in json_response['tactics']:
                        if not all(key in tactic.keys() for key in ['tactic','reason']):
                            tactics_valid = False
                            break
                    if tactics_valid:
                        return json_response
                    else:
                        attempts += 1
                        if attempts >= max_attempts:
                            warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
                            return {'tactics': []}
                        response = await llm_call(prompt + "\nResponse must contain 'tactics' as the key and each tactic must contain 'tactic' and 'reason' as the keys. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
                        continue
            
            elif refine_mode:
                if all(key in json_response.keys() for key in ['refined_tactic', 'reason']):
                    return json_response
            else:
                if any(key in json_response.keys() for key in ['info', 'tactics']):
                    if 'tactics' in json_response.keys():
                        tactics_valid = True
                        if not if_reason:
                            if isinstance(json_response['tactics'], list):
                                return json_response
                        for tactic in json_response['tactics']:
                            if not all(key in tactic.keys() for key in ['tactic','reason']):
                                tactics_valid = False
                                break
                        if not tactics_valid:
                            attempts += 1
                            if attempts >= max_attempts:
                                warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
                                return {'tactics': []}
                            response = await llm_call(prompt + "\nResponse must contain 'tactics' as the key and each tactic must contain 'tactic' and 'reason' as the keys. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
                            continue
                    return json_response
                
            attempts += 1
            if attempts >= max_attempts:
                print('Format gen response error: \n', response)
                if refine_mode:
                    return {'refined_tactic': '', 'reason': 'llm response is not a valid JSON object some error occurred'}
                warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
                return {'tactics': []}
            
            if force_tactics:
                response = await llm_call(prompt + "\nResponse must contain 'tactics' as the key. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
            elif refine_mode:
                response = await llm_call(prompt + "\nResponse must contain 'refined_tactic' and 'reason' as the key. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
            else:
                response = await llm_call(prompt + "\nResponse must contain either 'info' or 'tactics' as the key. If 'tactics' is present, it must contain 'tactic' and 'reason' as the keys. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
        
        except json.JSONDecodeError:
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx + 1]
                    json_response = json.loads(repair_json(json_str))
                        
                    json_response = {k.lower(): v for k, v in json_response.items()}
                    if any(key in json_response.keys() for key in ['info', 'tactics']):
                        if 'tactics' in json_response.keys():
                            tactics_valid = True
                            for tactic in json_response['tactics']:
                                if not all(key in tactic.keys() for key in ['tactic','reason']):
                                    tactics_valid = False
                                    break
                            if not tactics_valid:
                                attempts += 1
                                if attempts >= max_attempts:
                                    warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
                                    return {'tactics': []}
                                response = await llm_call(prompt + "\nResponse must contain 'tactics' as the key and each tactic must contain 'tactic' and 'reason' as the keys. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
                                continue
                        return json_response
                    elif all(key in json_response.keys() for key in ['refined_tactic', 'reason']):
                        return json_response
            except:
                pass
            
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx + 1]
                    json_response = json5.loads(repair_json(json_str))
                    if any(key in json_response.keys() for key in ['info', 'tactics']):
                        if 'tactics' in json_response.keys():
                            tactics_valid = True
                            for tactic in json_response['tactics']:
                                if not all(key in tactic.keys() for key in ['tactic','reason']):
                                    tactics_valid = False
                                    break
                            if not tactics_valid:
                                attempts += 1
                                if attempts >= max_attempts:
                                    warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
                                    return {'tactics': []}
                                response = await llm_call(prompt + "\nResponse must contain 'tactics' as the key and each tactic must contain 'tactic' and 'reason' as the keys. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
                                continue
                        return json_response
                    elif all(key in json_response.keys() for key in ['refined_tactic', 'reason']):
                        return json_response
            except:
                pass
            
            print('tactics json format error: \n', response)
            attempts += 1
            if attempts >= max_attempts:
                if refine_mode:
                    return {'refined_tactic': '', 'reason': 'llm response is not a valid JSON object some error occurred'}
                warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
                return {'tactics': []}
            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
        
        except Exception as e:
            print('Format gen response error, some unexpected error occurred: \n', response)
            print(e)
            attempts += 1
            if attempts >= max_attempts:
                if refine_mode:
                    return {'refined_tactic': '', 'reason': 'llm response is not a valid JSON object some error occurred'}
                warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
                return {'tactics': []}
            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
    
    warnings.warn('Format gen response error return empty tactics, some unexpected error occurred: \n', response)
    return {'tactics': []}

async def format_method_response(response, prompt):
    client = client_use
    attempts = 0
    max_attempts = 3
    while attempts < max_attempts:
        response = refine_response(response)
        if 'brief strategy' in response.lower():
            return response, response.lower().rsplit('brief strategy',1)[1].strip()
        else:
            print('method json format error: \n', response)
            attempts += 1
            if attempts >= max_attempts:
                return None
            response = await llm_call(prompt + "\nPlease ensure the last section starts with 'brief strategy'.", client)
    
    return None
        
async def format_selection_response(response, prompt):
    client = client_use
    attempts = 0
    max_attempts = 3
    print('do format selection response, attempt: ', attempts)
    while attempts < max_attempts:
        original_response = response
        response = refine_response(response)
        try:
            json_response = json.loads(response)
            if 'states' in json_response:
                if len(json_response['states']) > 3:
                    print('selection number error: \n', len(json_response['states']))
                    attempts += 1
                    if attempts >= max_attempts:
                        response = json_response['states']
                        random.shuffle(response)
                        return response[:3]
                    response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object and contains 'states' as the key and a list of state numbers as the value. You can only select up to 3 states.", client)
                    continue
                return json_response['states']
            else:
                print('selection json key error: \n', original_response)
                attempts += 1
                if attempts >= max_attempts:
                    return None
                response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object and contains 'states' as the key and a list of state numbers as the value.", client)
        except json.JSONDecodeError:
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx + 1]
                    json_response = json.loads(json_str)
                    if 'states' in json_response.keys():
                        if len(json_response['states']) > 3:
                            print('selection number error: \n', len(json_response['states']))
                            attempts += 1
                            if attempts >= max_attempts:
                                response = json_response['states']
                                random.shuffle(response)
                                return response[:3]
                            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object and contains 'states' as the key and a list of state numbers as the value. You can only select up to 3 states.", client)
                            continue
                        return json_response['states']
            except:
                pass
            
            print('selection json format error: \n', original_response)
            attempts += 1
            if attempts >= max_attempts:
                return None
            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object.", client)
    
    return None

async def format_note_response(response, prompt):
    client = client_use
    attempts = 0
    max_attempts = 2
    while attempts < max_attempts:
        response = refine_response(response)
        try:
            json_response = json.loads(response)
            if 'add' in json_response and 'remove' in json_response:
                return json_response
            else:
                print('note json key error: \n', response)
                attempts += 1
                if attempts >= max_attempts:
                    return None
                response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object and contains 'add' and 'remove' as the keys.", client)
        except json.JSONDecodeError:
            try:
                start_idx = response.find('{')
                end_idx = response.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx + 1]
                    json_response = json.loads(json_str)
                    if 'add' in json_response and 'remove' in json_response:
                        return json_response
            except:
                pass
            print('note json format error: \n', response)
            attempts += 1
            if attempts >= max_attempts:
                return None
            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object.", client)
    
    return None

async def format_state_explanation(prompt, client, mode='state'):
    response = await llm_call(prompt, client)
    max_attempts = 3
    for i in range(max_attempts+1):
        response = refine_response(response)
        response = extract_json(response, mode)
        if response is not None:
            return response
        print('do state explanation attempt: ', i)
        if mode == 'state':
            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object and contains 'before', 'after', and 'tactic' as the keys. Each of these keys must contain 'zh' and 'en' as the keys. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
        else:
            response = await llm_call(prompt + "\nPlease ensure the response is a valid JSON object and contains 'proof_trace', 'steps', and 'score' as the keys. Remember to properly escape nested quotes using backslash (\) if necessary for a valid JSON object.", client)
    print('state explanation failed, fallback instead')
    return get_error_result(mode)


def extract_json(text: str, mode: str):
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx + 1]
            json_response = json.loads(json_str)
            if mode == 'state':
                if not all(key in json_response for key in ['before', 'after', 'tactic']):
                    return None
                for section in ['before', 'after', 'tactic']:
                    if not all(lang in json_response[section] for lang in ['zh', 'en']):
                        return None
                return json_response
            elif mode == 'strategy':
                if not all(key in json_response for key in ['proof_trace', 'steps', 'score']):
                    return None
                return json_response
    except:
        pass

    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx:end_idx + 1]
            json_response = json5.loads(repair_json(json_str))
            if mode == 'state':
                if not all(key in json_response for key in ['before', 'after', 'tactic']):
                    return None
                for section in ['before', 'after', 'tactic']:
                    if not all(lang in json_response[section] for lang in ['zh', 'en']):
                        return None
                return json_response
            elif mode == 'strategy':
                if not all(key in json_response for key in ['proof_trace', 'steps', 'score']):
                    return None
                return json_response
    except:
        pass

    return None

def get_error_result(mode: str):
    if mode == 'state':
            return {
                'before': {'zh': '解析失败：未能识别初始状态', 'en': 'Failed to identify initial state'},
                'after': {'zh': '解析失败：未能识别结果状态', 'en': 'Failed to identify result state'},
                'tactic': {'zh': '解析失败：未能识别tactic效果', 'en': 'Failed to identify tactic effect'}
            }
    else:
        return {
            'proof_trace': 'Failed to generate proof trace summary',
            'steps': 0,
            'score': 0
        }
        
def log_prompt_response(prompt,response):
    with open('prompt_response.jsonl', 'a', encoding='utf-8') as f:
        f.write(json.dumps({'prompt': prompt, 'response': response}) + '\n')

async def llm_response(prompt, ifGen=False, force_tactics=False, refine_mode=False, use_ft_model=False, if_reason=True):
    client = client_use if not use_ft_model else client_ft
    
    if use_ft_model:
        if 'reorganize' in ft_name:
            max_tokens = 2048
        else:
            max_tokens = 20480
        tokens = tokenizer.encode(prompt)
        
        if len(tokens) > max_tokens:
            prefix_tokens = tokens[:max_tokens//2]
            suffix_tokens = tokens[-max_tokens//2:]
            trimmed_tokens = prefix_tokens + suffix_tokens
            prompt = tokenizer.decode(trimmed_tokens)
    
    response = await llm_call(prompt, client)
    if use_ft_model:
        time_start = time.strftime("%H:%M:%S", time.localtime())
        print('do format gen response', time_start)
        return response
    if ifGen:
        # log_prompt_response(prompt, response)
        time_start = time.strftime("%H:%M:%S", time.localtime())
        print('do format gen response', time_start)
        return await format_gen_response(response, prompt, force_tactics=force_tactics, refine_mode=refine_mode, if_reason=if_reason)
    else:
        # log_prompt_response(prompt, response)
        time_start = time.strftime("%H:%M:%S", time.localtime())
        print('do format method response', time_start)
        return await format_method_response(response, prompt)
    
async def llm_simplify_response(prompt):
    client = client_use
    response = await llm_call(prompt, client)
    # log_prompt_response(prompt, response)
    return await format_simplify_respose(response, prompt)

async def llm_selection_response(prompt):
    client = client_use
    response = await llm_call(prompt, client)
    # log_prompt_response(prompt, response)
    return await format_selection_response(response, prompt)

async def llm_note_response(prompt):
    client = client_use
    response = await llm_call(prompt, client)
    # log_prompt_response(prompt, response)
    return await format_note_response(response, prompt)

async def llm_reorganize_response(prompt):
    client = client_use
    print('do reorganize response')
    response = await llm_call(prompt, client)
    return response

async def llm_state_explanation(prompts, mode='state'):
    if isinstance(prompts, str):
        prompts = [prompts]
    elif isinstance(prompts, list):
        prompts = prompts
    else:
        raise ValueError("Invalid prompts type, must be str or list")
    
    client = client_explanation
    time_start = time.strftime("%H:%M:%S", time.localtime())
    print('state explanation start', time_start)
    tasks = [format_state_explanation(prompt, client, mode=mode) for prompt in prompts]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    time_end = time.strftime("%H:%M:%S", time.localtime())
    print('state explanation response finished, time: ', time_end)
    return responses

async def llm_normal(prompt,sample_size=10,logprobs=False):
    client = client_use
    all_responses = []
    
    if logprobs:
        tasks = [llm_call_logprobs(prompt, client) for _ in range(sample_size)]
        results = await asyncio.gather(*tasks)
        return results
    
    num_calls = (sample_size + max_sample_n - 1) // max_sample_n
    tasks = [llm_call(prompt, client, sample_size=max_sample_n) for _ in range(num_calls)]
    responses = await asyncio.gather(*tasks)
    for response in responses:
        if isinstance(response, list):
            all_responses.extend(response)
        elif isinstance(response, str):
            all_responses.append(response)
        else:
            raise ValueError("Invalid response type, must be str or list")
   
    return all_responses[:sample_size]