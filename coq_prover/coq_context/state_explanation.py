import asyncio
import uuid
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Union, List, Optional, Dict
import re
import json
import json5
from json_repair import repair_json
from vllm import LLM, SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

@dataclass
class ProcessResult:
    index: int = 0
    success: bool = False
    result: Optional[Dict[str, Union[str, Dict[str, str]]]] = None
    raw_text: Optional[str] = None

class StateExplanation:
    def __init__(self, model_name, tp_size=4, max_model_len=20000):
        self.model_name = model_name
        self.tp_size = tp_size
        self.max_model_len = max_model_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enforce_eager=True)
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.tp_size,
            max_model_len=self.max_model_len,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
            gpu_memory_utilization=0.9
        )
        self.model = AsyncLLMEngine.from_engine_args(engine_args)
        self.sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=1024,
            stop_token_ids=[self.tokenizer.eos_token_id]
        )

        self.state_keys = ['before', 'after', 'tactic']
        self.strategy_keys = ['proof_trace', 'steps', 'score']
        self.required_lang_keys = ['zh', 'en']
    
    async def generate(self, state: Union[str, List[str]], mode: str = 'state', max_retries=5):
        if mode not in ['state', 'strategy']:
            raise ValueError("Invalid mode. Expected 'state' or 'strategy'.")
        
        if isinstance(state, str):
            state = [state]
        elif isinstance(state, list):
            state = state
        else:
            raise ValueError("Invalid input type. Expected a state or a list of states.")
        
        state = [s[:10000] + '\n...\n' + s[-10000:] if len(s) > 20000 else s for s in state]
    
        all_results = [None] * len(state)
        states_to_process = [(i, s) for i, s in enumerate(state)]
        
        for attempt in range(max_retries):
            if not states_to_process:
                break
            tasks = []
            for idx, prompt in states_to_process:
                request_id = f"{uuid.uuid4()}-{idx}"
                task = self._generate_single(prompt, request_id, mode)
                tasks.append((idx, task))
            
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            print(f"mode {mode} attempt {attempt} results: ", results)

            failed_states = []
            for (idx, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    print(f"Request {idx} failed with error: {result}")
                    failed_states.append((idx, state[idx]))
                    continue
                
                if result is None:
                    failed_states.append((idx, state[idx]))
                    continue
                    
                all_results[idx] = result
        
            states_to_process = failed_states
                
        for index, _ in states_to_process:
            all_results[index] = self._get_error_result(mode)

        return all_results
    
    async def _generate_single(self, 
                         prompt: str, 
                         request_id: str,
                         mode: str) -> Optional[Dict]:
        try:
            results_generator = self.model.generate(
                prompt,
                self.sampling_params,
                request_id=request_id
            )
            
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
                
            if not final_output:
                return None
                
            output_text = final_output.outputs[0].text
            process_result = self._extract_json(output_text, mode)
            
            if process_result.success:
                return process_result.result
            return None
            
        except asyncio.CancelledError:
            await self.model.abort(request_id)
            raise
        except Exception as e:
            raise e
    
    def _extract_json(self, text: str, mode: str) -> ProcessResult:
        try:
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                return ProcessResult(success=False, raw_text=text)
                
            json_str = json_match.group()
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                try:
                    data = json5.loads(repair_json(json_str))
                except Exception as e:
                    print(f"Failed to parse JSON with json5 as well: {e}")
                    return ProcessResult(success=False, raw_text=text)
            
            if mode == 'state':
                if not all(key in data for key in self.state_keys):
                    return ProcessResult(success=False, raw_text=text)
                
                for section in self.state_keys:
                    if not all(lang in data[section] for lang in self.required_lang_keys):
                        return ProcessResult(success=False, raw_text=text)
            elif mode == 'strategy':
                if not all(key in data for key in self.strategy_keys):
                        return ProcessResult(success=False, raw_text=text)
            return ProcessResult(success=True, result=data)
            
        except json.JSONDecodeError:
            return ProcessResult(success=False, raw_text=text)
    
    def _get_error_result(self, mode: str) -> Dict:
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