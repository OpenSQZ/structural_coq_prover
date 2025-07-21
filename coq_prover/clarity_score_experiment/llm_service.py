import os
from typing import Optional
from openai import AsyncOpenAI

class LLMService:
    def __init__(self) -> None:
        llm_env_config = self._get_llm_env_config()
        self._validate_llm_env_config(llm_env_config)
        
        self.client_huoshan = AsyncOpenAI(api_key=llm_env_config['api_key'], base_url=llm_env_config['base_url'], timeout=120, max_retries=2)
        self.model_id = llm_env_config['model_id']

    def _get_llm_env_config(self) -> dict:
        return {
            'model_id': os.getenv('MODEL_REASONING'),
            'api_key': os.getenv('API_KEY_REASONING'),
            'base_url': os.getenv('BASE_URL_REASONING')
        }

    def _validate_llm_env_config(self, config: dict):
        missing_vars = []
        error_messages = {
            'model_id': 'MODEL_REASONING haven\'t been set, please set it before running',
            'api_key': 'API_KEY_REASONING haven\'t been set, please set it before running',
            'base_url': 'BASE_URL_REASONING haven\'t been set, please set it before running'
        }
        
        for key, value in config.items():
            if not value:
                missing_vars.append(error_messages[key])
        
        if missing_vars:
            error_msg = "LLM service initialization failed, missing necessary environment variables:\n" + "\n".join(f"- {msg}" for msg in missing_vars)
            error_msg += "\n\nPlease refer to the README file to set the relevant environment variables."
            raise ValueError(error_msg)

    def _extract_llm_definition(self, result: Optional[str], extract_prompt: str) -> Optional[str]:
        if result is None:
            return None
        
        try:
            extracted = result.split(f"```{extract_prompt}")[1].split("```")[0]
            return extracted
        except Exception:
            # return the original result if extract failed
            return result

    async def _get_completion(self, prompt: str, max_tokens: int = 1000):
        try:
            response = await self.client_huoshan.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are an expert in Coq formal proof system."}, 
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=20
            )
            if response.choices[0].logprobs is None:
                raise ValueError("logprobs is None")
            return response.choices[0].message.content, response.choices[0].logprobs.content
        except Exception as e:
            print(f"Error in _get_completion: {str(e)}")
            raise e
    
    async def get_zh_def(self, prompt: str):
        result, _ = await self._get_completion(prompt, max_tokens=1000)
        return self._extract_llm_definition(result, 'explanation')

    async def get_llm_definition(self, prompt: str):
        result, _ = await self._get_completion(prompt, max_tokens=1000)
        return self._extract_llm_definition(result, 'coq')

    async def check_equivalence_with_logprobs(self, check_equivalence_prompt: str):
        return await self._get_completion(check_equivalence_prompt, max_tokens=10)
