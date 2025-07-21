"""
API Testing Module for Coq Prover

This module provides functionality for:
1. Testing API in normal conditions
2. Beam search functionality (API-based)
3. Entropy calculation for response analysis
"""

import asyncio
import time
import numpy as np
from openai import OpenAI, AsyncOpenAI

# Test prompt for API testing
test_prompt = """
I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===

# Hypotheses:
s:## Internal: Coq.Strings.String.string.string
## Origin: Coq.Strings.String.string

ts:## Internal: ( Coq.Init.Datatypes.list.list Ceres.theories.CeresParserRoundtrip.Token.t.t )
## Origin: ( Coq.Init.Datatypes.list Ceres.theories.CeresParserRoundtrip.Token.t )

more:## Internal: Coq.Init.Datatypes.bool.bool
## Origin: Coq.Init.Datatypes.bool

# Goal:
## Internal: forall ( _Anonymous : ( Ceres.theories.CeresParserRoundtrip.token_string.token_string more ts s ) ) -> ( Ceres.theories.CeresParserRoundtrip.token_string.token_string Coq.Init.Datatypes.bool.false ( Coq.Init.Datatypes.app Ceres.theories.CeresParserRoundtrip.Token.t.t ts ( Coq.Init.Datatypes.list.cons Ceres.theories.CeresParserRoundtrip.Token.t.t Ceres.theories.CeresParserRoundtrip.Token.t.Close ( Coq.Init.Datatypes.list.nil Ceres.theories.CeresParserRoundtrip.Token.t.t ) ) ) ( Coq.Strings.String.append s ( Coq.Strings.String.string.String ( Coq.Strings.Ascii.ascii.Ascii Coq.Init.Datatypes.bool.true Coq.Init.Datatypes.bool.false Coq.Init.Datatypes.bool.false Coq.Init.Datatypes.bool.true Coq.Init.Datatypes.bool.false Coq.Init.Datatypes.bool.true Coq.Init.Datatypes.bool.false Coq.Init.Datatypes.bool.false ) Coq.Strings.String.string.EmptyString ) ) )
## Origin: ( Ceres.theories.CeresParserRoundtrip.token_string more ts s -> Ceres.theories.CeresParserRoundtrip.token_string Coq.Init.Datatypes.false ( ts ++ [Ceres.theories.CeresParserRoundtrip.Token.Close] ) ( s ++ ")" ) )

==========================
Suggest a list of 10 tactics to try - prefer single atomic tactics over compound ones unless the combination is highly confident. I will provide the compiler's response for each
Your response must be in this json format:
{
    tactics: [tactic1, tactic2, ...]
}
Ensure your response is a valid JSON without any other text.
"""


def logprobs_to_entropy(top_logprobs, first_token=False):
    """Calculate entropy from log probabilities"""
    logps = np.array([item.logprob for item in top_logprobs])
    ps = np.exp(logps - np.max(logps))  # Prevent overflow
    ps = ps / ps.sum()
    
    if first_token:
        std = np.std(logps)
        print(f"std: {std:.4f}")
    
    entropy = -np.sum(ps * np.log(ps + 1e-8))
    return float(entropy)


def test_api_normal():
    """Test API under normal conditions"""
    print("Testing API in normal conditions...")
    
    # Initialize clients
    client = OpenAI(
        api_key="", 
        base_url=""
    )
    
    try:
        response = client.chat.completions.create(
            model='deepseek-v3-0324',
            messages=[
                {"role": "system", "content": "You are an expert in Coq formal proof system."}, 
                {"role": "user", "content": test_prompt}
            ],
            temperature=0.3,
            max_tokens=100,
            logprobs=True,
            top_logprobs=20
        )
        
        result = response.choices[0].message.content
        print("Response:", result)
        
        # Calculate entropy
        entropy = logprobs_to_entropy(
            response.choices[0].logprobs.content[0].top_logprobs, 
            first_token=True
        )
        print(f"Entropy: {entropy}")
        
        # Calculate all entropies for analysis
        all_entropies = [
            logprobs_to_entropy(item.top_logprobs)
            for item in response.choices[0].logprobs.content
        ]
        
        mean_entropy = np.mean(all_entropies)
        std_entropy = np.std(all_entropies)
        print(f"Average entropy: {mean_entropy:.4f}, std: {std_entropy:.4f}")
        
        return True
        
    except Exception as e:
        print(f"API test failed: {e}")
        return False


async def beam_search_api():
    """Beam search functionality using API"""
    print("Running beam search...")
    
    client = AsyncOpenAI(
        api_key="", 
        base_url=""
    )
    
    try:
        response = await client.chat.completions.create(
            n=10,
            model='qwen-ins-2.5-32b-20480-old',
            messages=[{"role": "user", "content": test_prompt}],
            temperature=0.3,
            max_tokens=100,
            extra_body={'use_beam_search': True, 'length_penalty': 1.0}
        )
        
        print(f"Generated {len(response.choices)} beam search results")
        
        result_list = []
        for choice in response.choices:
            result_list.append(choice.message.content)
            
        return result_list
        
    except Exception as e:
        print(f"Beam search failed: {e}")
        return []


async def run_beam_search_test():
    """Run beam search test with timing"""
    start_time = time.time()
    results = await beam_search_api()
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time} seconds")
    print(f"Results: {results}")
    
    return results


def main():
    """Main testing function"""
    print("=== API Testing Module ===")
    
    # Test normal API functionality
    print("\n1. Testing normal API...")
    api_success = test_api_normal()
    
    # Test beam search
    print("\n2. Testing beam search...")
    asyncio.run(run_beam_search_test())
    
    print(f"\nAPI test result: {'PASSED' if api_success else 'FAILED'}")


if __name__ == "__main__":
    main()