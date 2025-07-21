import numpy as np
from typing import Dict, Optional

def calculate_scores(logprobs: list) -> Optional[Dict[str, float]]:
    """calculate the probability score of YES/NO"""
    # get the result token and top20 logprobs
    result_token = logprobs[0].token.strip().upper()
    top_logprobs = logprobs[0].top_logprobs

    # if the result is not strict YES or NO, return None
    if result_token != 'YES' and result_token != 'NO':
        return None
    
    # initialize the logprob values
    yes_logprob = float('-inf')
    no_logprob = float('-inf')
    
    # iterate to find the logprob values of YES and NO
    for item in top_logprobs:
        token = item.token.strip().upper()
        if token == 'YES':
            yes_logprob = item.logprob
        elif token == 'NO':
            no_logprob = item.logprob
    
    # use the logprob of the actual returned result
    if result_token == 'YES':
        yes_logprob = logprobs[0].logprob
    elif result_token == 'NO':
        no_logprob = logprobs[0].logprob
        
    # calculate the score
    exp_yes = np.exp(yes_logprob)
    exp_no = np.exp(no_logprob)
    lm_score = exp_yes / (exp_yes + exp_no)
    
    return {
        'lm_score': round(float(lm_score), 2),
        'yes_prob': round(float(exp_yes / (exp_yes + exp_no)) if (exp_yes + exp_no) > 0 else 0, 2),
        'no_prob': round(float(exp_no / (exp_yes + exp_no)) if (exp_yes + exp_no) > 0 else 0, 2)
    }