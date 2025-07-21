EXTRACT_COQ_ESSENCE_PROMPT_JSON = """You are an expert in formal methods and theorem proving, specializing in Coq. Your task is to analyze and extract the essential information from Coq premises, which may include definitions, theorems, lemmas, or other constructs.

Given a Coq premise and its context, provide a structured analysis as follows:

1. Mathematical Domains: List the mathematical or logical domains this premise belongs to, from broad to specific (e.g., ["Algebra", "Abstract Algebra", "Group Theory"]).

2. Key Concepts: List 3-5 fundamental concepts or constructs in Coq or mathematics that are central to this premise.

3. Concept Relations: Briefly describe how the key concepts relate to each other in the context of this premise.

4. Intuitive Explanation: Give a concise, intuitive explanation of what this premise means or does, in plain language.

5. Dependent Premises: List 2-3 other premises or constructs that this premise likely depends on or builds upon.

6. Potential Applications: Suggest 2-3 other premises or theorems that might use or build upon this premise in future proofs or constructions.

Ensure your analysis is precise, concise, and captures the essence of the Coq premise without introducing external information not present in the given text. Provide your output in JSON format, with keys corresponding to each of the above points.

########
Output example:
{{
  "mathematical_domains": ["Number Theory", "Elementary Number Theory", "Divisibility Theory"],
  "key_concepts": ["Prime numbers", "Divisibility", "Fundamental Theorem of Arithmetic", "Unique factorization"],
  "concept_relations": "This premise relates prime numbers to the concept of divisibility, leveraging the Fundamental Theorem of Arithmetic to establish a unique factorization property.",
  "intuitive_explanation": "This theorem states that every positive integer can be represented uniquely as a product of prime numbers, up to the order of the factors.",
  "dependent_premises": ["Definition of prime numbers", "Definition of divisibility", "Fundamental Theorem of Arithmetic"],
  "potential_applications": ["Proofs involving factorization of integers", "Theorems about greatest common divisors", "Cryptographic algorithms based on prime factorization"]
}}
########

######################
-Real Data-
Your response MUST be in valid JSON format. Do not include any text outside of the JSON structure. Ensure that your output can be directly parsed as JSON.
######################
Coq File: {file_name}
Premise: {premise_text}
######################
JSON Output:
"""

DEFINITION_FORMAT = """
## Name: {name}
## Content: 
{content}
## Intuition: {intuition}
"""

DEFINITION_FORMAT_NO_INTUITION = """
## Name: {name}
## Content: 
{content}
"""

TACTIC_FORMAT = """
### Tactic Name: {name}

### Example:
#### Before State
{before_state}

#### After State
{after_states}"""

TACTIC_STATE_FORMAT = """
##### Hypotheses:
{hyps}

##### Goal:
    {goal}"""

STATE_NODEF_FORMAT = """
# Hypotheses:
{hyps}

# Goal:
    {goal}"""

STATE_FORMAT = """
# Hypotheses:
{hyps}

# Goal:
{goal}

Global definitions referenced:
# Glob def:
{glob_def}
"""

CONCEPT_FORMAT = """
## Concept Name: 
  {concept}

## Related Tactic:
{tactics}
"""

TACTIC_STR_FORMAT = """
## Tactic Name: {name}
## Context: {context}
## Intuition: {intuition}
"""

TACTIC_STR_FORMAT_NO_INTUITION = """
## Tactic Name: {name}
## Context: {context}
"""

INTUITION_FORMAT = """### Before State: {before_state}
### After State: {after_state}
### Tactic: {tactic}"""

TACTIC_STRATEGY_FORMAT = """## Strategy: {strategy}
## Tactic: {tactic}
## Intuition: {intuition}"""

INTERNAL_ORIGIN_MIXED_FORMAT = """## Internal: {internal}
## Origin: {origin}"""

####### prompt module #######
PROOF_TRACING_FORMAT = """=== Proof Tracing ===
This shows how we reached the current state through previous tactics:

Tactics: {tactic_seq}
{proof_summary}"""

PUBLIC_NOTES_FORMAT = """=== Public Notes ===
Curated insights relevant to current proof:
{public_notes}"""

RETRIEVE_INFO_FORMAT = """=== Related Premises ===
Potentially relevant premises (for reference only):
{premises}

=== Related Tactic ===
Commonly used tactics for similar proofstates (for reference only):
{tactics}
"""

STRATEGY_FORMAT = """=== Hint ===
Some hints may help you to understand the proof:

{hint}
"""

GENERATE_METHOD4NEXT_STEP_PROMPT_FT_DATA = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{public_notes}

{proof_tracing}

=== Background Knowledge ===
Examples of concepts appearing in the current state
Helpful for understanding the coq concepts:
{concepts}
Describe your proof strategy and thought process by addressing the following aspects:

1. What is the core mathematical concept or property you aim to prove or use in the next step?

2. Which specific lemma, theorem, or property could be most useful for advancing your proof at this stage? Outline the form of this result without needing to refer to specific names.

3. What proof technique do you plan to apply? (e.g., induction, contradiction, case analysis)

4. What relationships between the terms and hypotheses in the current state are key to moving forward with the proof?

5. What are the immediate transformations or intermediate steps required to simplify or break down the problem toward your goal?

Please structure your response as clear, focused statements about your proof strategy. Avoid discussing implementation details or specific Coq tactics. The goal is to identify the logical flow and strategies that will guide the next part of the proof.

After answering the above questions, please provide a **brief strategy** for the next step. Start this section with the phrase **'brief strategy'** and focus on the general approach or method that will guide you forward in the proof. Keep this section concise and high-level, without going into too much detail.

Please ensure the last section starts with 'brief strategy'.

======================
[NOTICE]: I am using this to generate fine-tuning data, so at the end I will provide the correct answer. Please construct your responses to the above questions based on the following actual answer. The purpose of showing you this answer is to guide your reasoning in the right direction, not to have you reveal the solution directly.
[ANSWER]: {answer}
"""

GENERATE_METHOD4NEXT_STEP_PROMPT_NO_BACKGROUND_FT_DATA = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{public_notes}

{proof_tracing}

Describe your proof strategy and thought process by addressing the following aspects:

1. What is the core mathematical concept or property you aim to prove or use in the next step?

2. Which specific lemma, theorem, or property could be most useful for advancing your proof at this stage? Outline the form of this result without needing to refer to specific names.

3. What proof technique do you plan to apply? (e.g., induction, contradiction, case analysis)

4. What relationships between the terms and hypotheses in the current state are key to moving forward with the proof?

5. What are the immediate transformations or intermediate steps required to simplify or break down the problem toward your goal?

Please structure your response as clear, focused statements about your proof strategy. Avoid discussing implementation details or specific Coq tactics. The goal is to identify the logical flow and strategies that will guide the next part of the proof.

After answering the above questions, please provide a **brief strategy** for the next step. Start this section with the phrase **'brief strategy'** and focus on the general approach or method that will guide you forward in the proof. Keep this section concise and high-level, without going into too much detail.

Please ensure the last section starts with 'brief strategy'.

======================
[NOTICE]: I am using this to generate fine-tuning data, so at the end I will provide the correct answer. Please construct your responses to the above questions based on the following actual answer. The purpose of showing you this answer is to guide your reasoning in the right direction, not to have you reveal the solution directly.
[ANSWER]: {answer}
"""


GENERATE_METHOD4NEXT_STEP_PROMPT = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{public_notes}

{proof_tracing}

=== Background Knowledge ===
Examples of concepts appearing in the current state
Helpful for understanding the coq concepts:
{concepts}
Describe your proof strategy and thought process by addressing the following aspects:

1. What is the core mathematical concept or property you aim to prove or use in the next step?

2. Which specific lemma, theorem, or property could be most useful for advancing your proof at this stage? Outline the form of this result without needing to refer to specific names.

3. What proof technique do you plan to apply? (e.g., induction, contradiction, case analysis)

4. What relationships between the terms and hypotheses in the current state are key to moving forward with the proof?

5. What are the immediate transformations or intermediate steps required to simplify or break down the problem toward your goal?

Please structure your response as clear, focused statements about your proof strategy. Avoid discussing implementation details or specific Coq tactics. The goal is to identify the logical flow and strategies that will guide the next part of the proof.

After answering the above questions, please provide a **brief strategy** for the next step. Start this section with the phrase **'brief strategy'** and focus on the general approach or method that will guide you forward in the proof. Keep this section concise and high-level, without going into too much detail.

Please ensure the last section starts with 'brief strategy'.
"""

GENERATE_METHOD4NEXT_STEP_PROMPT_NO_BACKGROUND = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{public_notes}

{proof_tracing}

Describe your proof strategy and thought process by addressing the following aspects:

1. What is the core mathematical concept or property you aim to prove or use in the next step?

2. Which specific lemma, theorem, or property could be most useful for advancing your proof at this stage? Outline the form of this result without needing to refer to specific names.

3. What proof technique do you plan to apply? (e.g., induction, contradiction, case analysis)

4. What relationships between the terms and hypotheses in the current state are key to moving forward with the proof?

5. What are the immediate transformations or intermediate steps required to simplify or break down the problem toward your goal?

Please structure your response as clear, focused statements about your proof strategy. Avoid discussing implementation details or specific Coq tactics. The goal is to identify the logical flow and strategies that will guide the next part of the proof.

After answering the above questions, please provide a **brief strategy** for the next step. Start this section with the phrase **'brief strategy'** and focus on the general approach or method that will guide you forward in the proof. Keep this section concise and high-level, without going into too much detail.

Please ensure the last section starts with 'brief strategy'.
"""

GENERATE_METHOD4NEXT_STEP_PROMPT_RECONSIDER = """{previous_method_prompt}

======= PREVIOUS METHODOLOGY =======
{previous_method}

======= PREVIOUS TACTIC AND ERROR =======
Here is the tactic we have tried and did not work:
{previous_tactic}

Given this error, please reconsider your proof strategy. Address the same 5 questions as before, but with adjustments that account for why the previous approach failed:

1. What is the core mathematical concept or property you aim to prove or use in the next step?

2. Which specific lemma, theorem, or property could be most useful for advancing your proof at this stage? Outline the form of this result without needing to refer to specific names.

3. What proof technique do you plan to apply? (e.g., induction, contradiction, case analysis)

4. What relationships between the terms and hypotheses in the current state are key to moving forward with the proof?

5. What are the immediate transformations or intermediate steps required to simplify or break down the problem toward your goal?

Please structure your response as clear, focused statements about your proof strategy. Avoid discussing implementation details or specific Coq tactics. The goal is to identify the logical flow and strategies that will guide the next part of the proof.

After answering the above questions, please provide a **brief strategy** for the next step. Start this section with the phrase **'brief strategy'** and focus on the general approach or method that will guide you forward in the proof. Keep this section concise and high-level, without going into too much detail.

Please ensure the last section starts with 'brief strategy'.
"""

GENERATE_NEXT_ACTION_PROMPT_FT_DATA = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{proof_tracing}

{retrieve_info}

{public_notes}

=== Background Knowledge ===
Examples of concepts appearing in the current state,
Helpful for understanding the coq concepts:
{concepts}

{strategy}
==========================
Now please respond tactics:

[TACTIC]: """

GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND_FT_DATA = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{proof_tracing}

{retrieve_info}

{public_notes}

{strategy}
==========================
Now please respond tactics:

[TACTIC]: """

GENERATE_NEXT_ACTION_PROMPT = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{proof_tracing}

{retrieve_info}

{public_notes}

=== Background Knowledge ===
Examples of concepts appearing in the current state,
Helpful for understanding the coq concepts:
{concepts}

{strategy}

=== Available Actions ===

Please choose ONE of the following actions:

1. Request more information about specific concepts/tactics mentioned above
Your response must be in this format:
{{
  "info": ["concept_name1", "concept_name2", "tactic1", "tactic2", ...]
}}

2. Suggest a list of up to 10 tactics to try - prefer single atomic tactics over compound ones unless the combination is highly confident. I will provide the compiler's response for each
Your response must be in this format:
{{
  tactics: [
    {{"tactic": "tactic1", "reason": "explanation for why this specific tactic is recommended"}},
    {{"tactic": "tactic2", "reason": "explanation for why this specific tactic is recommended"}},
    ...
  ]
}}

Ensure your response is a valid JSON. For option 2, ensure each element in the "tactics" list contains both "tactic" and "reason" fields.

Please respond using ONLY one of the above formats.
"""

GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{proof_tracing}

{retrieve_info}

{public_notes}

{strategy}

=== Available Actions ===

Please choose ONE of the following actions:

1. Request more information about specific concepts/tactics mentioned above
Your response must be in this format:
{{
  "info": ["concept_name1", "concept_name2", "tactic1", "tactic2", ...]
}}

2. Suggest a list of up to 10 tactics to try - prefer single atomic tactics over compound ones unless the combination is highly confident. I will provide the compiler's response for each
Your response must be in this format:
{{
  tactics: [
    {{"tactic": "tactic1", "reason": "explanation for why this specific tactic is recommended"}},
    {{"tactic": "tactic2", "reason": "explanation for why this specific tactic is recommended"}},
    ...
  ]
}}

Ensure your response is a valid JSON. For option 2, ensure each element in the "tactics" list contains both "tactic" and "reason" fields.

Please respond using ONLY one of the above formats.
"""

GENERATE_NEXT_ACTION_PROMPT_NO_BACKGROUND_TEST = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{states}
{proof_tracing}

{retrieve_info}

=== Available Actions ===

Please follow the following format:

Suggest a list of up to 10 tactics to try - prefer single atomic tactics over compound ones unless the combination is highly confident. I will provide the compiler's response for each
Your response must be in this format:
{{
  tactics: [
    {{"tactic": "tactic1", "reason": "explanation for why this specific tactic is recommended"}},
    {{"tactic": "tactic2", "reason": "explanation for why this specific tactic is recommended"}},
    ...
  ]
}}

Ensure your response is a valid JSON. Ensure each element in the "tactics" list contains both "tactic" and "reason" fields.

Please respond using ONLY the above format.
"""

PLAIN_PROMPT_FT_DATA = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{state}

Now please respond tactics:

[TACTIC]: 
"""

PLAIN_PROMPT = """I am currently working on a formal proof in Coq. Here is my current state and context:

=== Current Proof States ===
{state}
=== Action ===
Suggest a list of up to 10 tactics to try
Your response must be in this format:
{{
  "tactics": ["tactic1", "tactic2", ...]
}}

Please respond using ONLY the above formats and ensure your response is a valid JSON.
"""

PS_SIMPLIFY_PROMPT="""Explain Coq concepts and tactics, simplify the examples to their essential meaning using natural language and mathematical notation. Focus on conveying understanding rather than strict Coq syntax.
Key points to include:
1. Core mathematical/logical meaning
2. Important assumptions and goals
3. How the tactic transforms the proof state
4. Key reasoning steps
5. Critical relationships between concepts
6. Brief comments explaining the meaning

Elements to simplify:
1. Replace Unnecessary Coq syntax with natural language when clearer
2. Use standard mathematical notation instead of function names
3. Remove implementation details
4. Focus on intuitive understanding
5. Use plain English to explain complex relationships

Always explain:
1. What we start with
2. What we're trying to prove
3. How the tactic helps
4. What changes after applying the tactic
5. Why this is useful

Rather than maintaining strict Coq syntax, prioritize clear communication of:
1. Mathematical concepts
2. Logical relationships
3. Proof strategies
4. Key transformations
5. Important patterns

=== OUTPUT FORMAT ===
INPUT:
## Concept Name: 
  Coq.Init.Logic.eq

### Tactic Name: induction p

### Example:
#### Before State

##### Hypotheses:
      p: Coq.Numbers.BinNums.positive.positive

##### Goal:
    forall ( q : Coq.Numbers.BinNums.positive.positive ) -> ( Coq.Init.Logic.eq.eq Coq.Numbers.BinNums.positive.positive ( Coq.PArith.BinPos.Pos.add_carry<ker>Coq.PArith.BinPosDef.Pos.add_carry p q ) ( Coq.PArith.BinPos.Pos.succ<ker>Coq.PArith.BinPosDef.Pos.succ ( Coq.PArith.BinPos.Pos.add<ker>Coq.PArith.BinPosDef.Pos.add p q ) ) )

#### After State

##### Hypotheses:
      IHp: forall ( q : Coq.Numbers.BinNums.positive.positive ) -> ( Coq.Init.Logic.eq.eq Coq.Numbers.BinNums.positive.positive ( Coq.PArith.BinPos.Pos.add_carry<ker>Coq.PArith.BinPosDef.Pos.add_carry p q ) ( Coq.PArith.BinPos.Pos.succ<ker>Coq.PArith.BinPosDef.Pos.succ ( Coq.PArith.BinPos.Pos.add<ker>Coq.PArith.BinPosDef.Pos.add p q ) ) )
      p: Coq.Numbers.BinNums.positive.positive

##### Goal:
    forall ( q : Coq.Numbers.BinNums.positive.positive ) -> ( Coq.Init.Logic.eq.eq Coq.Numbers.BinNums.positive.positive ( Coq.PArith.BinPos.Pos.add_carry<ker>Coq.PArith.BinPosDef.Pos.add_carry ( Coq.Numbers.BinNums.positive.xI p ) q ) ( Coq.PArith.BinPos.Pos.succ<ker>Coq.PArith.BinPosDef.Pos.succ ( Coq.PArith.BinPos.Pos.add<ker>Coq.PArith.BinPosDef.Pos.add ( Coq.Numbers.BinNums.positive.xI p ) q ) ) )

##### Hypotheses:
      IHp: forall ( q : Coq.Numbers.BinNums.positive.positive ) -> ( Coq.Init.Logic.eq.eq Coq.Numbers.BinNums.positive.positive ( Coq.PArith.BinPos.Pos.add_carry<ker>Coq.PArith.BinPosDef.Pos.add_carry p q ) ( Coq.PArith.BinPos.Pos.succ<ker>Coq.PArith.BinPosDef.Pos.succ ( Coq.PArith.BinPos.Pos.add<ker>Coq.PArith.BinPosDef.Pos.add p q ) ) )
      p: Coq.Numbers.BinNums.positive.positive

##### Goal:
    forall ( q : Coq.Numbers.BinNums.positive.positive ) -> ( Coq.Init.Logic.eq.eq Coq.Numbers.BinNums.positive.positive ( Coq.PArith.BinPos.Pos.add_carry<ker>Coq.PArith.BinPosDef.Pos.add_carry ( Coq.Numbers.BinNums.positive.xO p ) q ) ( Coq.PArith.BinPos.Pos.succ<ker>Coq.PArith.BinPosDef.Pos.succ ( Coq.PArith.BinPos.Pos.add<ker>Coq.PArith.BinPosDef.Pos.add ( Coq.Numbers.BinNums.positive.xO p ) q ) ) )

##### Hypotheses:


##### Goal:
    forall ( q : Coq.Numbers.BinNums.positive.positive ) -> ( Coq.Init.Logic.eq.eq Coq.Numbers.BinNums.positive.positive ( Coq.PArith.BinPos.Pos.add_carry<ker>Coq.PArith.BinPosDef.Pos.add_carry Coq.Numbers.BinNums.positive.xH q ) ( Coq.PArith.BinPos.Pos.succ<ker>Coq.PArith.BinPosDef.Pos.succ ( Coq.PArith.BinPos.Pos.add<ker>Coq.PArith.BinPosDef.Pos.add Coq.Numbers.BinNums.positive.xH q ) ) )

### Tactic Name: rewrite

### Example:
xxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxx

## Concept Name: 
  Coq.Classes.Morphisms.respectful

### Tactic Name: xxxx
xxxxxxxx
xxxxxxxx

OUTPUT:
{{
  "Coq.Init.Logic.eq": {{
    "core_meaning": "Defines when two terms are considered equal",
    "key_properties": [
      "Reflexive: x = x",
      "Symmetric: if x = y then y = x",
      "Transitive: if x = y and y = z then x = z"
    ],
    "tactics": [
      {{
        "tactic_name": "induction p",
        "purpose": "Proves properties by mathematical induction on binary numbers",
        "before_state": {{
          "hypotheses": [
            "p: positive  // a positive binary number"
          ],
          "goal": "For any q: add_with_carry(p,q) = add(p,q) + 2  // Property about binary addition with carry"
        }},
        "after_states": [
          {{
            "case": "Base case (p = 1)",
            "hypotheses": [],
            "goal": "add_with_carry(1,q) = add(1,q) + 2"
          }},
          {{
            "case": "Even case (p = 2k)",
            "hypotheses": [
              "IH: add_with_carry(k,q) = add(k,q) + 2  // induction hypothesis for k"
            ],
            "goal": "add_with_carry(2k,q) = add(2k,q) + 2  // need to prove for even numbers"
          }},
          {{
            "case": "Odd case (p = 2k+1)", 
            "hypotheses": [
              "IH: add_with_carry(k,q) = add(k,q) + 2  // induction hypothesis for k"
            ],
            "goal": "add_with_carry(2k+1,q) = add(2k+1,q) + 2  // need to prove for odd numbers"
          }}
        ]
      }},
      {{
        "tactic_name": "rewrite",
        "purpose": "Replaces equals with equals",
        "before_state": {{
          "hypotheses": [
            "H: x + 0 = x  // additive identity property"
          ],
          "goal": "2 + (x + 0) = 2 + x  // want to simplify expression"
        }},
        "after_states": [
          {{
            "case": "After rewrite H",
            "hypotheses": [
              "H: x + 0 = x"
            ],
            "goal": "2 + x = 2 + x  // simplified to obvious equality"
          }}
        ]
      }}
    ]
  }},
  "Coq.Classes.Morphisms.respectful": {{
    "core_meaning": "Defines when a function preserves relationships between elements",
    "key_properties": [
      "Preserves relations between inputs and outputs",
      "Used for proving function properties"
    ],
    "tactics": [
      // ... similar structure for each tactic
    ]
  }}
}}

=== IMPORTANT NOTES ABOUT OUTPUT FORMAT ===

The output MUST be a single JSON object where:
- Keys are concept names (exactly as given in input)
- Each concept contains core_meaning, key_properties, and tactics
- Each tactic contains tactic_name, purpose, before_state, and after_states
- before_state has hypotheses and goal
- after_states is an array of cases, each with case name, hypotheses, and goal

Follow the example output format exactly to ensure consistency and clarity.

ACTUAL INPUT:
{input}
"""

THEOREM_FORMAT = """
Name: {name}

Content: {content}

State: 
{state}
"""

SELECT_STATE_FORMAT = """
NUMBER STATE: {number}

CURRENT STATE: {state}

REMAINING GOALS: {remaining_goals}

TACTIC: {tactic}

PROOF TRACE: {proof_trace}
"""

PS_SELECTION_PROMPT = """You are now a Coq theorem proving expert. I will present you with:
1. A theorem T that needs to be proved, expressed in both Coq syntax and internal representation.
2. A list of current proof states (State 1, State 2, ..., State N) that I have reached through different sequences of tactics. Each state will include:
 - Current hypotheses and goals
 - The sequence of tactics used to reach this state
 - [Optional] Additional context or constraints
3. A proof trace that shows a concise summary highlighting the most crucial transitions and logical flow of the proof progress.

====== CURRENT THEOREM ======
{theorem}

Your task is to select up to 3 most promising states based on these criteria:
1. Proof Progress:
 - Whether each tactic makes meaningful progress toward proof completion
 - Absence of redundant or circular tactic sequences
 - No trivial transformations that don't advance the proof
2. Goal Reduction Quality:
 - How effectively the original goal has been broken down
 - Whether the subgoals are genuinely simpler
 - Maintenance of provability
3. State Evolution:
 - Clear logical progression from initial to current state
 - Each transformation contributes to proof development
 - Avoidance of state bloat or unnecessary complexity
4. Tactical Efficiency
 - No unnecessary back-and-forth transformations
 - Each tactic serves a clear purpose
 - Minimal redundancy in the tactic sequence

Output format:
{{
  "states": [list up to 3 state numbers, e.g., "4", "5", "3"]
}}

====== CURRENT PROOF STATES ======
{states}

Please ensure the response is a valid JSON object and contains 'states' as the key and a list of state numbers as the value.
"""

EXPLANATION_FORMAT = """You are an expert in formal methods and theorem proving, specializing in Coq.
Given a Coq proof state with:
{state}

Please analyze the states and provide your response as a JSON object in the following format:

{{
    "before": {{"zh": "用一句话概括这个证明状态在证明什么", "en": "give a short description of what this proof state is proving"}},
    "after": {{"zh": "用一句话概括转换后的证明状态在证明什么", "en": "give a short description of what this proof state is proving"}},
    "tactic": {{"zh": "简要说明这个tactic做了什么，它如何连接了前后两个状态", "en": "give a short description of what this tactic does, and how it connects the two states"}}
}}

Use clear and concise English (English)/Chinese (中文) descriptions that help readers quickly understand:
- What is being proved in the initial state
- What is being proved in the resulting state
- How the tactic transformed the proof state

Please ensure the response is a valid JSON object and contains 'before', 'after', and 'tactic' as the keys. Each of these keys must contain 'zh' and 'en' as the keys.
"""

STATE_INTUITION_FORMAT = """  ## tactic: {tactic}
  ## before: {before}
  ## after: {after}"""

SINGLE_STRATEGY_FORMAT = """# tactic: {tactic}
# brief_strategy: {brief_strategy}
# state_intuition:
{state_intuition}"""

STRATEGY_SUMMARY_FORMAT = """You are an expert in formal methods and theorem proving, specializing in Coq.

You are given the following information:
1. The list of tactics applied.
2. The key idea behind each tactic.
3. The explanation of the proof state before and after each tactic is applied.

====== PROOF TRACE ======
{proof_trace}

You need to condense this information into a summarized proof trace, highlighting the most crucial transitions between the states and focusing on the logical flow of the proof.

After summarizing the proof trace:
1. Estimate the number of remaining steps required to complete the proof of the goal.
2. Assign a score to evaluate the quality of this proof path (e.g., from 1 to 10, where 10 indicates a highly promising approach that's making good progress, and 1 indicates a less effective path that might need reconsideration).

Output format:
{{
    "proof_trace": "A concise summary of the proof trace, highlighting the most crucial transitions between the states and focusing on the logical flow of the proof."
    "steps": "The number of remaining steps required to complete the proof of the goal."
    "score": "The score evaluating the quality and effectiveness of this proof approach (1-10, where 10 means this is a very promising path and 1 suggests considering alternative approaches)."
}}

Please ensure the response is a valid JSON object and contains 'proof_trace', 'steps', and 'score' as the keys.
"""

TACTIC_STATUS_FORMAT = """TACTIC: {tactic}
STATUS: {status}
"""

RECONSIDER_TACTIC_PROMPT = """{before_prompt_tactic}
=== Current Step Attempts and Errors ===
The following tactics were tried and their results:
{previous_tactics_and_errors}

Please suggest new tactics for THIS STEP (prefer single atomic tactics over compound ones unless the combination is highly confident) that haven't been tried above. Consider both the successful and failed attempts as reference. If the successful tactics so far appear sufficient for current step, you can return an empty list.

Output format:
{{
  "tactics": [
    {{"tactic": "tactic1", "reason": "explanation for why this specific tactic is recommended"}},
    {{"tactic": "tactic2", "reason": "explanation for why this specific tactic is recommended"}},
    ...
  ]
}}

Ensure your response is a valid JSON object and each element in the "tactics" list contains both "tactic" and "reason" fields.
"""

TACTIC_ERROR_FORMAT = """Attempted tactic: {tactic}
Reason: {reason}
Error Message: {error}"""

TACTIC_ERROR_FORMAT_NOREASON = """
======== CURRENT TACTIC AND ERROR =======
Attempted tactic: {tactic}
Error Message: {error}

Please refine THIS SPECIFIC TACTIC to address the error. Do not suggest completely new approaches - instead, improve the existing tactic to make it successful. Focus on correcting mistakes, addressing edge cases, or making necessary adjustments to the current approach.
"""

RECONSIDER_HIERARCHICAL_FORMAT = """{before_prompt_tactic}
=== Current Tactic and Error ===
Attempted tactic: {current_tactic}
Reason: {reason}
Error Message: {error_message}

=== Previous Refinement Attempts ===
{previous_refinement_attempts}

Please refine THIS SPECIFIC TACTIC to address the error. Do not suggest completely new approaches - instead, improve the existing tactic to make it successful. Focus on correcting mistakes, addressing edge cases, or making necessary adjustments to the current approach.

IMPORTANT: 
1. Carefully review the previous refinement attempts to avoid suggesting modifications that have already been tried and failed. Your refinement should be distinctly different from previous attempts while still maintaining the core approach.

2. Prefer single atomic tactics over compound ones unless the combination is highly confident.

3. When possible, make minimal changes to fix the specific issue rather than creating complex compound tactics. If a simple fix can work, prefer that over a complex solution.

If you believe this tactic approach is fundamentally flawed and cannot be refined to succeed, you may return an empty refined_tactic with a reason explaining why.

Output format:
{{
  "refined_tactic": "improved_tactic",
  "reason": "Explanation of what was wrong with the original tactic and why this refinement addresses the issue"
}}

If you believe the tactic cannot be refined, respond with:
{{
  "refined_tactic": "",
  "reason": "Explanation of why this tactic approach is fundamentally flawed and cannot be fixed"
}}

Ensure your response is a valid JSON object and contains both refined_tactic and reason as keys.
"""

NOTE_ITEM_FORMAT = """ID: {id}
CONTENT: {content}"""

PUBLIC_NOTE_FORMAT = """You are an expert in formal methods and theorem proving, specializing in Coq. Your task is to maintain a concise, high-value collection of insights to help prove the current goal.

====== CURRENT GOAL ======
{current_state}
====== CURRENT NOTEBOOK ======
Items in the notebook, you can remove some items if they are not that helpful.

{current_notebook}

====== NEW CANDIDATES ======
New items that could potentially be added if they are high value.

{new_candidates}

Your task is to maintain a collection of the most valuable insights for proving the current goal. Evaluate entries based on:

1. Relevance to Current Goal:
   - Direct applicability to the current proof state
   - Potential usefulness in upcoming proof steps
   - Connection to the mathematical concepts involved

2. General Value:
   - Generalizability to similar proof patterns
   - Clarity and actionability of the insight
   - Uniqueness of the contribution

Notes:
- Total entries must not exceed 15
- Prioritize insights directly relevant to proving the current goal
- Maintain some general proof strategies that could help with subgoals
- Keep entries that might help with similar proof patterns

====== AVAILABLE ITEMS ======
Below is a summary of all available items above:
You can choose items to remove if they are not helpful for the current goal:

{available_items_note}

Select items to add if they provide valuable insights:

{available_items_candidate}

Please provide your decisions in the following format:
{{
    "remove": [1,2,7], # ID numbers of entries to remove from current notebook
    "add": [3,4,5], # ID numbers of new candidates to add
}}

Please ensure the response is a valid JSON object and contains 'add' and 'remove' as the keys.
"""

REORGANIZE_PROMPT_FORMAT = """================= Input prompt ================
{prompt_tactic}

=========== TASK ==============
You are an expert in formal methods and theorem proving, specializing in Coq. Your task is to reorganize the following prompt into a more concise format optimized for generating the next Coq tactic:

1. Preserve all information useful for selecting the next proof tactic

2. Structure content in this order:
   - Current proof state (goals and hypotheses)
   - Relevant definitions, theorems, lemmas and tactics
   - Anything that could suggest useful tactics

3. Remove clearly redundant or irrelevant information that cannot help advance the proof
   
4. Ensure the reorganized prompt remains comprehensive and self-contained, maintaining all potentially useful elements

5. Return only the reorganized prompt without additional commentary
"""


DEFINITION_RELATIONSHIP_PROMPT = """You are an expert in formal methods and theorem proving, specializing in Coq.
Given a Coq definition with:
{definition}

And the following concepts:
{concepts}

Please analyze and explain in one paragraph how these concepts relate to each other within the context of this definition, and their roles in the definition.

If you cannot determine the relationships or if the concepts are not sufficiently related in this context, only respond: "INSUFFICIENT_CONTEXT", do not give any other information.
"""

DEFINITION_UNDERSTANDING_PROMPT = """You are an expert in formal methods and theorem proving, specializing in Coq.
Given a Coq definition with:
{definition}

Please give an comprehensive explanation of the definition in one paragraph, including its purpose, usage, and key properties.

If you cannot understand the definition, only respond: "INSUFFICIENT_CONTEXT", do not give any other information.
"""

DEFINITION_WITHOUT_DEFS_FORMAT = """CURRENT DEFINITION:
Name: {name}
Content: {definition}
"""

DEFINITION_WITH_DEFS_FORMAT = """CURRENT DEFINITION:
Name: {name}
Content: 
{definition}

Related References:
{related_references}
"""

PREMISE_SELECTION_PROMPT = """You are an expert in formal methods and theorem proving, specializing in Coq.
Given a theorem with its current proof state:

{theorem}
CURRENT STATE:
{state}

AVAILABLE PREMISES:
{premises}

Please select the single most likely useful premise for the current proof state. Only respond with the premise's id (a number), nothing else.
Ensure only one number is returned.
"""