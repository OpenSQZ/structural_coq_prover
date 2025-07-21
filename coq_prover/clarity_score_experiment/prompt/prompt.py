ZH_DEF_REQUEST_TEMPLATE = """
Please provide a Chinese explanation for the term {concept_content}.

Please wrap your answer with ```explanation```.
This is because I will automatically extract the content wrapped in ```explanation```, and other content will be ignored.
"""


COMPLETE_TEMPLATE = """
{item_info}

Please extract and provide the precise Coq definition of the concept named {concept_name} that appears above.
Please wrap your answer with ```coq```.
This is because I will automatically extract the content wrapped in ```coq```, and other content will be ignored.
For example:
If I want to output the definition of concept 'nat', I would output:

```coq
nat : Set := |  O : nat  |  S : nat -> nat
```
"""


FINAL_REQUEST = """
Now please output the strict definition of {concept_name}:
"""


NAME_FORMAT = """
## Name: {name}
## Content: 
"""

TACTIC_NAME_FORMAT = """
## Tactic Name:
{name}
"""

ORIGIN_FORMAT = """
## Origin:
{origin}
"""

TACTIC_ORIGIN_FORMAT = """
## Context:
{origin}
"""

INTERNAL_FORMAT = """
## Internal:
{internal}
"""

INTUITION_FORMAT = """
## Intuition:
{intuition}
"""


EQUIVALENCE_CHECK_TEMPLATE = """
I have a concept in Coq with this definition:

```coq
{definition1}
```

But there is another definition for it:

```coq
{definition2}
```

Please tell me if these two definitions are equivalent.
Output ONLY 'YES' if they are equivalent, otherwise output ONLY 'NO'.
"""
