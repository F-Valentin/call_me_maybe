from llm_sdk.llm_sdk import Small_LLM_Model
import numpy as np
import json
import os
def softmax(x):
    """
    Compute softmax values for each set of scores in x.
    
    Args:
        x: Input array of shape (batch_size, num_classes) or (num_classes,)
        
    Returns:
        Softmax probabilities of same shape as input
    """
    # For numerical stability, subtract the maximum value from each input vector
    # This prevents overflow when calculating exp(x)
    shifted_x = x - np.max(x, axis=-1, keepdims=True)
    
    # Calculate exp(x) for each element
    exp_x = np.exp(shifted_x)
    
    # Calculate the sum of exp(x) for normalization
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    
    # Normalize to get probabilities
    probabilities = exp_x / sum_exp_x
    
    return probabilities

# function to find a token in the vocab.
# doit utiliser get_path_to_vocab_file ainsi que l'id du mot
# doit return un str (un token)


def add_next_word(prompt: str) ->str:
# pour encoder un text (prend un str || return un Torch.tensor)
    input_IDs = LLM.encode(prompt).tolist()[0]

# call le llm pour predire le prochain mot. (return un raw logits).
    logits = LLM.get_logits_from_input_ids(input_IDs)

#  convertir le raw logits en proba avec softmax.
    proba = softmax(logits)

#   le constraint decoding se fait ici ! avant le choix du token 
    next_token_id = int(np.argmax(proba))
    
    news_id = input_IDs + [next_token_id]
    return LLM.decode(news_id)


def build_prompt(user_message: str, functions: dict) -> str:
    tools_json = json.dumps(functions)
    
    return f"""<|im_start|>system
# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_json}
</tools>
For each function call, return a json object with the prompt, function name and parameters within <tool_call></tool_call> XML tags:
<tool_call>
{{"prompt":<original-prompt>, "name": <function-name>, "parameters": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{user_message}<|im_end|>
"""

def generate(prompt: str, max_tokens: int = 500) -> str:
    current = prompt
    print(current)
    for _ in range(max_tokens):
        current = add_next_word(current)
        # print(current)
        generated = current[len(prompt):]
        print(generated)
        if "</tool_call>" in generated:
            break
    return current

def parse_result(result: str) -> dict | None:
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    
    start = result.rfind(start_tag)
    end = result.rfind(end_tag)
    if start == -1 or end == -1:
        return None
    
    json_str = result[start + len(start_tag):end].strip()
    return json.loads(json_str)

LLM = Small_LLM_Model()

with open("data/input/functions_definition.json") as file:
    functions = json.load(file)


prompt =  "Add two numbers together and return their sum."


# 1) Construire le prompt
prompt = build_prompt(prompt, functions)

# 2) generer jusqu'a </tool_call>
result = generate(prompt)
# 3) extraire et afficher le json
tool_call = parse_result(result)
print(tool_call)


os.makedirs("data/output", exist_ok=True)
# 4) ecrire le resultat dans: data/output/function_calling_results.json
with open("data/output/result.json", "a") as f:
    json.dump(tool_call, f, indent=2)