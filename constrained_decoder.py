from function_selector import FunctionDefinition, apply_mask
from llm_sdk.llm_sdk import Small_LLM_Model

def build_arguments_prompt(user_prompt: str, function: FunctionDefinition, current_params: dict, next_param: str) -> str:
    # On montre au modèle ce qu'on a déjà trouvé pour lui donner du contexte
    past_extractions = ", ".join([f"{k}: {v}" for k, v in current_params.items()])
    
    return (
        f"<|im_start|>system\n"
        f"You are a data extraction engine. Extract the exact numbers from the user prompt for the function '{function.name}'. "
        f"Do NOT calculate sums. Only extract raw values.\n"
        f"Available parameters: {list(function.parameters.keys())}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"{past_extractions}{', ' if past_extractions else ''}{next_param}: "
    )


def generate_arguments(prompt: str, function: FunctionDefinition, llm: Small_LLM_Model, vocab: dict[int, str]) -> dict:
    results = {}
    for param_name, param_info in function.parameters.items():
        # On passe 'results' pour que le prompt de 'b' sache ce qu'on a mis dans 'a'
        full_prompt = build_arguments_prompt(prompt, function, results, param_name)
        input_ids = llm.encode(full_prompt)[0].tolist()
        
        value = generate_value(param_info["type"], input_ids, llm, vocab)
        results[param_name] = value
    return results


def get_valid_next_tokens_number(generated_so_far: str, vocab: dict[int, str]) -> set[int]:
    valid = set()
    # On définit ce qui permet de sortir
    stops = {",", "}", " ", "\n", "<|endoftext|>", "<|im_end|>"}

    # Analyse du nombre en cours
    has_dot = "." in generated_so_far
    
    for token_id, token_str in vocab.items():
        # 1. Autoriser TOUJOURS les sorties
        if any(s in token_str for s in stops):
            valid.add(token_id)
            continue

        # 2. Gestion des chiffres
        # On n'autorise un nouveau token que s'il respecte la syntaxe float
        candidate = generated_so_far + token_str
        
        # Astuce : On refuse les tokens qui ne sont QUE des zéros 
        # si on a déjà pas mal de décimales (ex: 3). 
        # Ça force l'IA à utiliser un token de 'stops' au lieu de boucler.
        if has_dot and generated_so_far.split(".")[1].endswith("000"):
             continue

        try:
            float(candidate)
            valid.add(token_id)
        except ValueError:
            if (generated_so_far == "" and candidate == "-") or (candidate.endswith(".") and not has_dot):
                valid.add(token_id)
                
    return valid

def generate_number(input_ids: list[int], llm: Small_LLM_Model, vocab: dict[int, str]) -> float:
    generated = ""
    max_tokens = 10 # Un nombre dépasse rarement 10 tokens

    while True:
        valid_ids = get_valid_next_tokens_number(generated, vocab)
        logits = llm.get_logits_from_input_ids(input_ids)
        logits = apply_mask(logits, valid_ids)

        next_token_id = logits.index(max(logits))
        next_token_str = vocab[next_token_id]

        # Liste élargie des signaux d'arrêt
        if any(s in next_token_str for s in {",", "}", " ", "\n", "<|endoftext|>", "<|im_end|>"}):
            break
        print(next_token_str) 
        generated += next_token_str
        input_ids.append(next_token_id)

    # Nettoyage pour éviter l'erreur float() sur une chaîne vide ou juste "-"
    final_str = generated.strip()
    try:
        return float(final_str)
    except ValueError:
        return 0.0 # Valeur par défaut en cas d'échec d'extraction

def generate_value(
    param_type: str,
    input_ids: list[int],
    llm: Small_LLM_Model,
    vocab: dict[int, str]
) -> float | None:

    # Selon le type, appliquer un masque différent à chaque step
    # et accumuler les tokens jusqu'à un token de fin valide

    if param_type == "number":
        # tokens valides : chiffres, "-", ".", pas deux "." etc.
        return generate_number(input_ids, llm, vocab)
    # elif param_type == "string":
    #     # tokens valides : tout sauf guillemet non échappé
    #     # s'arrêter sur guillemet fermant
    #     ...
    # elif param_type == "boolean":
    #     ...
    return None