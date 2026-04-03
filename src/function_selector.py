from pydantic import BaseModel, ConfigDict
from llm_sdk.llm_sdk import Small_LLM_Model


class FunctionDefinition(BaseModel):
    model_config = ConfigDict(strict=True)

    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


def build_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    functions_desc = "\n".join(
        f"- {f.name}: {f.description}" for f in functions
    )
    return (
        "<|im_start|>system\n"
        "You are a function calling assistant. "
        "Given a user request, you must:"
        "respond with exactly one function name.\n"
        f"Available functions:\n{functions_desc}\n"
        "Respond with only the function name, nothing else.<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_valid_next_tokens(
    generated_so_far: str,
    functions: list[FunctionDefinition],
    vocab: dict[int, str]
) -> set[int]:

    valid = set()
    function_names = [f.name for f in functions]

    for token_id, token_str in vocab.items():
        candidate = generated_so_far + str(token_str)
        # Valide si candidate est un préfixe d'au moins un nom
        if any(name.startswith(candidate) for name in function_names):
            valid.add(token_id)

    return valid


def apply_mask(logits: list[float], valid_token_ids: set[int]) -> list[float]:
    mask = [float("-inf")] * len(logits)

    for idx in valid_token_ids:
        mask[idx] = logits[idx]

    return mask


def select_function(
    prompt: str,
    functions: list[FunctionDefinition],
    llm: Small_LLM_Model,
    vocab: dict[int, str]  # token_id -> string
) -> FunctionDefinition:
    full_prompt = build_selection_prompt(prompt, functions)

    input_ids = llm.encode(full_prompt)[0].tolist()

    generated = ""
    while True:
        logits = llm.get_logits_from_input_ids(input_ids)
        valid_token_ids = get_valid_next_tokens(generated, functions, vocab)
        logits = apply_mask(logits, valid_token_ids)

        next_token_id = logits.index(max(logits))
        next_token_str = vocab[next_token_id]
        generated += next_token_str

        if generated in [f.name for f in functions]:
            break

        input_ids = input_ids + [next_token_id]
    return next(f for f in functions if f.name == generated)
