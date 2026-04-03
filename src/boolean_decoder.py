from llm_sdk.llm_sdk import Small_LLM_Model
from src.function_selector import apply_mask


def generate_boolean(
        input_ids: list[int], llm: Small_LLM_Model, vocab: dict[int, str]
) -> bool:
    generated = ""
    valid_values = ["true", "false"]

    while True:
        valid_ids = get_valid_next_tokens_boolean(
            generated, valid_values, vocab)
        logits = llm.get_logits_from_input_ids(input_ids)
        logits = apply_mask(logits, valid_ids)

        next_token_id = logits.index(max(logits))
        next_token_str = vocab[next_token_id]
        generated += next_token_str

        if generated in valid_values:
            break

        input_ids = input_ids + [next_token_id]

    return generated == "true"


def get_valid_next_tokens_boolean(
    generated_so_far: str,
    valid_values: list[str],
    vocab: dict[int, str]
) -> set[int]:
    valid = set()

    for token_id, token_str in vocab.items():
        candidate = generated_so_far + token_str
        if any(v.startswith(candidate) for v in valid_values):
            valid.add(token_id)

    return valid
