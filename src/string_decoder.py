from llm_sdk.llm_sdk import Small_LLM_Model
from src.function_selector import apply_mask


def get_valid_next_tokens_string(
    generated_so_far: str,
    vocab: dict[int, str]
) -> set[int]:

    valid = set()

    for token_id, token_str in vocab.items():
        if token_str == '"' and len(generated_so_far) > 0:
            valid.add(token_id)
            continue

        if '"' not in token_str:
            valid.add(token_id)

    return valid


def generate_string(
        input_ids: list[int], llm: Small_LLM_Model, vocab: dict[int, str]
) -> str:
    generated_ids: list[int] = []
    max_token = 50

    for _ in range(max_token):
        generated_str = llm.decode(generated_ids)
        valid_ids = get_valid_next_tokens_string(generated_str, vocab)
        logits = llm.get_logits_from_input_ids(input_ids)
        logits = apply_mask(logits, valid_ids)

        next_token_id = logits.index(max(logits))
        next_token_str = vocab[next_token_id]

        if next_token_str == '"':
            break

        generated_ids.append(next_token_id)
        input_ids = input_ids + [next_token_id]

    return llm.decode(generated_ids)
