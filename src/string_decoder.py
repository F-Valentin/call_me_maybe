from llm_sdk.llm_sdk import Small_LLM_Model
from src.function_selector import apply_mask
from typing import cast


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
    max_tokens = 50

    valid_ids_empty = get_valid_next_tokens_string("", vocab)
    valid_ids_started = get_valid_next_tokens_string("x", vocab)

    for _ in range(max_tokens):
        valid_ids = valid_ids_empty if not generated_ids else valid_ids_started
        logits = llm.get_logits_from_input_ids(input_ids)
        logits = apply_mask(logits, valid_ids)

        next_token_id = logits.index(max(logits))
        next_token_str = vocab[next_token_id]

        if next_token_str == '"':
            break
        generated_ids.append(next_token_id)
        input_ids = input_ids + [next_token_id]
    else:
        raise RuntimeError(
            f"generate_string: max_tokens ({max_tokens})"
            "reached without closing quote"
        )

    return cast(str, llm.decode(generated_ids))
