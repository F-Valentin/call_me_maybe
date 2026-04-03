from llm_sdk.llm_sdk import Small_LLM_Model
from src.function_selector import apply_mask


def get_valid_next_tokens_number(
        generated_so_far: str, vocab: dict[int, str]) -> set[int]:
    valid = set()
    stops = {",", "}", " ", "\n", "<|endoftext|>", "<|im_end|>"}

    has_dot = "." in generated_so_far

    for token_id, token_str in vocab.items():
        if token_str.strip() in stops or token_str in stops:
            valid.add(token_id)
            continue

        candidate = generated_so_far + token_str

        if has_dot and generated_so_far.split(".")[1].endswith("000"):
            continue

        try:
            float(candidate)
            valid.add(token_id)
        except ValueError:
            if (generated_so_far == "" and candidate ==
                    "-") or (candidate.endswith(".") and not has_dot):
                valid.add(token_id)

    return valid


def generate_number(
    input_ids: list[int], llm: Small_LLM_Model, vocab: dict[int, str]
) -> float:
    generated = ""

    while True:
        valid_ids = get_valid_next_tokens_number(generated, vocab)
        logits = llm.get_logits_from_input_ids(input_ids)
        logits = apply_mask(logits, valid_ids)

        next_token_id = logits.index(max(logits))
        next_token_str = vocab[next_token_id]

        if any(s in next_token_str for s in {
               ",", "}", " ", "\n", "<|endoftext|>", "<|im_end|>"}):
            break
        generated += next_token_str
        input_ids.append(next_token_id)

    final_str = generated.strip()
    try:
        return float(final_str)
    except ValueError:
        return 0.0
