from src.function_selector import FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model
from src.number import generate_number
from src.string_decoder import generate_string
from src.boolean_decoder import generate_boolean
from typing import Any


def build_arguments_prompt(user_prompt: str, function: FunctionDefinition,
                           current_params: dict, next_param: str) -> str:
    past_extractions = ", ".join(
        [f'{k}: "{v}"' for k, v in current_params.items()])
    param_type = function.parameters[next_param]["type"]

    return (
        f"<|im_start|>system\n"
        "You are a data extraction engine. "
        "Extract the exact argument values from the user prompt "
        f"for the function '{function.name}': {function.description}\n"
        f"Current parameter to extract: '{next_param}' (type: {param_type})\n"
        "Extract only the raw input value as it appears in the prompt, "
        "not the result of the function. Do not execute the function. "
        "Do not calculate, do not explain,"
        "do not add extra content.<|im_end|>\n"

        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{past_extractions}{', ' if past_extractions else ''}{next_param}: \""
    )


def generate_arguments(prompt: str, function: FunctionDefinition,
                       llm: Small_LLM_Model, vocab: dict[int, str]) -> dict:
    results: dict[str, Any] = {}
    for param_name, param_info in function.parameters.items():
        full_prompt = build_arguments_prompt(
            prompt, function, results, param_name)
        input_ids = llm.encode(full_prompt)[0].tolist()
        value = generate_value(param_info["type"], input_ids, llm, vocab)
        results[param_name] = value
    return results


def generate_value(
    param_type: str,
    input_ids: list[int],
    llm: Small_LLM_Model,
    vocab: dict[int, str]
) -> float | str | bool | None:
    if param_type == "number":
        return generate_number(input_ids, llm, vocab)
    elif param_type == "string":
        return generate_string(input_ids, llm, vocab)
    elif param_type == "boolean":
        return generate_boolean(input_ids, llm, vocab)
    return None
