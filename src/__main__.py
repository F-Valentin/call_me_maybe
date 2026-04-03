import argparse
import json
import sys

from src.function_selector import select_function, FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError
from src.constrained_decoder import generate_arguments
from pathlib import Path


def write_output(
    results: list[dict],
    output_path: str
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe - Function Calling Tool")

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Path to the output results JSON file"
    )

    args = parser.parse_args()

    print(f"Chargement des fonctions : {args.functions_definition}")
    print(f"Lecture des tests : {args.input}")
    print(f"Destination : {args.output}")

    prompt = "What is the sum of 2 and 3?"
    llm = Small_LLM_Model()

    def load_functions(path: str) -> list[FunctionDefinition]:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return [FunctionDefinition(**fn) for fn in data]
        except (FileNotFoundError, ValidationError, json.JSONDecodeError) as e:
            print(f"Erreur lors du chargement : {e}")
            sys.exit(1)
    with open(llm.get_path_to_vocab_file()) as f:
        raw_vocab = json.load(f)
        vocab_inv = {v: k for k, v in raw_vocab.items()}

    output = []
    functions = load_functions(args.functions_definition)
    with open(args.input) as f:
        data = json.load(f)
        prompts = [p["prompt"] for p in data]
    for prompt in prompts:
        function = select_function(prompt, functions, llm, vocab_inv)
        parameters = generate_arguments(prompt, function, llm, vocab_inv)
        output.append({
            "prompt": prompt,
            "name": function.name,
            "parameters": parameters
        })
        print(f"prompt: {prompt}, name: {function.name}, parameters: {parameters}")
    write_output(output, args.output)


if __name__ == "__main__":
    main()
