import argparse
import json
import sys

from function_selector import select_function, FunctionDefinition
from llm_sdk.llm_sdk import Small_LLM_Model
from pydantic import ValidationError

def main() -> None:
    # 1. Créer le parser
    parser = argparse.ArgumentParser(description="Call Me Maybe - Function Calling Tool")

    # 2. Ajouter les arguments (en respectant les noms imposés par le sujet)
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

    # 3. Récupérer les arguments
    args = parser.parse_args()

    # 4. Utiliser les chemins pour charger tes données
    print(f"Chargement des fonctions : {args.functions_definition}")
    print(f"Lecture des tests : {args.input}")
    print(f"Destination : {args.output}")

    # Ici, tu appelles ta logique de traitement
    # exemple: run_pipeline(args.functions_definition, args.input, args.output)
    prompt = "What is the sum of 2 and 3?"
    llm = Small_LLM_Model()
    def load_functions(path: str) -> list[FunctionDefinition]:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Validation Pydantic pour chaque fonction
                return [FunctionDefinition(**fn) for fn in data]
        except (FileNotFoundError, ValidationError, json.JSONDecodeError) as e:
            print(f"Erreur lors du chargement : {e}")
            sys.exit(1) # Quitter proprement comme demandé
    with open(llm.get_path_to_vocab_file()) as f:
        raw_vocab = json.load(f)
        vocab = {v: k for k, v in raw_vocab.items()}

    functions = load_functions(args.functions_definition)
    with open(args.input) as f:
        data = json.load(f)
        prompts = [p["prompt"] for p in data]
    for prompt in prompts:
        t = select_function(prompt, functions, llm, vocab)

if __name__ == "__main__":
    main()