import argparse
import sys

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

if __name__ == "__main__":
    main()