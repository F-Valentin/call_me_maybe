import json
import os


# def build_prompt(user_message: str, functions: dict) -> str:
#     tools_json = json.dumps(functions)
    
#     return f"""<|im_start|>system
# # Tools
# You may call one or more functions to assist with the user query.
# You are provided with function signatures within <tools></tools> XML tags:
# <tools>
# {tools_json}
# </tools>
# <|im_start|>user
# {user_message}<|im_end|>
# """


# start = dict["prompt"] = "user message"
# Name = dict["name"] = "generer"
# paremetres = dict["param"] = dict qui peut avoir plusieurs elements
# elem : nom du param et valeur "key": "value"
# plus de param terminer

# LLM = Small_LLM_Model()


def load_functions(path: str) -> list[FunctionSchema]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            # Validation Pydantic pour chaque fonction
            return [FunctionSchema(**fn) for fn in data]
    except (FileNotFoundError, ValidationError, json.JSONDecodeError) as e:
        print(f"Erreur lors du chargement : {e}")
        sys.exit(1) # Quitter proprement comme demandé