```markdown
*This project has been created as part of the 42 curriculum by vafechte.*

# Call Me Maybe - Function Calling Tool

---

## 📝 Description
**Call Me Maybe** is a Function Calling tool designed to translate natural language queries (e.g., "Add 2 and 3") into structured JSON function calls. 

The core challenge is ensuring 100% reliability in the output structure despite using a Small Language Model (**Qwen3-0.6B**), which is naturally prone to syntax errors. To solve this, the project implements **Constrained Decoding**: instead of letting the model generate text freely, we intercept the token probability scores (logits) to allow only those that satisfy JSON syntax and the specific function schema.

## 🚀 Instructions

### Prerequisites
The project uses `uv` for modern and fast dependency management.

```bash
# Install dependencies (pydantic, numpy, etc.)
uv sync
```

### Execution
To run the program using default files:
```bash
uv run python -m src
```

To specify custom file paths:
```bash
uv run python -m src --functions_definition <path_to_json> --input <path_to_tests> --output <path_to_results>
```

## 🛠️ Project Architecture

The codebase is modularized to handle each data type in isolation:

* **`function_selector.py`**: Uses a logit mask to force the model to pick a valid function name from the provided list.
* **`constrained_decoder.py`**: Orchestrates the extraction logic. It builds specific prompts for each parameter to guide the model.
* **Specialized Decoders**:
    * `number.py`: Filters tokens to guarantee a valid numerical format (float/int).
    * `string_decoder.py`: Captures text until the next closing quotation mark.
    * `boolean_decoder.py`: Restricts choices strictly to tokens forming `true` or `false`.

## 📚 Resources
* **Pydantic**: Used for strict validation of function definitions (`FunctionDefinition` model) and internal objects.
* **Logit Masking**: The `apply_mask` function replaces forbidden token scores with `-inf`, making their selection impossible during generation.
* **ChatML Formatting**: Implements the `<|im_start|>` and `<|im_end|>` format to clearly separate system instructions and user queries.
* **Documentation**: Refer to `call_me_maybe.pdf` for the full technical subject and constraints.

## ✅ Code Quality
The project follows 42's rigorous standards:
* **Flake8** compliant for coding style.
* Static typing with **Mypy**.
* Clean resource management (Context Managers) and robust JSON exception handling.


*Note on AI usage: AI was used during this project as a pedagogical tool to better understand the underlying concepts of constrained decoding and logit manipulation.*
