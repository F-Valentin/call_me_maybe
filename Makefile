# Variables
PYTHON = uv run python
MODULE = src
# Valeurs par défaut selon l'énoncé
FUNC_DEF = data/input/functions_definition.json
INPUT_FILE = data/input/function_calling_tests.json
OUTPUT_FILE = data/output/function_calls.json

.PHONY: install run debug clean lint

# Installation des dépendances avec uv
install:
	uv sync

# Règle run respectant strictement la syntaxe : 
# uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
run:
	$(PYTHON) -m $(MODULE) \
		--functions_definition $(FUNC_DEF) \
		--input $(INPUT_FILE) \
		--output $(OUTPUT_FILE)

# Mode débug avec l'outil pdb de python
debug:
	$(PYTHON) -m pdb -m $(MODULE) \
		--functions_definition $(FUNC_DEF) \
		--input $(INPUT_FILE) \
		--output $(OUTPUT_FILE)

# Nettoyage des fichiers temporaires et caches
clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Linting avec les flags obligatoires du sujet (IV.2)
lint:
	uv run flake8 $(MODULE)
	uv run mypy $(MODULE) \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs