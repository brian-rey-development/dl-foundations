.DEFAULT_GOAL := help

.PHONY: help install run serve figures test lint format typecheck check docker clean

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (including dev tools)
	uv sync

run: ## Train the network, save plots and the model artifact to outputs/
	uv run foundations

serve: ## Serve the trained model with FastAPI on http://127.0.0.1:8000 (docs at /docs)
	uv run foundations-api

figures: ## Regenerate every figure used in docs/
	uv run python scripts/make_figures.py

test: ## Run the test suite
	uv run pytest

lint: ## Lint with ruff
	uv run ruff check src tests

format: ## Format code with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck: ## Type-check with ty
	uv run ty check src tests

check: lint typecheck test ## Run every quality gate

docker: ## Build and run the API in a container
	docker build -t foundations-api .
	docker run --rm -p 8000:8000 foundations-api

clean: ## Remove caches and generated plots
	rm -rf .pytest_cache .ruff_cache outputs/*.png outputs/*.npz
	find . -type d -name __pycache__ -exec rm -rf {} +
