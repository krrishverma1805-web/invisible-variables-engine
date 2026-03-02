# =============================================================================
# Invisible Variables Engine - Makefile
# =============================================================================

.PHONY: help setup dev down test test-unit test-integration test-statistical \
        lint format typecheck migrate makemigrations seed clean logs shell \
        build worker flower docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE  := \033[0;34m
GREEN := \033[0;32m
RESET := \033[0m

# Docker compose command
DC := docker compose
SRC := src

help: ## Show this help message
	@echo "$(BLUE)Invisible Variables Engine - Available Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup: ## Install dependencies (Poetry) and set up pre-commit hooks
	@echo "$(BLUE)Installing dependencies...$(RESET)"
	pip install poetry
	poetry install --with dev
	poetry run pre-commit install
	cp -n .env.example .env || true
	@echo "$(GREEN)Setup complete! Edit .env before running services.$(RESET)"

setup-pip: ## Install via pip instead of Poetry
	pip install -e ".[dev]"
	cp -n .env.example .env || true

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------
dev: ## Start all services (API, worker, postgres, redis, streamlit)
	$(DC) up --build -d
	@echo "$(GREEN)Services started:$(RESET)"
	@echo "  API:       http://localhost:8000"
	@echo "  Docs:      http://localhost:8000/docs"
	@echo "  Streamlit: http://localhost:8501"
	@echo "  Flower:    http://localhost:5555"

dev-local: ## Run API locally (without Docker, requires local Postgres/Redis)
	poetry run uvicorn ive.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src

worker-local: ## Run Celery worker locally
	poetry run celery -A ive.workers.celery_app worker --loglevel=INFO --concurrency=2

streamlit-local: ## Run Streamlit app locally
	poetry run streamlit run streamlit_app/app.py --server.port 8501

down: ## Stop all Docker services
	$(DC) down

down-volumes: ## Stop all services AND remove volumes (destructive!)
	$(DC) down -v

restart: ## Restart all services
	$(DC) restart

build: ## Build Docker images
	$(DC) build

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test: ## Run ALL tests (unit + integration + statistical)
	poetry run pytest tests/ -v --tb=short --cov=src/ive \
		--cov-report=term-missing --cov-report=html:htmlcov -n auto

test-unit: ## Run unit tests only
	poetry run pytest tests/unit/ -v --tb=short -n auto

test-integration: ## Run integration tests (requires running services)
	poetry run pytest tests/integration/ -v --tb=short

test-statistical: ## Run statistical validation tests
	poetry run pytest tests/statistical/ -v --tb=short -s

test-fast: ## Run tests excluding slow statistical tests
	poetry run pytest tests/unit/ tests/integration/ -v --tb=short -n auto -m "not slow"

test-coverage: ## Generate HTML coverage report
	poetry run pytest tests/ --cov=src/ive --cov-report=html:htmlcov
	open htmlcov/index.html

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint: ## Run all linters (ruff + mypy)
	@echo "$(BLUE)Running ruff linter...$(RESET)"
	poetry run ruff check $(SRC) tests
	@echo "$(BLUE)Running mypy type checker...$(RESET)"
	poetry run mypy $(SRC)

format: ## Auto-format code with ruff
	poetry run ruff format $(SRC) tests
	poetry run ruff check --fix $(SRC) tests

typecheck: ## Run mypy type checking only
	poetry run mypy $(SRC)

check: lint typecheck ## Run all quality checks

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
migrate: ## Run pending Alembic migrations
	poetry run alembic upgrade head

makemigrations: ## Create a new Alembic migration (usage: make makemigrations MSG="add users table")
	poetry run alembic revision --autogenerate -m "$(MSG)"

migration-history: ## Show migration history
	poetry run alembic history --verbose

migration-current: ## Show current migration state
	poetry run alembic current

rollback: ## Rollback one migration
	poetry run alembic downgrade -1

seed: ## Seed the database with development data
	poetry run python scripts/seed_db.py

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
generate-data: ## Generate synthetic datasets for testing
	poetry run python scripts/generate_synthetic_data.py

logs: ## Tail logs from all services
	$(DC) logs -f

logs-api: ## Tail API logs
	$(DC) logs -f api

logs-worker: ## Tail worker logs
	$(DC) logs -f worker

shell: ## Open a Python shell in the API container
	$(DC) exec api python -c "import ive; print('IVE shell ready')"

flower: ## Open Celery Flower in the browser
	open http://localhost:5555

docs-serve: ## Serve docs locally (requires mkdocs)
	poetry run mkdocs serve

clean: ## Remove build artifacts, caches, and __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build *.egg-info
	@echo "$(GREEN)Cleaned!$(RESET)"

ps: ## Show running service status
	$(DC) ps
