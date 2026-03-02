# Testing Architecture

The Invisible Variables Engine uses a multi-layered testing architecture to ensure high code quality.

## Layers of Defense

### 1. Local Pre-commit Hooks

We use `pre-commit` to catch simple errors before they are committed to history.

**Installation (Done once during setup):**

```bash
make setup
```

**What it runs:**

- Whitespace cleanup
- YAML/TOML validation
- Giant file check
- **Ruff** (Linter & Auto-formatter)
- **Mypy** (Static Type Checker)

To run hooks manually across the entire codebase:

```bash
poetry run pre-commit run --all-files
```

### 2. Make Commands

The `Makefile` simplifies running core quality and testing tools locally:

- `make check`: Runs Ruff linting and Mypy type-checking.
- `make format`: Auto-formats code with Ruff.
- `make test-unit`: Runs only fast unit tests.
- `make test-integration`: Runs integration tests (requires Docker services).
- `make test-statistical`: Runs slow statistical validity tests.
- `make test-fast`: Runs unit + integration tests (what the CI workflow runs).
- `make test`: Runs the absolute full suite.

### 3. CI/CD GitHub Actions

Every Pull Request and push to `main` triggers our GitHub Actions pipeline (`.github/workflows/ci.yml`).

**The Pipeline:**

1. Sets up Python 3.11 and Poetry.
2. Installs dependencies.
3. Runs Linters (`make check` -> Ruff & Mypy).
4. Spins up isolated Postgres & Redis services.
5. Runs Fast Tests (`make test-fast`).
6. Uploads the HTML coverage report.

**Required Checks:**
Branch protection rules should be enforced on the repository to require `test` (the CI job name) to pass successfully before merging any code into `main`.
