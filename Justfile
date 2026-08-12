# Default deterministic repository validation.
default: check-all

# =============================================================================
# Stable repository validation API
# =============================================================================

# Lint the backend package and backend tests.
lint target="kagya tests":
    @echo "==> Running Ruff on {{target}}..."
    uv run ruff check {{target}}

# Format backend source/tests and apply safe Ruff fixes.
format target="kagya tests":
    @echo "==> Formatting {{target}}..."
    uv run ruff format {{target}}
    uv run ruff check --fix {{target}}

# Type-check the actual backend package tree.
typecheck target="kagya":
    @echo "==> Running Mypy on {{target}}..."
    uv run mypy {{target}}

# Run backend tests.
test target="tests/":
    @echo "==> Running Pytest on {{target}}..."
    uv run pytest {{target}} -v

# Run the deterministic frontend test suite.
frontend-test:
    @echo "==> Running frontend tests..."
    cd frontend && npm test

# Build the frontend in production mode, including its type checks.
frontend-build:
    @echo "==> Building frontend..."
    cd frontend && npm run build

# Start the FastAPI server.
api:
    @echo "==> Starting PROJECT-KAGYA FastAPI server..."
    uv run python -m kagya.api.server

# =============================================================================
# Composite validation
# =============================================================================

# Repository-wide deterministic gate for the implementation that exists today.
# Later rebuild PRs may extend this only after their implementation/tests exist.
check-all: lint typecheck test frontend-test frontend-build
    @echo "==> [OK] All deterministic repository checks passed."

# =============================================================================
# Dependency management
# =============================================================================

# Reproduce the committed backend dependency graph.
sync:
    @echo "==> Syncing locked backend dependencies..."
    uv sync --locked
