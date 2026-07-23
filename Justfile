# デフォルトのアクション（引数なしで実行された場合）
# エージェントが迷ったときや、コミット前に全体を検証するための安全なフォールバックです。
default: check-all

# =============================================================================
# Agent API: 以下のコマンド群のみを CLAUDE.md でエージェントに露出させます
# =============================================================================

# コードの静的解析（Lint）を実行します
lint target="kagya tests":
    @echo "==> Running Linter (Ruff) on {{target}}..."
    uv run ruff check {{target}}

# コードの自動フォーマットを実行します
format target="kagya tests":
    @echo "==> Running Formatter (Ruff) on {{target}}..."
    uv run ruff format {{target}}
    uv run ruff check --fix {{target}}

# 静的型チェックを実行します
typecheck target="kagya":
    @echo "==> Running Type Checker (Mypy) on {{target}}..."
    uv run mypy {{target}}

# テストを実行します
test target="tests/":
    @echo "==> Running Tests (Pytest) on {{target}}..."
    uv run pytest {{target}} -v

# Backend OpenAPI schema and frontend API type drift checks
schema-check:
    @echo "==> Checking backend schema and frontend API type alignment..."
    uv run pytest tests/test_frontend_api_schema_sync.py -q

# Validate runtime configuration and compatibility notes
config-check config="config.yaml":
    @echo "==> Checking PROJECT-KAGYA config {{config}}..."
    uv run python -m kagya.config.check {{config}}

# Opt-in smoke test for real Hugging Face Transformers models.
transformers-smoke config="config.yaml":
    @echo "==> Running opt-in Transformers provider smoke check..."
    uv run python -m kagya.models.transformers_smoke --config {{config}}

# Opt-in smoke test for both primary and fallback Transformers models.
transformers-smoke-fallback config="config.yaml":
    @echo "==> Running opt-in Transformers provider fallback smoke check..."
    uv run python -m kagya.models.transformers_smoke --config {{config}} --check-fallback

# Opt-in actual subject runtime evaluation against a base model and candidate adapter.
behavioral-real-model config adapter_id evaluation_id:
    @echo "==> Running opt-in real-model subject runtime evaluation..."
    KAGYA_RUN_REAL_MODEL_BEHAVIORAL=1 uv run python -m kagya.learning.real_model_runtime_behavioral --config {{config}} --adapter-id {{adapter_id}} --evaluation-id {{evaluation_id}}

# Deterministic real-runtime behavioral harness and recovery checks.
behavioral-runtime:
    uv run pytest tests/test_runtime_behavioral_harness.py tests/test_runtime_behavioral_runner.py tests/test_behavioral_artifact_reconciliation.py -v

# Check non-dry-run QLoRA production prerequisites without starting training.
qlora-prod-check config="config.yaml":
    @echo "==> Checking production QLoRA prerequisites..."
    uv run python -m kagya.learning.qlora_requirements --config {{config}}

# FastAPI サーバーを起動します
api:
    @echo "==> Starting PROJECT-KAGYA FastAPI server..."
    uv run python -m kagya.api.server

# =============================================================================
# 複合タスク (Pipelines)
# =============================================================================

# プルリクエスト作成前や、大きな変更の後にエージェントに実行させる一括検証
check-all: lint typecheck test behavioral-runtime config-check schema-check
    @echo "==> [OK] All checks passed successfully."

# =============================================================================
# ユーティリティ (依存関係管理など)
# =============================================================================

# 依存関係の同期（.envrc からも呼ばれますが、明示的なAPIとしても提供）
sync:
    @echo "==> Syncing dependencies with uv..."
    uv sync
