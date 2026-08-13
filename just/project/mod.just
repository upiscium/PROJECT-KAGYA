root := justfile_directory()
bootstrap_script := root / ".automation/bin/project_bootstrap.py"
environment_script := root / "just/project/environment.py"

mod python 'python.just'
mod? repository 'repository.just'

[working-directory: root]
bootstrap name='':
    python3 {{quote(bootstrap_script)}} {{quote(name)}}

[working-directory: root]
environment:
    python3 {{quote(environment_script)}} report

[working-directory: root]
doctor:
    python3 {{quote(environment_script)}} exec uv --version >/dev/null
    python3 {{quote(environment_script)}} exec ruff --version >/dev/null
    python3 {{quote(environment_script)}} exec uv run --no-sync mypy --version >/dev/null
    python3 {{quote(environment_script)}} exec uv run --no-sync pytest --version >/dev/null
    python3 {{quote(environment_script)}} exec node --version >/dev/null
    python3 {{quote(environment_script)}} exec npm --version >/dev/null
    @echo "Python Project Adapter doctor: PASS"

[working-directory: root]
format-check:
    @echo "Python Project Adapter format-check: SKIPPED (repository has no clean Ruff format baseline)"

[working-directory: root]
lint:
    python3 {{quote(environment_script)}} exec uv run --no-sync ruff check kagya tests
    python3 {{quote(environment_script)}} exec uv run --no-sync mypy kagya

[working-directory: root]
test:
    python3 {{quote(environment_script)}} exec uv run --no-sync pytest
    python3 {{quote(environment_script)}} exec npm --prefix frontend test

[working-directory: root]
eval:
    python3 {{quote(environment_script)}} exec uv run --no-sync python -m kagya.config.check config.yaml
    python3 {{quote(environment_script)}} exec uv run --no-sync pytest tests/test_frontend_api_schema_sync.py -q

[working-directory: root]
build:
    python3 {{quote(environment_script)}} exec npm --prefix frontend run build

[working-directory: root]
check: doctor format-check lint test eval build
    @echo "Python Project Adapter check: PASS"
