"""Operator-safe system metadata routes."""

from importlib.metadata import PackageNotFoundError, version
import os
import subprocess

from fastapi import APIRouter, Depends

from kagya.api.dependencies import get_api_settings
from kagya.api.schemas.system import (
    BuildInfoSchema,
    RuntimeInfoSchema,
    SystemInfoResponse,
)
from kagya.config import Settings


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/info", response_model=SystemInfoResponse)
def system_info(settings: Settings = Depends(get_api_settings)) -> SystemInfoResponse:
    """Return public-safe deployment metadata for operators."""

    return SystemInfoResponse(
        project=settings.project.name,
        status="ok",
        build=BuildInfoSchema(
            version=_package_version(),
            commit=_build_commit(),
        ),
        runtime=RuntimeInfoSchema(
            environment=settings.project.environment,
            provider=settings.model.provider,
            primary_model_id=settings.model.primary_id,
            fallback_configured=bool(settings.model.fallback_id),
            transformers_4bit=settings.model.load_in_4bit,
            qlora_dry_run=settings.qlora.dry_run,
            admin_token_configured=bool(os.getenv(settings.api.admin_token_env)),
        ),
    )


def _package_version() -> str:
    try:
        return version("project-kagya")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _build_commit() -> str | None:
    for env_name in ("KAGYA_BUILD_COMMIT", "GIT_COMMIT"):
        value = os.getenv(env_name)
        if value:
            return value[:12]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None
