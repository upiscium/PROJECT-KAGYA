import re
from pathlib import Path

from kagya.api.server import create_app


FRONTEND_API_PATH = Path(__file__).resolve().parents[1] / "frontend" / "src" / "lib" / "api.ts"


SCHEMA_TO_FRONTEND_TYPE = {
    "EmotionSchema": "Emotion",
    "ModelSchema": "ModelInfo",
    "ChatResponse": "ChatResponse",
    "DebugChatResponse": "DebugChatResponse",
    "MemorySearchResponse": "MemorySearchResponse",
    "AdapterResponse": "Adapter",
    "AdapterListResponse": "AdapterListResponse",
    "AdapterEvaluateResponse": "AdapterEvaluateResponse",
    "EvaluationResultSummary": "EvaluationResultSummary",
    "EvaluationResultListResponse": "EvaluationResultListResponse",
    "AdapterEvaluationHistoryResponse": "AdapterEvaluationHistoryResponse",
    "EvaluationResultDetail": "EvaluationResultDetail",
    "TrainingJobResponse": "TrainingJob",
    "TrainingJobListResponse": "TrainingJobListResponse",
    "BuildInfoSchema": "BuildInfo",
    "RuntimeInfoSchema": "RuntimeInfo",
    "SystemInfoResponse": "SystemInfoResponse",
    "RuntimeEventSchema": "RuntimeEvent",
    "RuntimeEventListResponse": "RuntimeEventListResponse",
}


def test_frontend_api_types_include_backend_schema_fields() -> None:
    openapi_schemas = create_app().openapi()["components"]["schemas"]
    frontend_api = FRONTEND_API_PATH.read_text(encoding="utf-8")

    missing: list[str] = []
    for schema_name, frontend_type in SCHEMA_TO_FRONTEND_TYPE.items():
        frontend_body = _frontend_type_body(frontend_api, frontend_type)
        if "ChatResponse" in frontend_body:
            frontend_body += _frontend_type_body(frontend_api, "ChatResponse")
        backend_fields = set(openapi_schemas[schema_name].get("properties", {}))
        for field in sorted(backend_fields):
            if not re.search(rf"\b{re.escape(field)}\b", frontend_body):
                missing.append(f"{frontend_type}.{field} from {schema_name}")

    assert missing == []


def test_frontend_api_client_exposes_backend_routes() -> None:
    frontend_api = FRONTEND_API_PATH.read_text(encoding="utf-8")

    expected_route_snippets = [
        "/api/chat",
        "/chat/debug",
        "/memory/search",
        "/sleep/jobs",
        "/training/datasets",
        "/adapters",
        "/evaluations",
        "/evaluations/adapters/",
        "/api-proxy/system/info",
        "/system/events",
        "/experiences",
        "/beliefs",
        "/motivation",
    ]
    missing = [snippet for snippet in expected_route_snippets if snippet not in frontend_api]

    assert missing == []


def _frontend_type_body(source: str, type_name: str) -> str:
    match = re.search(rf"export type {re.escape(type_name)}\s*=\s*", source)
    assert match is not None, f"frontend type {type_name} is missing"
    start = match.end()
    depth = 0
    for index in range(start, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
        elif source[index] == ";" and depth == 0:
            return source[start:index]
    raise AssertionError(f"frontend type {type_name} is not closed")
