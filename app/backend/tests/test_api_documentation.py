from fastapi.routing import APIRoute

from main import app

HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def test_every_endpoint_is_documented_for_swagger():
    undocumented = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        missing = []
        if not route.tags:
            missing.append("tags")
        if not route.summary:
            missing.append("summary")
        if not (route.description or "").strip():
            missing.append("description")
        if route.response_model is None:
            missing.append("response_model")
        if missing:
            undocumented.append(f"{sorted(route.methods)} {route.path}: {missing}")
    assert not undocumented, f"Endpoints missing Swagger docs: {undocumented}"


def test_openapi_schema_exposes_all_operations():
    schema = app.openapi()
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            if method not in HTTP_METHODS:
                continue
            assert operation.get("tags"), f"{method.upper()} {path} has no tags"
            assert operation.get("summary"), f"{method.upper()} {path} has no summary"
            assert operation.get(
                "description"
            ), f"{method.upper()} {path} has no description"
