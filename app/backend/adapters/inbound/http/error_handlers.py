from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from domain.errors import DomainError


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainError)
    async def handle_domain_error(request: Request, exc: DomainError) -> JSONResponse:
        content: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.details is not None:
            content["details"] = exc.details
        return JSONResponse(status_code=exc.status_code, content=content)
