import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}
UPLOAD_PATH = "/api/studies"
WINDOW_SECONDS = 60.0


def client_ip(request: Request) -> str:
    """Return the caller address; Cloud Run appends the real one last."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add the browser protection headers to every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


class UploadRateLimitMiddleware(BaseHTTPMiddleware):
    """Answer 429 when one client uploads more than ``limit`` studies a minute."""

    def __init__(self, app: FastAPI, limit: int) -> None:
        super().__init__(app)
        self._limit = limit
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "POST" and request.url.path == UPLOAD_PATH:
            now = time.monotonic()
            hits = self._hits[client_ip(request)]
            while hits and now - hits[0] > WINDOW_SECONDS:
                hits.popleft()
            if len(hits) >= self._limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "code": "TOO_MANY_UPLOADS",
                        "message": "Too many uploads. Try again in a minute.",
                    },
                )
            hits.append(now)
        return await call_next(request)
