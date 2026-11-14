"""Request logging middleware."""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from traffic_predictor.utils.logger import get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed_ms:.1f}ms)")
        return response
