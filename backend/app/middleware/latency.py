"""HTTP request latency logging and Prometheus metrics."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

import structlog
from app.config import settings
from prometheus_client import Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

_latency_lock = Lock()
_latency_samples: deque[float] = deque(maxlen=1000)


def record_latency_sample(duration_ms: float) -> None:
    with _latency_lock:
        _latency_samples.append(duration_ms)


def latency_percentiles() -> dict[str, float | int]:
    with _latency_lock:
        samples = sorted(_latency_samples)
    if not samples:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}

    def percentile(p: float) -> float:
        index = min(int(len(samples) * p), len(samples) - 1)
        return round(samples[index], 2)

    return {
        "count": len(samples),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return route.path
    return request.url.path


class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        if not settings.enable_latency_logging:
            return await call_next(request)

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        endpoint = _endpoint_label(request)

        REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration_ms / 1000)
        record_latency_sample(duration_ms)

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            endpoint=endpoint,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        return response
