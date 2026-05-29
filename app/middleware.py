"""
app/middleware.py — Custom ASGI middleware for the RAG Evaluation System.

RequestTimingMiddleware
-----------------------
Measures wall-clock time for every HTTP request and:
  1. Adds an ``X-Response-Time`` header (value in milliseconds, e.g. ``42.3ms``).
    2. Logs a WARNING for any request that exceeds the configured threshold
         (``settings.SLOW_REQUEST_THRESHOLD_MS``).

         Register in main.py::

             from app.middleware import RequestTimingMiddleware
                 app.add_middleware(RequestTimingMiddleware)
                 """

                 from __future__ import annotations

                 import logging
                 import time

                 from starlette.middleware.base import BaseHTTPMiddleware
                 from starlette.requests import Request
                 from starlette.responses import Response

                 logger = logging.getLogger(__name__)


                 class RequestTimingMiddleware(BaseHTTPMiddleware):
                     """ASGI middleware that measures and reports HTTP request latency.

                         Attributes
                             ----------
                                 threshold_ms:
                                         Requests exceeding this duration (milliseconds) are logged as warnings.
                                                 ``None`` disables slow-request logging.
                                                     """

                                                         def __init__(self, app, threshold_ms: float | None = None) -> None:
                                                                 super().__init__(app)
                                                                         if threshold_ms is None:
                                                                                     from app.config import settings
                                                                                                 threshold_ms = settings.SLOW_REQUEST_THRESHOLD_MS or None
                                                                                                         self._threshold_ms = threshold_ms
                                                                                                         
                                                                                                             async def dispatch(self, request: Request, call_next) -> Response:
                                                                                                                     start = time.perf_counter()
                                                                                                                             response: Response = await call_next(request)
                                                                                                                                     elapsed_ms = (time.perf_counter() - start) * 1000
                                                                                                                                     
                                                                                                                                             response.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
                                                                                                                                             
                                                                                                                                                     if self._threshold_ms and elapsed_ms > self._threshold_ms:
                                                                                                                                                                 logger.warning(
                                                                                                                                                                                 "Slow request: %s %s took %.1fms (threshold: %.0fms)",
                                                                                                                                                                                                 request.method,
                                                                                                                                                                                                                 request.url.path,
                                                                                                                                                                                                                                 elapsed_ms,
                                                                                                                                                                                                                                                 self._threshold_ms,
                                                                                                                                                                                                                                                             )
                                                                                                                                                                                                                                                                     else:
                                                                                                                                                                                                                                                                                 logger.debug(
                                                                                                                                                                                                                                                                                                 "%s %s — %.1fms",
                                                                                                                                                                                                                                                                                                                 request.method,
                                                                                                                                                                                                                                                                                                                                 request.url.path,
                                                                                                                                                                                                                                                                                                                                                 elapsed_ms,
                                                                                                                                                                                                                                                                                                                                                             )
                                                                                                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                     return response
