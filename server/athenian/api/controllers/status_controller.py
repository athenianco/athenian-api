import time

import aiohttp.web
import prometheus_client

from athenian.api import metadata
from athenian.api.metadata import __package__


@aiohttp.web.middleware
async def instrument(request, handler):
    """Middleware to count requests and record the elapsed time."""
    start_time = time.time()
    request.app["request_in_progress"].labels(
        __package__, request.path, request.method).inc()
    try:
        response = await handler(request)
        return response
    finally:
        request.app["request_latency"].labels(
            __package__, request.path).observe(time.time() - start_time)
        request.app["request_in_progress"].labels(
            __package__, request.path, request.method).dec()
        try:
            code = response.status
        except NameError:
            code = 500
        request.app["request_count"].labels(
            __package__, request.method, request.path, code).inc()


class StatusRenderer:
    """Render the status page with Prometheus."""

    def __init__(self, registry: prometheus_client.CollectorRegistry):
        """Record the registry where the metrics are maintained."""
        self._registry = registry

    async def __call__(self, request: aiohttp.web.Request) -> aiohttp.web.Response:
        """Endpoint handler to output the current Prometheus state."""
        resp = aiohttp.web.Response(body=prometheus_client.generate_latest(self._registry))
        resp.content_type = prometheus_client.CONTENT_TYPE_LATEST
        return resp


def setup_status(app) -> prometheus_client.CollectorRegistry:
    """Add /status to serve Prometheus-driven runtime metrics."""
    registry = prometheus_client.CollectorRegistry(auto_describe=True)
    app["request_count"] = prometheus_client.Counter(
        "requests_total", "Total Request Count",
        ["app_name", "method", "endpoint", "http_status"],
        registry=registry,
    )
    app["request_latency"] = prometheus_client.Histogram(
        "request_latency_seconds", "Request latency",
        ["app_name", "endpoint"],
        registry=registry,
    )
    app["request_in_progress"] = prometheus_client.Gauge(
        "requests_in_progress_total", "Requests in progress",
        ["app_name", "endpoint", "method"],
        registry=registry,
    )
    prometheus_client.Info("server", "API server version", registry=registry).info({
        "version": metadata.__version__,
        "commit": getattr(metadata, "__commit__", "null"),
        "build_date": getattr(metadata, "__date__", "null"),
    })
    app.middlewares.insert(0, instrument)
    # passing StatusRenderer(registry) without __call__ triggers a spurious DeprecationWarning
    # FIXME(vmarkovtsev): https://github.com/aio-libs/aiohttp/issues/4519
    app.router.add_get("/status", StatusRenderer(registry).__call__)
    return registry
