import time

import aiohttp.web
import prometheus_client

from athenian.api.metadata import __package__


@aiohttp.web.middleware
async def instrument(request, handler):
    """Middleware to count requests and record the elapsed time."""
    start_time = time.time()
    request.app["REQUEST_IN_PROGRESS"].labels(
        __package__, request.path, request.method).inc()
    try:
        response = await handler(request)
        return response
    finally:
        request.app["REQUEST_LATENCY"].labels(
            __package__, request.path).observe(time.time() - start_time)
        request.app["REQUEST_IN_PROGRESS"].labels(
            __package__, request.path, request.method).dec()
        try:
            code = response.status
        except NameError:
            code = 500
        request.app["REQUEST_COUNT"].labels(
            __package__, request.method, request.path, code).inc()


class StatusRenderer:
    """Render the status page with Prometheus."""

    def __init__(self, registry: prometheus_client.CollectorRegistry):
        """Record the registry where the metrics are maintained."""
        self._registry = registry

    async def __call__(self, request):
        """Endpoint handler to output the current Prometheus state."""
        resp = aiohttp.web.Response(body=prometheus_client.generate_latest(self._registry))
        resp.content_type = prometheus_client.CONTENT_TYPE_LATEST
        return resp


def setup_status(app):
    """Add /status to serve Prometheus-driven runtime metrics."""
    registry = prometheus_client.CollectorRegistry(auto_describe=True)
    app["REQUEST_COUNT"] = prometheus_client.Counter(
        "requests_total", "Total Request Count",
        ["app_name", "method", "endpoint", "http_status"],
        registry=registry,
    )
    app["REQUEST_LATENCY"] = prometheus_client.Histogram(
        "request_latency_seconds", "Request latency",
        ["app_name", "endpoint"],
        registry=registry,
    )
    app["REQUEST_IN_PROGRESS"] = prometheus_client.Gauge(
        "requests_in_progress_total", "Requests in progress",
        ["app_name", "endpoint", "method"],
        registry=registry,
    )
    app.middlewares.insert(0, instrument)
    app.router.add_get("/status", StatusRenderer(registry))
