from collections import defaultdict
from contextvars import ContextVar
from itertools import chain
import logging
import re
from time import time
from typing import Optional

from aiohttp import web
import prometheus_client
import sentry_sdk

from athenian.api import metadata
from athenian.api.metadata import __package__, __version__


PROMETHEUS_REGISTRY_VAR_NAME = "prometheus_registry"
METRICS_CALCULATOR_VAR_NAME = "metrics_calculator"
elapsed_error_threshold = 60
measure_latency_re = re.compile("/v1/(metrics|histograms|get|diff|filter|paginate)")


def _after_response(request: web.Request,
                    response: Optional[web.Response],
                    start_time: float,
                    ) -> None:
    account = getattr(request, "account", "N/A")
    if account is None:
        account = "N/A"
    db_elapsed = request.app["db_elapsed"].get()
    cache_context = request.app["cache_context"]
    pdb_context = request.app["pdb_context"]
    sdb_elapsed, mdb_elapsed, pdb_elapsed, rdb_elapsed = (
        db_elapsed[x + "db"] for x in ("s", "m", "p", "r"))
    if response is not None:
        response.headers.add(
            "X-Performance-DB",
            "s %.3f, m %.3f, p %.3f, r %.3f" % (
                sdb_elapsed, mdb_elapsed, pdb_elapsed, rdb_elapsed))
        for k, v in cache_context.items():
            s = sorted("%s %d" % (f.replace("athenian.api.", ""), n)
                       for f, n in v.get().items())
            response.headers.add("X-Performance-Cache-%s" % k.capitalize(), ", ".join(s))
        for k, v in pdb_context.items():
            s = sorted("%s %d" % p for p in v.get().items())
            response.headers.add("X-Performance-Precomputed-%s" % k.capitalize(), ", ".join(s))
        with sentry_sdk.configure_scope() as scope:
            for k, v in response.headers.items():
                scope.set_extra(k, v)
        if "X-Requested-Load" in response.headers:
            request.app["requests_rejected_by_load"] \
                .labels(__package__, __version__) \
                .inc()
        code = response.status
    else:
        code = 500
    request.app["request_in_progress"] \
        .labels(__package__, __version__, request.path, request.method) \
        .dec()
    request.app["request_count"] \
        .labels(__package__, __version__, request.method, request.path, code, account) \
        .inc()
    elapsed = (time() - start_time) or 0.001
    if measure_latency_re.match(request.path):
        request.app["request_latency"] \
            .labels(__package__, __version__, request.path, account) \
            .observe(elapsed)
        db_latency = request.app["db_latency"]
        db_latency \
            .labels(__package__, __version__, request.path, "state") \
            .observe(sdb_elapsed)
        db_latency \
            .labels(__package__, __version__, request.path, "metadata") \
            .observe(mdb_elapsed)
        db_latency \
            .labels(__package__, __version__, request.path, "precomputed") \
            .observe(pdb_elapsed)
        db_latency \
            .labels(__package__, __version__, request.path, "persistentdata") \
            .observe(rdb_elapsed)
    if elapsed > elapsed_error_threshold:
        with sentry_sdk.push_scope() as scope:
            scope.fingerprint = ["{{ default }}", request.path]
            logging.getLogger(f"{__package__}.instrument").error(
                "%s took %ds -> HTTP %d", request.path, int(elapsed), code)


@web.middleware
async def instrument(request: web.Request, handler) -> web.Response:
    """Middleware to count requests, record the elapsed time and track features flags."""
    start_time = time()
    request.app["request_in_progress"] \
        .labels(__package__, __version__, request.path, request.method) \
        .inc()
    request.app["db_elapsed"].set(defaultdict(float))
    request.app[METRICS_CALCULATOR_VAR_NAME].set(defaultdict(str))
    for v in chain(request.app["cache_context"].values(), request.app["pdb_context"].values()):
        v.set(defaultdict(int))
    try:
        response = await handler(request)  # type: web.Response
        return response
    finally:
        if request.method != "OPTIONS":
            try:
                response
            except NameError:
                response = None
            _after_response(request, response, start_time)


def setup_prometheus(app: web.Application) -> None:
    """Initialize the Prometheus registry, the common metrics, and add the tracking middleware."""
    registry = prometheus_client.CollectorRegistry(auto_describe=True)
    app[PROMETHEUS_REGISTRY_VAR_NAME] = registry
    app[METRICS_CALCULATOR_VAR_NAME] = ContextVar(METRICS_CALCULATOR_VAR_NAME, default=None)
    app["load"] = prometheus_client.Gauge(
        "load", "Current server load in abstract units",
        ["app_name", "version"],
        registry=registry,
    )
    app["request_count"] = prometheus_client.Counter(
        "requests_total", "Total request count",
        ["app_name", "version", "method", "endpoint", "http_status", "account"],
        registry=registry,
    )
    app["requests_rejected_by_load"] = requests_rejected_by_load_total = prometheus_client.Counter(
        "requests_rejected_by_load", "Number of requests rejected for exceeding the maximum load",
        ["app_name", "version"],
        registry=registry,
    )
    # explicitly initialize this metric so it appears at startup
    requests_rejected_by_load_total.labels(__package__, __version__)
    app["request_latency"] = prometheus_client.Histogram(
        "request_latency_seconds", "Request latency",
        ["app_name", "version", "endpoint", "account"],
        buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0,
                 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                 12.0, 15.0, 20.0, 25.0, 30.0,
                 45.0, 60.0, 120.0, 180.0, 240.0, prometheus_client.metrics.INF],
        registry=registry,
    )
    app["request_in_progress"] = prometheus_client.Gauge(
        "requests_in_progress_total", "Requests in progress",
        ["app_name", "version", "endpoint", "method"],
        registry=registry,
    )
    app["db_latency"] = prometheus_client.Histogram(
        "db_latency_seconds",
        "How much time was spent in each DB.",
        ["app_name", "version", "endpoint", "database"],
        buckets=[
            0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
            1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            12.0, 15.0, 20.0, 25.0, 30.0,
            45.0, 60.0, 120.0, 180.0, 240.0, prometheus_client.metrics.INF,
        ],
        registry=registry,
    )
    app["db_elapsed"] = ContextVar("db_elapsed", default=None)
    prometheus_client.Info("server", "API server metadata", registry=registry).info({
        "version": __version__,
        "commit": getattr(metadata, "__commit__", "null"),
        "build_date": getattr(metadata, "__date__", "null"),
        "max_load": str(app["max_load"]),
    })
    app.middlewares.insert(0, instrument)
