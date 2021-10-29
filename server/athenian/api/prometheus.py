from collections import defaultdict
from contextvars import ContextVar
from itertools import chain
import logging
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


def _after_response(request: web.Request,
                    response: Optional[web.Response],
                    start_time: float,
                    ) -> None:
    db_elapsed = request.app["db_elapsed"].get()
    metrics_calculator = request.app[METRICS_CALCULATOR_VAR_NAME].get()
    cache_context = request.app["cache_context"]
    pdb_context = request.app["pdb_context"]
    sdb_elapsed, mdb_elapsed, pdb_elapsed, rdb_elapsed = (
        db_elapsed[x + "db"] for x in ("s", "m", "p", "r"))
    if response is not None:
        response.headers.add(
            "X-Performance-DB",
            "s %.3f, m %.3f, p %.3f, r %.3f" % (
                sdb_elapsed, mdb_elapsed, pdb_elapsed, rdb_elapsed))
        response.headers.add("X-Metrics-Calculator",
                             ",".join(f"{k}={v}" for k, v in metrics_calculator.items()))
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
    request.app["state_db_latency"] \
        .labels(__package__, __version__, request.path) \
        .observe(sdb_elapsed)
    request.app["metadata_db_latency"] \
        .labels(__package__, __version__, request.path) \
        .observe(mdb_elapsed)
    request.app["precomputed_db_latency"] \
        .labels(__package__, __version__, request.path) \
        .observe(pdb_elapsed)
    request.app["persistentdata_db_latency"] \
        .labels(__package__, __version__, request.path) \
        .observe(rdb_elapsed)
    request.app["request_in_progress"] \
        .labels(__package__, __version__, request.path, request.method) \
        .dec()
    elapsed = (time() - start_time) or 0.001
    request.app["request_latency"] \
        .labels(__package__, __version__, request.path) \
        .observe(elapsed)
    request.app["state_db_latency_ratio"] \
        .labels(__package__, __version__, request.path) \
        .observe(sdb_elapsed / elapsed)
    request.app["metadata_db_latency_ratio"] \
        .labels(__package__, __version__, request.path) \
        .observe(mdb_elapsed / elapsed)
    request.app["precomputed_db_latency_ratio"] \
        .labels(__package__, __version__, request.path) \
        .observe(pdb_elapsed / elapsed)
    request.app["persistentdata_db_latency_ratio"] \
        .labels(__package__, __version__, request.path) \
        .observe(rdb_elapsed / elapsed)
    code = response.status if response is not None else 500
    if elapsed > elapsed_error_threshold:
        with sentry_sdk.push_scope() as scope:
            scope.fingerprint = ["{{ default }}", request.path]
            logging.getLogger(f"{__package__}.instrument").error(
                "%s took %ds -> HTTP %d", request.path, int(elapsed), code)
    request.app["request_count"] \
        .labels(__package__, __version__, request.method, request.path, code) \
        .inc()


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
        return (response := await handler(request))  # type: web.Response
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
    app["request_count"] = prometheus_client.Counter(
        "requests_total", "Total request count",
        ["app_name", "version", "method", "endpoint", "http_status"],
        registry=registry,
    )
    app["request_latency"] = prometheus_client.Histogram(
        "request_latency_seconds", "Request latency",
        ["app_name", "version", "endpoint"],
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
    db_latency_buckets = [
        0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0,
        1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        12.0, 15.0, 20.0, 25.0, 30.0,
        45.0, 60.0, 120.0, 180.0, 240.0, prometheus_client.metrics.INF]
    for db in ("state", "metadata", "precomputed", "persistentdata"):
        app["%s_db_latency" % db] = prometheus_client.Histogram(
            "%s_db_latency_seconds" % db,
            "%s%s DB latency" % (db[0].upper(), db[1:]),
            ["app_name", "version", "endpoint"],
            buckets=db_latency_buckets,
            registry=registry,
        )
    db_ratio_buckets = [
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
        0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
        0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for db in ("state", "metadata", "precomputed", "persistentdata"):
        app["%s_db_latency_ratio" % db] = prometheus_client.Histogram(
            "%s_db_latency_ratio" % db,
            "%s%s DB latency ratio to request time" % (db[0].upper(), db[1:]),
            ["app_name", "version", "endpoint"],
            buckets=db_ratio_buckets,
            registry=registry,
        )
    app["db_elapsed"] = ContextVar("db_elapsed", default=None)
    prometheus_client.Info("server", "API server metadata", registry=registry).info({
        "version": __version__,
        "commit": getattr(metadata, "__commit__", "null"),
        "build_date": getattr(metadata, "__date__", "null"),
    })
    app.middlewares.insert(0, instrument)
