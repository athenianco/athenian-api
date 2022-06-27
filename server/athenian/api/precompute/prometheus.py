"""Prometheus support module for precompute batch commands."""
from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Histogram, push_to_gateway


def push_metrics(gateway: str) -> None:
    """Push current metrics to Prometheus Pushgateway."""
    push_to_gateway(gateway, _PROMETHEUS_JOB, _registry)


def get_metrics() -> Metrics:
    """Get the singleton collection of metrics used by precomputer batch jobs."""
    return _metrics


@dataclass
class Metrics:
    """Collection of metrics used by precomputer batch jobs."""

    precompute_account_seconds: Histogram
    precompute_account_successes_total: Counter
    precompute_account_failures_total: Counter


_PROMETHEUS_JOB = "precomputer"

_registry = CollectorRegistry()


def _build_metrics() -> Metrics:
    return Metrics(
        precompute_account_seconds=Histogram(
            "precompute_account_seconds",
            "Duration of the precompute operation on a single account.",
            ["account", "github_account", "is_fresh"],
            registry=_registry,
            buckets=[1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 180, 300, 600, 1200],
        ),
        precompute_account_successes_total=Counter(
            "precompute_account_successes_total",
            "Times the precompute operation ended successfully.",
            ["account", "github_account", "is_fresh"],
            registry=_registry,
        ),
        precompute_account_failures_total=Counter(
            "precompute_account_failures_total",
            "Times the precompute operation failed.",
            ["account", "github_account", "is_fresh"],
            registry=_registry,
        ),
    )


_metrics = _build_metrics()
