from athenian.api.models.web.base_model_ import Model


class AccountHealth(Model):
    """Map from health metric names to metric values."""

    broken_branches: list[int]
    broken_dags: list[int]
    deployments: list[int]
    empty_releases: list[int]
    endpoint_p50: dict[str, list[float]]
    endpoint_p95: dict[str, list[float]]
    event_releases: list[int]
    inconsistent_nodes: dict[str, list[int]]
    pending_fetch_branches: list[int]
    pending_fetch_prs: list[int]
    prs_count: list[int]
    released_prs_ratio: list[float]
    reposet_problems: list[int]
    repositories_count: list[int]
    unresolved_deployments: list[int]
    unresolved_releases: list[int]
