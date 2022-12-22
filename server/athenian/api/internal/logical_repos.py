from typing import Iterable

from athenian.api.internal.logical_accelerated import drop_logical_repo


def is_logical_repo(repo: str) -> bool:
    """Return the value indicating whether the repository name is logical."""
    return repo.count("/") > 1


def coerce_logical_repos(repos: Iterable[str]) -> dict[str, set[str]]:
    """Remove the logical part of the repository names + deduplicate."""
    result: dict[str, set[str]] = {}
    for r in repos:
        result.setdefault(drop_logical_repo(r), set()).add(r)
    return result


def contains_logical_repos(repos: Iterable[str]) -> bool:
    """Check whether at least one repository name is logical."""
    return any(r.count("/") > 1 for r in repos)
