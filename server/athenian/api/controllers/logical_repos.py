from typing import Dict, Iterable, Set


def drop_logical_repo(repo: str) -> str:
    """Remove the logical part of the repository name."""
    return "/".join(repo.split("/", 2)[:2])


def coerce_logical_repos(repos: Iterable[str]) -> Dict[str, Set[str]]:
    """Remove the logical part of the repository names + deduplicate."""
    result = {}
    for r in repos:
        result.setdefault(drop_logical_repo(r), set()).add(r)
    return result


def contains_logical_repos(repos: Iterable[str]) -> bool:
    """Check whether at least one repository name is logical."""
    return any(r.count("/") > 1 for r in repos)
