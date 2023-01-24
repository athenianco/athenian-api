from __future__ import annotations

import logging
from typing import Optional, Sequence

from athenian.api import metadata
from athenian.api.internal.prefixer import Prefixer, RepositoryName, RepositoryReference
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.web import InvalidRequestError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError


def dereference_db_repositories(
    repos: list[tuple[int, Optional[str]]],
    prefixer: Prefixer,
    logical_settings: LogicalRepositorySettings,
) -> tuple[RepositoryName, ...]:
    """Dereference the repository IDs into a RepositoryName sequence."""
    references = parse_db_repositories(repos)
    resolved = []
    deleted_logical = []
    for name in prefixer.dereference_repositories(references):
        if name.logical and not logical_settings.repo_exists(name):
            deleted_logical.append(str(name))
            continue
        resolved.append(name)
    if deleted_logical:
        if deleted_logical:
            log = logging.getLogger(f"{metadata.__package__}.dereference_db_repositories")
            log.warning("ignoring deleted logical repositories: %s", deleted_logical)
    return tuple(resolved)


def parse_db_repositories(
    val: Optional[list[tuple[int, str | None]]],
) -> Optional[list[RepositoryReference]]:
    """Parse the raw value in DB repositories JSON column as list of `RepositoryReference`."""
    if val is None:
        return val
    return [
        RepositoryReference("github.com", repo_id, logical_name or "")
        for repo_id, logical_name in val
    ]


def dump_db_repositories(
    repo_idents: Optional[Sequence[RepositoryReference]],
) -> Optional[list[tuple[int, str]]]:
    """Dump the sequence of RepositoryReference-s in the format used by DB repositories JSON \
    column."""
    if repo_idents is None:
        return None
    return [(ident.node_id, ident.logical_name) for ident in repo_idents]


async def parse_request_repositories(
    repo_names: Optional[list[str]],
    request: AthenianWebRequest,
    account: int,
    pointer: str = ".repositories",
) -> Optional[list[tuple[int, str]]]:
    """Resolve repository node IDs from the prefixed names."""
    if repo_names is None:
        return None
    prefixer = await Prefixer.from_request(request, account)
    try:
        return dump_db_repositories(prefixer.reference_repositories(repo_names))
    except ValueError as e:
        raise ResponseError(InvalidRequestError(".repositories", str(e)))
