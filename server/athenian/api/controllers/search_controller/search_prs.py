from __future__ import annotations

from collections.abc import Collection
import dataclasses
from datetime import datetime, timedelta, timezone
import logging
from typing import Optional

from aiohttp import web
import aiomcache
import sentry_sdk

from athenian.api.async_utils import gather
from athenian.api.db import DatabaseLike
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import PRFactsCalculator
from athenian.api.internal.jira import get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.pull_request import fetch_prs_numbers
from athenian.api.internal.miners.types import (
    JIRAEntityToFetch,
    PRParticipants,
    PRParticipationKind,
    PullRequestFacts,
)
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import resolve_withgroups
from athenian.api.models.web import (
    PullRequestDigest,
    SearchPullRequestsRequest,
    SearchPullRequestsResponse,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response
from athenian.api.tracing import sentry_span

log = logging.getLogger(__name__)


async def search_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Search pull requests that satisfy the query."""
    search_request = model_from_body(SearchPullRequestsRequest, body)
    connectors = _SearchPRsConnectors(request.mdb, request.pdb, request.rdb, request.cache)
    account_info = await _build_account_info(search_request.account, request)
    search_filter = await _build_filter(search_request, account_info, request)
    repos_settings = await _build_repos_settings(account_info, search_filter.repositories)
    pr_digests = await _search_pr_digests(search_filter, account_info, repos_settings, connectors)
    return model_response(SearchPullRequestsResponse(pull_requests=pr_digests))


@sentry_span
async def _build_account_info(account: int, request: AthenianWebRequest) -> _SearchPRsAccountInfo:
    meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_account(
        account, prefixer, request.sdb, request.mdb, request.cache, request.app["slack"],
    )
    account_bots = await bots(account, meta_ids, request.mdb, request.sdb, request.cache)
    return _SearchPRsAccountInfo(account, meta_ids, prefixer, account_bots, settings)


@sentry_span
async def _build_repos_settings(
    account_info: _SearchPRsAccountInfo,
    repos: Collection[RepositoryName] | None,
) -> _SearchPRsReposSettings:
    repos_prefixed = None if repos is None else [str(r) for r in repos]
    release_settings, logical_settings = await gather(
        account_info.settings.list_release_matches(repos_prefixed),
        account_info.settings.list_logical_repositories(repos_prefixed),
    )
    return _SearchPRsReposSettings(release_settings, logical_settings)


@sentry_span
async def _build_filter(
    search_req: SearchPullRequestsRequest,
    account_info: _SearchPRsAccountInfo,
    request: AthenianWebRequest,
) -> _SearchPRsFilter:
    min_time = datetime.min.time()
    time_from = datetime.combine(search_req.date_from, min_time, tzinfo=timezone.utc)
    time_to = datetime.combine(
        search_req.date_to + timedelta(days=1), min_time, tzinfo=timezone.utc,
    )

    async def _resolve_repos():
        if search_req.repositories is None:
            return None
        repos, _ = await resolve_repos_with_request(
            search_req.repositories,
            account_info.account,
            request,
            meta_ids=account_info.meta_ids,
            prefixer=account_info.prefixer,
            pointer=".repositories",
        )
        return repos

    async def _resolve_participants():
        if search_req.participants is None:
            return None
        groups = await resolve_withgroups(
            [search_req.participants],
            PRParticipationKind,
            False,
            account_info.account,
            None,
            ".with",
            account_info.prefixer,
            request.sdb,
            group_type=set,
        )
        return groups[0] if groups else {}

    async def _resolve_jira():
        if search_req.jira is None:
            return None
        jira_conf = await get_jira_installation(
            account_info.account, request.sdb, request.mdb, request.cache,
        )
        return JIRAFilter.from_web(search_req.jira, jira_conf)

    repositories, participants, jira = await gather(
        _resolve_repos(), _resolve_participants(), _resolve_jira(),
    )

    return _SearchPRsFilter(time_from, time_to, repositories, participants, jira)


@dataclasses.dataclass
class _SearchPRsFilter:
    time_from: datetime
    time_to: datetime
    repositories: Optional[Collection[RepositoryName]] = None
    participants: Optional[PRParticipants] = None
    jira: Optional[JIRAFilter] = None


@dataclasses.dataclass
class _SearchPRsAccountInfo:
    account: int
    meta_ids: tuple[int, ...]
    prefixer: Prefixer
    bots: set[str]
    settings: Settings


@dataclasses.dataclass
class _SearchPRsReposSettings:
    release_settings: ReleaseSettings
    logical_settings: LogicalRepositorySettings


@dataclasses.dataclass
class _SearchPRsConnectors:
    mdb: DatabaseLike
    pdb: DatabaseLike
    rdb: DatabaseLike
    cache: Optional[aiomcache.Client]


@sentry_span
async def _search_pr_digests(
    search_filter: _SearchPRsFilter,
    account_info: _SearchPRsAccountInfo,
    repos_settings: _SearchPRsReposSettings,
    connectors: _SearchPRsConnectors,
) -> list[PullRequestDigest]:
    mdb, pdb, rdb, cache = connectors.mdb, connectors.pdb, connectors.rdb, connectors.cache
    if search_filter.repositories is None:
        repos = set(repos_settings.release_settings.native.keys())
    else:
        repos = {rname.unprefixed for rname in search_filter.repositories}

    calc = PRFactsCalculator(
        account_info.account, account_info.meta_ids, mdb, pdb, rdb, cache=cache,
    )
    pr_facts = await calc(
        search_filter.time_from,
        search_filter.time_to,
        repos,
        search_filter.participants or {},
        LabelFilter.empty(),
        search_filter.jira or JIRAFilter.empty(),
        exclude_inactive=True,
        bots=account_info.bots,
        release_settings=repos_settings.release_settings,
        logical_settings=repos_settings.logical_settings,
        prefixer=account_info.prefixer,
        fresh=False,
        with_jira=JIRAEntityToFetch.NOTHING,
    )

    prs_numbers = await fetch_prs_numbers(
        pr_facts[PullRequestFacts.f.node_id].values, account_info.meta_ids, mdb,
    )
    known_mask = prs_numbers != 0

    prefix_logical_repo = account_info.prefixer.prefix_logical_repo
    repo_mapping = {r: prefix_logical_repo(r) for r in repos}
    with sentry_sdk.start_span(op="materialize models", description=str(len(pr_facts))):
        pr_digests = [
            PullRequestDigest(number=number, repository=repo_mapping[repository_full_name])
            for node_id, repository_full_name, number in zip(
                pr_facts[PullRequestFacts.f.node_id].values[known_mask],
                pr_facts[PullRequestFacts.f.repository_full_name].values[known_mask],
                prs_numbers[known_mask],
            )
        ]

    unknown_prs = pr_facts[PullRequestFacts.f.node_id].values[~known_mask]
    if len(unknown_prs):
        log.error(
            "Cannot fetch PR numbers, probably missing entries in node_pullrequest table; PR node"
            " IDs: %s",
            ",".join(map(str, unknown_prs)),
        )
    return pr_digests
