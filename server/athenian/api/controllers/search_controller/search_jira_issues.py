from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from aiohttp import web
import aiomcache

from athenian.api.async_utils import gather
from athenian.api.db import Database
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import ParticipantsMerge
from athenian.api.internal.jira import JIRAConfig, get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.jira.issue import fetch_jira_issues
from athenian.api.internal.miners.participation import JIRAParticipants
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import resolve_jira_with
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.web import (
    JIRAIssueDigest,
    SearchJIRAIssuesRequest,
    SearchJIRAIssuesResponse,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response


async def search_jira_issues(request: AthenianWebRequest, body: dict) -> web.Response:
    """Search Jira issues that satisfy the filters."""
    search_request = model_from_body(SearchJIRAIssuesRequest, body)
    account_info = await _AccountInfo.from_request(search_request.account, request)
    connectors = _Connectors(request.sdb, request.mdb, request.pdb, request.cache)
    search_filter = await _SearchFilter.build(search_request, account_info, connectors)
    digests = await _search_jira_issue_digests(search_filter, account_info, connectors)
    return model_response(SearchJIRAIssuesResponse(jira_issues=digests))


@dataclass
class _Connectors:
    sdb: Database
    mdb: Database
    pdb: Database
    cache: aiomcache.Client | None


@dataclass
class _AccountInfo:
    account: int
    meta_ids: tuple[int, ...]
    jira_conf: JIRAConfig
    release_settings: ReleaseSettings
    logical_settings: LogicalRepositorySettings
    default_branches: dict[str, str]

    @classmethod
    async def from_request(cls, account: int, request: AthenianWebRequest) -> "_AccountInfo":
        sdb, pdb, mdb, cache = request.sdb, request.pdb, request.mdb, request.cache
        meta_ids = await get_metadata_account_ids(account, sdb, cache)
        prefixer = await Prefixer.load(meta_ids, mdb, cache)
        settings = Settings.from_account(account, prefixer, sdb, mdb, cache, request.app["slack"])
        release_settings, logical_settings, (_, default_branches), jira_conf = await gather(
            settings.list_release_matches(None),
            settings.list_logical_repositories(None),
            BranchMiner.load_branches(
                None, prefixer, account, meta_ids, mdb, pdb, cache, strip=True,
            ),
            get_jira_installation(account, request.sdb, request.mdb, request.cache),
        )

        return cls(
            account, meta_ids, jira_conf, release_settings, logical_settings, default_branches,
        )


@dataclass
class _SearchFilter:
    time_from: datetime | None
    time_to: datetime | None
    jira: JIRAFilter
    participants: JIRAParticipants

    @classmethod
    async def build(
        cls,
        search_req: SearchJIRAIssuesRequest,
        acc_info: _AccountInfo,
        conns: _Connectors,
    ) -> "_SearchFilter":
        time_from = time_to = None
        if from_ := search_req.date_from:
            time_from = datetime.combine(from_, time.min, tzinfo=timezone.utc)
        if to_ := search_req.date_to:
            time_to = datetime.combine(to_ + timedelta(days=1), time.min, tzinfo=timezone.utc)

        if search_req.filter:
            jira = JIRAFilter.from_web(search_req.filter, acc_info.jira_conf)
        else:
            jira = JIRAFilter.from_jira_config(acc_info.jira_conf).replace(custom_projects=False)

        if search_req.with_:
            all_participants = await resolve_jira_with(
                [search_req.with_], acc_info.account, conns.sdb, conns.mdb, conns.cache,
            )
            participants = all_participants[0]
        else:
            participants = {}

        return cls(time_from, time_to, jira, participants)


async def _search_jira_issue_digests(
    search_filter: _SearchFilter,
    acc_info: _AccountInfo,
    conns: _Connectors,
) -> list[JIRAIssueDigest]:
    reporters, assignees, commenters = ParticipantsMerge.jira([search_filter.participants])
    issues = await fetch_jira_issues(
        search_filter.time_from,
        search_filter.time_to,
        search_filter.jira,
        True,
        reporters,
        assignees,
        commenters,
        False,
        acc_info.default_branches,
        acc_info.release_settings,
        acc_info.logical_settings,
        acc_info.account,
        acc_info.meta_ids,
        conns.mdb,
        conns.pdb,
        conns.cache,
        extra_columns=[Issue.key],
    )

    return [JIRAIssueDigest(id=issue_key) for issue_key in issues[Issue.key.name].values]
