from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain, repeat
import logging
import operator
from typing import Any, Callable, Generator, Iterable, Mapping, Optional, Sequence, Set, TypeVar

from aiohttp import web
import aiomcache
from dateutil.parser import parse as parse_datetime
import numpy as np
import pandas as pd
import sentry_sdk

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.cache import expires_header, middle_term_exptime, short_term_exptime
from athenian.api.db import Database
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.github.check_run_filter import filter_check_runs
from athenian.api.internal.features.github.pull_request_filter import (
    fetch_pull_requests,
    filter_pull_requests,
)
from athenian.api.internal.jira import (
    JIRAConfig,
    get_jira_installation_or_none,
    load_mapped_jira_users,
)
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import FilterCommitsProperty, extract_commits
from athenian.api.internal.miners.github.contributors import mine_contributors
from athenian.api.internal.miners.github.deployment import (
    deployment_facts_extract_mentioned_people,
    load_jira_issues_for_deployments,
    mine_deployments,
)
from athenian.api.internal.miners.github.deployment_light import (
    NoDeploymentNotificationsError,
    mine_environments,
)
from athenian.api.internal.miners.github.label import mine_labels
from athenian.api.internal.miners.github.release_mine import (
    diff_releases as mine_diff_releases,
    mine_releases,
    mine_releases_by_name,
)
from athenian.api.internal.miners.github.repository import mine_repositories
from athenian.api.internal.miners.github.user import UserAvatarKeys, mine_user_avatars
from athenian.api.internal.miners.jira.issue import fetch_jira_issues_by_keys
from athenian.api.internal.miners.participation import (
    PRParticipants,
    PRParticipationKind,
    ReleaseParticipationKind,
)
from athenian.api.internal.miners.types import (
    Deployment,
    DeploymentConclusion,
    DeploymentFacts,
    JIRAEntityToFetch,
    PullRequestEvent,
    PullRequestJIRAIssueItem,
    PullRequestListItem,
    PullRequestStage,
    ReleaseFacts,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import (
    compile_developers,
    fetch_teams_map,
    resolve_withgroups,
    scan_for_teams,
)
from athenian.api.models.metadata.github import PullRequest, PushCommit, Release, User
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)
from athenian.api.models.web import (
    BadRequestError,
    Commit,
    CommitSignature,
    CommitsList,
    CommitsListInclude,
    CommonFilterProperties,
    DeployedComponent as WebDeployedComponent,
    DeployedPullRequest,
    DeployedRelease,
    DeploymentAnalysisCode,
    DeploymentNotification as WebDeploymentNotification,
    DeveloperSummary,
    DeveloperUpdates,
    FilterCommitsRequest,
    FilterContributorsRequest,
    FilterDeploymentsRequest,
    FilteredDeployment,
    FilteredEnvironment,
    FilteredLabel,
    FilteredRelease,
    FilterEnvironmentsRequest,
    FilterLabelsRequest,
    FilterPullRequestsRequest,
    FilterReleasesRequest,
    FilterRepositoriesRequest,
    ForbiddenError,
    GetPullRequestsRequest,
    GetReleasesRequest,
    IncludedNativeUser,
    InvalidRequestError,
    LinkedJIRAIssue,
    NoSourceDataError,
    PullRequest as WebPullRequest,
    PullRequestLabel,
    PullRequestParticipant,
    PullRequestSet,
    PullRequestSetInclude,
    ReleasedPullRequest,
    ReleaseSet,
    ReleaseSetInclude,
    StageTimings,
)
from athenian.api.models.web.code_check_run_statistics import CodeCheckRunStatistics
from athenian.api.models.web.diff_releases_request import DiffReleasesRequest
from athenian.api.models.web.diffed_releases import DiffedReleases
from athenian.api.models.web.filter_code_checks_request import FilterCodeChecksRequest
from athenian.api.models.web.filtered_code_check_run import FilteredCodeCheckRun
from athenian.api.models.web.filtered_code_check_runs import FilteredCodeCheckRuns
from athenian.api.models.web.filtered_deployments import FilteredDeployments
from athenian.api.models.web.release_diff import ReleaseDiff
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass_asdict
from athenian.api.unordered_unique import unordered_unique


@weight(2.5)
async def filter_contributors(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find developers that made an action within the given timeframe."""
    try:
        filt = FilterContributorsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    (
        time_from,
        time_to,
        repos,
        meta_ids,
        prefixer,
        logical_settings,
    ) = await _common_filter_preprocess(filt, filt.in_, request, strip_prefix=False)
    settings = Settings.from_request(request, filt.account, prefixer)
    release_settings = await settings.list_release_matches(repos)
    repos = [r.split("/", 1)[1] for r in repos]
    users = await mine_contributors(
        repos,
        time_from,
        time_to,
        True,
        filt.as_ or [],
        release_settings,
        logical_settings,
        prefixer,
        filt.account,
        meta_ids,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )
    mapped_jira = await load_mapped_jira_users(
        filt.account,
        [u[User.node_id.name] for u in users],
        request.sdb,
        request.mdb,
        request.cache,
    )
    model = [
        DeveloperSummary(
            login=prefixer.user_node_to_prefixed_login[u[User.node_id.name]],
            avatar=u[User.avatar_url.name],
            name=u[User.name.name],
            updates=DeveloperUpdates(
                **{
                    k: v
                    for k, v in u["stats"].items()
                    # TODO(se7entyse7en): make `DeveloperUpdates` support all the stats we can get instead of doing this filtering. See also `mine_contributors`.  # noqa
                    if k in DeveloperUpdates.attribute_types
                },
            ),
            jira_user=mapped_jira.get(u[User.node_id.name]),
        )
        for u in sorted(users, key=operator.itemgetter("login"))
    ]
    return model_response(model)


@expires_header(short_term_exptime)
@weight(0.5)
async def filter_repositories(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find repositories that were updated within the given timeframe."""
    try:
        filt = FilterRepositoriesRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    (
        time_from,
        time_to,
        repos,
        meta_ids,
        prefixer,
        logical_settings,
    ) = await _common_filter_preprocess(filt, filt.in_, request, strip_prefix=False)
    settings = Settings.from_request(request, filt.account, prefixer)
    release_settings = await settings.list_release_matches(repos)
    repos = [r.split("/", 1)[1] for r in repos]
    repos = await mine_repositories(
        repos,
        time_from,
        time_to,
        filt.exclude_inactive,
        release_settings,
        prefixer,
        filt.account,
        meta_ids,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )
    repos = prefixer.prefix_repo_names(repos)
    return web.json_response(repos)


async def _common_filter_preprocess(
    filt: CommonFilterProperties,
    repos: Optional[list[str]],
    request: AthenianWebRequest,
    strip_prefix=True,
) -> tuple[datetime, datetime, Set[str], tuple[int, ...], Prefixer, LogicalRepositorySettings]:
    if filt.date_to < filt.date_from:
        raise ResponseError(
            InvalidRequestError(
                detail="date_from may not be greater than date_to",
                pointer=".date_from",
            ),
        )
    min_time = datetime.min.time()
    time_from = datetime.combine(filt.date_from, min_time, tzinfo=timezone.utc)
    time_to = datetime.combine(filt.date_to + timedelta(days=1), min_time, tzinfo=timezone.utc)
    if filt.timezone is not None:
        tzoffset = timedelta(minutes=-filt.timezone)
        time_from += tzoffset
        time_to += tzoffset
    repos, meta_ids, prefixer, logical_settings = await _repos_preprocess(
        repos, filt.account, request, strip_prefix=strip_prefix,
    )
    return time_from, time_to, repos, meta_ids, prefixer, logical_settings


async def _repos_preprocess(
    repos: Optional[list[str]],
    account: int,
    request: AthenianWebRequest,
    strip_prefix=True,
) -> tuple[Set[str], tuple[int, ...], Prefixer, LogicalRepositorySettings]:
    meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, account, prefixer)
    (repos, _), logical_settings = await gather(
        resolve_repos_with_request(
            repos,
            account,
            request,
            meta_ids=meta_ids,
            prefixer=prefixer,
        ),
        settings.list_logical_repositories(),
    )
    if strip_prefix:
        repos = [r.unprefixed for r in repos]
    else:
        repos = [str(r) for r in repos]
    return repos, meta_ids, prefixer, logical_settings


async def resolve_filter_prs_parameters(
    filt: FilterPullRequestsRequest,
    request: AthenianWebRequest,
) -> tuple[
    datetime,
    datetime,
    Set[str],
    Set[str],
    Set[str],
    PRParticipants,
    LabelFilter,
    JIRAFilter,
    Set[str],
    ReleaseSettings,
    LogicalRepositorySettings,
    Prefixer,
    tuple[int, ...],
]:
    """Infer all the required PR filters from the request."""
    (
        time_from,
        time_to,
        repos,
        meta_ids,
        prefixer,
        logical_settings,
    ) = await _common_filter_preprocess(filt, filt.in_, request, strip_prefix=False)
    events = {getattr(PullRequestEvent, e.upper()) for e in (filt.events or [])}
    stages = {getattr(PullRequestStage, s.upper()) for s in (filt.stages or [])}
    if not events and not stages:
        raise ResponseError(
            InvalidRequestError(
                detail="Either `events` or `stages` must be specified and be not empty.",
                pointer=".stages",
            ),
        )
    participants = await resolve_withgroups(
        [filt.with_],
        PRParticipationKind,
        False,
        filt.account,
        None,
        ".with",
        prefixer,
        request.sdb,
        group_type=set,
    )
    participants = participants[0] if participants else {}
    settings = Settings.from_request(request, filt.account, prefixer)
    release_settings, jira, account_bots = await gather(
        settings.list_release_matches(repos),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
        bots(filt.account, meta_ids, request.mdb, request.sdb, request.cache),
    )
    repos = {r.split("/", 1)[1] for r in repos}
    labels = LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude)
    if jira is not None:
        jira = JIRAFilter.from_web(filt.jira, jira)
    else:
        jira = JIRAFilter.empty()
    return (
        time_from,
        time_to,
        repos,
        events,
        stages,
        participants,
        labels,
        jira,
        account_bots,
        release_settings,
        logical_settings,
        prefixer,
        meta_ids,
    )


@expires_header(short_term_exptime)
@weight(6)
async def filter_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """List pull requests that satisfy the query."""
    try:
        filt = FilterPullRequestsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    (
        time_from,
        time_to,
        repos,
        events,
        stages,
        participants,
        labels,
        jira,
        account_bots,
        release_settings,
        logical_settings,
        prefixer,
        meta_ids,
    ) = await resolve_filter_prs_parameters(filt, request)
    updated_min, updated_max = _bake_updated_min_max(filt)
    prs, deployments = await filter_pull_requests(
        events,
        stages,
        time_from,
        time_to,
        repos,
        participants,
        labels,
        jira,
        filt.environments,
        filt.exclude_inactive,
        account_bots,
        release_settings,
        logical_settings,
        updated_min,
        updated_max,
        prefixer,
        filt.account,
        meta_ids,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )
    return await _build_github_prs_response(
        prs, deployments, prefixer, meta_ids, request.mdb, request.cache,
    )


def _bake_updated_min_max(filt: FilterPullRequestsRequest) -> tuple[datetime, datetime]:
    if (filt.updated_from is None) != (filt.updated_to is None):
        raise ResponseError(
            InvalidRequestError(
                ".updated_from",
                "`updated_from` and `updated_to` must be both either specified or not",
            ),
        )
    if filt.updated_from is not None:
        updated_min = datetime.combine(filt.updated_from, datetime.min.time(), tzinfo=timezone.utc)
        updated_max = datetime.combine(
            filt.updated_to, datetime.min.time(), tzinfo=timezone.utc,
        ) + timedelta(days=1)
    else:
        updated_min = updated_max = None
    return updated_min, updated_max


T = TypeVar("T")


def web_pr_from_struct(
    prs: Iterable[PullRequestListItem],
    prefixer: Prefixer,
    log: logging.Logger,
    postprocess: Callable[[WebPullRequest, PullRequestListItem], T] = lambda w, _: w,
) -> Generator[T, None, None]:
    """Convert an intermediate PR representation to the web model."""
    for pr in prs:
        props = dict(dataclass_asdict(pr))
        del props["node_id"]

        props["repository"] = prefixer.prefix_logical_repo(props["repository"])
        if props["repository"] is None:
            # deleted repository
            continue
        if pr.events_time_machine is not None:
            props["events_time_machine"] = sorted(p.name.lower() for p in pr.events_time_machine)
        if pr.stages_time_machine is not None:
            props["stages_time_machine"] = sorted(p.name.lower() for p in pr.stages_time_machine)
        props["events_now"] = sorted(p.name.lower() for p in pr.events_now)
        props["stages_now"] = sorted(p.name.lower() for p in pr.stages_now)
        props["stage_timings"] = StageTimings(**pr.stage_timings)
        participants = defaultdict(list)
        for pk, pids in sorted(pr.participants.items()):
            pkweb = pk.name.lower()
            for pid in pids:
                try:
                    participants[prefixer.user_node_to_prefixed_login[pid]].append(pkweb)
                except KeyError:
                    log.error("Failed to resolve user %s", pid)
        props["participants"] = sorted(
            PullRequestParticipant(id=id_, status=status) for id_, status in participants.items()
        )
        if pr.labels is not None:
            props["labels"] = [PullRequestLabel(**dataclass_asdict(label)) for label in pr.labels]
        if pr.jira is not None:
            props["jira"] = jira = [LinkedJIRAIssue(**dataclass_asdict(iss)) for iss in pr.jira]
            for issue in jira:
                if issue.labels is not None:
                    # it is a set, must be a list
                    issue.labels = sorted(issue.labels)
        yield postprocess(WebPullRequest(**props), pr)


def _nan_to_none(val):
    if val != val:
        return None
    return val


@expires_header(short_term_exptime)
@weight(1.5)
async def filter_commits(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find commits that match the specified query."""
    try:
        filt = FilterCommitsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    time_from, time_to, repos, meta_ids, prefixer, _ = await _common_filter_preprocess(
        filt, filt.in_, request,
    )
    teams = set()
    scan_for_teams(filt.with_author, teams, ".with_author")
    scan_for_teams(filt.with_committer, teams, ".with_committer")
    teams_map = await fetch_teams_map(teams, filt.account, request.sdb)
    with_author = compile_developers(
        filt.with_author, teams_map, None, False, prefixer, ".with_author",
    )
    with_committer = compile_developers(
        filt.with_committer, teams_map, None, False, prefixer, ".with_committer",
    )
    log = logging.getLogger(f"{metadata.__package__}.filter_commits")
    commits, deployments = await extract_commits(
        FilterCommitsProperty(filt.property),
        time_from,
        time_to,
        repos,
        with_author,
        with_committer,
        filt.only_default_branch,
        BranchMiner(),
        prefixer,
        filt.account,
        meta_ids,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )
    prefix_logical_repo = prefixer.prefix_logical_repo
    model = CommitsList(
        data=[],
        include=CommitsListInclude(
            users={},
            deployments={
                key: webify_deployment(val, prefix_logical_repo)
                for key, val in sorted(deployments.items())
            },
        ),
    )
    users = model.include.users
    utc = timezone.utc
    repo_name_map, user_login_map = (
        prefixer.repo_name_to_prefixed_name,
        prefixer.user_login_to_prefixed_login,
    )
    with sentry_sdk.start_span(op="filter_commits/generate response"):
        for (
            author_login,
            committer_login,
            repository_full_name,
            sha,
            children,
            deployments,
            message,
            additions,
            deletions,
            changed_files,
            author_name,
            author_email,
            authored_date,
            committer_name,
            committer_email,
            committed_date,
            author_date,
            commit_date,
            author_avatar_url,
            committer_avatar_url,
        ) in zip(
            commits[PushCommit.author_login.name].values,
            commits[PushCommit.committer_login.name].values,
            commits[PushCommit.repository_full_name.name].values,
            commits[PushCommit.sha.name].values,
            commits["children"].values,
            commits["deployments"].values,
            commits[PushCommit.message.name].values,
            commits[PushCommit.additions.name].values,
            commits[PushCommit.deletions.name].values,
            commits[PushCommit.changed_files.name].values,
            commits[PushCommit.author_name.name].values,
            commits[PushCommit.author_email.name].values,
            commits[PushCommit.authored_date.name],
            commits[PushCommit.committer_name.name].values,
            commits[PushCommit.committer_email.name].values,
            commits[PushCommit.committed_date.name],
            commits[PushCommit.author_date.name],
            commits[PushCommit.commit_date.name],
            commits[PushCommit.author_avatar_url.name],
            commits[PushCommit.committer_avatar_url.name],
        ):
            obj = Commit(
                repository=repo_name_map[repository_full_name],
                hash=sha,
                children=children,
                deployments=deployments,
                message=message,
                size_added=_nan_to_none(additions),
                size_removed=_nan_to_none(deletions),
                files_changed=_nan_to_none(changed_files),
                author=CommitSignature(
                    login=(user_login_map[author_login]) if author_login else None,
                    name=author_name,
                    email=author_email,
                    timestamp=authored_date.replace(tzinfo=utc),
                    timezone=0,
                ),
                committer=CommitSignature(
                    login=(user_login_map[committer_login]) if committer_login else None,
                    name=committer_name,
                    email=committer_email,
                    timestamp=committed_date.replace(tzinfo=utc),
                    timezone=0,
                ),
            )
            try:
                dt = parse_datetime(author_date)
                obj.author.timezone = dt.tzinfo.utcoffset(dt).total_seconds() / 3600
            except ValueError:
                log.warning("Failed to parse the author timestamp of %s", obj.hash)
            try:
                dt = parse_datetime(commit_date)
                obj.committer.timezone = dt.tzinfo.utcoffset(dt).total_seconds() / 3600
            except ValueError:
                log.warning("Failed to parse the committer timestamp of %s", obj.hash)
            if obj.author.login and obj.author.login not in users:
                users[obj.author.login] = IncludedNativeUser(avatar=author_avatar_url)
            if obj.committer.login and obj.committer.login not in users:
                users[obj.committer.login] = IncludedNativeUser(avatar=committer_avatar_url)
            model.data.append(obj)
    return model_response(model)


@expires_header(short_term_exptime)
@weight(1)
async def filter_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find releases that were published in the given time fram in the given repositories."""
    try:
        filt = FilterReleasesRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    (
        time_from,
        time_to,
        repos,
        meta_ids,
        prefixer,
        logical_settings,
    ) = await _common_filter_preprocess(filt, filt.in_, request, strip_prefix=False)
    participants = await resolve_withgroups(
        [filt.with_],
        ReleaseParticipationKind,
        True,
        filt.account,
        None,
        ".with",
        prefixer,
        request.sdb,
    )
    participants = participants[0] if participants else {}
    stripped_repos = [r.split("/", 1)[1] for r in repos]
    settings = Settings.from_request(request, filt.account, prefixer)
    release_settings, jira_ids, (branches, default_branches) = await gather(
        settings.list_release_matches(repos),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
        BranchMiner.extract_branches(
            stripped_repos, prefixer, meta_ids, request.mdb, request.cache,
        ),
    )
    releases, avatars, _, deployments = await mine_releases(
        repos=stripped_repos,
        participants=participants,
        branches=branches,
        default_branches=default_branches,
        time_from=time_from,
        time_to=time_to,
        labels=LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude),
        jira=JIRAFilter.from_web(filt.jira, jira_ids),
        release_settings=release_settings,
        logical_settings=logical_settings,
        prefixer=prefixer,
        account=filt.account,
        meta_ids=meta_ids,
        mdb=request.mdb,
        pdb=request.pdb,
        rdb=request.rdb,
        cache=request.cache,
        with_extended_pr_details=True,
        with_jira=JIRAEntityToFetch.ISSUES,
    )
    return await _build_release_set_response(
        releases, avatars, deployments, prefixer, jira_ids, request.mdb,
    )


async def _load_jira_issues_for_releases(
    releases: pd.DataFrame,
    jira_ids: Optional[JIRAConfig],
    mdb: Database,
) -> dict[str, LinkedJIRAIssue]:
    if releases.empty or jira_ids is None:
        return {}
    issue_keys = unordered_unique(np.concatenate(releases["jira_ids"].values))
    rows = await fetch_jira_issues_by_keys(issue_keys, jira_ids, mdb)
    issues = {row["id"]: LinkedJIRAIssue(**row) for row in rows}
    return issues


async def _build_release_set_response(
    releases: pd.DataFrame,
    avatars: Iterable[tuple[int, str]],
    deployments: dict[str, Deployment],
    prefixer: Prefixer,
    jira_ids: Optional[JIRAConfig],
    mdb: Database,
) -> web.Response:
    issues = await _load_jira_issues_for_releases(releases, jira_ids, mdb)
    prefix_logical_repo = prefixer.prefix_logical_repo
    user_node_to_login = prefixer.user_node_to_prefixed_login.get
    data = _filtered_releases_from_df(releases, prefixer)
    model = ReleaseSet(
        data=data,
        include=ReleaseSetInclude(
            users={
                pl: IncludedNativeUser(avatar=a)
                for u, a in avatars
                if (pl := user_node_to_login(u)) is not None
            },
            jira=issues,
            deployments={
                key: webify_deployment(val, prefix_logical_repo)
                for key, val in sorted(deployments.items())
            }
            or None,
        ),
    )
    return model_response(model)


def _filtered_releases_from_df(df: pd.DataFrame, prefixer: Prefixer) -> list[FilteredRelease]:
    if df.empty:
        return []
    repo_name_to_prefixed_name = prefixer.prefix_logical_repo
    user_node_to_prefixed_login = prefixer.user_node_to_prefixed_login.get
    return [
        FilteredRelease(
            name=name,
            sha=sha,
            repository=repo_name_to_prefixed_name(repo),
            url=url,
            publisher=user_node_to_prefixed_login(publisher),
            published=pd.Timestamp(published, tzinfo=timezone.utc),
            age=age,
            added_lines=additions,
            deleted_lines=deletions,
            commits=commits_count,
            commit_authors=sorted(user_node_to_prefixed_login(u) for u in commit_authors),
            prs=_extract_released_prs(*prs_columns, prefixer=prefixer),
            deployments=deployments,
        )
        for (
            name,
            sha,
            repo,
            url,
            publisher,
            published,
            age,
            additions,
            deletions,
            commits_count,
            commit_authors,
            *prs_columns,
            deployments,
        ) in zip(
            df[ReleaseFacts.f.name].values,
            df[ReleaseFacts.f.sha].values,
            df[ReleaseFacts.f.repository_full_name].values,
            df[ReleaseFacts.f.url].values,
            df[ReleaseFacts.f.publisher].values,
            df[ReleaseFacts.f.published].values,
            df[ReleaseFacts.f.age].values,
            df[ReleaseFacts.f.additions].values,
            df[ReleaseFacts.f.deletions].values,
            df[ReleaseFacts.f.commits_count].values,
            df[ReleaseFacts.f.commit_authors].values,
            df["prs_" + PullRequest.number.name].values,
            df["prs_" + PullRequest.title.name].values,
            df["prs_" + PullRequest.created_at.name].values,
            df["prs_" + PullRequest.additions.name].values,
            df["prs_" + PullRequest.deletions.name].values,
            df["prs_" + PullRequest.user_node_id.name].values,
            df["jira_ids"].values,
            df["jira_pr_offsets"].values,
            df[ReleaseFacts.f.deployments].values,
        )
    ]


def _extract_released_prs(
    *pr_columns: np.ndarray,
    prefixer: Prefixer,
) -> list[ReleasedPullRequest]:
    user_node_to_prefixed_login = prefixer.user_node_to_prefixed_login
    *pr_columns, jira_ids, jira_pr_offsets = pr_columns
    return [
        ReleasedPullRequest(
            number=number,
            title=title,
            created=created,
            additions=adds,
            deletions=dels,
            author=user_node_to_prefixed_login.get(author),
            jira=jira_ids[jira_offset_beg:jira_offset_end]
            if jira_offset_end > jira_offset_beg
            else None,
        )
        for number, title, created, adds, dels, author, jira_offset_beg, jira_offset_end in zip(
            *pr_columns, jira_pr_offsets[:-1], jira_pr_offsets[1:],
        )
    ]


@expires_header(short_term_exptime)
@weight(0.5)
async def get_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """List pull requests by repository and number."""
    req = GetPullRequestsRequest.from_dict(body)
    prs_by_repo: dict[str, set[int]] = {}
    for p in req.prs:
        prs_by_repo.setdefault(p.repository, set()).update(p.numbers)
    (
        release_settings,
        logical_settings,
        prefixer,
        account_bots,
        meta_ids,
        prs_by_repo,
    ) = await _check_github_repos(request, req.account, prs_by_repo, ".prs")
    prs, deployments = await fetch_pull_requests(
        prs_by_repo,
        account_bots,
        release_settings,
        logical_settings,
        req.environments,
        prefixer,
        req.account,
        meta_ids,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )

    # order PRs the same as they were requested by user
    indexed_prs = {(prefixer.prefix_logical_repo(pr.repository), pr.number): pr for pr in prs}
    ordered_prs = [
        pr
        for pr_group in req.prs
        for pr_number in pr_group.numbers
        if (pr := indexed_prs.get((pr_group.repository, pr_number))) is not None
    ]

    return await _build_github_prs_response(
        ordered_prs, deployments, prefixer, meta_ids, request.mdb, request.cache,
    )


async def _check_github_repos(
    request: AthenianWebRequest,
    account: int,
    prefixed_repos: Mapping[str, Any],
    pointer: str,
) -> tuple[
    ReleaseSettings,
    LogicalRepositorySettings,
    Prefixer,
    Set[str],
    tuple[int, ...],
    dict[str, Any],
]:
    meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    try:
        repos = {k.split("/", 1)[1]: v for k, v in prefixed_repos.items()}
    except IndexError:
        raise ResponseError(
            InvalidRequestError(detail="Invalid format of repositories.", pointer=pointer),
        ) from None

    async def check():
        checker = access_classes["github.com"](
            account, meta_ids, request.sdb, request.mdb, request.cache,
        )
        await checker.load()
        try:
            if denied := await checker.check(coerce_logical_repos(repos).keys()):
                raise ResponseError(
                    ForbiddenError(
                        detail="Account %d is access denied to repos %s" % (account, denied),
                    ),
                )
        except IndexError:
            raise ResponseError(
                BadRequestError(detail="Invalid repositories: %s" % prefixed_repos),
            ) from None
        return meta_ids

    settings = Settings.from_request(request, account, prefixer)
    meta_ids, release_settings, logical_settings, account_bots = await gather(
        check(),
        settings.list_release_matches(prefixed_repos),
        settings.list_logical_repositories(prefixed_repos, pointer=pointer),
        bots(account, meta_ids, request.mdb, request.sdb, request.cache),
        op="_check_github_repos",
    )
    return release_settings, logical_settings, prefixer, account_bots, meta_ids, repos


def webify_deployment(val: Deployment, prefix_logical_repo) -> WebDeploymentNotification:
    """Convert a Deployment to the web representation."""
    return WebDeploymentNotification(
        name=val.name,
        environment=val.environment,
        conclusion=val.conclusion.name,
        url=val.url,
        date_started=val.started_at,
        date_finished=val.finished_at,
        components=[
            WebDeployedComponent(
                repository=prefix_logical_repo(c.repository_full_name),
                reference=f"{c.reference} ({c.sha})"
                if not c.sha.startswith(c.reference)
                else c.sha,
            )
            for c in val.components
        ],
        labels=val.labels,
    )


@sentry_span
async def _build_github_prs_response(
    prs: Sequence[PullRequestListItem],
    deployments: dict[str, Deployment],
    prefixer: Prefixer,
    meta_ids: tuple[int, ...],
    mdb: Database,
    cache: Optional[aiomcache.Client],
) -> web.Response:
    log = logging.getLogger(f"{metadata.__package__}._build_github_prs_response")
    prefix_logical_repo = prefixer.prefix_logical_repo
    web_prs: list[WebPullRequest] = list(web_pr_from_struct(prs, prefixer, log))
    users = set(chain.from_iterable(chain.from_iterable(pr.participants.values()) for pr in prs))
    avatars = await mine_user_avatars(
        UserAvatarKeys.PREFIXED_LOGIN, meta_ids, mdb, cache, nodes=users,
    )
    model = PullRequestSet(
        include=PullRequestSetInclude(
            users={login: IncludedNativeUser(avatar=avatar) for login, avatar in avatars},
            deployments={
                key: webify_deployment(val, prefix_logical_repo)
                for key, val in sorted(deployments.items())
            }
            or None,
        ),
        data=web_prs,
    )
    return model_response(model, native=True)


@expires_header(middle_term_exptime)
@weight(0.5)
async def filter_labels(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find labels used in the given repositories."""
    try:
        filt = FilterLabelsRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    repos, meta_ids, _, _ = await _repos_preprocess(filt.repositories, filt.account, request)
    labels = await mine_labels(repos, meta_ids, request.mdb, request.cache)
    labels = [FilteredLabel(**dataclass_asdict(label)) for label in labels]
    return model_response(labels)


@expires_header(short_term_exptime)
@weight(1)
async def get_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """List releases by repository and name."""
    body = GetReleasesRequest.from_dict(body)
    releases_by_repo = {}
    for p in body.releases:
        releases_by_repo.setdefault(p.repository, set()).update(p.names)
    try:
        (
            release_settings,
            logical_settings,
            prefixer,
            _,
            meta_ids,
            releases_by_repo,
        ), jira_ids = await gather(
            _check_github_repos(request, body.account, releases_by_repo, ".releases"),
            get_jira_installation_or_none(body.account, request.sdb, request.mdb, request.cache),
        )
    except KeyError:
        return model_response(ReleaseSet())
    releases, avatars, deployments = await mine_releases_by_name(
        releases_by_repo,
        release_settings,
        logical_settings,
        prefixer,
        body.account,
        meta_ids,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )
    return await _build_release_set_response(
        releases, avatars, deployments, prefixer, jira_ids, request.mdb,
    )


@expires_header(short_term_exptime)
@weight(1)
async def diff_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find releases between the two given ones per repository."""
    body = DiffReleasesRequest.from_dict(body)
    borders = {}
    for repo, border in body.borders.items():
        borders[repo] = [(pair.old, pair.new) for pair in border]
    try:
        (
            release_settings,
            logical_settings,
            prefixer,
            _,
            meta_ids,
            borders,
        ), jira_ids = await gather(
            _check_github_repos(request, body.account, borders, ".borders"),
            get_jira_installation_or_none(body.account, request.sdb, request.mdb, request.cache),
        )
    except KeyError:
        return model_response(ReleaseSet())
    releases, avatars = await mine_diff_releases(
        borders,
        release_settings,
        logical_settings,
        prefixer,
        body.account,
        meta_ids,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
    )
    if all_diffs := [r[-1] for rr in releases.values() for r in rr]:
        all_diffs = pd.concat(all_diffs, ignore_index=True)
        issues = await _load_jira_issues_for_releases(all_diffs, jira_ids, request.mdb)
    else:
        issues = {}

    result = DiffedReleases(
        data={},
        include=ReleaseSetInclude(
            users={u: IncludedNativeUser(avatar=a) for u, a in avatars},
            jira=issues,
        ),
    )
    for repo, diffs in releases.items():
        result.data[prefixer.repo_name_to_prefixed_name[repo]] = repo_result = []
        for diff in diffs:
            repo_result.append(
                ReleaseDiff(
                    old=diff[0],
                    new=diff[1],
                    releases=_filtered_releases_from_df(diff[2], prefixer),
                ),
            )
    return model_response(result)


@expires_header(short_term_exptime)
@weight(1)
async def filter_code_checks(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find code check runs that match the specified query."""
    try:
        filt = FilterCodeChecksRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    (time_from, time_to, repos, meta_ids, prefixer, logical_settings), jira_ids = await gather(
        _common_filter_preprocess(filt, filt.in_, request, strip_prefix=True),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
    )
    teams = set()
    scan_for_teams(filt.triggered_by, teams, ".triggered_by")
    teams_map = await fetch_teams_map(teams, filt.account, request.sdb)
    triggered_by = compile_developers(
        filt.triggered_by, teams_map, None, False, prefixer, ".triggered_by",
    )
    timeline, check_runs = await filter_check_runs(
        time_from,
        time_to,
        repos,
        triggered_by,
        LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude),
        JIRAFilter.from_web(filt.jira, jira_ids),
        filt.quantiles or [0, 1],
        logical_settings,
        meta_ids,
        request.mdb,
        request.cache,
    )
    model = FilteredCodeCheckRuns(
        timeline=timeline,
        items=[
            FilteredCodeCheckRun(
                title=cr.title,
                repository=prefixer.prefix_logical_repo(cr.repository),
                last_execution_time=cr.last_execution_time,
                last_execution_url=cr.last_execution_url,
                size_groups=cr.size_groups,
                total_stats=CodeCheckRunStatistics(**dataclass_asdict(cr.total_stats)),
                prs_stats=CodeCheckRunStatistics(**dataclass_asdict(cr.prs_stats)),
            )
            for cr in check_runs
        ],
    )
    return model_response(model)


async def filter_deployments(request: AthenianWebRequest, body: dict) -> web.Response:
    """
    List the deployments that satisfy the provided filters.

    We submit new deployments using `/events/deployments`.
    """
    try:
        filt = FilterDeploymentsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    (time_from, time_to, repos, meta_ids, prefixer, logical_settings), jira_ids = await gather(
        _common_filter_preprocess(filt, filt.in_, request, strip_prefix=False),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
    )
    repos = [r.split("/", 1)[1] for r in repos]
    participants = await resolve_withgroups(
        [filt.with_],
        ReleaseParticipationKind,
        True,
        filt.account,
        None,
        ".with",
        prefixer,
        request.sdb,
    )
    participants = participants[0] if participants else {}
    settings = Settings.from_request(request, filt.account, prefixer)
    # all the repos, because we don't know what else is released in the matched deployments
    release_settings, (branches, default_branches) = await gather(
        settings.list_release_matches(),
        BranchMiner.extract_branches(None, prefixer, meta_ids, request.mdb, request.cache),
    )
    deployments = await mine_deployments(
        repositories=repos,
        participants=participants,
        time_from=time_from,
        time_to=time_to,
        environments=filt.environments or [],
        conclusions=[DeploymentConclusion[c] for c in (filt.conclusions or [])],
        with_labels=filt.with_labels or {},
        without_labels=filt.without_labels or {},
        pr_labels=LabelFilter.from_iterables(filt.pr_labels_include, filt.pr_labels_exclude),
        jira=JIRAFilter.from_web(filt.jira, jira_ids),
        release_settings=release_settings,
        logical_settings=logical_settings,
        branches=branches,
        default_branches=default_branches,
        prefixer=prefixer,
        account=filt.account,
        jira_ids=jira_ids,
        meta_ids=meta_ids,
        mdb=request.mdb,
        pdb=request.pdb,
        rdb=request.rdb,
        cache=request.cache,
        with_extended_prs=True,
        with_jira=True,
    )
    people = deployment_facts_extract_mentioned_people(deployments)
    avatars, issues = await gather(
        mine_user_avatars(
            UserAvatarKeys.PREFIXED_LOGIN, meta_ids, request.mdb, request.cache, nodes=people,
        ),
        load_jira_issues_for_deployments(deployments, jira_ids, request.mdb),
    )
    model = await _build_deployments_response(deployments, avatars, issues, prefixer)
    return model_response(model)


async def _build_deployments_response(
    df: pd.DataFrame,
    people: list[tuple[str, str]],
    issues: dict[str, PullRequestJIRAIssueItem],
    prefixer: Prefixer,
) -> [FilteredDeployment]:
    if df.empty:
        return []
    prefix_logical_repo = prefixer.prefix_logical_repo
    user_node_to_prefixed_login = prefixer.user_node_to_prefixed_login
    return FilteredDeployments(
        deployments=[
            FilteredDeployment(
                name=name,
                environment=environment,
                url=url,
                date_started=started_at,
                date_finished=finished_at,
                conclusion=conclusion,
                components=[
                    WebDeployedComponent(repository=prefix_logical_repo(repo_name), reference=ref)
                    for repo_name, ref in zip(
                        components_df[DeployedComponent.repository_full_name].values,
                        components_df[DeployedComponent.reference.name].values,
                    )
                ],
                labels={
                    key: val
                    for key, val in zip(
                        labels_df[DeployedLabel.key.name].values,
                        labels_df[DeployedLabel.value.name].values,
                    )
                }
                if not labels_df.empty
                else None,
                code=DeploymentAnalysisCode(
                    prs=dict(
                        zip(
                            resolved_repos := [prefix_logical_repo(r) for r in repos],
                            np.diff(prs_offsets, prepend=0, append=len(prs)),
                        ),
                    ),
                    lines_prs=dict(zip(resolved_repos, lines_prs)),
                    lines_overall=dict(zip(resolved_repos, lines_overall)),
                    commits_prs=dict(zip(resolved_repos, commits_prs)),
                    commits_overall=dict(zip(resolved_repos, commits_overall)),
                    jira={
                        r: keys
                        for r, keys in zip(
                            resolved_repos, np.split(jira_by_repo, jira_repo_offsets),
                        )
                        if keys is not None
                    }
                    if len(jira_by_repo)
                    else None,
                ),
                prs=[
                    DeployedPullRequest(
                        number=pr_number,
                        title=pr_title,
                        created=pr_created,
                        additions=pr_adds,
                        deletions=pr_dels,
                        author=user_node_to_prefixed_login[pr_author] if pr_author else None,
                        jira=pr_jira if len(pr_jira) else None,
                        repository=pr_repo,
                    )
                    for (
                        pr_number,
                        pr_title,
                        pr_created,
                        pr_adds,
                        pr_dels,
                        pr_author,
                        pr_jira,
                        pr_repo,
                    ) in zip(
                        pr_numbers,
                        pr_titles,
                        pr_createds,
                        pr_additions,
                        pr_deletions,
                        pr_user_node_ids,
                        np.split(pr_jiras, pr_jira_offsets) if len(pr_jiras) else repeat([]),
                        np.repeat(
                            resolved_repos,
                            np.diff(prs_offsets, prepend=0, append=len(prs)),
                        ),
                    )
                ],
                releases=[
                    DeployedRelease(
                        name=rel_name,
                        sha=rel_sha,
                        repository=rel_repo,
                        url=rel_url,
                        publisher=rel_author,
                        published=rel_date,
                        age=rel_age,
                        added_lines=rel_additions,
                        deleted_lines=rel_deletions,
                        commits=rel_commits_count,
                        commit_authors=[
                            user_node_to_prefixed_login[u] for u in rel_commit_authors
                        ],
                        prs=len(rel_pr_numbers),
                    )
                    for (
                        rel_name,
                        rel_sha,
                        rel_repo,
                        rel_url,
                        rel_author,
                        rel_date,
                        rel_age,
                        rel_additions,
                        rel_deletions,
                        rel_commits_count,
                        rel_commit_authors,
                        rel_pr_numbers,
                    ) in zip(
                        releases_df[Release.name.name].values,
                        releases_df[Release.sha.name].values,
                        releases_df.index.get_level_values(1).values,
                        releases_df[Release.url.name].values,
                        releases_df[Release.author.name].values,
                        releases_df[Release.published_at.name].values,
                        releases_df[ReleaseFacts.f.age].values,
                        releases_df[ReleaseFacts.f.additions].values,
                        releases_df[ReleaseFacts.f.deletions].values,
                        releases_df[ReleaseFacts.f.commits_count].values,
                        releases_df[ReleaseFacts.f.commit_authors].values,
                        releases_df["prs_" + PullRequest.number.name].values,
                    )
                ]
                if not releases_df.empty
                else None,
            )
            for (
                name,
                environment,
                components_df,
                url,
                started_at,
                finished_at,
                conclusion,
                labels_df,
                releases_df,
                repos,
                prs,
                prs_offsets,
                lines_prs,
                lines_overall,
                commits_prs,
                commits_overall,
                pr_numbers,
                pr_titles,
                pr_createds,
                pr_additions,
                pr_deletions,
                pr_user_node_ids,
                pr_jiras,
                pr_jira_offsets,
                jira_by_repo,
                jira_repo_offsets,
            ) in zip(
                df.index.values,
                df[DeploymentNotification.environment.name].values,
                df["components"].values,
                df[DeploymentNotification.url.name].values,
                df[DeploymentNotification.started_at.name].values,
                df[DeploymentNotification.finished_at.name].values,
                df[DeploymentNotification.conclusion.name].values,
                df["labels"].values,
                df["releases"].values,
                df[DeploymentFacts.f.repositories].values,
                df[DeploymentFacts.f.prs].values,
                df[DeploymentFacts.f.prs_offsets].values,
                df[DeploymentFacts.f.lines_prs].values,
                df[DeploymentFacts.f.lines_overall].values,
                df[DeploymentFacts.f.commits_prs].values,
                df[DeploymentFacts.f.commits_overall].values,
                df[DeploymentFacts.f.prs_number].values,
                df[DeploymentFacts.f.prs_title].values,
                df[DeploymentFacts.f.prs_created_at].values,
                df[DeploymentFacts.f.prs_additions].values,
                df[DeploymentFacts.f.prs_deletions].values,
                df[DeploymentFacts.f.prs_user_node_id].values,
                df[DeploymentFacts.f.prs_jira_ids].values,
                df[DeploymentFacts.f.prs_jira_offsets].values,
                df[DeploymentFacts.f.jira_ids].values,
                df[DeploymentFacts.f.jira_offsets].values,
            )
        ],
        include=ReleaseSetInclude(
            users={u: IncludedNativeUser(avatar=a) for u, a in people},
            jira={k: LinkedJIRAIssue(**dataclass_asdict(v)) for k, v in issues.items()},
        ),
    )


async def filter_environments(request: AthenianWebRequest, body: dict) -> web.Response:
    """List the deployment environments."""
    try:
        filt = FilterEnvironmentsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    time_from, time_to, repos, _, prefixer, logical_settings = await _common_filter_preprocess(
        filt, filt.repositories, request,
    )
    try:
        envs = await mine_environments(
            repos if filt.repositories else None,
            time_from,
            time_to,
            prefixer,
            logical_settings,
            filt.account,
            request.rdb,
            request.cache,
        )
    except NoDeploymentNotificationsError as e:
        raise ResponseError(
            NoSourceDataError(
                detail="Submit at least one deployment notification with `/events/deployments`.",
            ),
        ) from e
    envs = [FilteredEnvironment(**dataclass_asdict(env)) for env in envs]
    return model_response(envs)
