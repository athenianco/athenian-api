from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import operator
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from aiohttp import web
import aiomcache
from dateutil.parser import parse as parse_datetime
import numpy as np
import pandas as pd
from sqlalchemy import and_, select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.features.github.check_run_filter import filter_check_runs
from athenian.api.controllers.features.github.pull_request_filter import fetch_pull_requests, \
    filter_pull_requests
from athenian.api.controllers.jira import get_jira_installation, get_jira_installation_or_none, \
    load_mapped_jira_users
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.controllers.miners.github.deployment import load_jira_issues_for_deployments, \
    mine_deployments
from athenian.api.controllers.miners.github.label import mine_labels
from athenian.api.controllers.miners.github.release_mine import \
    diff_releases as mine_diff_releases, mine_releases, mine_releases_by_name
from athenian.api.controllers.miners.github.repository import mine_repositories
from athenian.api.controllers.miners.github.user import mine_user_avatars, UserAvatarKeys
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues_for_prs
from athenian.api.controllers.miners.types import Deployment, DeploymentConclusion, \
    DeploymentFacts, PRParticipants, PRParticipationKind, PullRequestEvent, PullRequestListItem, \
    PullRequestStage, ReleaseFacts
from athenian.api.controllers.prefixer import Prefixer, PrefixerPromise
from athenian.api.controllers.release import extract_release_participants
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.controllers.settings import ReleaseSettings, Settings
from athenian.api.db import ParallelDatabase
from athenian.api.models.metadata.github import NodeRepository, PullRequest, PushCommit, Release, \
    User
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel, \
    DeploymentNotification
from athenian.api.models.web import BadRequestError, Commit, CommitSignature, CommitsList, \
    DeployedComponent as WebDeployedComponent, DeploymentAnalysisCode, \
    DeploymentNotification as WebDeploymentNotification, DeveloperSummary, \
    DeveloperUpdates, FilterCommitsRequest, FilterContributorsRequest, FilterDeploymentsRequest, \
    FilteredDeployment, FilteredLabel, FilteredRelease, FilterLabelsRequest, \
    FilterPullRequestsRequest, FilterReleasesRequest, FilterRepositoriesRequest, ForbiddenError, \
    GetPullRequestsRequest, GetReleasesRequest, IncludedNativeUser, IncludedNativeUsers, \
    InvalidRequestError, LinkedJIRAIssue, PullRequest as WebPullRequest, PullRequestLabel, \
    PullRequestParticipant, PullRequestSet, PullRequestSetInclude, ReleasedPullRequest, \
    ReleaseSet, ReleaseSetInclude, StageTimings
from athenian.api.models.web.code_check_run_statistics import CodeCheckRunStatistics
from athenian.api.models.web.diff_releases_request import DiffReleasesRequest
from athenian.api.models.web.diffed_releases import DiffedReleases
from athenian.api.models.web.filter_code_checks_request import FilterCodeChecksRequest
from athenian.api.models.web.filtered_code_check_run import FilteredCodeCheckRun
from athenian.api.models.web.filtered_code_check_runs import FilteredCodeCheckRuns
from athenian.api.models.web.filtered_deployments import FilteredDeployments
from athenian.api.models.web.release_diff import ReleaseDiff
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span


@weight(2.5)
async def filter_contributors(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find developers that made an action within the given timeframe."""
    try:
        filt = FilterContributorsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    time_from, time_to, repos, meta_ids, prefixer = await _common_filter_preprocess(
        filt, request, strip_prefix=False)
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    repos = [r.split("/", 1)[1] for r in repos]
    users = await mine_contributors(
        repos, time_from, time_to, True, filt.as_ or [], release_settings, prefixer,
        filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
    mapped_jira = await load_mapped_jira_users(
        filt.account, [u[User.node_id.name] for u in users],
        request.sdb, request.mdb, request.cache)
    prefixer = await prefixer.load()
    model = [
        DeveloperSummary(
            login=prefixer.user_node_to_prefixed_login[u[User.node_id.name]],
            avatar=u[User.avatar_url.name],
            name=u[User.name.name],
            updates=DeveloperUpdates(**{
                k: v for k, v in u["stats"].items()
                # TODO(se7entyse7en): make `DeveloperUpdates` support all the stats we can get instead of doing this filtering. See also `mine_contributors`.  # noqa
                if k in DeveloperUpdates.openapi_types
            }),
            jira_user=mapped_jira.get(u[User.node_id.name]),
        )
        for u in sorted(users, key=operator.itemgetter("login"))
    ]
    return model_response(model)


@weight(0.5)
async def filter_repositories(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find repositories that were updated within the given timeframe."""
    try:
        filt = FilterRepositoriesRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    time_from, time_to, repos, meta_ids, prefixer = await _common_filter_preprocess(
        filt, request, strip_prefix=False)
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    repos = [r.split("/", 1)[1] for r in repos]
    repos = await mine_repositories(
        repos, time_from, time_to, filt.exclude_inactive, release_settings, prefixer,
        filt.account, meta_ids, request.mdb, request.pdb, request.cache)
    prefixer = await prefixer.load()
    repos = prefixer.prefix_repo_names(repos)
    return web.json_response(repos)


async def _common_filter_preprocess(filt: Union[FilterReleasesRequest,
                                                FilterRepositoriesRequest,
                                                FilterPullRequestsRequest,
                                                FilterCommitsRequest,
                                                FilterCodeChecksRequest,
                                                FilterDeploymentsRequest],
                                    request: AthenianWebRequest,
                                    strip_prefix=True,
                                    ) -> Tuple[datetime,
                                               datetime,
                                               Set[str],
                                               Tuple[int, ...],
                                               PrefixerPromise]:
    if filt.date_to < filt.date_from:
        raise ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        ))
    min_time = datetime.min.time()
    time_from = datetime.combine(filt.date_from, min_time, tzinfo=timezone.utc)
    time_to = datetime.combine(filt.date_to + timedelta(days=1), min_time, tzinfo=timezone.utc)
    if filt.timezone is not None:
        tzoffset = timedelta(minutes=-filt.timezone)
        time_from += tzoffset
        time_to += tzoffset

    async def login_loader() -> str:
        return (await request.user()).login

    repos, meta_ids = await resolve_repos(
        filt.in_, filt.account, request.uid, login_loader,
        request.sdb, request.mdb, request.cache, request.app["slack"],
        strip_prefix=strip_prefix)
    prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    return time_from, time_to, repos, meta_ids, prefixer


async def resolve_filter_prs_parameters(filt: FilterPullRequestsRequest,
                                        request: AthenianWebRequest,
                                        ) -> Tuple[datetime, datetime,
                                                   Set[str], Set[str], Set[str],
                                                   PRParticipants, LabelFilter, JIRAFilter,
                                                   ReleaseSettings,
                                                   PrefixerPromise,
                                                   Tuple[int, ...]]:
    """Infer all the required PR filters from the request."""
    time_from, time_to, repos, meta_ids, prefixer = await _common_filter_preprocess(
        filt, request, strip_prefix=False)
    events = set(getattr(PullRequestEvent, e.upper()) for e in (filt.events or []))
    stages = set(getattr(PullRequestStage, s.upper()) for s in (filt.stages or []))
    if not events and not stages:
        raise ResponseError(InvalidRequestError(
            detail="Either `events` or `stages` must be specified and be not empty.",
            pointer=".stages"))
    participants = {PRParticipationKind[k.upper()]: {d.rsplit("/", 1)[1] for d in v}
                    for k, v in (filt.with_ or {}).items() if v}
    settings = await Settings.from_request(request, filt.account).list_release_matches(repos)
    repos = {r.split("/", 1)[1] for r in repos}
    labels = LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude)
    try:
        jira = JIRAFilter.from_web(
            filt.jira,
            await get_jira_installation(filt.account, request.sdb, request.mdb, request.cache))
    except ResponseError:
        jira = JIRAFilter.empty()
    return time_from, time_to, repos, events, stages, participants, labels, jira, settings, \
        prefixer, meta_ids


@weight(6)
async def filter_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """List pull requests that satisfy the query."""
    try:
        filt = FilterPullRequestsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    time_from, time_to, repos, events, stages, participants, labels, jira, settings, prefixer, \
        meta_ids = await resolve_filter_prs_parameters(filt, request)
    updated_min, updated_max = _bake_updated_min_max(filt)
    prs, deployments = await filter_pull_requests(
        events, stages, time_from, time_to, repos, participants, labels, jira,
        filt.environment, filt.exclude_inactive, settings, updated_min, updated_max,
        prefixer, filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
    prefixer = await prefixer.load()  # no-op
    return await _build_github_prs_response(
        prs, deployments, prefixer, meta_ids, request.mdb, request.cache)


def _bake_updated_min_max(filt: FilterPullRequestsRequest) -> Tuple[datetime, datetime]:
    if (filt.updated_from is None) != (filt.updated_to is None):
        raise ResponseError(InvalidRequestError(
            ".updated_from",
            "`updated_from` and `updated_to` must be both either specified or not"))
    if filt.updated_from is not None:
        updated_min = datetime.combine(filt.updated_from, datetime.min.time(), tzinfo=timezone.utc)
        updated_max = datetime.combine(filt.updated_to, datetime.min.time(), tzinfo=timezone.utc) \
            + timedelta(days=1)
    else:
        updated_min = updated_max = None
    return updated_min, updated_max


def web_pr_from_struct(pr: PullRequestListItem,
                       prefixer: Prefixer,
                       log: logging.Logger,
                       ) -> WebPullRequest:
    """Convert an intermediate PR representation to the web model."""
    props = dict(pr)
    del props["node_id"]
    del props["deployments"]
    props["repository"] = prefixer.repo_name_to_prefixed_name[props["repository"]]
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
                participants[prefixer.user_login_to_prefixed_login[pid]].append(pkweb)
            except KeyError:
                log.error("Failed to resolve user %s", pid)
    props["participants"] = sorted(PullRequestParticipant(*p) for p in participants.items())
    if pr.labels is not None:
        props["labels"] = [PullRequestLabel(**label) for label in pr.labels]
    if pr.jira is not None:
        props["jira"] = jira = [LinkedJIRAIssue(**issue) for issue in pr.jira]
        for issue in jira:
            if issue.labels is not None:
                # it is a set, must be a list
                issue.labels = sorted(issue.labels)
    return WebPullRequest(**props)


def _nan_to_none(val):
    if val != val:
        return None
    return val


@weight(1.5)
async def filter_commits(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find commits that match the specified query."""
    try:
        filt = FilterCommitsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    time_from, time_to, repos, meta_ids, prefixer = await _common_filter_preprocess(filt, request)
    with_author = [s.rsplit("/", 1)[1] for s in (filt.with_author or [])]
    with_committer = [s.rsplit("/", 1)[1] for s in (filt.with_committer or [])]
    log = logging.getLogger("filter_commits")
    commits = await extract_commits(
        FilterCommitsProperty(filt.property), time_from, time_to, repos,
        with_author, with_committer, filt.only_default_branch,
        BranchMiner(), filt.account, meta_ids, request.mdb, request.pdb, request.cache)
    model = CommitsList(data=[], include=IncludedNativeUsers(users={}))
    users = model.include.users
    utc = timezone.utc
    prefixer = await prefixer.load()
    repo_name_map, user_login_map = \
        prefixer.repo_name_to_prefixed_name, prefixer.user_login_to_prefixed_login
    for author_login, committer_login, repository_full_name, sha, message, \
            additions, deletions, changed_files, author_name, author_email, authored_date, \
            committer_name, committer_email, committed_date, author_date, commit_date, \
            author_avatar_url, committer_avatar_url in zip(
            commits[PushCommit.author_login.name].values,
            commits[PushCommit.committer_login.name].values,
            commits[PushCommit.repository_full_name.name].values,
            commits[PushCommit.sha.name].values,
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
            commits[PushCommit.committer_avatar_url.name]):
        obj = Commit(
            repository=repo_name_map[repository_full_name],
            hash=sha,
            message=message,
            size_added=_nan_to_none(additions),
            size_removed=_nan_to_none(deletions),
            files_changed=_nan_to_none(changed_files),
            author=CommitSignature(
                login=(user_login_map[author_login]) if author_login else None,
                name=author_name,
                email=author_email,
                timestamp=authored_date.replace(tzinfo=utc),
            ),
            committer=CommitSignature(
                login=(user_login_map[committer_login]) if committer_login else None,
                name=committer_name,
                email=committer_email,
                timestamp=committed_date.replace(tzinfo=utc),
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
            users[obj.author.login] = IncludedNativeUser(
                avatar=author_avatar_url)
        if obj.committer.login and obj.committer.login not in users:
            users[obj.committer.login] = IncludedNativeUser(committer_avatar_url)
        model.data.append(obj)
    return model_response(model)


@weight(1)
async def filter_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find releases that were published in the given time fram in the given repositories."""
    try:
        filt = FilterReleasesRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    time_from, time_to, repos, meta_ids, prefixer = await _common_filter_preprocess(
        filt, request, strip_prefix=False)
    stripped_repos = [r.split("/", 1)[1] for r in repos]
    release_settings, jira_ids, (branches, default_branches), participants = await gather(
        Settings.from_request(request, filt.account).list_release_matches(repos),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
        BranchMiner.extract_branches(stripped_repos, meta_ids, request.mdb, request.cache),
        extract_release_participants(filt.with_, meta_ids, request.mdb),
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
        settings=release_settings,
        prefixer=prefixer,
        account=filt.account,
        meta_ids=meta_ids,
        mdb=request.mdb,
        pdb=request.pdb,
        rdb=request.rdb,
        cache=request.cache,
        with_pr_titles=True)
    prefixer = await prefixer.load()
    avatars = [(prefixer.user_node_to_prefixed_login[u], url) for u, url in avatars]
    return await _build_release_set_response(
        releases, avatars, deployments, prefixer, jira_ids, meta_ids, request.mdb)


async def _load_jira_issues_for_releases(jira_ids: Optional[Tuple[int, List[str]]],
                                         releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                                         meta_ids: Tuple[int, ...],
                                         mdb: ParallelDatabase) -> Dict[str, LinkedJIRAIssue]:
    if jira_ids is None:
        for (_, facts) in releases:
            facts.prs_jira = np.full(len(facts["prs_" + PullRequest.node_id.name]), None)
        return {}

    pr_to_ix = {}
    for ri, (_, facts) in enumerate(releases):
        node_ids = facts["prs_" + PullRequest.node_id.name]
        facts.prs_jira = [[] for _ in range(len(node_ids))]
        for pri, node_id in enumerate(node_ids):
            pr_to_ix[node_id] = ri, pri
    rows = await fetch_jira_issues_for_prs(pr_to_ix, meta_ids, jira_ids, mdb)
    issues = {}
    for r in rows:
        key = r["key"]
        issues[key] = LinkedJIRAIssue(
            id=key, title=r["title"], epic=r["epic"], labels=r["labels"], type=r["type"])
        ri, pri = pr_to_ix[r["node_id"]]
        releases[ri][1].prs_jira[pri].append(key)
    return issues


async def _build_release_set_response(releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                                      avatars: List[Tuple[str, str]],
                                      deployments: Dict[str, Deployment],
                                      prefixer: Prefixer,
                                      jira_ids: Optional[Tuple[int, List[str]]],
                                      meta_ids: Tuple[int, ...],
                                      mdb: ParallelDatabase,
                                      ) -> web.Response:
    issues = await _load_jira_issues_for_releases(jira_ids, releases, meta_ids, mdb)
    repo_node_to_prefixed_name = prefixer.repo_node_to_prefixed_name.get
    data = [_filtered_release_from_tuple(t, prefixer) for t in releases]
    model = ReleaseSet(data=data, include=ReleaseSetInclude(
        users={u: IncludedNativeUser(avatar=a) for u, a in avatars},
        jira=issues,
        deployments={
            key: webify_deployment(val, repo_node_to_prefixed_name)
            for key, val in sorted(deployments.items())
        } or None,
    ))
    return model_response(model)


def _filtered_release_from_tuple(t: Tuple[Dict[str, Any], ReleaseFacts],
                                 prefixer: Prefixer,
                                 ) -> FilteredRelease:
    details, facts = t
    user_node_to_prefixed_login = prefixer.user_node_to_prefixed_login
    return FilteredRelease(
        name=details[Release.name.name],
        repository=details[Release.repository_full_name.name],
        url=details[Release.url.name],
        publisher=user_node_to_prefixed_login.get(facts.publisher),
        published=facts.published.item().replace(tzinfo=timezone.utc),
        age=facts.age,
        added_lines=facts.additions,
        deleted_lines=facts.deletions,
        commits=facts.commits_count,
        commit_authors=sorted(user_node_to_prefixed_login.get(u) for u in facts.commit_authors),
        prs=_extract_release_prs(facts, prefixer),
        deployments=facts.deployments,
    )


def _extract_release_prs(facts: ReleaseFacts, prefixer: Prefixer) -> List[ReleasedPullRequest]:
    user_node_to_prefixed_login = prefixer.user_node_to_prefixed_login
    return [
        ReleasedPullRequest(
            number=number,
            title=title,
            additions=adds,
            deletions=dels,
            author=user_node_to_prefixed_login.get(author),
            jira=jira or None,
        )
        for number, title, adds, dels, author, jira in zip(
            facts["prs_" + PullRequest.number.name],
            facts["prs_" + PullRequest.title.name],
            facts["prs_" + PullRequest.additions.name],
            facts["prs_" + PullRequest.deletions.name],
            facts["prs_" + PullRequest.user_node_id.name],
            facts.prs_jira,
        )
    ]


@weight(0.5)
async def get_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """List pull requests by repository and number."""
    body = GetPullRequestsRequest.from_dict(body)
    prs_by_repo = {}
    for p in body.prs:
        prs_by_repo.setdefault(p.repository, set()).update(p.numbers)
    settings, meta_ids, prs_by_repo = await _check_github_repos(
        request, body.account, prs_by_repo, ".prs")
    prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    prs, deployments = await fetch_pull_requests(
        prs_by_repo, settings, body.environment, prefixer, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    prefixer = await prefixer.load()  # no-op
    return await _build_github_prs_response(
        prs, deployments, prefixer, meta_ids, request.mdb, request.cache)


async def _check_github_repos(request: AthenianWebRequest,
                              account: int,
                              prefixed_repos: Mapping[str, Any],
                              pointer: str,
                              ) -> Tuple[ReleaseSettings,
                                         Tuple[int, ...],
                                         Dict[str, Any]]:
    try:
        repos = {k.split("/", 1)[1]: v for k, v in prefixed_repos.items()}
    except IndexError:
        raise ResponseError(InvalidRequestError(
            detail="Invalid repositories.", pointer=pointer,
        )) from None

    async def check():
        async with request.sdb.connection() as sdb_conn:
            meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
            checker = access_classes["github"](
                account, meta_ids, sdb_conn, request.mdb, request.cache)
            await checker.load()
            try:
                if denied := await checker.check(repos.keys()):
                    raise ResponseError(ForbiddenError(
                        detail="Account %d is access denied to repos %s" % (account, denied)))
            except IndexError:
                raise ResponseError(BadRequestError(
                    detail="Invalid repositories: %s" % prefixed_repos)) from None
        return meta_ids

    async def load_settings():
        return await Settings.from_request(request, account).list_release_matches(prefixed_repos)

    meta_ids, settings = await gather(check(), load_settings(), op="_check_github_repos")
    # the order is reversed for performance reasons
    return settings, meta_ids, repos


def webify_deployment(val: Deployment, repo_node_to_prefixed_name) -> WebDeploymentNotification:
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
                repository=repo_node_to_prefixed_name(c.repository_id),
                reference=f"{c.reference} ({c.sha})"
                if not c.sha.startswith(c.reference) else c.sha,
            )
            for c in val.components
        ],
        labels=val.labels,
    )


@sentry_span
async def _build_github_prs_response(prs: List[PullRequestListItem],
                                     deployments: Dict[str, Deployment],
                                     prefixer: Prefixer,
                                     meta_ids: Tuple[int, ...],
                                     mdb: ParallelDatabase,
                                     cache: Optional[aiomcache.Client],
                                     ) -> web.Response:
    log = logging.getLogger(f"{metadata.__package__}._build_github_prs_response")
    repo_node_to_prefixed_name = prefixer.repo_node_to_prefixed_name.get
    web_prs = sorted(web_pr_from_struct(pr, prefixer, log) for pr in prs)
    users = set(chain.from_iterable(chain.from_iterable(pr.participants.values()) for pr in prs))
    avatars = await mine_user_avatars(users, UserAvatarKeys.PREFIXED_LOGIN, meta_ids, mdb, cache)
    model = PullRequestSet(include=PullRequestSetInclude(
        users={
            login: IncludedNativeUser(avatar=avatar) for login, avatar in avatars
        },
        deployments={
            key: webify_deployment(val, repo_node_to_prefixed_name)
            for key, val in sorted(deployments.items())
        } or None,
    ), data=web_prs)
    return model_response(model)


@weight(0.5)
async def filter_labels(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find labels used in the given repositories."""
    body = FilterLabelsRequest.from_dict(body)

    async def login_loader() -> str:
        return (await request.user()).login

    repos, meta_ids = await resolve_repos(
        body.repositories, body.account, request.uid, login_loader,
        request.sdb, request.mdb, request.cache, request.app["slack"])
    labels = await mine_labels(repos, meta_ids, request.mdb, request.cache)
    labels = [FilteredLabel(**label) for label in labels]
    return model_response(labels)


@weight(1)
async def get_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """List releases by repository and name."""
    body = GetReleasesRequest.from_dict(body)
    releases_by_repo = {}
    for p in body.releases:
        releases_by_repo.setdefault(p.repository, set()).update(p.names)
    try:
        (settings, meta_ids, releases_by_repo), jira_ids = await gather(
            _check_github_repos(request, body.account, releases_by_repo, ".releases"),
            get_jira_installation_or_none(body.account, request.sdb, request.mdb, request.cache),
        )
    except KeyError:
        return model_response(ReleaseSet())
    prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    releases, avatars, deployments = await mine_releases_by_name(
        releases_by_repo, settings, prefixer, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    return await _build_release_set_response(
        releases, avatars, deployments, await prefixer.load(), jira_ids, meta_ids, request.mdb)


@weight(1)
async def diff_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find releases between the two given ones per repository."""
    body = DiffReleasesRequest.from_dict(body)
    borders = {}
    for repo, border in body.borders.items():
        borders[repo] = [(pair.old, pair.new) for pair in border]
    try:
        (settings, meta_ids, borders), jira_ids = await gather(
            _check_github_repos(request, body.account, borders, ".borders"),
            get_jira_installation_or_none(body.account, request.sdb, request.mdb, request.cache),
        )
    except KeyError:
        return model_response(ReleaseSet())
    prefixer = await Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    releases, avatars = await mine_diff_releases(
        borders, settings, prefixer, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    issues = await _load_jira_issues_for_releases(
        jira_ids, list(chain.from_iterable(chain.from_iterable(r[-1] for r in rr)
                                           for rr in releases.values())),
        meta_ids, request.mdb)
    result = DiffedReleases(data={}, include=ReleaseSetInclude(
        users={u: IncludedNativeUser(avatar=a) for u, a in avatars},
        jira=issues,
    ))
    prefixer = await prefixer.load()
    for repo, diffs in releases.items():
        result.data[prefixer.repo_name_to_prefixed_name[repo]] = repo_result = []
        for diff in diffs:
            repo_result.append(ReleaseDiff(
                old=diff[0], new=diff[1],
                releases=[_filtered_release_from_tuple(t, prefixer) for t in diff[2]]))
    return model_response(result)


@weight(1)
async def filter_code_checks(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find code check runs that match the specified query."""
    try:
        filt = FilterCodeChecksRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    (time_from, time_to, repos, meta_ids, prefixer), jira_ids = await gather(
        _common_filter_preprocess(filt, request, strip_prefix=True),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
    )
    timeline, check_runs = await filter_check_runs(
        time_from, time_to, repos, {d.rsplit("/", 1)[1] for d in (filt.triggered_by or [])},
        LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude),
        JIRAFilter.from_web(filt.jira, jira_ids), filt.quantiles or [0, 1],
        meta_ids, request.mdb, request.cache)
    prefixer = await prefixer.load()
    model = FilteredCodeCheckRuns(timeline=timeline, items=[
        FilteredCodeCheckRun(
            title=cr.title,
            repository=prefixer.repo_name_to_prefixed_name[cr.repository],
            last_execution_time=cr.last_execution_time,
            last_execution_url=cr.last_execution_url,
            size_groups=cr.size_groups,
            total_stats=CodeCheckRunStatistics(**cr.total_stats),
            prs_stats=CodeCheckRunStatistics(**cr.prs_stats),
        ) for cr in check_runs
    ])
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
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    (time_from, time_to, repos, meta_ids, prefixer), jira_ids = await gather(
        _common_filter_preprocess(filt, request, strip_prefix=False),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
    )
    stripped_repos = [r.split("/", 1)[1] for r in repos]
    repo_node_rows, release_settings, (branches, default_branches), participants = await gather(
        request.mdb.fetch_all(
            select([NodeRepository.node_id])
            .where(and_(NodeRepository.acc_id.in_(meta_ids),
                        NodeRepository.name_with_owner.in_(stripped_repos)))),
        Settings.from_request(request, filt.account).list_release_matches(repos),
        BranchMiner.extract_branches(stripped_repos, meta_ids, request.mdb, request.cache),
        extract_release_participants(filt.with_, meta_ids, request.mdb),
    )
    deployments, people = await mine_deployments(
        repo_node_ids=[r[0] for r in repo_node_rows],
        participants=participants,
        time_from=time_from,
        time_to=time_to,
        environments=filt.environments or [],
        conclusions=[DeploymentConclusion[c] for c in (filt.conclusions or [])],
        with_labels=filt.with_labels or {},
        without_labels=filt.without_labels or {},
        pr_labels=LabelFilter.from_iterables(filt.pr_labels_include, filt.pr_labels_exclude),
        jira=JIRAFilter.from_web(filt.jira, jira_ids),
        settings=release_settings,
        branches=branches,
        default_branches=default_branches,
        prefixer=prefixer,
        account=filt.account,
        meta_ids=meta_ids,
        mdb=request.mdb,
        pdb=request.pdb,
        rdb=request.rdb,
        cache=request.cache,
    )
    prefixer = await prefixer.load()
    user_node_to_login = prefixer.user_node_to_login.get
    avatars, issues = await gather(
        mine_user_avatars([user_node_to_login(u) for u in people], UserAvatarKeys.PREFIXED_LOGIN,
                          meta_ids, request.mdb, request.cache),
        load_jira_issues_for_deployments(deployments, jira_ids, meta_ids, request.mdb),
    )
    model = await _build_deployments_response(deployments, avatars, issues, prefixer)
    return model_response(model)


async def _build_deployments_response(df: pd.DataFrame,
                                      people: List[Tuple[str, str]],
                                      issues: Dict[str, LinkedJIRAIssue],
                                      prefixer: Prefixer,
                                      ) -> [FilteredDeployment]:
    if df.empty:
        return []
    repo_node_to_prefixed_name = prefixer.repo_node_to_prefixed_name
    user_node_to_prefixed_login = prefixer.user_node_to_prefixed_login
    return FilteredDeployments(deployments=[
        FilteredDeployment(
            name=name,
            environment=environment,
            url=url,
            date_started=started_at,
            date_finished=finished_at,
            conclusion=conclusion,
            components=[
                WebDeployedComponent(repository=repo_node_to_prefixed_name[repo_node_id],
                                     reference=ref)
                for repo_node_id, ref in zip(
                    components_df[DeployedComponent.repository_node_id.name].values,
                    components_df[DeployedComponent.reference.name].values)
            ],
            labels={
                key: val for key, val in zip(
                    labels_df[DeployedLabel.key.name].values,
                    labels_df[DeployedLabel.value.name].values)
            } if not labels_df.empty else None,
            code=DeploymentAnalysisCode(
                prs=dict(zip(resolved_repos := [
                    repo_node_to_prefixed_name.get(r, f"unidentified_{i}")
                    for i, r in enumerate(repos)
                ], np.diff(prs_offsets, prepend=0, append=len(prs)))),
                lines_prs=dict(zip(resolved_repos, lines_prs)),
                lines_overall=dict(zip(resolved_repos, lines_overall)),
                commits_prs=dict(zip(resolved_repos, commits_prs)),
                commits_overall=dict(zip(resolved_repos, commits_overall)),
                jira={r: keys.astype("U") for r, keys in zip(resolved_repos, jira)
                      if keys is not None},
            ),
            releases=[
                FilteredRelease(
                    name=rel_name,
                    repository=rel_repo,
                    url=rel_url,
                    publisher=rel_author,
                    published=rel_date,
                    age=rel_age,
                    added_lines=rel_additions,
                    deleted_lines=rel_deletions,
                    commits=rel_commits_count,
                    commit_authors=[
                        user_node_to_prefixed_login[u]
                        for u in rel_commit_authors
                    ],
                    prs=[
                        ReleasedPullRequest(
                            number=pr_number,
                            title=pr_title,
                            additions=pr_adds,
                            deletions=pr_dels,
                            author=user_node_to_prefixed_login[pr_author] if pr_author else None,
                            jira=pr_jira.astype("U")
                            if pr_jira is not None and len(pr_jira)
                            else None,
                        )
                        for pr_number, pr_title, pr_adds, pr_dels, pr_author, pr_jira in zip(
                            pr_numbers, pr_titles, pr_additions, pr_deletions,
                            pr_user_node_ids, pr_jiras,
                        )
                    ],
                )
                for rel_name, rel_repo, rel_url, rel_author, rel_date, rel_age, rel_additions,
                rel_deletions, rel_commits_count, rel_commit_authors,
                pr_numbers, pr_titles, pr_additions, pr_deletions, pr_user_node_ids,
                pr_jiras
                in zip(releases_df[Release.name.name].values,
                       releases_df[Release.repository_full_name.name].values,
                       releases_df[Release.url.name].values,
                       releases_df[Release.author.name].values,
                       releases_df[Release.published_at.name].values,
                       releases_df[ReleaseFacts.f.age].values,
                       releases_df[ReleaseFacts.f.additions].values,
                       releases_df[ReleaseFacts.f.deletions].values,
                       releases_df[ReleaseFacts.f.commits_count].values,
                       releases_df[ReleaseFacts.f.commit_authors].values,
                       releases_df["prs_" + PullRequest.number.name].values,
                       releases_df["prs_" + PullRequest.title.name].values,
                       releases_df["prs_" + PullRequest.additions.name].values,
                       releases_df["prs_" + PullRequest.deletions.name].values,
                       releases_df["prs_" + PullRequest.user_node_id.name].values,
                       releases_df["prs_jira"].values)
            ] if not releases_df.empty else None,
        )
        for name, environment, components_df, url, started_at, finished_at, conclusion,
        labels_df, releases_df, repos, prs, prs_offsets, lines_prs, lines_overall,
        commits_prs, commits_overall, jira
        in zip(
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
            df["jira"].values,
        )
    ], include=ReleaseSetInclude(
        users={u: IncludedNativeUser(avatar=a) for u, a in people},
        jira=issues,
    ))
