from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import operator
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from aiohttp import web
import aiomcache
import databases
from dateutil.parser import parse as parse_datetime
import numpy as np
from sqlalchemy import and_, join, outerjoin, select
from sqlalchemy.orm import aliased

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
from athenian.api.controllers.miners.github.label import mine_labels
from athenian.api.controllers.miners.github.release_mine import \
    diff_releases as mine_diff_releases, mine_releases, mine_releases_by_name
from athenian.api.controllers.miners.github.repository import mine_repositories
from athenian.api.controllers.miners.github.user import mine_user_avatars
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind, \
    PullRequestEvent, PullRequestListItem, PullRequestStage, ReleaseFacts, ReleaseParticipationKind
from athenian.api.controllers.prefixer import Prefixer, PrefixerPromise
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.controllers.settings import ReleaseSettings, Settings
from athenian.api.models.metadata.github import NodePullRequestJiraIssues, PullRequest, \
    PushCommit, Release, User
from athenian.api.models.metadata.jira import Epic, Issue
from athenian.api.models.web import BadRequestError, Commit, CommitSignature, CommitsList, \
    DeveloperSummary, DeveloperUpdates, FilterCommitsRequest, FilterContributorsRequest, \
    FilteredLabel, FilteredRelease, FilterLabelsRequest, FilterPullRequestsRequest, \
    FilterReleasesRequest, FilterRepositoriesRequest, ForbiddenError, GetPullRequestsRequest, \
    GetReleasesRequest, IncludedNativeUser, IncludedNativeUsers, InvalidRequestError, \
    LinkedJIRAIssue, PullRequest as WebPullRequest, PullRequestLabel, PullRequestParticipant, \
    PullRequestSet, ReleasedPullRequest, ReleaseSet, ReleaseSetInclude, StageTimings
from athenian.api.models.web.code_check_run_statistics import CodeCheckRunStatistics
from athenian.api.models.web.diff_releases_request import DiffReleasesRequest
from athenian.api.models.web.diffed_releases import DiffedReleases
from athenian.api.models.web.filter_code_checks_request import FilterCodeChecksRequest
from athenian.api.models.web.filtered_code_check_run import FilteredCodeCheckRun
from athenian.api.models.web.filtered_code_check_runs import FilteredCodeCheckRuns
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
        filt.account, [u[User.node_id.key] for u in users],
        request.sdb, request.mdb, request.cache)
    prefixer = await prefixer.load()
    model = [
        DeveloperSummary(
            login=prefixer.user_node_to_prefixed_login[u[User.node_id.key]],
            avatar=u[User.avatar_url.key],
            name=u[User.name.key],
            updates=DeveloperUpdates(**{
                k: v for k, v in u["stats"].items()
                # TODO(se7entyse7en): make `DeveloperUpdates` support all the stats we can get instead of doing this filtering. See also `mine_contributors`.  # noqa
                if k in DeveloperUpdates.openapi_types
            }),
            jira_user=mapped_jira.get(u[User.node_id.key]),
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
                                                FilterCodeChecksRequest],
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
    prefixer = Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
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
    prs = await filter_pull_requests(
        events, stages, time_from, time_to, repos, participants, labels, jira,
        filt.exclude_inactive, settings, updated_min, updated_max,
        prefixer, filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
    prefixer = await prefixer.load()  # no-op
    return await _build_github_prs_response(prs, prefixer, meta_ids, request.mdb, request.cache)


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
            commits[PushCommit.author_login.key].values,
            commits[PushCommit.committer_login.key].values,
            commits[PushCommit.repository_full_name.key].values,
            commits[PushCommit.sha.key].values,
            commits[PushCommit.message.key].values,
            commits[PushCommit.additions.key].values,
            commits[PushCommit.deletions.key].values,
            commits[PushCommit.changed_files.key].values,
            commits[PushCommit.author_name.key].values,
            commits[PushCommit.author_email.key].values,
            commits[PushCommit.authored_date.key],
            commits[PushCommit.committer_name.key].values,
            commits[PushCommit.committer_email.key].values,
            commits[PushCommit.committed_date.key],
            commits[PushCommit.author_date.key],
            commits[PushCommit.commit_date.key],
            commits[PushCommit.author_avatar_url.key],
            commits[PushCommit.committer_avatar_url.key]):
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


@weight(0.5)
async def filter_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find releases that were published in the given time fram in the given repositories."""
    try:
        filt = FilterReleasesRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    time_from, time_to, repos, meta_ids, prefixer = await _common_filter_preprocess(
        filt, request, strip_prefix=False)
    participants = {
        rpk: getattr(filt.with_, attr) or []
        for attr, rpk in (("releaser", ReleaseParticipationKind.RELEASER),
                          ("pr_author", ReleaseParticipationKind.PR_AUTHOR),
                          ("commit_author", ReleaseParticipationKind.COMMIT_AUTHOR))
    } if filt.with_ is not None else {}
    tasks = [
        Settings.from_request(request, filt.account).list_release_matches(repos),
        get_jira_installation_or_none(filt.account, request.sdb, request.mdb, request.cache),
    ]
    settings, jira_ids = await gather(*tasks)
    repos = [r.split("/", 1)[1] for r in repos]
    branches, default_branches = await BranchMiner.extract_branches(
        repos, meta_ids, request.mdb, request.cache)
    releases, avatars, _ = await mine_releases(
        repos, participants, branches, default_branches, time_from, time_to,
        LabelFilter.from_iterables(filt.labels_include, filt.labels_exclude),
        JIRAFilter.from_web(filt.jira, jira_ids), settings, prefixer, filt.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache, with_pr_titles=True)
    return await _build_release_set_response(releases, avatars, jira_ids, meta_ids, request.mdb)


async def _load_jira_issues(jira_ids: Optional[Tuple[int, List[str]]],
                            releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                            meta_ids: Tuple[int, ...],
                            mdb: databases.Database) -> Dict[str, LinkedJIRAIssue]:
    if jira_ids is None:
        for (_, facts) in releases:
            facts.prs_jira = np.full(len(facts["prs_" + PullRequest.node_id.key]), None)
        return {}

    pr_to_ix = {}
    for ri, (_, facts) in enumerate(releases):
        node_ids = facts["prs_" + PullRequest.node_id.key]
        facts.prs_jira = [[] for _ in range(len(node_ids))]
        for pri, node_id in enumerate(node_ids):
            pr_to_ix[node_id.decode()] = ri, pri
    regiss = aliased(Issue, name="regular")
    epiciss = aliased(Epic, name="epic")
    prmap = aliased(NodePullRequestJiraIssues, name="m")
    rows = await mdb.fetch_all(
        select([prmap.node_id.label("node_id"),
                regiss.key.label("key"),
                regiss.title.label("title"),
                regiss.labels.label("labels"),
                regiss.type.label("type"),
                epiciss.key.label("epic")])
        .select_from(outerjoin(
            join(regiss, prmap, and_(regiss.id == prmap.jira_id,
                                     regiss.acc_id == prmap.jira_acc)),
            epiciss, and_(epiciss.id == regiss.epic_id,
                          epiciss.acc_id == regiss.acc_id)))
        .where(and_(prmap.node_id.in_(pr_to_ix),
                    prmap.node_acc.in_(meta_ids),
                    regiss.project_id.in_(jira_ids[1]),
                    regiss.is_deleted.is_(False))))
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
                                      jira_ids: Optional[Tuple[int, List[str]]],
                                      meta_ids: Tuple[int, ...],
                                      mdb: databases.Database,
                                      ) -> web.Response:
    issues = await _load_jira_issues(jira_ids, releases, meta_ids, mdb)
    data = [_filtered_release_from_tuple(t) for t in releases]
    model = ReleaseSet(data=data, include=ReleaseSetInclude(
        users={u: IncludedNativeUser(avatar=a) for u, a in avatars},
        jira=issues,
    ))
    return model_response(model)


def _filtered_release_from_tuple(t: Tuple[Dict[str, Any], ReleaseFacts]) -> FilteredRelease:
    details, facts = t
    return FilteredRelease(name=details[Release.name.key],
                           repository=details[Release.repository_full_name.key],
                           url=details[Release.url.key],
                           publisher=facts.publisher,
                           published=facts.published.item().replace(tzinfo=timezone.utc),
                           age=facts.age,
                           added_lines=facts.additions,
                           deleted_lines=facts.deletions,
                           commits=facts.commits_count,
                           commit_authors=[u.decode() for u in facts.commit_authors],
                           prs=_extract_release_prs(facts))


def _extract_release_prs(facts: ReleaseFacts) -> List[ReleasedPullRequest]:
    return [
        ReleasedPullRequest(
            number=number,
            title=title,
            additions=adds,
            deletions=dels,
            author=author.decode() or None,
            jira=jira or None,
        )
        for number, title, adds, dels, author, jira in zip(
            facts["prs_" + PullRequest.number.key],
            facts["prs_" + PullRequest.title.key],
            facts["prs_" + PullRequest.additions.key],
            facts["prs_" + PullRequest.deletions.key],
            facts["prs_" + PullRequest.user_login.key],
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
    prefixer = Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    prs = await fetch_pull_requests(
        prs_by_repo, settings, prefixer, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    prefixer = await prefixer.load()  # no-op
    return await _build_github_prs_response(prs, prefixer, meta_ids, request.mdb, request.cache)


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


@sentry_span
async def _build_github_prs_response(prs: List[PullRequestListItem],
                                     prefixer: Prefixer,
                                     meta_ids: Tuple[int, ...],
                                     mdb: databases.Database,
                                     cache: Optional[aiomcache.Client],
                                     ) -> web.Response:
    log = logging.getLogger(f"{metadata.__package__}._build_github_prs_response")
    web_prs = sorted(web_pr_from_struct(pr, prefixer, log) for pr in prs)
    users = set(chain.from_iterable(chain.from_iterable(pr.participants.values()) for pr in prs))
    avatars = await mine_user_avatars(users, False, meta_ids, mdb, cache)
    model = PullRequestSet(include=IncludedNativeUsers(users={
        prefixer.user_login_to_prefixed_login[login]: IncludedNativeUser(avatar=avatar)
        for login, avatar in avatars
    }), data=web_prs)
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
    prefixer = Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    releases, avatars = await mine_releases_by_name(
        releases_by_repo, settings, prefixer, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    return await _build_release_set_response(releases, avatars, jira_ids, meta_ids, request.mdb)


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
    prefixer = Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    releases, avatars = await mine_diff_releases(
        borders, settings, prefixer, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    issues = await _load_jira_issues(
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
                releases=[_filtered_release_from_tuple(t) for t in diff[2]]))
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
    raise NotImplementedError
