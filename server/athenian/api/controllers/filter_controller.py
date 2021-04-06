from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import operator
from typing import Any, Collection, Dict, List, Optional, Set, Tuple, Union

from aiohttp import web
import aiomcache
import databases
from dateutil.parser import parse as parse_datetime
import numpy as np
from sqlalchemy import and_, join, outerjoin, select
from sqlalchemy.orm import aliased

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.features.github.pull_request_filter import fetch_pull_requests, \
    filter_pull_requests
from athenian.api.controllers.jira import get_jira_installation, get_jira_installation_or_none, \
    load_mapped_jira_users
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.controllers.miners.github.contributors import mine_contributors
from athenian.api.controllers.miners.github.label import mine_labels
from athenian.api.controllers.miners.github.release_mine import \
    diff_releases as mine_diff_releases, mine_releases, mine_releases_by_name
from athenian.api.controllers.miners.github.repositories import mine_repositories
from athenian.api.controllers.miners.github.users import mine_user_avatars
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind, \
    PullRequestEvent, PullRequestListItem, PullRequestStage, ReleaseFacts, ReleaseParticipationKind
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.controllers.settings import ReleaseMatchSetting, Settings
from athenian.api.models.metadata import PREFIXES
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
from athenian.api.models.web.diff_releases_request import DiffReleasesRequest
from athenian.api.models.web.diffed_releases import DiffedReleases
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
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    repos, meta_ids = await _common_filter_preprocess(filt, request, strip_prefix=False)
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    repos = [r.split("/", 1)[1] for r in repos]
    users = await mine_contributors(
        repos, filt.date_from, filt.date_to, True, filt.as_ or [], release_settings,
        filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
    mapped_jira = await load_mapped_jira_users(
        filt.account, [u[User.node_id.key] for u in users],
        request.sdb, request.mdb, request.cache)
    model = [
        DeveloperSummary(
            login=f"{PREFIXES['github']}{u[User.login.key]}",
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
    repos, meta_ids = await _common_filter_preprocess(filt, request, strip_prefix=False)
    release_settings = \
        await Settings.from_request(request, filt.account).list_release_matches(repos)
    repos = [r.split("/", 1)[1] for r in repos]
    repos = await mine_repositories(
        repos, filt.date_from, filt.date_to, filt.exclude_inactive, release_settings,
        filt.account, meta_ids, request.mdb, request.pdb, request.cache)
    return web.json_response(repos)


async def _common_filter_preprocess(filt: Union[FilterReleasesRequest,
                                                FilterRepositoriesRequest,
                                                FilterPullRequestsRequest,
                                                FilterCommitsRequest],
                                    request: AthenianWebRequest,
                                    strip_prefix=True) -> Tuple[Set[str], Tuple[int, ...]]:
    if filt.date_to < filt.date_from:
        raise ResponseError(InvalidRequestError(
            detail="date_from may not be greater than date_to",
            pointer=".date_from",
        ))
    filt.date_from = datetime.combine(filt.date_from, datetime.min.time(), tzinfo=timezone.utc)
    filt.date_to = datetime.combine(
        filt.date_to + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    if filt.timezone is not None:
        tzoffset = timedelta(minutes=-filt.timezone)
        filt.date_from += tzoffset
        filt.date_to += tzoffset

    async def login_loader() -> str:
        return (await request.user()).login

    return await resolve_repos(
        filt.in_, filt.account, request.uid, login_loader,
        request.sdb, request.mdb, request.cache, request.app["slack"],
        strip_prefix=strip_prefix)


async def resolve_filter_prs_parameters(filt: FilterPullRequestsRequest,
                                        request: AthenianWebRequest,
                                        ) -> Tuple[Set[str], Set[str], Set[str], PRParticipants,
                                                   LabelFilter, JIRAFilter,
                                                   Dict[str, ReleaseMatchSetting],
                                                   Tuple[int, ...]]:
    """Infer all the required PR filters from the request."""
    repos, meta_ids = await _common_filter_preprocess(filt, request, strip_prefix=False)
    events = set(getattr(PullRequestEvent, e.upper()) for e in (filt.events or []))
    stages = set(getattr(PullRequestStage, s.upper()) for s in (filt.stages or []))
    if not events and not stages:
        raise ResponseError(InvalidRequestError(
            detail="Either `events` or `stages` must be specified and be not empty.",
            pointer=".stages"))
    participants = {PRParticipationKind[k.upper()]: {d.split("/", 1)[1] for d in v}
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
    return repos, events, stages, participants, labels, jira, settings, meta_ids


@weight(6)
async def filter_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """List pull requests that satisfy the query."""
    try:
        filt = FilterPullRequestsRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        raise ResponseError(InvalidRequestError("?", detail=str(e)))
    repos, events, stages, participants, labels, jira, settings, meta_ids = \
        await resolve_filter_prs_parameters(filt, request)
    updated_min, updated_max = _bake_updated_min_max(filt)
    prs = await filter_pull_requests(
        events, stages, filt.date_from, filt.date_to, repos, participants, labels, jira,
        filt.exclude_inactive, settings, updated_min, updated_max,
        filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
    return await _build_github_prs_response(prs, meta_ids, request.mdb, request.cache)


def _bake_updated_min_max(filt: FilterPullRequestsRequest) -> Tuple[datetime, datetime]:
    if (filt.updated_from is None) != (filt.updated_to is None):
        raise ResponseError(InvalidRequestError(
            ".updated_from",
            "`updated_from` and `updated_to` must be both either specified or not"))
    if filt.updated_from is not None:
        updated_min = datetime.combine(filt.updated_from, datetime.min.time(), tzinfo=timezone.utc)
        updated_max = datetime.combine(filt.updated_to, datetime.min.time(), tzinfo=timezone.utc)
    else:
        updated_min = updated_max = None
    return updated_min, updated_max


def web_pr_from_struct(pr: PullRequestListItem) -> WebPullRequest:
    """Convert an intermediate PR representation to the web model."""
    props = dict(pr)
    del props["node_id"]
    if pr.events_time_machine is not None:
        props["events_time_machine"] = sorted(p.name.lower() for p in pr.events_time_machine)
    if pr.stages_time_machine is not None:
        props["stages_time_machine"] = sorted(p.name.lower() for p in pr.stages_time_machine)
    props["events_now"] = sorted(p.name.lower() for p in pr.events_now)
    props["stages_now"] = sorted(p.name.lower() for p in pr.stages_now)
    props["stage_timings"] = StageTimings(**pr.stage_timings)
    participants = defaultdict(list)
    prefix = PREFIXES["github"]
    for pk, pids in sorted(pr.participants.items()):
        pkweb = pk.name.lower()
        for pid in pids:
            participants[prefix + pid].append(pkweb)
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
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    repos, meta_ids = await _common_filter_preprocess(filt, request)
    with_author = [s.split("/", 1)[1] for s in (filt.with_author or [])]
    with_committer = [s.split("/", 1)[1] for s in (filt.with_committer or [])]
    log = logging.getLogger("filter_commits")
    commits = await extract_commits(
        FilterCommitsProperty(filt.property), filt.date_from, filt.date_to, repos,
        with_author, with_committer, meta_ids, request.mdb, request.cache)
    model = CommitsList(data=[], include=IncludedNativeUsers(users={}))
    users = model.include.users
    utc = timezone.utc
    prefix = PREFIXES["github"]
    for commit in commits.itertuples():
        author_login = getattr(commit, PushCommit.author_login.key)
        committer_login = getattr(commit, PushCommit.committer_login.key)
        obj = Commit(
            repository=getattr(commit, PushCommit.repository_full_name.key),
            hash=getattr(commit, PushCommit.sha.key),
            message=getattr(commit, PushCommit.message.key),
            size_added=_nan_to_none(getattr(commit, PushCommit.additions.key)),
            size_removed=_nan_to_none(getattr(commit, PushCommit.deletions.key)),
            files_changed=_nan_to_none(getattr(commit, PushCommit.changed_files.key)),
            author=CommitSignature(
                login=(prefix + author_login) if author_login else None,
                name=getattr(commit, PushCommit.author_name.key),
                email=getattr(commit, PushCommit.author_email.key),
                timestamp=getattr(commit, PushCommit.authored_date.key).replace(tzinfo=utc),
            ),
            committer=CommitSignature(
                login=(prefix + committer_login) if committer_login else None,
                name=getattr(commit, PushCommit.committer_name.key),
                email=getattr(commit, PushCommit.committer_email.key),
                timestamp=getattr(commit, PushCommit.committed_date.key).replace(tzinfo=utc),
            ),
        )
        try:
            dt = parse_datetime(getattr(commit, PushCommit.author_date.key))
            obj.author.timezone = dt.tzinfo.utcoffset(dt).total_seconds() / 3600
        except ValueError:
            log.warning("Failed to parse the author timestamp of %s", obj.hash)
        try:
            dt = parse_datetime(getattr(commit, PushCommit.commit_date.key))
            obj.committer.timezone = dt.tzinfo.utcoffset(dt).total_seconds() / 3600
        except ValueError:
            log.warning("Failed to parse the committer timestamp of %s", obj.hash)
        if obj.author.login and obj.author.login not in users:
            users[obj.author.login] = IncludedNativeUser(
                avatar=getattr(commit, PushCommit.author_avatar_url.key))
        if obj.committer.login and obj.committer.login not in users:
            users[obj.committer.login] = IncludedNativeUser(
                getattr(commit, PushCommit.committer_avatar_url.key))
        model.data.append(obj)
    return model_response(model)


@weight(0.5)
async def filter_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find releases that were published in the given time fram in the given repositories."""
    try:
        filt = FilterReleasesRequest.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    repos, meta_ids = await _common_filter_preprocess(filt, request, strip_prefix=False)
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
    branches, default_branches = await extract_branches(
        repos, meta_ids, request.mdb, request.cache)
    releases, avatars, _ = await mine_releases(
        repos, participants, branches, default_branches, filt.date_from, filt.date_to,
        JIRAFilter.from_web(filt.jira, jira_ids), settings,
        filt.account, meta_ids, request.mdb, request.pdb, request.rdb, request.cache)
    return await _build_release_set_response(releases, avatars, jira_ids, meta_ids, request.mdb)


async def _load_jira_issues(jira_ids: Optional[Tuple[int, List[str]]],
                            releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                            meta_ids: Tuple[int, ...],
                            mdb: databases.Database) -> Dict[str, LinkedJIRAIssue]:
    if jira_ids is None:
        for (_, facts) in releases:
            facts.prs["jira"] = np.full(len(facts.prs[PullRequest.node_id.key]), None)
        return {}

    pr_to_ix = {}
    for ri, (_, facts) in enumerate(releases):
        node_ids = facts.prs[PullRequest.node_id.key]
        facts.prs["jira"] = [[] for _ in range(len(node_ids))]
        for pri, node_id in enumerate(node_ids):
            pr_to_ix[node_id] = ri, pri
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
                    regiss.project_id.in_(jira_ids[1]))))
    issues = {}
    for r in rows:
        key = r["key"]
        issues[key] = LinkedJIRAIssue(
            id=key, title=r["title"], epic=r["epic"], labels=r["labels"], type=r["type"])
        ri, pri = pr_to_ix[r["node_id"]]
        releases[ri][1].prs["jira"][pri].append(key)
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
                           published=facts.published,
                           age=facts.age,
                           added_lines=facts.additions,
                           deleted_lines=facts.deletions,
                           commits=facts.commits_count,
                           commit_authors=facts.commit_authors,
                           prs=_extract_release_prs(facts.prs))


def _extract_release_prs(prs: Dict[str, np.ndarray]) -> List[ReleasedPullRequest]:
    return [
        ReleasedPullRequest(
            number=number,
            title=title,
            additions=adds,
            deletions=dels,
            author=author,
            jira=jira or None,
        )
        for number, title, adds, dels, author, jira in zip(
            prs[PullRequest.number.key],
            prs[PullRequest.title.key],
            prs[PullRequest.additions.key],
            prs[PullRequest.deletions.key],
            prs[PullRequest.user_login.key],
            prs["jira"],
        )
    ]


@weight(0.5)
async def get_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """List pull requests by repository and number."""
    body = GetPullRequestsRequest.from_dict(body)
    repos = {}
    for p in body.prs:
        repos.setdefault(p.repository, set()).update(p.numbers)
    try:
        github_repos, settings, meta_ids = await _get_github_repos(request, body.account, repos)
    except KeyError:
        return model_response(PullRequestSet())
    github_prs = {r: repos[PREFIXES["github"] + r] for r in github_repos}
    prs = await fetch_pull_requests(
        github_prs, settings, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    return await _build_github_prs_response(prs, meta_ids, request.mdb, request.cache)


async def _get_github_repos(request: AthenianWebRequest,
                            account: int,
                            repos: Collection[str],
                            ) -> Tuple[List[str],
                                       Dict[str, ReleaseMatchSetting],
                                       Tuple[int, ...]]:
    checkers = {}
    repos_by_service = {}
    reverse_prefixes = {v: k for k, v in PREFIXES.items()}
    async with request.sdb.connection() as sdb_conn:
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
        async with request.mdb.connection() as mdb_conn:
            for repo in repos:
                for prefix, service in reverse_prefixes.items():  # noqa: B007
                    if repo.startswith(prefix):
                        break
                else:
                    raise ResponseError(BadRequestError(
                        detail="Repository %s is unsupported" % repo))
                try:
                    checker = checkers[service]
                except KeyError:
                    checker = checkers[service] = access_classes[service](
                        account, meta_ids, sdb_conn, mdb_conn, request.cache)
                    await checker.load()
                repo = repo[len(prefix):]
                if await checker.check({repo}):
                    raise ResponseError(ForbiddenError(
                        detail="Account %d is access denied to repo %s" % (account, repo)))
                repos_by_service.setdefault(service, []).append(repo)
    settings = await Settings.from_request(request, account).list_release_matches(repos)
    return repos_by_service["github"], settings, meta_ids


@sentry_span
async def _build_github_prs_response(prs: List[PullRequestListItem],
                                     meta_ids: Tuple[int, ...],
                                     mdb: databases.Database,
                                     cache: Optional[aiomcache.Client]) -> web.Response:
    web_prs = sorted(web_pr_from_struct(pr) for pr in prs)
    users = set(chain.from_iterable(chain.from_iterable(pr.participants.values()) for pr in prs))
    avatars = await mine_user_avatars(users, meta_ids, mdb, cache)
    prefix = PREFIXES["github"]
    model = PullRequestSet(include=IncludedNativeUsers(users={
        prefix + login: IncludedNativeUser(avatar=avatar) for login, avatar in avatars
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
    repos = {}
    for p in body.releases:
        repos.setdefault(p.repository, set()).update(p.names)
    try:
        (github_repos, settings, meta_ids), jira_ids = await gather(
            _get_github_repos(request, body.account, repos),
            get_jira_installation_or_none(body.account, request.sdb, request.mdb, request.cache),
        )
    except KeyError:
        return model_response(ReleaseSet())
    github_releases = {r: repos[PREFIXES["github"] + r] for r in github_repos}
    releases, avatars = await mine_releases_by_name(
        github_releases, settings, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    return await _build_release_set_response(releases, avatars, jira_ids, meta_ids, request.mdb)


@weight(1)
async def diff_releases(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find releases between the two given ones per repository."""
    prefix = PREFIXES["github"]
    body = DiffReleasesRequest.from_dict(body)
    borders = {}
    for repo, border in body.borders.items():
        borders[repo] = [(pair.old, pair.new) for pair in border]
    try:
        (github_repos, settings, meta_ids), jira_ids = await gather(
            _get_github_repos(request, body.account, borders),
            get_jira_installation_or_none(body.account, request.sdb, request.mdb, request.cache),
        )
    except KeyError:
        return model_response(ReleaseSet())
    github_borders = {r: borders[prefix + r] for r in github_repos}
    releases, avatars = await mine_diff_releases(
        github_borders, settings, body.account, meta_ids,
        request.mdb, request.pdb, request.rdb, request.cache)
    issues = await _load_jira_issues(
        jira_ids, list(chain.from_iterable(chain.from_iterable(r[-1] for r in rr)
                                           for rr in releases.values())),
        meta_ids, request.mdb)
    result = DiffedReleases(data={}, include=ReleaseSetInclude(
        users={u: IncludedNativeUser(avatar=a) for u, a in avatars},
        jira=issues,
    ))
    for repo, diffs in releases.items():
        result.data[prefix + repo] = repo_result = []
        for diff in diffs:
            repo_result.append(ReleaseDiff(
                old=diff[0], new=diff[1],
                releases=[_filtered_release_from_tuple(t) for t in diff[2]]))
    return model_response(result)
