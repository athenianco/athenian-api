import asyncio
from datetime import datetime, timezone
import logging
import pickle
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Set

import aiomcache
import databases
from dateutil.rrule import DAILY, rrule
import numpy as np
import pandas as pd
from sqlalchemy import and_, insert, join, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.sql import ClauseElement

from athenian.api import metadata
from athenian.api.async_read_sql_query import read_sql_query
from athenian.api.cache import cached
from athenian.api.controllers.miners.github.released_pr import matched_by_column, \
    new_released_prs_df
from athenian.api.controllers.miners.types import MinedPullRequest, Participants, \
    ParticipationKind, PullRequestTimes
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.db import add_pdb_hits
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestLabel, PullRequestReview, PullRequestReviewRequest, Release
from athenian.api.models.precomputed.models import GitHubMergedPullRequest, GitHubPullRequestTimes
from athenian.api.tracing import sentry_span


def _create_common_filters(time_from: datetime,
                           time_to: datetime,
                           repos: Collection[str]) -> List[ClauseElement]:
    assert isinstance(time_from, datetime)
    assert isinstance(time_to, datetime)
    ghprt = GitHubPullRequestTimes
    return [
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        ghprt.repository_full_name.in_(repos),
        ghprt.pr_created_at < time_to,
        ghprt.pr_done_at >= time_from,
    ]


def _check_release_match(repo: str,
                         release_match: str,
                         release_settings: Dict[str, ReleaseMatchSetting],
                         default_branches: Dict[str, str],
                         prefix: str,
                         result: Any,
                         ambiguous: Dict[str, Any]) -> Optional[Any]:
    if release_match == "rejected":
        dump = result
    else:
        match_name, match_by = release_match.split("|", 1)
        match = ReleaseMatch[match_name]
        required_release_match = release_settings[prefix + repo]
        if required_release_match.match != ReleaseMatch.tag_or_branch:
            if match != required_release_match.match:
                return None
            dump = result
        else:
            dump = ambiguous[match_name]
        if match == ReleaseMatch.tag:
            target = required_release_match.tags
        elif match == ReleaseMatch.branch:
            target = required_release_match.branches.replace(
                default_branch_alias, default_branches.get(repo, default_branch_alias))
        else:
            raise AssertionError("Precomputed DB may not contain Match.tag_or_branch")
        if target != match_by:
            return None
    return dump


@sentry_span
async def load_precomputed_done_candidates(time_from: datetime,
                                           time_to: datetime,
                                           repos: Collection[str],
                                           default_branches: Dict[str, str],
                                           release_settings: Dict[str, ReleaseMatchSetting],
                                           pdb: databases.Database,
                                           ) -> Set[str]:
    """
    Load the set of released PR identifiers.

    We find all the released PRs for a given time frame, repositories, and release match settings.
    """
    ghprt = GitHubPullRequestTimes
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match]
    filters = _create_common_filters(time_from, time_to, repos)
    rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
    prefix = PREFIXES["github"]
    result = set()
    ambiguous = {ReleaseMatch.tag.name: set(), ReleaseMatch.branch.name: set()}
    for row in rows:
        dump = _check_release_match(
            row[1], row[2], release_settings, default_branches, prefix, result, ambiguous)
        if dump is None:
            continue
        dump.add(row[0])
    result.update(ambiguous[ReleaseMatch.tag.name])
    result.update(ambiguous[ReleaseMatch.branch.name])
    return result


def _build_participants_filters(participants: Participants,
                                filters: list,
                                selected: list,
                                postgres: bool) -> None:
    ghprt = GitHubPullRequestTimes
    if postgres:
        developer_filters_single = []
        for col, pk in ((ghprt.author, ParticipationKind.AUTHOR),
                        (ghprt.merger, ParticipationKind.MERGER),
                        (ghprt.releaser, ParticipationKind.RELEASER)):
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            developer_filters_single.append(col.in_(col_parts))
        # do not send the same array several times
        for f in developer_filters_single[1:]:
            f.right = developer_filters_single[0].right
        developer_filters_multiple = []
        for col, pk in ((ghprt.commenters, ParticipationKind.COMMENTER),
                        (ghprt.reviewers, ParticipationKind.REVIEWER),
                        (ghprt.commit_authors, ParticipationKind.COMMIT_AUTHOR),
                        (ghprt.commit_committers, ParticipationKind.COMMIT_COMMITTER)):
            col_parts = participants.get(pk)
            if not col_parts:
                continue
            developer_filters_multiple.append(col.has_any(col_parts))
        # do not send the same array several times
        for f in developer_filters_multiple[1:]:
            f.right = developer_filters_multiple[0].right
        filters.append(or_(*developer_filters_single, *developer_filters_multiple))
    else:
        selected.extend([
            ghprt.author, ghprt.merger, ghprt.releaser, ghprt.reviewers, ghprt.commenters,
            ghprt.commit_authors, ghprt.commit_committers])


def _build_labels_filters(labels: Collection[str],
                          filters: list,
                          selected: list,
                          postgres: bool) -> None:
    if postgres:
        filters.append(GitHubPullRequestTimes.labels.has_any(labels))
    else:
        selected.append(GitHubPullRequestTimes.labels)


def _check_participants(row: Mapping, participants: Participants) -> bool:
    ghprt = GitHubPullRequestTimes
    for col, pk in ((ghprt.author, ParticipationKind.AUTHOR),
                    (ghprt.merger, ParticipationKind.MERGER),
                    (ghprt.releaser, ParticipationKind.RELEASER)):
        dev = row[col.key]
        if dev and dev in participants.get(pk, set()):
            return True
    for col, pk in ((ghprt.reviewers, ParticipationKind.REVIEWER),
                    (ghprt.commenters, ParticipationKind.COMMENTER),
                    (ghprt.commit_authors, ParticipationKind.COMMIT_AUTHOR),
                    (ghprt.commit_committers, ParticipationKind.COMMIT_COMMITTER)):
        devs = set(row[col.key])
        if devs.intersection(participants.get(pk, set())):
            return True
    return False


@sentry_span
async def load_precomputed_done_times(time_from: datetime,
                                      time_to: datetime,
                                      repos: Collection[str],
                                      participants: Participants,
                                      labels: Collection[str],
                                      default_branches: Dict[str, str],
                                      exclude_inactive: bool,
                                      release_settings: Dict[str, ReleaseMatchSetting],
                                      pdb: databases.Database,
                                      ) -> Dict[str, PullRequestTimes]:
    """Load PullRequestTimes belonging to released or rejected PRs from the precomputed DB."""
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    ghprt = GitHubPullRequestTimes
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                ghprt.data]
    filters = _create_common_filters(time_from, time_to, repos)
    if len(participants) > 0:
        _build_participants_filters(participants, filters, selected, postgres)
    if len(labels) > 0:
        _build_labels_filters(labels, filters, selected, postgres)
        if not postgres and not isinstance(labels, set):
            labels = set(labels)
    if exclude_inactive:
        # timezones: date_from and date_to may be not exactly 00:00
        date_from_day = datetime.combine(
            time_from.date(), datetime.min.time(), tzinfo=timezone.utc)
        date_to_day = datetime.combine(
            time_to.date(), datetime.min.time(), tzinfo=timezone.utc)
        # date_to_day will be included
        date_range = rrule(DAILY, dtstart=date_from_day, until=date_to_day)
        if postgres:
            filters.append(ghprt.activity_days.overlap(list(date_range)))
        else:
            selected.append(ghprt.activity_days)
            date_range = set(date_range)
    rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
    prefix = PREFIXES["github"]
    result = {}
    ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
    for row in rows:
        dump = _check_release_match(
            row[1], row[2], release_settings, default_branches, prefix, result, ambiguous)
        if dump is None:
            continue
        if not postgres:
            if len(participants) > 0 and not _check_participants(row, participants):
                continue
            if len(labels) > 0 and not labels.intersection(row[ghprt.labels.key]):
                continue
            if exclude_inactive:
                activity_days = {datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                 for d in row[ghprt.activity_days.key]}
                if not activity_days.intersection(date_range):
                    continue
        dump[row[0]] = pickle.loads(row[3])
    result.update(ambiguous[ReleaseMatch.tag.name])
    for node_id, times in ambiguous[ReleaseMatch.branch.name].items():
        if node_id not in result:
            result[node_id] = times
    return result


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda prs, default_branches, release_settings, **_: (
        ",".join(sorted(prs)), sorted(default_branches.items()), release_settings,
    ),
    refresh_on_access=True,
)
async def load_precomputed_pr_releases(prs: Iterable[str],
                                       time_to: datetime,
                                       matched_bys: Dict[str, ReleaseMatch],
                                       default_branches: Dict[str, str],
                                       release_settings: Dict[str, ReleaseMatchSetting],
                                       pdb: databases.Database,
                                       cache: Optional[aiomcache.Client]) -> pd.DataFrame:
    """
    Load the releases mentioned in the specified PRs.

    Each PR is represented by a node_id, a repository name, and a required release match.
    """
    log = logging.getLogger("%s.load_precomputed_pr_releases" % metadata.__package__)
    ghprt = GitHubPullRequestTimes
    prs = await pdb.fetch_all(
        select([ghprt.pr_node_id, ghprt.pr_done_at, ghprt.releaser, ghprt.release_url,
                ghprt.repository_full_name, ghprt.release_match])
        .where(and_(ghprt.pr_node_id.in_(prs),
                    ghprt.releaser.isnot(None),
                    ghprt.pr_done_at < time_to)))
    prefix = PREFIXES["github"]
    records = []
    utc = timezone.utc
    for pr in prs:
        repo = pr[ghprt.repository_full_name.key]
        match_name, match_by = pr[ghprt.release_match.key].split("|", 1)
        release_match = ReleaseMatch[match_name]
        try:
            if release_match != matched_bys[repo]:
                continue
        except KeyError:
            # pdb thinks this PR was released but our current release matching settings disagree
            log.warning("Alternative release matching detected: %s", dict(pr))
            continue
        if release_match == ReleaseMatch.tag:
            if match_by != release_settings[prefix + repo].tags:
                continue
        elif release_match == ReleaseMatch.branch:
            branches = release_settings[prefix + repo].branches.replace(
                default_branch_alias, default_branches[repo])
            if match_by != branches:
                continue
        else:
            raise AssertionError("Unsupported release match in the precomputed DB: " + match_name)
        records.append((pr[ghprt.pr_node_id.key], pr[ghprt.pr_done_at.key].replace(tzinfo=utc),
                        pr[ghprt.releaser.key].rstrip(), pr[ghprt.release_url.key],
                        pr[ghprt.repository_full_name.key],
                        ReleaseMatch[pr[ghprt.release_match.key].split("|", 1)[0]]))
    return new_released_prs_df(records)


@sentry_span
async def store_precomputed_done_times(prs: Iterable[MinedPullRequest],
                                       pr_times: Iterable[PullRequestTimes],
                                       default_branches: Dict[str, str],
                                       release_settings: Dict[str, ReleaseMatchSetting],
                                       pdb: databases.Database,
                                       ) -> None:
    """Store PullRequestTimes belonging to released or rejected PRs to the precomputed DB."""
    inserted = []
    prefix = PREFIXES["github"]
    sqlite = pdb.url.dialect == "sqlite"
    for pr, times in zip(prs, pr_times):
        activity_days = set()
        if not times.released:
            if not times.closed or times.merged:
                continue
            done_at = times.closed.best
        else:
            done_at = times.released.best
            activity_days.add(times.released.best.date())
        activity_days.add(times.created.best.date())
        activity_days.add(times.closed.best.date())
        repo = pr.pr[PullRequest.repository_full_name.key]
        if pr.release[matched_by_column] is not None:
            release_match = release_settings[prefix + repo]
            match = ReleaseMatch(pr.release[matched_by_column])
            if match == ReleaseMatch.branch:
                branch = release_match.branches.replace(
                    default_branch_alias, default_branches[repo])
                release_match = "|".join((match.name, branch))
            elif match == ReleaseMatch.tag:
                release_match = "|".join((match.name, release_match.tags))
            else:
                raise AssertionError("Unhandled release match strategy: " + match.name)
        else:
            release_match = "rejected"
        participants = pr.participants()
        # if they are empty the column dtype is sometimes an object so .dt raises an exception
        if not pr.review_requests.empty:
            activity_days.update(
                pr.review_requests[PullRequestReviewRequest.created_at.key].dt.date)
        if not pr.reviews.empty:
            activity_days.update(pr.reviews[PullRequestReview.created_at.key].dt.date)
        if not pr.comments.empty:
            activity_days.update(pr.comments[PullRequestComment.created_at.key].dt.date)
        if not pr.commits.empty:
            activity_days.update(pr.commits[PullRequestCommit.committed_date.key].dt.date)
        if sqlite:
            activity_days = [d.strftime("%Y-%m-%d") for d in sorted(activity_days)]
        else:
            # Postgres is "clever" enough to localize them otherwise
            activity_days = [datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc)
                             for d in activity_days]
        inserted.append(GitHubPullRequestTimes(
            pr_node_id=pr.pr[PullRequest.node_id.key],
            release_match=release_match,
            repository_full_name=repo,
            pr_created_at=times.created.best,
            pr_done_at=done_at,
            number=pr.pr[PullRequest.number.key],
            release_url=pr.release[Release.url.key],
            author=_flatten_set(participants[ParticipationKind.AUTHOR]),
            merger=_flatten_set(participants[ParticipationKind.MERGER]),
            releaser=_flatten_set(participants[ParticipationKind.RELEASER]),
            commenters={k: "" for k in participants[ParticipationKind.COMMENTER]},
            reviewers={k: "" for k in participants[ParticipationKind.REVIEWER]},
            commit_authors={k: "" for k in participants[ParticipationKind.COMMIT_AUTHOR]},
            commit_committers={k: "" for k in participants[ParticipationKind.COMMIT_COMMITTER]},
            labels={label: "" for label in pr.labels[PullRequestLabel.name.key]},
            activity_days=activity_days,
            data=pickle.dumps(times),
        ).create_defaults().explode(with_primary_keys=True))
    if pdb.url.dialect in ("postgres", "postgresql"):
        sql = postgres_insert(GitHubPullRequestTimes).on_conflict_do_nothing()
    elif pdb.url.dialect == "sqlite":
        sql = insert(GitHubPullRequestTimes).prefix_with("OR IGNORE")
    else:
        raise AssertionError("Unsupported database dialect: %s" % pdb.url.dialect)
    await pdb.execute_many(sql, inserted)


def _flatten_set(s: set) -> Optional[Any]:
    if not s:
        return None
    assert len(s) == 1
    return next(iter(s))


@sentry_span
async def discover_unreleased_prs(prs: pd.DataFrame,
                                  releases: pd.DataFrame,
                                  matched_bys: Dict[str, ReleaseMatch],
                                  default_branches: Dict[str, str],
                                  release_settings: Dict[str, ReleaseMatchSetting],
                                  pdb: databases.Database) -> List[str]:
    """
    Load the list PR node identifiers which we are sure are not released in one of `releases`.

    For each merged PR we maintain the set of releases that do include that PR.
    """
    filters = []
    prefix = PREFIXES["github"]
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    for repo in prs[PullRequest.repository_full_name.key].unique():
        try:
            matched_by = matched_bys[repo]
        except KeyError:
            # no new releases
            continue
        release_setting = release_settings[prefix + repo]
        if matched_by == ReleaseMatch.tag:
            release_match = "tag|" + release_setting.tags
        elif matched_by == ReleaseMatch.branch:
            branch = release_setting.branches.replace(
                default_branch_alias, default_branches[repo])
            release_match = "branch|" + branch
        else:
            raise AssertionError("Unsupported release match %s" % matched_by)
        repo_filters = [GitHubMergedPullRequest.repository_full_name == repo,
                        GitHubMergedPullRequest.release_match == release_match]
        if postgres:
            repo_releases = releases[Release.id.key].take(
                np.where(releases[Release.repository_full_name.key] == repo)[0])
            repo_filters.append(GitHubMergedPullRequest.checked_releases.has_all(repo_releases))
        filters.append(and_(*repo_filters))
    selected = [GitHubMergedPullRequest.pr_node_id]
    if not postgres:
        selected.extend([GitHubMergedPullRequest.checked_releases,
                         GitHubMergedPullRequest.repository_full_name])
    rows = await pdb.fetch_all(
        select(selected)
        .where(and_(GitHubMergedPullRequest.pr_node_id.in_(prs.index),
                    or_(*filters))))
    if not postgres:
        filtered_rows = []
        grouped = {}
        for r in rows:
            grouped.setdefault(r[GitHubMergedPullRequest.repository_full_name.key], []).append(r)
        for repo, rows in grouped.items():
            repo_releases = set(releases[Release.id.key].take(
                np.where(releases[Release.repository_full_name.key] == repo)[0]))
            for r in rows:
                if not (repo_releases - set(r[GitHubMergedPullRequest.checked_releases.key])):
                    filtered_rows.append(r)
        rows = filtered_rows
    return [r[GitHubMergedPullRequest.pr_node_id.key] for r in rows]


@sentry_span
async def update_unreleased_prs(merged_prs: pd.DataFrame,
                                released_prs: pd.DataFrame,
                                releases: pd.DataFrame,
                                labels: Dict[str, List[str]],
                                matched_bys: Dict[str, ReleaseMatch],
                                default_branches: Dict[str, str],
                                release_settings: Dict[str, ReleaseMatchSetting],
                                pdb: databases.Database) -> None:
    """
    Append new releases which do *not* include the specified merged PRs.

    :param merged_prs: Merged PRs to update in the pdb.
    :param released_prs: Released PRs, we shouldn't write their releases in the pdb. However, \
                         any earlier release in `releases` should be appended.
    :param releases: All the releases that were considered in mapping `merged_prs` to \
                     `released_prs`.
    """
    updates = []
    prefix = PREFIXES["github"]
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    if not postgres:
        assert pdb.url.dialect == "sqlite"
    release_dates_by_pr = dict(zip(released_prs.index, released_prs[Release.published_at.key]))
    for repo, repo_prs in merged_prs.groupby(PullRequest.repository_full_name.key, sort=False):
        try:
            matched_by = matched_bys[repo]
        except KeyError:
            # no new releases
            continue
        repo_releases = releases[[Release.id.key, Release.published_at.key]].take(
            np.where(releases[Release.repository_full_name.key] == repo)[0])
        repo_releases_dict = {r: "" for r in repo_releases[Release.id.key].values}
        release_setting = release_settings[prefix + repo]
        if matched_by == ReleaseMatch.tag:
            release_match = "tag|" + release_setting.tags
        elif matched_by == ReleaseMatch.branch:
            branch = release_setting.branches.replace(
                default_branch_alias, default_branches[repo])
            release_match = "branch|" + branch
        else:
            raise AssertionError("Unsupported release match %s" % matched_by)
        if postgres:
            sql = postgres_insert(GitHubMergedPullRequest)
            sql = sql.on_conflict_do_update(
                constraint=GitHubMergedPullRequest.__table__.primary_key,
                set_={
                    GitHubMergedPullRequest.checked_releases.key:
                        GitHubMergedPullRequest.checked_releases + sql.excluded.checked_releases,
                    GitHubMergedPullRequest.updated_at.key: sql.excluded.updated_at,
                },
            )
        else:
            # this is wrong but we just cannot update SQLite properly
            # nothing will break though
            sql = insert(GitHubMergedPullRequest).prefix_with("OR REPLACE")
        values = []
        for node_id, merged_at, author, merger in zip(
                repo_prs.index.values, repo_prs[PullRequest.merged_at.key],
                repo_prs[PullRequest.user_login.key].values,
                repo_prs[PullRequest.merged_by_login.key].values):
            release_date = release_dates_by_pr.get(node_id)
            if release_date is None:
                pr_releases = repo_releases_dict
            else:
                pr_releases = {
                    r: "" for r in repo_releases[Release.id.key].values[
                        repo_releases[Release.published_at.key] < release_date]
                }
            values.append(GitHubMergedPullRequest(
                pr_node_id=node_id,
                release_match=release_match,
                repository_full_name=repo,
                checked_releases=pr_releases,
                merged_at=merged_at,
                author=author,
                merger=merger,
                labels={label: "" for label in labels.get(node_id, [])},
            ).create_defaults().explode(with_primary_keys=True))
        updates.append(pdb.execute_many(sql, values))
    errors = await asyncio.gather(*updates, return_exceptions=True)
    for r in errors:
        if isinstance(r, Exception):
            raise r from None


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda time_from, time_to, repos, participants, labels, default_branches, release_settings, **_: (  # noqa
        time_from.timestamp(), time_to.timestamp(), ",".join(sorted(repos)),
        sorted((k.name.lower(), sorted(v)) for k, v in participants.items()),
        ",".join(sorted(labels)), sorted(default_branches.items()), release_settings,
    ),
    refresh_on_access=True,
)
async def load_inactive_merged_unreleased_prs(time_from: datetime,
                                              time_to: datetime,
                                              repos: Collection[str],
                                              participants: Participants,
                                              labels: Collection[str],
                                              default_branches: Dict[str, str],
                                              release_settings: Dict[str, ReleaseMatchSetting],
                                              mdb: databases.Database,
                                              pdb: databases.Database,
                                              cache: Optional[aiomcache.Client],
                                              ) -> pd.DataFrame:
    """Discover PRs which were merged before `time_from` and still not released."""
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    selected = [GitHubMergedPullRequest.pr_node_id,
                GitHubMergedPullRequest.repository_full_name,
                GitHubMergedPullRequest.release_match]
    filters = [
        or_(GitHubPullRequestTimes.pr_done_at.is_(None),
            GitHubPullRequestTimes.pr_done_at >= time_to),
        GitHubMergedPullRequest.repository_full_name.in_(repos),
        GitHubMergedPullRequest.merged_at < time_from,
    ]
    for role, col in ((ParticipationKind.AUTHOR, GitHubMergedPullRequest.author),
                      (ParticipationKind.MERGER, GitHubMergedPullRequest.merger)):
        people = participants.get(role)
        if people:
            filters.append(col.in_(people))
    if len(labels) > 0:
        if postgres:
            filters.append(GitHubMergedPullRequest.labels.has_any(labels))
        else:
            selected.append(GitHubMergedPullRequest.labels)
            if not isinstance(labels, set):
                labels = set(labels)
    body = join(GitHubMergedPullRequest, GitHubPullRequestTimes, and_(
        GitHubPullRequestTimes.pr_node_id == GitHubMergedPullRequest.pr_node_id,
        GitHubPullRequestTimes.release_match == GitHubMergedPullRequest.release_match,
        GitHubPullRequestTimes.repository_full_name.in_(repos),
        GitHubPullRequestTimes.pr_created_at < time_from,
    ), isouter=True)
    rows = await pdb.fetch_all(select(selected).select_from(body).where(and_(*filters)))
    prefix = PREFIXES["github"]
    ambiguous = {ReleaseMatch.tag.name: set(), ReleaseMatch.branch.name: set()}
    node_ids = []
    for row in rows:
        dump = _check_release_match(
            row[1], row[2], release_settings, default_branches, prefix, "whatever", ambiguous)
        if dump is None:
            continue
        # we do not care about the exact release match
        if len(labels) > 0 and not postgres and \
                not labels.intersection(row[GitHubMergedPullRequest.labels.key]):
            continue
        node_ids.append(row[0])
    add_pdb_hits(pdb, "inactive_merged_unreleased", len(node_ids))
    return await read_sql_query(select([PullRequest])
                                .where(PullRequest.node_id.in_(node_ids))
                                .order_by(PullRequest.node_id),
                                mdb, PullRequest, index=PullRequest.node_id.key)
