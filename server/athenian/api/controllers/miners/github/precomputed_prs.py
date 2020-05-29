import asyncio
from datetime import datetime, timezone
import pickle
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import aiomcache
import databases
from dateutil.rrule import DAILY, rrule
from sqlalchemy import and_, insert, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql import ClauseElement

from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.pull_request import MinedPullRequest, PullRequestTimes
from athenian.api.controllers.miners.github.release import matched_by_column
from athenian.api.controllers.miners.pull_request_list_item import Participants, ParticipationKind
from athenian.api.controllers.settings import default_branch_alias, Match, ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, PullRequestReviewRequest
from athenian.api.models.precomputed.models import GitHubPullRequestTimes
from athenian.api.tracing import sentry_span


def _create_common_filters(date_from: datetime,
                           date_to: datetime,
                           repos: Collection[str]) -> List[ClauseElement]:
    assert isinstance(date_from, datetime)
    assert isinstance(date_to, datetime)
    ghprt = GitHubPullRequestTimes
    return [
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        ghprt.repository_full_name.in_(repos),
        ghprt.pr_created_at < date_to,
        ghprt.pr_done_at >= date_from,
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
        match = Match[match_name]
        required_release_match = release_settings[prefix + repo]
        if required_release_match.match != Match.tag_or_branch:
            if match != required_release_match.match:
                return None
            dump = result
        else:
            dump = ambiguous[match_name]
        if match == Match.tag:
            target = required_release_match.tags
        elif match == Match.branch:
            target = required_release_match.branches.replace(
                default_branch_alias, default_branches.get(repo, default_branch_alias))
        else:
            raise AssertionError("Precomputed DB may not contain Match.tag_or_branch")
        if target != match_by:
            return None
    return dump


async def _fetch_rows_and_default_branches(selected: List[InstrumentedAttribute],
                                           filters: List[ClauseElement],
                                           repos: Collection[str],
                                           mdb: databases.Database,
                                           pdb: databases.Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> Tuple[List[Mapping], Dict[str, str]]:
    rows, dbt = await asyncio.gather(pdb.fetch_all(select(selected).where(and_(*filters))),
                                     extract_branches(repos, mdb, cache),
                                     return_exceptions=True)
    if isinstance(rows, Exception):
        raise rows from None
    if isinstance(dbt, Exception):
        raise dbt from None
    return rows, dbt[1]


@sentry_span
async def load_precomputed_done_candidates(date_from: datetime,
                                           date_to: datetime,
                                           repos: Collection[str],
                                           release_settings: Dict[str, ReleaseMatchSetting],
                                           mdb: databases.Database,
                                           pdb: databases.Database,
                                           cache: Optional[aiomcache.Client],
                                           ) -> Set[str]:
    """
    Load the set of released PR identifiers.

    We find all the released PRs for a given time frame, repositories, and release match settings.
    """
    ghprt = GitHubPullRequestTimes
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match]
    filters = _create_common_filters(date_from, date_to, repos)
    rows, default_branches = await _fetch_rows_and_default_branches(
        selected, filters, repos, mdb, pdb, cache)
    prefix = PREFIXES["github"]
    result = set()
    ambiguous = {Match.tag.name: set(), Match.branch.name: set()}
    for row in rows:
        dump = _check_release_match(
            row[1], row[2], release_settings, default_branches, prefix, result, ambiguous)
        if dump is None:
            continue
        dump.add(row[0])
    result.update(ambiguous[Match.tag.name])
    result.update(ambiguous[Match.branch.name])
    return result


@sentry_span
async def load_precomputed_done_times(date_from: datetime,
                                      date_to: datetime,
                                      repos: Collection[str],
                                      participants: Participants,
                                      exclude_inactive: bool,
                                      release_settings: Dict[str, ReleaseMatchSetting],
                                      mdb: databases.Database,
                                      pdb: databases.Database,
                                      cache: Optional[aiomcache.Client],
                                      ) -> Dict[str, PullRequestTimes]:
    """Load PullRequestTimes belonging to released or rejected PRs from the precomputed DB."""
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    ghprt = GitHubPullRequestTimes
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                ghprt.data]
    filters = _create_common_filters(date_from, date_to, repos)
    if len(participants) > 0:
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
    if exclude_inactive:
        # timezones: date_from and date_to may be not exactly 00:00
        date_from_day = datetime.combine(
            date_from.date(), datetime.min.time(), tzinfo=timezone.utc)
        date_to_day = datetime.combine(
            date_to.date(), datetime.min.time(), tzinfo=timezone.utc)
        # date_to_day will be included
        date_range = rrule(DAILY, dtstart=date_from_day, until=date_to_day)
        if postgres:
            filters.append(ghprt.activity_days.overlap(list(date_range)))
        else:
            selected.append(ghprt.activity_days)
            date_range = set(date_range)
    rows, default_branches = await _fetch_rows_and_default_branches(
        selected, filters, repos, mdb, pdb, cache)
    prefix = PREFIXES["github"]
    result = {}
    ambiguous = {Match.tag.name: {}, Match.branch.name: {}}
    for row in rows:
        dump = _check_release_match(
            row[1], row[2], release_settings, default_branches, prefix, result, ambiguous)
        if dump is None:
            continue
        if not postgres:
            if len(participants) > 0:
                passed = False
                for col, pk in ((ghprt.author, ParticipationKind.AUTHOR),
                                (ghprt.merger, ParticipationKind.MERGER),
                                (ghprt.releaser, ParticipationKind.RELEASER)):
                    dev = row[col.key]
                    if dev and dev in participants.get(pk, set()):
                        passed = True
                        break
                if not passed:
                    for col, pk in ((ghprt.reviewers, ParticipationKind.REVIEWER),
                                    (ghprt.commenters, ParticipationKind.COMMENTER),
                                    (ghprt.commit_authors, ParticipationKind.COMMIT_AUTHOR),
                                    (ghprt.commit_committers, ParticipationKind.COMMIT_COMMITTER)):
                        devs = set(row[col.key])
                        if devs.intersection(participants.get(pk, set())):
                            passed = True
                            break
                if not passed:
                    continue
            if exclude_inactive:
                activity_days = {datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                 for d in row[ghprt.activity_days.key]}
                if not activity_days.intersection(date_range):
                    continue
        dump[row[0]] = pickle.loads(row[3])
    result.update(ambiguous[Match.tag.name])
    for node_id, times in ambiguous[Match.branch.name].items():
        if node_id not in result:
            result[node_id] = times
    return result


@sentry_span
async def store_precomputed_done_times(prs: Iterable[MinedPullRequest],
                                       times: Iterable[PullRequestTimes],
                                       release_settings: Dict[str, ReleaseMatchSetting],
                                       mdb: databases.Database,
                                       pdb: databases.Database,
                                       cache: Optional[aiomcache.Client],
                                       ) -> None:
    """Store PullRequestTimes belonging to released or rejected PRs to the precomputed DB."""
    inserted = []
    prefix = PREFIXES["github"]
    repos = {pr.pr[PullRequest.repository_full_name.key] for pr in prs}
    _, default_branches = await extract_branches(repos, mdb, cache)
    sqlite = pdb.url.dialect == "sqlite"
    for pr, times in zip(prs, times):
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
            match = Match(pr.release[matched_by_column])
            if match == Match.branch:
                branch = release_match.branches.replace(
                    default_branch_alias, default_branches.get(repo, default_branch_alias))
                release_match = "|".join((match.name, branch))
            elif match == Match.tag:
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
            author=_flatten_set(participants[ParticipationKind.AUTHOR]),
            merger=_flatten_set(participants[ParticipationKind.MERGER]),
            releaser=_flatten_set(participants[ParticipationKind.RELEASER]),
            commenters={k: "" for k in participants[ParticipationKind.COMMENTER]},
            reviewers={k: "" for k in participants[ParticipationKind.REVIEWER]},
            commit_authors={k: "" for k in participants[ParticipationKind.COMMIT_AUTHOR]},
            commit_committers={k: "" for k in participants[ParticipationKind.COMMIT_COMMITTER]},
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
