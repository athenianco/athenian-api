from datetime import datetime, timezone
from itertools import chain
import pickle
from typing import Any, Collection, Dict, Iterable, Optional

import aiomcache
import databases
from dateutil.rrule import DAILY, rrule
from sqlalchemy import and_, insert, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.pull_request import MinedPullRequest, PullRequestTimes
from athenian.api.controllers.miners.github.release import matched_by_column
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind
from athenian.api.controllers.settings import default_branch_alias, Match, ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest, PullRequestComment, \
    PullRequestCommit, PullRequestReview, PullRequestReviewRequest
from athenian.api.models.precomputed.models import GitHubPullRequestTimes


async def load_precomputed_done_times(date_from: datetime,
                                      date_to: datetime,
                                      repos: Collection[str],
                                      developers: Collection[str],
                                      exclude_inactive: bool,
                                      release_settings: Dict[str, ReleaseMatchSetting],
                                      mdb: databases.Database,
                                      pdb: databases.Database,
                                      cache: Optional[aiomcache.Client],
                                      ) -> Dict[str, PullRequestTimes]:
    """Load PullRequestTimes belonging to released or rejected PRs from the precomputed DB."""
    assert isinstance(date_from, datetime)
    assert isinstance(date_to, datetime)
    postgres = pdb.url.dialect in ("postgres", "postgresql")
    ghprt = GitHubPullRequestTimes
    selected = [ghprt.pr_node_id,
                ghprt.repository_full_name,
                ghprt.release_match,
                ghprt.data]
    filters = [
        ghprt.format_version == ghprt.__table__.columns[ghprt.format_version.key].default.arg,
        ghprt.repository_full_name.in_(repos),
        ghprt.pr_created_at < date_to,
        ghprt.pr_done_at >= date_from,
    ]
    if len(developers) > 0:
        if postgres:
            developer_filters = [
                ghprt.author.in_(developers),
                ghprt.merger.in_(developers),
                ghprt.releaser.in_(developers),
                ghprt.commenters.has_any(developers),
                ghprt.reviewers.has_any(developers),
                ghprt.commit_authors.has_any(developers),
                ghprt.commit_committers.has_any(developers),
            ]
            # do not send the same array 3 times
            for f in developer_filters[1:3]:
                f.right = developer_filters[0].right
            # do not send the same array 4 times
            for f in developer_filters[4:]:
                f.right = developer_filters[3].right
            filters.append(or_(*developer_filters))
        else:
            selected.extend([
                ghprt.author, ghprt.merger, ghprt.releaser, ghprt.reviewers, ghprt.commenters,
                ghprt.commit_authors, ghprt.commit_committers])
            developers = set(developers)
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
    rows = await pdb.fetch_all(select(selected).where(and_(*filters)))
    prefix = PREFIXES["github"]
    result = {}
    ambiguous = {Match.tag.name: {}, Match.branch.name: {}}
    _, default_branches = await extract_branches(repos, mdb, cache)
    for row in rows:
        node_id, repo, release_match, data = row[0], row[1], row[2], row[3]
        if release_match == "rejected":
            dump = result
        else:
            match_name, match_by = release_match.split("|", 1)
            match = Match[match_name]
            required_release_match = release_settings[prefix + repo]
            if required_release_match.match != Match.tag_or_branch:
                if match != required_release_match.match:
                    continue
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
                continue
        if not postgres:
            if len(developers) > 0:
                pr_parts = set(chain(row[4:7], *row[7:11]))  # sqlite Record supports slices
                if not pr_parts.intersection(developers):
                    continue
            if exclude_inactive:
                activity_days = {datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                                 for d in row[ghprt.activity_days.key]}
                if not activity_days.intersection(date_range):
                    continue
        dump[node_id] = pickle.loads(data)
    result.update(ambiguous[Match.tag.name])
    for node_id, times in ambiguous[Match.branch.name].items():
        if node_id not in result:
            result[node_id] = times
    return result


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
        participants = pr.participants(with_prefix=False)
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
