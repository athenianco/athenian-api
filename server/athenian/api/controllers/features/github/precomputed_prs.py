from datetime import datetime
import pickle
from typing import Collection, Dict, Iterable

import databases
from sqlalchemy import and_, insert, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api.controllers.miners.github.pull_request import MinedPullRequest, PullRequestTimes
from athenian.api.controllers.miners.github.release import matched_by_column
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.precomputed.models import GitHubPullRequestTimes


async def load_precomputed_done_times(date_from: datetime,
                                      date_to: datetime,
                                      repos: Collection[str],
                                      developers: Collection[str],
                                      release_settings: Dict[str, ReleaseMatchSetting],
                                      db: databases.Database,
                                      ) -> Dict[str, PullRequestTimes]:
    """Load PullRequestTimes belonging to released or rejected PRs from the precomputed DB."""
    assert isinstance(date_from, datetime)
    assert isinstance(date_to, datetime)
    postgres = db.url.dialect in ("postgres", "postgresql")
    selected = [GitHubPullRequestTimes.pr_node_id,
                GitHubPullRequestTimes.repository_full_name,
                GitHubPullRequestTimes.release_match,
                GitHubPullRequestTimes.data]
    filters = [GitHubPullRequestTimes.format_version == 1,
               GitHubPullRequestTimes.repository_full_name.in_(repos),
               GitHubPullRequestTimes.pr_created_at < date_to,
               GitHubPullRequestTimes.pr_done_at >= date_from]
    if len(developers) > 0:
        if postgres:
            filters.append(GitHubPullRequestTimes.developers.has_any(developers))
        else:
            selected.append(GitHubPullRequestTimes.developers)
            developers = set(developers)
    rows = await db.fetch_all(select(selected).where(and_(*filters)))
    prefix = PREFIXES["github"]
    result = {}
    ambiguous = {Match.tag.name: {}, Match.branch.name: {}}
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
        if not postgres and len(developers) > 0:
            if not set(row[4]).intersection(developers):
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
                                       db: databases.Database,
                                       ) -> None:
    """Store PullRequestTimes belonging to released or rejected PRs to the precomputed DB."""
    inserted = []
    prefix = PREFIXES["github"]
    for pr, times in zip(prs, times):
        if not times.released:
            if not times.closed or times.merged:
                continue
            done_at = times.closed.best
        else:
            done_at = times.released.best
        repo = pr.pr[PullRequest.repository_full_name.key]
        if pr.release[matched_by_column] is not None:
            release_match = release_settings[prefix + repo]
            match = Match(pr.release[matched_by_column])
            if match == Match.branch:
                release_match = "|".join((match.name, release_match.branches))
            elif match == Match.tag:
                release_match = "|".join((match.name, release_match.tags))
            else:
                raise AssertionError("Unhandled release match strategy: " + match.name)
        else:
            release_match = "rejected"
        participants = {}
        for kind, people in sorted(pr.participants(with_prefix=False).items()):
            for p in people:
                participants.setdefault(p, []).append(str(kind.value))
        participants = {k: ",".join(v) for k, v in participants.items()}
        inserted.append(GitHubPullRequestTimes(
            pr_node_id=pr.pr[PullRequest.node_id.key],
            release_match=release_match,
            repository_full_name=repo,
            pr_created_at=times.created.best,
            pr_done_at=done_at,
            developers=participants,
            data=pickle.dumps(times),
        ).create_defaults().explode(with_primary_keys=True))
    if db.url.dialect in ("postgres", "postgresql"):
        sql = postgres_insert(GitHubPullRequestTimes).on_conflict_do_nothing()
    else:
        sql = insert(GitHubPullRequestTimes)
    await db.execute_many(sql, inserted)
