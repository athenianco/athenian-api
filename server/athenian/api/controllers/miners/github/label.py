import pickle
from typing import List, Optional, Set, Tuple

import aiomcache
from sqlalchemy import func, select

from athenian.api.cache import cached
from athenian.api.models.metadata.github import PullRequestLabel
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike, dataclass


@dataclass(slots=True, frozen=True)
class LabelDetails:
    """Name, description, color, and used_prs - number of times the labels was used in PRs."""

    name: str
    description: Optional[str]
    color: str
    used_prs: int


@sentry_span
@cached(
    exptime=60 * 60,  # 1 hour
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
)
async def mine_labels(accounts: Tuple[int, ...],
                      repos: Set[str],
                      mdb: DatabaseLike,
                      cache: Optional[aiomcache.Client],
                      ) -> List[LabelDetails]:
    """Collect PR labels and count the number of PRs where they were used."""
    rows = await mdb.fetch_all(
        select([PullRequestLabel.name,
                func.min(PullRequestLabel.color).label("color"),
                func.max(PullRequestLabel.description).label("description"),
                func.count(PullRequestLabel.pull_request_node_id).label("used_prs")])
        .where(PullRequestLabel.repository_full_name.in_(repos))
        .group_by(PullRequestLabel.name))
    result = [LabelDetails(name=row[PullRequestLabel.name.key], color=row["color"],
                           description=row["description"], used_prs=row["used_prs"])
              for row in rows]
    result.sort(key=lambda label: label.used_prs, reverse=True)
    return result
