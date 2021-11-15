import pickle
from typing import Collection, List, Optional, Set, Tuple

import aiomcache
import pandas as pd
from sqlalchemy import and_, func, select

from athenian.api.async_utils import read_sql_query
from athenian.api.cache import cached
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import PullRequestLabel
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass


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
async def mine_labels(repos: Set[str],
                      meta_ids: Tuple[int, ...],
                      mdb: DatabaseLike,
                      cache: Optional[aiomcache.Client],
                      ) -> List[LabelDetails]:
    """Collect PR labels and count the number of PRs where they were used."""
    rows = await mdb.fetch_all(
        select([PullRequestLabel.name,
                func.min(PullRequestLabel.color).label("color"),
                func.max(PullRequestLabel.description).label("description"),
                func.count(PullRequestLabel.pull_request_node_id).label("used_prs")])
        .where(and_(PullRequestLabel.repository_full_name.in_(repos),
                    PullRequestLabel.acc_id.in_(meta_ids)))
        .group_by(PullRequestLabel.name))
    result = [LabelDetails(name=row[PullRequestLabel.name.name], color=row["color"],
                           description=row["description"], used_prs=row["used_prs"])
              for row in rows]
    result.sort(key=lambda label: label.used_prs, reverse=True)
    return result


async def fetch_labels_to_filter(prs: Collection[int],
                                 meta_ids: Tuple[int, ...],
                                 mdb: DatabaseLike,
                                 ) -> pd.DataFrame:
    """
    Load PR labels from mdb for filtering purposes.

    :return: DataFrame, the index is PR node IDs and the only column is lowercase label names.
    """
    lcols = [
        PullRequestLabel.pull_request_node_id,
        func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name),
    ]
    return await read_sql_query(
        select(lcols)
        .where(and_(PullRequestLabel.pull_request_node_id.in_(prs),
                    PullRequestLabel.acc_id.in_(meta_ids))),
        mdb, lcols, index=PullRequestLabel.pull_request_node_id.name)
