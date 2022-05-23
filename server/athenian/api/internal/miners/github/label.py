import pickle
from typing import Collection, List, Optional, Sequence, Set, Tuple

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select

from athenian.api.async_utils import read_sql_query
from athenian.api.cache import cached, middle_term_exptime
from athenian.api.db import DatabaseLike
from athenian.api.internal.miners.filters import LabelFilter
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
    exptime=middle_term_exptime,
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


@sentry_span
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
                    PullRequestLabel.acc_id.in_(meta_ids)))
        .with_statement_hint(f"Rows(prl label #{len(prs)})"),
        mdb, lcols, index=PullRequestLabel.pull_request_node_id.name)


def find_left_prs_by_labels(full_index: pd.Index,
                            df_labels_index: pd.Index,
                            df_labels_names: Sequence[str],
                            labels: LabelFilter) -> pd.Index:
    """
    Filter PRs by their labels.

    :param full_index: All the PR node IDs, not just those that correspond to labeled PRs.
    :param df_labels_index: (PR node ID, label name) DataFrame index. There may be several \
                            rows for the same PR node ID.
    :param df_labels_names: (PR node ID, label name) DataFrame column.
    """
    left_include = left_exclude = None
    if labels.include:
        singles, multiples = LabelFilter.split(labels.include)
        left_include = df_labels_index.take(
            np.nonzero(np.in1d(df_labels_names, singles))[0],
        ).unique()
        for group in multiples:
            passed = df_labels_index
            for label in group:
                passed = passed.intersection(
                    df_labels_index.take(np.nonzero(df_labels_names == label)[0]))
                if passed.empty:
                    break
            left_include = left_include.union(passed)
    if labels.exclude:
        left_exclude = full_index.difference(df_labels_index.take(
            np.nonzero(np.in1d(df_labels_names, list(labels.exclude)))[0],
        ).unique())
    if labels.include:
        if labels.exclude:
            left = left_include.intersection(left_exclude)
        else:
            left = left_include
    else:
        left = left_exclude
    return left
