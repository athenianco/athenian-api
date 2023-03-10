from dataclasses import dataclass
import pickle
from typing import Collection, Optional

import aiomcache
import medvedi as md
import numpy as np
from numpy import typing as npt
from sqlalchemy import func, select

from athenian.api.async_utils import read_sql_query
from athenian.api.cache import cached, middle_term_exptime
from athenian.api.db import DatabaseLike
from athenian.api.internal.miners.filters import LabelFilter
from athenian.api.models.metadata.github import PullRequestLabel
from athenian.api.tracing import sentry_span


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
async def mine_labels(
    repos: set[str],
    meta_ids: tuple[int, ...],
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> list[LabelDetails]:
    """Collect PR labels and count the number of PRs where they were used."""
    rows = await mdb.fetch_all(
        select(
            PullRequestLabel.name,
            func.min(PullRequestLabel.color).label("color"),
            func.max(PullRequestLabel.description).label("description"),
            func.count(PullRequestLabel.pull_request_node_id).label("used_prs"),
        )
        .where(
            PullRequestLabel.acc_id.in_(meta_ids),
            PullRequestLabel.repository_full_name.in_(repos),
        )
        .group_by(PullRequestLabel.name),
    )
    result = [
        LabelDetails(
            name=row[PullRequestLabel.name.name],
            color=row["color"],
            description=row["description"],
            used_prs=row["used_prs"],
        )
        for row in rows
    ]
    result.sort(key=lambda label: label.used_prs, reverse=True)
    return result


@sentry_span
async def fetch_labels_to_filter(
    prs: Collection[int],
    meta_ids: tuple[int, ...],
    mdb: DatabaseLike,
) -> md.DataFrame:
    """
    Load PR labels from mdb for filtering purposes.

    :return: DataFrame, the index is PR node IDs and the only column is lowercase label names.
    """
    in_values = len(prs) > 100
    lcols = [
        PullRequestLabel.pull_request_node_id,
        func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name),
    ]
    query = select(*lcols).where(
        PullRequestLabel.acc_id.in_(meta_ids),
        PullRequestLabel.pull_request_node_id.in_any_values(prs)
        if in_values
        else PullRequestLabel.pull_request_node_id.in_(prs),
    )
    if in_values:
        # fmt: off
        query = (
            query
            .with_statement_hint("Leading(*VALUES* prl label)")
            .with_statement_hint("Rows(*VALUES* prl label *100)")
        )
        # fmt: on
    else:
        query = query.with_statement_hint(f"Rows(prl label #{len(prs) // 10})")

    return await read_sql_query(
        query,
        mdb,
        lcols,
        index=PullRequestLabel.pull_request_node_id.name,
    )


def find_left_prs_by_labels(
    full_index: npt.NDArray[int],
    df_labels_index: npt.NDArray[int],
    df_labels_names: npt.NDArray[object],
    labels: LabelFilter,
) -> npt.NDArray[int]:
    """
    Filter PRs by their labels.

    :param full_index: All the PR node IDs, not just those that correspond to labeled PRs.
    :param df_labels_index: PR node IDs. There may be several rows for the same PR node ID.
    :param df_labels_names: (PR node ID, label name) DataFrame column.
    :return: PR node IDs that satisfy the filter.
    """
    assert full_index.dtype == int
    assert df_labels_index.dtype == int
    left_include = left_exclude = None
    if labels.include:
        singles, multiples = LabelFilter.split(labels.include)
        left_include = np.unique(df_labels_index[np.in1d(df_labels_names, singles)])
        for group in multiples:
            passed = df_labels_index
            for label in group:
                passed = np.intersect1d(
                    passed,
                    df_labels_index[df_labels_names == label],
                )
                if len(passed) == 0:
                    break
            left_include = np.union1d(left_include, passed)
    if labels.exclude:
        left_exclude = np.setdiff1d(
            full_index,
            df_labels_index[np.in1d(df_labels_names, list(labels.exclude))],
        )
    if labels.include:
        if labels.exclude:
            left = np.intersect1d(left_include, left_exclude)
        else:
            left = left_include
    else:
        left = left_exclude
    return left
