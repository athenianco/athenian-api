from datetime import timezone
from typing import Collection

import numpy as np
import pandas as pd
from sqlalchemy import sql

from athenian.api.async_utils import read_sql_query
from athenian.api.db import Database, add_pdb_hits, add_pdb_misses, insert_or_ignore
from athenian.api.models.precomputed.models import GitHubRebasedPullRequest


def drop_undead_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate matches of rebased PRs according to the time heuristic."""
    df.sort_values(
        [
            GitHubRebasedPullRequest.matched_merge_commit_committed_date.name,
            GitHubRebasedPullRequest.matched_merge_commit_pushed_date.name,
        ],
        ascending=False,
        inplace=True,
        na_position="first",
    )
    df.drop_duplicates(GitHubRebasedPullRequest.pr_node_id.name, inplace=True)
    return df


async def load_undead_prs(
    suspects: Collection[int],
    account: int,
    pdb: Database,
) -> pd.DataFrame:
    """Fetch rebased PRs that are known to exist in the new DAG."""
    df = await read_sql_query(
        sql.select(GitHubRebasedPullRequest).where(
            GitHubRebasedPullRequest.acc_id == account,
            GitHubRebasedPullRequest.pr_node_id.in_(suspects),
        ),
        pdb,
        GitHubRebasedPullRequest,
    )
    if pdb.url.dialect == "sqlite":
        df[GitHubRebasedPullRequest.acc_id.name] = df[
            GitHubRebasedPullRequest.acc_id.name
        ].values.astype(np.int32)
    df = drop_undead_duplicates(df)
    add_pdb_hits(pdb, "dead_prs", len(df))
    add_pdb_misses(pdb, "dead_prs", len(suspects) - len(df))
    return df


async def store_undead_prs(
    df: pd.DataFrame,
    account: int,
    pdb: Database,
):
    """Persist the rebased PR matches to the pdb."""
    values = [
        GitHubRebasedPullRequest(
            acc_id=account,
            pr_node_id=pr_node_id,
            matched_merge_commit_id=matched_merge_commit_id,
            matched_merge_commit_sha=matched_merge_commit_sha.decode(),
            matched_merge_commit_committed_date=pd.Timestamp(
                matched_merge_commit_committed_date, tzinfo=timezone.utc,
            ),
            matched_merge_commit_pushed_date=pd.Timestamp(
                matched_merge_commit_pushed_date, tzinfo=timezone.utc,
            )
            if matched_merge_commit_pushed_date == matched_merge_commit_pushed_date
            else None,
        )
        .create_defaults()
        .explode(with_primary_keys=True)
        for (
            pr_node_id,
            matched_merge_commit_id,
            matched_merge_commit_sha,
            matched_merge_commit_committed_date,
            matched_merge_commit_pushed_date,
        ) in zip(
            df[GitHubRebasedPullRequest.pr_node_id.name].values,
            df[GitHubRebasedPullRequest.matched_merge_commit_id.name].values,
            df[GitHubRebasedPullRequest.matched_merge_commit_sha.name].values,
            df[GitHubRebasedPullRequest.matched_merge_commit_committed_date.name].values,
            df[GitHubRebasedPullRequest.matched_merge_commit_pushed_date.name].values,
        )
    ]
    await insert_or_ignore(GitHubRebasedPullRequest, values, "store_undead_prs", pdb)
