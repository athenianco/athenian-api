from collections import defaultdict
from datetime import timedelta
import re
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from sqlalchemy import and_, join, select, sql

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.db import (
    Database,
    add_pdb_hits,
    add_pdb_misses,
    greatest,
    insert_or_ignore,
    strpos,
)
from athenian.api.defer import defer
from athenian.api.internal.miners.github.precomputed_prs.dead_prs import (
    drop_undead_duplicates,
    store_undead_prs,
)
from athenian.api.models.metadata.github import NodeCommit, NodePullRequest
from athenian.api.models.precomputed.models import (
    GitHubRebaseCheckedCommit,
    GitHubRebasedPullRequest,
)
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import unordered_unique

commit_message_substr_len = 32
jira_key_re = re.compile(r"[A-Z][A-Z\d]+-[1-9]\d*")


async def first_line_of_commit_message(db: Database):
    """Select the first line of the commit message."""
    col = sql.func.substr(
        NodeCommit.message,
        1,
        sql.func.coalesce(
            sql.func.nullif((await strpos(db))(NodeCommit.message, "\n") - 1, -1),
            sql.func.length(NodeCommit.message),
        ),
    )
    return col


@sentry_span
async def match_rebased_prs(
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    commit_ids: Optional[npt.NDArray[int]] = None,
    commit_shas: Optional[npt.NDArray[bytes]] = None,
) -> pd.DataFrame:
    """Perform inverse rebased PR matching: from alive commits to PRs."""
    assert commit_ids is not None or commit_shas is not None
    if commit_ids is not None:
        commit_ids = unordered_unique(commit_ids)
        if len(commit_ids) == 0:
            return pd.DataFrame()
    else:
        commit_shas = unordered_unique(commit_shas)
        if len(commit_shas) == 0:
            return pd.DataFrame()
    batch_size = 100_000

    @sentry_span
    async def fetch_precomputed_rebased_prs() -> pd.DataFrame:
        tasks = []
        for p in range(0, len(commit_ids if commit_ids is not None else commit_shas), batch_size):
            batch = slice(p * batch_size, (p + 1) * batch_size)
            tasks.append(
                read_sql_query(
                    select(GitHubRebasedPullRequest).where(
                        GitHubRebasedPullRequest.acc_id == account,
                        GitHubRebasedPullRequest.matched_merge_commit_id.in_(commit_ids[batch])
                        if commit_ids is not None
                        else GitHubRebasedPullRequest.matched_merge_commit_sha.in_(
                            commit_shas[batch],
                        ),
                    ),
                    pdb,
                    GitHubRebasedPullRequest,
                ),
            )
        prs = await gather(*tasks, op="fetch_precomputed_rebased_prs/sql")
        if len(prs) == 1:
            prs = prs[0]
        else:
            prs = pd.concat(prs, ignore_index=True)
        prs = drop_undead_duplicates(prs)
        del prs[GitHubRebasedPullRequest.acc_id.name]
        del prs[GitHubRebasedPullRequest.updated_at.name]
        add_pdb_hits(pdb, "rebased_prs", len(prs))
        return prs

    @sentry_span
    async def fetch_checked_commits() -> npt.NDArray[int | bytes]:
        tasks = []
        for p in range(0, len(commit_ids if commit_ids is not None else commit_shas), batch_size):
            batch = slice(p * batch_size, (p + 1) * batch_size)
            column = (
                GitHubRebaseCheckedCommit.node_id
                if commit_ids is not None
                else GitHubRebaseCheckedCommit.sha
            )
            tasks.append(
                read_sql_query(
                    select(column).where(
                        GitHubRebaseCheckedCommit.acc_id == account,
                        GitHubRebaseCheckedCommit.node_id.in_(commit_ids[batch])
                        if commit_ids is not None
                        else GitHubRebaseCheckedCommit.sha.in_(commit_shas[batch]),
                    ),
                    pdb,
                    [column],
                ),
            )
        checked = await gather(*tasks, op="fetch_checked_commits/sql")
        if len(checked) == 1:
            checked = checked[0][column.name].values
        else:
            checked = np.concatenate([df[column.name].values for df in checked], casting="unsafe")
        add_pdb_hits(pdb, "rebase_checked_commits", len(checked))
        return checked

    precomputed_rebased_prs, checked_commits = await gather(
        fetch_precomputed_rebased_prs(),
        fetch_checked_commits(),
    )
    if not precomputed_rebased_prs.empty:
        known_commits = np.concatenate(
            [
                precomputed_rebased_prs[
                    (
                        GitHubRebasedPullRequest.matched_merge_commit_id
                        if commit_ids is not None
                        else GitHubRebasedPullRequest.matched_merge_commit_sha
                    ).name
                ].values,
                checked_commits,
            ],
        )
    else:
        known_commits = checked_commits
    if commit_ids is not None:
        commit_ids = np.setdiff1d(commit_ids, known_commits, assume_unique=True)
        left_count = len(commit_ids)
    else:
        commit_shas = np.setdiff1d(commit_shas, known_commits, assume_unique=True)
        left_count = len(commit_shas)
    add_pdb_misses(pdb, "rebase_checked_commits", left_count)
    if left_count == 0:
        add_pdb_misses(pdb, "rebased_prs", 0)
        return precomputed_rebased_prs

    new_prs = await _match_rebased_prs_from_scratch(
        account, meta_ids, mdb, pdb, commit_ids=commit_ids, commit_shas=commit_shas,
    )
    if not new_prs.empty:
        rebased_prs = pd.concat(
            [new_prs, precomputed_rebased_prs[new_prs.columns]], ignore_index=True,
        )
        del new_prs
    else:
        rebased_prs = precomputed_rebased_prs
    rebased_prs = drop_undead_duplicates(rebased_prs)
    add_pdb_misses(pdb, "rebased_prs", len(rebased_prs) - len(precomputed_rebased_prs))
    return rebased_prs


@sentry_span
async def _match_rebased_prs_from_scratch(
    account: int,
    meta_ids: tuple[int, ...],
    mdb: Database,
    pdb: Database,
    commit_ids: Optional[npt.NDArray[int]] = None,
    commit_shas: Optional[npt.NDArray[bytes]] = None,
) -> pd.DataFrame:
    tasks = []
    batch_size = 100_000
    for p in range(0, len(commit_ids if commit_ids is not None else commit_shas), batch_size):
        batch = slice(p * batch_size, (p + 1) * batch_size)
        tasks.append(
            read_sql_query(
                select(
                    NodeCommit.node_id,
                    NodeCommit.sha,
                    NodeCommit.committed_date,
                    NodeCommit.pushed_date,
                    NodeCommit.repository_id,
                    sql.func.substr(
                        NodeCommit.message,
                        1,
                        (await greatest(mdb))(
                            sql.func.coalesce(
                                sql.func.nullif(
                                    (await strpos(mdb))(NodeCommit.message, "\n") - 1, -1,
                                ),
                                sql.func.length(NodeCommit.message),
                            ),
                            commit_message_substr_len,
                        ),
                    ).label(NodeCommit.message.name),
                ).where(
                    NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.node_id.in_(commit_ids[batch])
                    if commit_ids is not None
                    else NodeCommit.sha.in_(commit_shas[batch]),
                ),
                mdb,
                [
                    NodeCommit.node_id,
                    NodeCommit.sha,
                    NodeCommit.committed_date,
                    NodeCommit.pushed_date,
                    NodeCommit.repository_id,
                    NodeCommit.message,
                ],
            ),
        )
    searched_commits = await gather(*tasks, op="_match_rebased_prs_from_scratch/searched_commits")
    del commit_ids
    del commit_shas
    if len(searched_commits) == 1:
        searched_commits = searched_commits[0]
    else:
        searched_commits = pd.concat(searched_commits, ignore_index=True)
    prefixes = []
    message_commit_map = defaultdict(lambda: defaultdict(list))
    message_pr_merge_map = defaultdict(lambda: defaultdict(list))
    merge_message_re = re.compile(r"Merge pull request #(\d+) from ")
    extra_prs = defaultdict(list)
    for i, (msg, repo_id) in enumerate(
        zip(
            searched_commits[NodeCommit.message.name].values,
            searched_commits[NodeCommit.repository_id.name].values,
        ),
    ):
        prefixes.append(msg[:commit_message_substr_len])
        if (nl := msg.find("\n")) >= 0:
            msg = msg[:nl]
        message_commit_map[repo_id][msg].append(i)
        if match := merge_message_re.match(msg):
            extra_prs[repo_id].append(int(match.group(1)))
            message_pr_merge_map[repo_id][match.group(0)].append(i)

    tasks = [
        read_sql_query(
            select(
                NodePullRequest.node_id,
                NodePullRequest.number,
                NodePullRequest.merge_commit_id,
                NodePullRequest.merged_at,
                NodePullRequest.repository_id,
                (await first_line_of_commit_message(mdb)).label(NodeCommit.message.name),
            )
            .select_from(
                join(
                    NodePullRequest,
                    NodeCommit,
                    and_(
                        NodePullRequest.acc_id == NodeCommit.acc_id,
                        NodePullRequest.merge_commit_id == NodeCommit.node_id,
                    ),
                ),
            )
            .where(
                NodeCommit.acc_id.in_(meta_ids),
                NodeCommit.repository_id.in_(message_commit_map),
                sql.func.substr(NodeCommit.message, 1, commit_message_substr_len).in_(
                    prefixes[p * batch_size : (p + 1) * batch_size],
                ),
            ),
            mdb,
            [
                NodePullRequest.node_id,
                NodePullRequest.number,
                NodePullRequest.merge_commit_id,
                NodePullRequest.merged_at,
                NodePullRequest.repository_id,
                NodeCommit.message,
            ],
        )
        for p in range(0, len(prefixes), batch_size)
    ] + [
        read_sql_query(
            select(
                NodePullRequest.node_id,
                NodePullRequest.number,
                NodePullRequest.merge_commit_id,
                NodePullRequest.merged_at,
                NodePullRequest.repository_id,
                (await first_line_of_commit_message(mdb)).label(NodeCommit.message.name),
            )
            .select_from(
                join(
                    NodePullRequest,
                    NodeCommit,
                    and_(
                        NodePullRequest.acc_id == NodeCommit.acc_id,
                        NodePullRequest.merge_commit_id == NodeCommit.node_id,
                        sql.func.substr(NodeCommit.message, 1, commit_message_substr_len).notlike(
                            "Merge pull request #% from %",
                        ),
                    ),
                ),
            )
            .where(
                NodePullRequest.acc_id.in_(meta_ids),
                NodePullRequest.repository_id == repo_id,
                NodePullRequest.number.in_(numbers),
            ),
            mdb,
            [
                NodePullRequest.node_id,
                NodePullRequest.number,
                NodePullRequest.merge_commit_id,
                NodePullRequest.merged_at,
                NodePullRequest.repository_id,
                NodeCommit.message,
            ],
        )
        for repo_id, numbers in extra_prs.items()
    ]

    rough_matches = await gather(*tasks, op="_match_rebased_prs_from_scratch/rough_matches")
    if len(rough_matches) == 1:
        rough_matches = rough_matches[0]
    else:
        rough_matches = pd.concat(rough_matches, ignore_index=True)
    if rough_matches.empty:
        add_pdb_misses(pdb, "rebased_prs", 0)
        return pd.DataFrame()
    matched_pr_node_ids = []
    matched_commits = []
    searched_commit_ids = searched_commits[NodeCommit.node_id.name].values
    searched_committed_ats = searched_commits[NodeCommit.committed_date.name].values
    for pr_node_id, pr_number, merge_commit_id, merged_at, repo_id, message in zip(
        rough_matches[NodePullRequest.node_id.name].values,
        rough_matches[NodePullRequest.number.name].values,
        rough_matches[NodePullRequest.merge_commit_id.name].values,
        rough_matches[NodePullRequest.merged_at.name].values,
        rough_matches[NodePullRequest.repository_id.name].values,
        rough_matches[NodeCommit.message.name].values,
    ):
        if not (f"#{pr_number}" in message or jira_key_re.search(message)):
            # last resort: there are possible merge commits which changed the message
            message = f"Merge pull request #{pr_number} from "
            lookup_map = message_pr_merge_map
        else:
            lookup_map = message_commit_map
        if indexes := lookup_map[repo_id][message]:
            if (searched_commit_ids[indexes] != merge_commit_id).all():
                indexes = np.array(indexes)[
                    searched_committed_ats[indexes]
                    > (pd.Timestamp(merged_at) - timedelta(seconds=10)).to_numpy()
                ]
                matched_commits.extend(indexes)
                matched_pr_node_ids.extend(pr_node_id for _ in indexes)
    if not matched_commits:
        add_pdb_misses(pdb, "rebased_prs", 0)
        return pd.DataFrame()
    unmatched_mask = np.ones(len(searched_commits), dtype=bool)
    if matched_commits:
        unmatched_mask[matched_commits] = False
    await defer(
        _store_rebase_checked_commits(
            searched_commits[NodeCommit.node_id.name].values[unmatched_mask],
            searched_commits[NodeCommit.sha.name].values[unmatched_mask],
            account,
            pdb,
        ),
        "_store_rebase_checked_commits",
    )
    extra_rebased_prs = searched_commits[
        [
            c.name
            for c in (
                NodeCommit.node_id,
                NodeCommit.sha,
                NodeCommit.committed_date,
                NodeCommit.pushed_date,
            )
        ]
    ].take(matched_commits)
    extra_rebased_prs.columns = [
        c.name
        for c in (
            GitHubRebasedPullRequest.matched_merge_commit_id,
            GitHubRebasedPullRequest.matched_merge_commit_sha,
            GitHubRebasedPullRequest.matched_merge_commit_committed_date,
            GitHubRebasedPullRequest.matched_merge_commit_pushed_date,
        )
    ]
    extra_rebased_prs[GitHubRebasedPullRequest.pr_node_id.name] = matched_pr_node_ids
    await defer(store_undead_prs(extra_rebased_prs, account, pdb), "match_rebased_prs/store_extra")
    return extra_rebased_prs


async def _store_rebase_checked_commits(
    ids: npt.NDArray[int],
    shas: npt.NDArray[bytes],
    account: int,
    pdb: Database,
) -> None:
    inserted = [
        GitHubRebaseCheckedCommit(
            node_id=node_id,
            sha=sha.decode(),
            acc_id=account,
        )
        .create_defaults()
        .explode(with_primary_keys=True)
        for node_id, sha in zip(ids, shas)
    ]
    await insert_or_ignore(
        GitHubRebaseCheckedCommit, inserted, "_store_rebase_checked_commits", pdb,
    )
