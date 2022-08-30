import argparse
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import func, select, text, update
from tqdm import tqdm

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.internal.miners.github.dag_accelerated import searchsorted_inrange
from athenian.api.models.metadata.github import NodePullRequest, PullRequestLabel
from athenian.api.models.precomputed.models import (
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
)
from athenian.api.precompute.context import PrecomputeContext


async def main(context: PrecomputeContext, args: argparse.Namespace) -> None:
    """Update the labels in the precomputed PRs."""
    log, mdb, pdb = context.log, context.mdb, context.pdb
    all_prs = await read_sql_query(
        select([NodePullRequest.id, NodePullRequest.acc_id]),
        mdb,
        [NodePullRequest.id, NodePullRequest.acc_id],
    )
    log.info("There are %d PRs in mdb", len(all_prs))
    all_node_ids = all_prs[NodePullRequest.id.name].values
    all_accounts = all_prs[NodePullRequest.acc_id.name].values
    del all_prs
    max_account = all_accounts.max()
    order = np.argsort(all_node_ids)
    all_node_ids = all_node_ids[order]
    all_accounts = all_accounts[order]
    del order
    all_pr_times_labels, all_pr_times_empty, all_merged_labels, all_merged_empty = await gather(
        read_sql_query(
            select(
                [
                    GitHubDonePullRequestFacts.acc_id,
                    GitHubDonePullRequestFacts.pr_node_id,
                    GitHubDonePullRequestFacts.labels,
                ],
            )
            .where(GitHubDonePullRequestFacts.labels != text("''::HSTORE"))
            .order_by(GitHubDonePullRequestFacts.acc_id),
            pdb,
            [
                GitHubDonePullRequestFacts.acc_id,
                GitHubDonePullRequestFacts.pr_node_id,
                GitHubDonePullRequestFacts.labels,
            ],
        ),
        read_sql_query(
            select(
                [
                    GitHubDonePullRequestFacts.acc_id,
                    GitHubDonePullRequestFacts.pr_node_id,
                ],
            )
            .where(GitHubDonePullRequestFacts.labels == text("''::HSTORE"))
            .order_by(GitHubDonePullRequestFacts.acc_id),
            pdb,
            [
                GitHubDonePullRequestFacts.acc_id,
                GitHubDonePullRequestFacts.pr_node_id,
            ],
        ),
        read_sql_query(
            select(
                [
                    GitHubMergedPullRequestFacts.acc_id,
                    GitHubMergedPullRequestFacts.pr_node_id,
                    GitHubMergedPullRequestFacts.labels,
                ],
            )
            .where(GitHubMergedPullRequestFacts.labels != text("''::HSTORE"))
            .order_by(GitHubMergedPullRequestFacts.acc_id),
            pdb,
            [
                GitHubMergedPullRequestFacts.acc_id,
                GitHubMergedPullRequestFacts.pr_node_id,
                GitHubMergedPullRequestFacts.labels,
            ],
        ),
        read_sql_query(
            select(
                [
                    GitHubMergedPullRequestFacts.acc_id,
                    GitHubMergedPullRequestFacts.pr_node_id,
                ],
            )
            .where(GitHubMergedPullRequestFacts.labels == text("''::HSTORE"))
            .order_by(GitHubMergedPullRequestFacts.acc_id),
            pdb,
            [
                GitHubMergedPullRequestFacts.acc_id,
                GitHubMergedPullRequestFacts.pr_node_id,
            ],
        ),
    )

    log.info(
        "Loaded %d+%d done and %d+%d merged precomputed PRs",
        len(all_pr_times_labels),
        len(all_pr_times_empty),
        len(all_merged_labels),
        len(all_merged_empty),
    )
    all_pr_times_empty[GitHubDonePullRequestFacts.labels.name] = np.full(
        len(all_pr_times_empty), {}, object,
    )
    all_merged_empty[GitHubMergedPullRequestFacts.labels.name] = np.full(
        len(all_merged_empty), {}, object,
    )
    unique_prs = np.unique(
        np.concatenate(
            [
                all_pr_times_labels[GitHubDonePullRequestFacts.pr_node_id.name].values,
                all_pr_times_empty[GitHubDonePullRequestFacts.pr_node_id.name].values,
                all_merged_labels[GitHubMergedPullRequestFacts.pr_node_id.name].values,
                all_merged_empty[GitHubMergedPullRequestFacts.pr_node_id.name].values,
            ],
        ),
    )
    found_account_indexes = searchsorted_inrange(all_node_ids, unique_prs)
    found_mask = all_node_ids[found_account_indexes] == unique_prs
    unique_prs = unique_prs[found_mask]
    unique_pr_acc_ids = all_accounts[found_account_indexes[found_mask]]
    del found_mask
    del found_account_indexes
    del all_node_ids
    del all_accounts
    if (prs_count := len(unique_prs)) == 0:
        return
    log.info("Querying labels in %d PRs", prs_count)
    order = np.argsort(unique_pr_acc_ids)
    unique_prs = unique_prs[order]
    unique_pr_acc_ids = unique_pr_acc_ids[order]
    del order
    unique_acc_ids, acc_id_counts = np.unique(unique_pr_acc_ids, return_counts=True)
    del unique_pr_acc_ids
    node_id_by_acc_id = np.split(unique_prs, np.cumsum(acc_id_counts))
    del unique_prs
    del acc_id_counts
    tasks = [
        read_sql_query(
            select(
                [
                    PullRequestLabel.pull_request_node_id,
                    func.lower(PullRequestLabel.name).label(PullRequestLabel.name.name),
                ],
            ).where(
                PullRequestLabel.pull_request_node_id.in_(node_ids),
                PullRequestLabel.acc_id == int(acc_id),
            ),
            mdb,
            [PullRequestLabel.pull_request_node_id, PullRequestLabel.name],
        )
        for acc_id, node_ids in zip(unique_acc_ids, node_id_by_acc_id)
    ]
    del node_id_by_acc_id
    dfs_acc_counts = [np.zeros(max_account + 1, dtype=int) for _ in range(4)]
    for i, col in enumerate(
        (
            all_pr_times_labels[GitHubDonePullRequestFacts.acc_id.name],
            all_pr_times_empty[GitHubDonePullRequestFacts.acc_id.name],
            all_merged_labels[GitHubMergedPullRequestFacts.acc_id.name],
            all_merged_empty[GitHubMergedPullRequestFacts.acc_id.name],
        ),
    ):
        accs, counts = np.unique(col.values, return_counts=True)
        dfs_acc_counts[i][accs] = counts
    batch_size = 20
    update_tasks = []
    for batch in tqdm(
        range(0, len(tasks), batch_size), total=(len(tasks) + batch_size - 1) // batch_size,
    ):
        dfs = await gather(*tasks[batch : batch + batch_size])
        actual_labels = defaultdict(dict)
        for df in dfs:
            for pr_node_id, label in zip(
                df[PullRequestLabel.pull_request_node_id.name].values,
                df[PullRequestLabel.name.name].values,
            ):
                actual_labels[pr_node_id][label] = ""
        log.info("Loaded labels for %d PRs", len(actual_labels))
        acc_ids = unique_acc_ids[batch : batch + batch_size]
        for (df, model), df_acc_size in zip(
            (
                (all_pr_times_labels, GitHubDonePullRequestFacts),
                (all_pr_times_empty, GitHubDonePullRequestFacts),
                (all_merged_labels, GitHubMergedPullRequestFacts),
                (all_merged_empty, GitHubMergedPullRequestFacts),
            ),
            dfs_acc_counts,
        ):
            indexes = np.searchsorted(df[model.acc_id.name].values, [acc_ids[0], acc_ids[-1]])
            indexes = slice(indexes[0], indexes[1] + df_acc_size[acc_ids[-1]])
            for pr_node_id, labels in zip(
                df[model.pr_node_id.name].values[indexes],
                df[model.labels.name].values[indexes],
            ):
                assert isinstance(labels, dict)
                if (pr_labels := actual_labels.get(pr_node_id, {})) != labels:
                    update_tasks.append(
                        pdb.execute(
                            update(model)
                            .where(model.pr_node_id == pr_node_id)
                            .values(
                                {
                                    model.labels: pr_labels,
                                    model.updated_at: datetime.now(timezone.utc),
                                },
                            ),
                        ),
                    )
    tasks = update_tasks
    if not tasks:
        return
    log.info("Updating %d records", len(tasks))
    batch_size = 1000
    bar = tqdm(total=len(tasks))
    try:
        while tasks:
            batch, tasks = tasks[:batch_size], tasks[batch_size:]
            await gather(*batch)
            bar.update(len(batch))
    finally:
        bar.close()
