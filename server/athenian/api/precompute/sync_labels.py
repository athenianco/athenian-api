import argparse
from collections import defaultdict
from datetime import datetime, timezone
from itertools import chain

import numpy as np
from sqlalchemy import and_, func, select, update
from tqdm import tqdm

from athenian.api.async_utils import gather
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
    tasks = []
    all_prs = await mdb.fetch_all(select([NodePullRequest.id, NodePullRequest.acc_id]))
    log.info("There are %d PRs in mdb", len(all_prs))
    all_node_ids = np.fromiter((pr[0] for pr in all_prs), int, len(all_prs))
    all_accounts = np.fromiter((pr[1] for pr in all_prs), np.uint32, len(all_prs))
    del all_prs
    order = np.argsort(all_node_ids)
    all_node_ids = all_node_ids[order]
    all_accounts = all_accounts[order]
    del order
    all_pr_times = await pdb.fetch_all(
        select([GitHubDonePullRequestFacts.pr_node_id, GitHubDonePullRequestFacts.labels]),
    )
    all_merged = await pdb.fetch_all(
        select([GitHubMergedPullRequestFacts.pr_node_id, GitHubMergedPullRequestFacts.labels]),
    )
    unique_prs = np.unique(np.array([pr[0] for pr in chain(all_pr_times, all_merged)]))
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
    for acc_id, node_ids in zip(unique_acc_ids, node_id_by_acc_id):
        tasks.append(
            mdb.fetch_all(
                select(
                    [PullRequestLabel.pull_request_node_id, func.lower(PullRequestLabel.name)],
                ).where(
                    and_(
                        PullRequestLabel.pull_request_node_id.in_(node_ids),
                        PullRequestLabel.acc_id == int(acc_id),
                    ),
                ),
            ),
        )
    del unique_acc_ids
    del node_id_by_acc_id
    task_results = await gather(*tasks)
    actual_labels = defaultdict(dict)
    for row in chain.from_iterable(task_results):
        actual_labels[row[0]][row[1]] = ""
    log.info("Loaded labels for %d PRs", len(actual_labels))
    tasks = []
    for rows, model in (
        (all_pr_times, GitHubDonePullRequestFacts),
        (all_merged, GitHubMergedPullRequestFacts),
    ):
        for row in rows:
            assert isinstance(row[1], dict)
            if (pr_labels := actual_labels.get(row[0], {})) != row[1]:
                tasks.append(
                    pdb.execute(
                        update(model)
                        .where(model.pr_node_id == row[0])
                        .values(
                            {
                                model.labels: pr_labels,
                                model.updated_at: datetime.now(timezone.utc),
                            },
                        ),
                    ),
                )
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
