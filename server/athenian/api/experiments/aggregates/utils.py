from collections import defaultdict
from typing import List

from sqlalchemy import select

from athenian.api import ParallelDatabase
from athenian.api.experiments.aggregates.typing_utils import RepositoryCollection
from athenian.api.models.state.models import RepositorySet


async def get_accounts_and_repos(
    sdb: ParallelDatabase, accounts: List[int]
) -> RepositoryCollection:
    """Return the repositories associated with each account."""
    query = select(
        [
            RepositorySet.owner_id,
            RepositorySet.items,
        ]
    )

    if accounts:
        query = query.where(RepositorySet.owner_id.in_(accounts))

    account_repos = await sdb.fetch_all(query.order_by(RepositorySet.owner_id.asc()))
    grouped_repos = defaultdict(set)
    for record in account_repos:
        repos = [repo[len("github.com/") :] for repo in record.get(1)]
        grouped_repos[record.get(0)].update(repos)

    return grouped_repos
