"""Tools to easily execute complex creations and insertions of DB objects."""

from bisect import bisect_right
from collections.abc import Collection
from datetime import datetime, timezone
from typing import Any, Sequence

import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.internal.settings import ReleaseMatch
from athenian.api.models.metadata.github import Repository
from athenian.api.models.state.models import RepositorySet
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import ReleaseSettingFactory


async def insert_repo(
    repository: Repository,
    mdb_cleaner: DBCleaner,
    mdb: Database,
    sdb: Database,
) -> None:
    """Insert rows in sdb and mdb in order to have a valid repository.

    RepositorySet row must exists before calling this wizard.
    """
    all_reposet_cond = [RepositorySet.owner_id == 1, RepositorySet.name == RepositorySet.ALL]
    reposet_items = await sdb.fetch_val(sa.select(RepositorySet.items).where(*all_reposet_cond))
    assert reposet_items is not None

    ref = ["github.com", repository.node_id, ""]
    reposet_items.insert(bisect_right(reposet_items, ref), ref)

    values = {
        RepositorySet.items.name: reposet_items,
        RepositorySet.updated_at: datetime.now(timezone.utc),
        RepositorySet.updates_count: RepositorySet.updates_count + 1,
    }
    await sdb.execute(sa.update(RepositorySet).where(*all_reposet_cond).values(values))
    await models_insert(
        sdb, ReleaseSettingFactory(repo_id=repository.node_id, match=ReleaseMatch.tag),
    )
    md_models = [
        repository,
        # AccountRepository rows are needed to pass GitHubAccessChecker
        md_factory.AccountRepositoryFactory(
            repo_full_name=repository.full_name, repo_graph_id=repository.node_id,
        ),
    ]
    mdb_cleaner.add_models(*md_models)
    await models_insert(mdb, *md_models)


def pr_models(
    repo_id: int,
    node_id: int,
    number: int,
    *,
    review_request: datetime | None = None,
    review_submit: datetime | None = None,
    commits: Collection[datetime] = (),
    **kwargs: Any,
) -> Sequence[Any]:
    """Return the models to insert in mdb to have a valid pull request.

    - `review_request`: if passed also a PullRequestReviewRequest model will be generated
    - `review_submit`: if passed also a PullRequestReview model will be generated
    - `commit`: a PullRequestCommit will be generated for every datetime passed

    All other parameters are passed to PullRequestFactory.
    """
    models = [
        md_factory.PullRequestFactory(node_id=node_id, repository_node_id=repo_id, **kwargs),
        md_factory.NodePullRequestFactory(node_id=node_id, number=number, repository_id=repo_id),
    ]
    if review_request is not None:
        models.append(
            md_factory.PullRequestReviewRequestFactory(
                created_at=review_request, pull_request_id=node_id,
            ),
        )
    if review_submit is not None:
        models.append(
            md_factory.PullRequestReviewFactory(
                submitted_at=review_submit,
                pull_request_node_id=node_id,
                repository_node_id=repo_id,
            ),
        )

    for commit in commits:
        models.append(
            md_factory.PullRequestCommitFactory(
                pull_request_node_id=node_id, committed_date=commit, repository_node_id=repo_id,
            ),
        )

    return models
