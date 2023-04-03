import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Collection

from aiohttp import web
import medvedi as md
import sqlalchemy as sa

from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.defer import launch_defer_from_request, wait_deferred
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import PRFactsCalculator
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.github.release_mine import mine_releases
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseSettings,
    Settings,
)
from athenian.api.models.persistentdata.models import DeployedComponent
from athenian.api.models.precomputed.models import (
    GitHubCommitDeployment,
    GitHubDeploymentFacts,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubPullRequestDeployment,
    GitHubReleaseDeployment,
    GitHubReleaseFacts,
)
from athenian.api.models.web import DeleteEventsCacheRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.tracing import sentry_span


@disable_default_user
@weight(10)
async def clear_precomputed_events(request: AthenianWebRequest, body: dict) -> web.Response:
    """Reset the precomputed data related to the pushed events."""
    launch_defer_from_request(request, detached=True)  # DEV-2798
    model = DeleteEventsCacheRequest.from_dict(body)

    meta_ids = await get_metadata_account_ids(model.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, model.account, prefixer)
    (prefixed_repos, _), logical_settings = await gather(
        resolve_repos_with_request(
            model.repositories,
            model.account,
            request,
            meta_ids=meta_ids,
            prefixer=prefixer,
            pointer=".repositories",
        ),
        settings.list_logical_repositories(),
    )
    repos = [r.unprefixed for r in prefixed_repos]
    (branches, default_branches), release_settings = await gather(
        BranchMiner.load_branches(
            repos, prefixer, model.account, meta_ids, request.mdb, request.pdb, request.cache,
        ),
        settings.list_release_matches([str(r) for r in prefixed_repos]),
    )
    tasks = [
        droppers[t](
            model.account,
            repos,
            prefixer,
            release_settings,
            logical_settings,
            branches,
            default_branches,
            request,
            meta_ids,
        )
        for t in model.targets
    ]
    await gather(*tasks, op="clear_precomputed_events/gather drops")
    await wait_deferred(final=True)
    return web.json_response({})


@sentry_span
async def _drop_precomputed_deployments(
    account: int,
    repos: Collection[str],
    prefixer: Prefixer,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: md.DataFrame,
    default_branches: dict[str, str],
    request: AthenianWebRequest,
    meta_ids: tuple[int, ...],
) -> None:
    pdb, rdb = request.pdb, request.rdb
    repo_name_to_node = prefixer.repo_name_to_node.get
    repo_node_ids = [repo_name_to_node(r, 0) for r in repos]
    deployments_to_kill = await rdb.fetch_all(
        sa.select(sa.distinct(DeployedComponent.deployment_name)).where(
            DeployedComponent.account_id == account,
            DeployedComponent.repository_node_id.in_(repo_node_ids),
        ),
    )
    deployments_to_kill = [r[0] for r in deployments_to_kill]
    await gather(
        pdb.execute(
            sa.delete(GitHubDeploymentFacts).where(
                GitHubDeploymentFacts.acc_id == account,
                GitHubDeploymentFacts.deployment_name.in_(deployments_to_kill),
            ),
        ),
        pdb.execute(
            sa.delete(GitHubReleaseDeployment).where(
                GitHubReleaseDeployment.acc_id == account,
                GitHubReleaseDeployment.deployment_name.in_(deployments_to_kill),
            ),
        ),
        pdb.execute(
            sa.delete(GitHubPullRequestDeployment).where(
                GitHubPullRequestDeployment.acc_id == account,
                GitHubPullRequestDeployment.deployment_name.in_(deployments_to_kill),
            ),
        ),
        pdb.execute(
            sa.delete(GitHubCommitDeployment).where(
                GitHubCommitDeployment.acc_id == account,
                GitHubCommitDeployment.deployment_name.in_(deployments_to_kill),
            ),
        ),
        op="remove %d deployments from pdb" % len(deployments_to_kill),
    )
    today = datetime.now(timezone.utc)
    await mine_deployments(
        repos,
        {},
        today - timedelta(days=730),
        today,
        [],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_settings,
        logical_settings,
        branches,
        default_branches,
        prefixer,
        account,
        None,
        meta_ids,
        request.mdb,
        pdb,
        rdb,
        None,
    )


@sentry_span
async def _drop_precomputed_event_releases(
    account: int,
    repos: Collection[str],
    prefixer: Prefixer,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: md.DataFrame,
    default_branches: dict[str, str],
    request: AthenianWebRequest,
    meta_ids: tuple[int, ...],
) -> None:
    pdb = request.pdb
    bots_task = asyncio.create_task(
        bots(account, meta_ids, request.mdb, request.sdb, request.cache),
        name="_drop_precomputed_event_releases/bots",
    )
    await gather(
        *(
            pdb.execute(
                sa.delete(table).where(
                    table.release_match == ReleaseMatch.event.name,
                    table.repository_full_name.in_(repos),
                    table.acc_id == account,
                ),
            )
            for table in (
                GitHubDonePullRequestFacts,
                GitHubMergedPullRequestFacts,
                GitHubReleaseFacts,
            )
        ),
        op="delete precomputed releases",
    )

    # preheat these repos
    mdb, pdb, rdb = request.mdb, request.pdb, request.rdb
    time_to = datetime.combine(
        date.today() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc,
    )
    no_time_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=365 * 2)
    await mine_releases(
        repos,
        {},
        branches,
        default_branches,
        no_time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        None,
        force_fresh=True,
        with_avatars=False,
        with_deployments=False,
        with_extended_pr_details=False,
    )
    await gather(wait_deferred(), bots_task)
    await PRFactsCalculator(account, meta_ids, mdb, pdb, rdb, cache=None)(
        time_from,
        time_to,
        set(repos),
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        bots_task.result(),
        release_settings,
        logical_settings,
        prefixer,
        True,
        0,
    )


droppers = {
    "release": _drop_precomputed_event_releases,
    "deployment": _drop_precomputed_deployments,
}
