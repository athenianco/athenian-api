import logging
from typing import Optional

from aiohttp import web
import aiomcache
import sentry_sdk
from sqlalchemy import delete, select, union_all, update

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.db import Database
from athenian.api.internal.account import get_metadata_account_ids, only_god
from athenian.api.internal.jira import fetch_jira_installation_progress, get_jira_id
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import fetch_precomputed_commit_history_dags
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.team_sync import sync_teams
from athenian.api.internal.user import load_user_accounts
from athenian.api.models.metadata.github import NodeCommit
from athenian.api.models.state.models import God, RepositorySet, UserAccount
from athenian.api.models.web import InvalidRequestError, NotFoundError, ResetRequest, ResetTarget
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span
from athenian.precomputer.db.models import (
    GitHubCommitDeployment,
    GitHubCommitHistory,
    GitHubDeploymentFacts,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubOpenPullRequestFacts,
    GitHubPullRequestDeployment,
    GitHubRebaseCheckedCommit,
    GitHubRebasedPullRequest,
    GitHubRelease,
    GitHubReleaseDeployment,
    GitHubReleaseFacts,
    GitHubReleaseMatchTimespan,
)


@only_god
async def become_user(request: AthenianWebRequest, id: str = "") -> web.Response:
    """God mode ability to turn into any user. The current user must be marked internally as \
    a super admin."""
    user_id = request.god_id
    async with request.sdb.connection() as conn:
        if (
            id
            and (await conn.fetch_one(select([UserAccount]).where(UserAccount.user_id == id)))
            is None
        ):
            raise ResponseError(NotFoundError(detail="User %s does not exist" % id))
        god = await conn.fetch_one(select([God]).where(God.user_id == user_id))
        god = God(**god).refresh()
        god.mapped_id = id or None
        await conn.execute(update(God).where(God.user_id == user_id).values(god.explode()))
    user = await request.app["auth"].get_user(id or user_id)
    user.accounts = await load_user_accounts(
        user.id,
        getattr(request, "god_id", user.id),
        request.sdb,
        request.mdb,
        request.rdb,
        request.app["slack"],
        request.user,
        request.cache,
    )
    return model_response(user)


@sentry_span
async def _reset_commits(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    physical_repos = coerce_logical_repos(repos)
    dags, prefixer = await gather(
        fetch_precomputed_commit_history_dags(physical_repos, account, pdb, cache),
        Prefixer.load(meta_ids, mdb, cache),
    )
    queries = [
        select(NodeCommit.node_id).where(
            NodeCommit.repository_id == prefixer.repo_name_to_node.get(repo),
            NodeCommit.acc_id.in_(meta_ids),
            NodeCommit.oid.in_any_values(dag[0]),
        )
        for repo, (_, dag) in dags.items()
    ]
    df = await read_sql_query(union_all(*queries), mdb, [NodeCommit.node_id])
    commit_ids = df[NodeCommit.node_id.name].values
    await gather(
        pdb.execute(
            delete(GitHubCommitHistory).where(
                GitHubCommitHistory.acc_id == account,
                GitHubCommitHistory.repository_full_name.in_(physical_repos),
            ),
        ),
        pdb.execute(
            delete(GitHubRebasedPullRequest).where(
                GitHubRebasedPullRequest.acc_id == account,
                GitHubRebasedPullRequest.matched_merge_commit_id.in_(commit_ids),
            ),
        ),
        pdb.execute(
            delete(GitHubRebaseCheckedCommit).where(
                GitHubRebaseCheckedCommit.acc_id == account,
                GitHubRebaseCheckedCommit.node_id.in_(commit_ids),
            ),
        ),
    )


@sentry_span
async def _reset_jira_account(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    await gather(
        get_jira_id.reset_cache(account, sdb, cache),
        fetch_jira_installation_progress.reset_cache(account, sdb, mdb, cache),
    )


@sentry_span
async def _reset_metadata_account(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    await get_metadata_account_ids.reset_cache(account, sdb, cache)


@sentry_span
async def _reset_teams(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    await sync_teams(account, meta_ids, sdb, mdb, force=True, unmapped=True)


@sentry_span
async def _reset_prs(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    await gather(
        pdb.execute(
            delete(GitHubDonePullRequestFacts).where(
                GitHubDonePullRequestFacts.acc_id == account,
                GitHubDonePullRequestFacts.repository_full_name.in_(repos),
            ),
        ),
        pdb.execute(
            delete(GitHubMergedPullRequestFacts).where(
                GitHubMergedPullRequestFacts.acc_id == account,
                GitHubMergedPullRequestFacts.repository_full_name.in_(repos),
            ),
        ),
        pdb.execute(
            delete(GitHubOpenPullRequestFacts).where(
                GitHubOpenPullRequestFacts.acc_id == account,
                GitHubOpenPullRequestFacts.repository_full_name.in_(repos),
            ),
        ),
    )


@sentry_span
async def _reset_reposet(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    await gather(
        sdb.execute(
            delete(RepositorySet).where(
                RepositorySet.owner_id == account, RepositorySet.name == RepositorySet.ALL,
            ),
        ),
        Prefixer.load.reset_cache(meta_ids, mdb, cache),
    )


@sentry_span
async def _reset_releases(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    await gather(
        pdb.execute(
            delete(GitHubRelease).where(
                GitHubRelease.acc_id == account, GitHubRelease.repository_full_name.in_(repos),
            ),
        ),
        pdb.execute(
            delete(GitHubReleaseMatchTimespan).where(
                GitHubReleaseMatchTimespan.acc_id == account,
                GitHubReleaseMatchTimespan.repository_full_name.in_(repos),
            ),
        ),
        pdb.execute(
            delete(GitHubReleaseFacts).where(
                GitHubReleaseFacts.acc_id == account,
                GitHubReleaseFacts.repository_full_name.in_(repos),
            ),
        ),
        BranchMiner.extract_branches.reset_cache(None, None, meta_ids, mdb, cache),
    )


@sentry_span
async def _reset_deployments(
    repos: list[str],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    await gather(
        pdb.execute(
            delete(GitHubCommitDeployment).where(
                GitHubCommitDeployment.acc_id == account,
                GitHubCommitDeployment.repository_full_name.in_(repos),
            ),
        ),
        pdb.execute(
            delete(GitHubPullRequestDeployment).where(
                GitHubPullRequestDeployment.acc_id == account,
                GitHubPullRequestDeployment.repository_full_name.in_(repos),
            ),
        ),
        pdb.execute(
            delete(GitHubReleaseDeployment).where(
                GitHubReleaseDeployment.acc_id == account,
                GitHubReleaseDeployment.repository_full_name.in_(repos),
            ),
        ),
        pdb.execute(delete(GitHubDeploymentFacts).where(GitHubDeploymentFacts.acc_id == account)),
    )


_resetters = {
    ResetTarget.COMMITS: _reset_commits,
    ResetTarget.JIRA_ACCOUNT: _reset_jira_account,
    ResetTarget.METADATA_ACCOUNT: _reset_metadata_account,
    ResetTarget.TEAMS: _reset_teams,
    ResetTarget.PRS: _reset_prs,
    ResetTarget.REPOSET: _reset_reposet,
    ResetTarget.RELEASES: _reset_releases,
    ResetTarget.DEPLOYMENTS: _reset_deployments,
}


@only_god
async def reset_account(request: AthenianWebRequest, body: dict) -> web.Response:
    """Clear the selected tables in precomputed DB, drop the related caches."""
    log = logging.getLogger(f"{metadata.__package__}.reset_account")
    try:
        request_model = ResetRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    meta_ids = await get_metadata_account_ids(request_model.account, request.sdb, request.cache)

    repos = (
        await resolve_repos_with_request(
            request_model.repositories,
            request_model.account,
            request,
            meta_ids,
        )
    )[0]
    log.info("reset %s on %s", request_model.targets, repos)
    await gather(
        *(
            _i_will_survive(
                log,
                target,
                _resetters[target](
                    repos,
                    request_model.account,
                    meta_ids,
                    request.sdb,
                    request.mdb,
                    request.pdb,
                    request.cache,
                ),
            )
            for target in request_model.targets
        ),
    )
    return web.Response()


async def _i_will_survive(log, target, coro) -> None:
    try:
        await coro
    except Exception:
        sentry_sdk.capture_exception()
        log.warning("reset failed: %s", target)
