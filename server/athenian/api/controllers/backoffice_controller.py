from datetime import datetime, timezone
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Optional

from aiohttp import web
import aiohttp.web
import aiomcache
from asyncpg import IntegrityConstraintViolationError
import sentry_sdk
from sqlalchemy import delete, insert, select, union_all, update

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.db import Database, dialect_specific_insert
from athenian.api.internal.account import (
    get_metadata_account_ids,
    get_user_account_status,
    get_user_account_status_from_request,
)
from athenian.api.internal.account_feature import get_account_features
from athenian.api.internal.jira import fetch_jira_installation_progress, get_jira_id
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import fetch_precomputed_commit_history_dags
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.team_sync import sync_teams
from athenian.api.internal.user import load_user_accounts
from athenian.api.models.metadata.github import NodeCommit
from athenian.api.models.state.models import (
    Account,
    AccountFeature,
    Feature,
    God,
    RepositorySet,
    UserAccount,
)
from athenian.api.models.web import (
    DatabaseConflict,
    ForbiddenError,
    InvalidRequestError,
    NotFoundError,
    ProductFeature,
    ResetRequest,
    ResetTarget,
    UserMoveRequest,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.serialization import deserialize_datetime
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


async def become_user(request: AthenianWebRequest, id: str = "") -> web.Response:
    """God mode ability to turn into any user. The current user must be marked internally as \
    a super admin."""
    user_id = request.god_id
    async with request.sdb.connection() as conn:
        if (
            id
            and (await conn.fetch_one(select(UserAccount).where(UserAccount.user_id == id)))
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
        # cached method is unbounded, add one more param for cls
        BranchMiner.extract_branches.reset_cache(None, None, None, meta_ids, mdb, cache),
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


async def reset_account(request: AthenianWebRequest, body: dict) -> web.Response:
    """Clear the selected tables in precomputed DB, drop the related caches."""
    log = logging.getLogger(f"{metadata.__package__}.reset_account")
    try:
        request_model = ResetRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    meta_ids = await get_metadata_account_ids(request_model.account, request.sdb, request.cache)

    repos = [
        r.unprefixed
        for r in (
            await resolve_repos_with_request(
                request_model.repositories,
                request_model.account,
                request,
                meta_ids=meta_ids,
                pointer=".repositories",
            )
        )[0]
    ]
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


async def move_user(request: AthenianWebRequest, body: dict) -> web.Response:
    """Move the user between accounts."""
    model = UserMoveRequest.from_dict(body)
    await get_user_account_status(
        model.user,
        model.old_account,
        request.sdb,
        request.mdb,
        request.user,
        request.app["slack"],
        request.cache,
        is_god=True,
    )
    if (model.new_account_admin is None) == (model.new_account_regular is None):
        raise ResponseError(
            InvalidRequestError(
                ".new_account_admin || .new_account_regular",
                detail=(
                    "Either `new_account_admin` or `new_account_regular` must be specified or "
                    "removed."
                ),
            ),
        )
    new_account = model.new_account_regular or model.new_account_admin
    if new_account == model.old_account:
        raise ResponseError(
            InvalidRequestError(
                ".old_account",
                detail="The old and the new acoounts may not match.",
            ),
        )
    try:
        async with request.sdb.connection() as sdb:
            async with sdb.transaction():
                await sdb.execute(
                    delete(UserAccount).where(
                        UserAccount.account_id == model.old_account,
                        UserAccount.user_id == model.user,
                    ),
                )
                await sdb.execute(
                    insert(UserAccount).values(
                        {
                            UserAccount.user_id: model.user,
                            UserAccount.account_id: new_account,
                            UserAccount.is_admin: model.new_account_admin is not None,
                            UserAccount.created_at: datetime.now(timezone.utc),
                        },
                    ),
                )
    except (IntegrityConstraintViolationError, IntegrityError, OperationalError) as e:
        raise ResponseError(DatabaseConflict(detail=str(e)))
    await gather(
        get_user_account_status.reset_cache(
            model.user, model.old_account, None, None, None, None, None,
        ),
        get_user_account_status.reset_cache(
            model.user, new_account, None, None, None, None, None,
        ),
    )
    return web.Response()


async def set_account_features(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Set account features if you are a god."""
    # we must check this even though we've already checked x-only-god
    # the god must be impersonating someone
    if getattr(request, "god_id", None) is None:
        raise ResponseError(
            ForbiddenError(
                detail="User %s is not allowed to set features of accounts" % request.uid,
            ),
        )
    features = [ProductFeature.from_dict(f) for f in body]
    async with request.sdb.connection() as conn:
        await get_user_account_status_from_request(request, id)
        for i, feature in enumerate(features):
            if feature.name == Account.expires_at.name:
                try:
                    expires_at = deserialize_datetime(feature.parameters, max_future_delta=None)
                except (TypeError, ValueError):
                    raise ResponseError(
                        InvalidRequestError(
                            pointer=f".[{i}].parameters",
                            detail=f"Invalid datetime string: {feature.parameters}",
                        ),
                    )
                await conn.execute(
                    update(Account)
                    .where(Account.id == id)
                    .values(
                        {
                            Account.expires_at: expires_at,
                        },
                    ),
                )
            else:
                if not isinstance(feature.parameters, dict) or not isinstance(
                    feature.parameters.get("enabled"), bool,
                ):
                    raise ResponseError(
                        InvalidRequestError(
                            pointer=f".[{i}].parameters",
                            detail='Parameters must be {"enabled": true|false, ...}',
                        ),
                    )
                fid = await conn.fetch_val(
                    select([Feature.id]).where(Feature.name == feature.name),
                )
                if fid is None:
                    raise ResponseError(
                        InvalidRequestError(
                            pointer=f".[{i}].name",
                            detail=f"Feature is not supported: {feature.name}",
                        ),
                    )
                query = (await dialect_specific_insert(conn))(AccountFeature)
                query = query.on_conflict_do_update(
                    index_elements=AccountFeature.__table__.primary_key.columns,
                    set_={
                        AccountFeature.enabled.name: query.excluded.enabled,
                        AccountFeature.parameters.name: query.excluded.parameters,
                    },
                )
                await conn.execute(
                    query.values(
                        AccountFeature(
                            account_id=id,
                            feature_id=fid,
                            enabled=feature.parameters["enabled"],
                            parameters=feature.parameters.get("parameters"),
                        )
                        .create_defaults()
                        .explode(with_primary_keys=True),
                    ),
                )
    return model_response(await get_account_features(id, request.sdb))


async def get_account_health(
    request: AthenianWebRequest,
    id: Optional[int] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> aiohttp.web.Response:
    """Return the account health metrics measured per hour."""
    raise NotImplementedError
