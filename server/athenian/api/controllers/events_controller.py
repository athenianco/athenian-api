from datetime import datetime, timezone
import re
from typing import List

from aiohttp import web
from sqlalchemy import and_, func, insert, select, union
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import PushCommit
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.web import ForbiddenError, InvalidRequestError, \
    ReleaseNotification as WebReleaseNotification
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError


commit_hash_re = re.compile(r"[a-f0-9]{7}([a-f0-9]{33})?")


@weight(0)
async def notify_release(request: AthenianWebRequest, body: List[dict]) -> web.Response:
    """Notify about new releases. The release settings must be set to "notification"."""
    # account is automatically checked at this point
    notifications = [WebReleaseNotification.from_dict(n) for n in body]
    account = request.account
    sdb, mdb, rdb = request.sdb, request.mdb, request.rdb
    prefix = PREFIXES["github"]

    async def main_flow():
        repos = set()
        full_commits = set()
        prefixed_commits = set()
        unique_notifications = set()
        for i, n in enumerate(notifications):
            try:
                repos.add(n.repository.split("/", 1)[1])
            except IndexError:
                raise ResponseError(InvalidRequestError(
                    "[%d].repository" % i, detail="repository name is invalid: %s" % n.repository))
            if not commit_hash_re.fullmatch(n.commit):
                raise ResponseError(InvalidRequestError(
                    "[%d].commit" % i, detail="invalid commit hash"))
            if len(n.commit) == 7:
                prefixed_commits.add(n.commit)
            else:
                full_commits.add(n.commit)
            if (key := (n.commit, n.repository)) in unique_notifications:
                raise ResponseError(InvalidRequestError(
                    "[%d]" % i, detail="duplicate release notification"))
            unique_notifications.add(key)
        meta_ids = await get_metadata_account_ids(account, sdb, request.cache)
        checker = await access_classes["github"](account, meta_ids, sdb, mdb, request.cache).load()
        if denied := await checker.check(repos):
            raise ResponseError(ForbiddenError(
                detail="the following repositories are access denied for %s: %s" %
                       (prefix, denied),
            ))

        # the commit may not exist yet in the metadata, but let's try to resolve what we can
        rows = await mdb.fetch_all(
            union(
                select([PushCommit.sha, PushCommit.node_id, PushCommit.repository_full_name])
                .where(and_(PushCommit.acc_id.in_(meta_ids),
                            PushCommit.sha.in_(full_commits))),
                select([PushCommit.sha, PushCommit.node_id, PushCommit.repository_full_name])
                .where(and_(PushCommit.acc_id.in_(meta_ids),
                            func.substr(PushCommit.sha, 1, 7).in_(prefixed_commits))),
            ))
        resolved_prefixed_commits = {}
        resolved_full_commits = {}
        for row in rows:
            commit, repo = row[PushCommit.sha.key], row[PushCommit.repository_full_name.key]
            resolved_prefixed_commits[(commit[:7], repo)] = \
                resolved_full_commits[(commit, repo)] = row
        return resolved_full_commits, resolved_prefixed_commits, checker.installed_repos()

    user, (resolved_full_commits, resolved_prefixed_commits, installed_repos) = \
        await gather(request.user(), main_flow())

    inserted = []
    for n in notifications:
        repo = n.repository.split("/", 1)[1]
        resolved = (
            resolved_full_commits if len(n.commit) == 40 else resolved_prefixed_commits
        ).get((n.commit, repo), {PushCommit.sha.key: None, PushCommit.node_id.key: None})
        if (author := (n.author or user.login)).startswith(prefix):
            author = author[len(prefix):]  # remove github.com/
        inserted.append(ReleaseNotification(
            account_id=account,
            repository_node_id=installed_repos[n.repository],
            commit_hash=resolved[PushCommit.sha.key] or n.commit,
            resolved_commit_hash=resolved[PushCommit.sha.key],
            resolved_commit_node_id=resolved[PushCommit.node_id.key],
            name=n.name,
            author=author,
            url=n.url,
            published_at=n.published_at or datetime.now(timezone.utc),
        ).create_defaults().explode(with_primary_keys=True))
    if rdb.url.dialect in ("postgres", "postgresql"):
        sql = postgres_insert(ReleaseNotification)
        sql = sql.on_conflict_do_update(
            constraint=ReleaseNotification.__table__.primary_key,
            set_={
                ReleaseNotification.name.key: sql.excluded.name,
                ReleaseNotification.author.key: sql.excluded.author,
                ReleaseNotification.url.key: sql.excluded.url,
                ReleaseNotification.published_at.key: sql.excluded.published_at,
            },
        )
    else:  # sqlite
        sql = insert(ReleaseNotification).prefix_with("OR REPLACE")
    async with rdb.connection() as perdata_conn:
        async with perdata_conn.transaction():
            await perdata_conn.execute_many(sql, inserted)
    return web.Response(status=200)
