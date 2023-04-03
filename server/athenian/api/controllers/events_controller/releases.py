from collections import defaultdict
from datetime import datetime, timezone
import logging
import re

from aiohttp import web
import sqlalchemy as sa

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.db import dialect_specific_insert
from athenian.api.defer import defer
from athenian.api.internal.account import get_installation_url_prefix, get_metadata_account_ids
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.miners.github.commit import compose_commit_url
from athenian.api.models.metadata.github import PushCommit, User
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.web import (
    ForbiddenError,
    InvalidRequestError,
    ReleaseNotification as WebReleaseNotification,
    ReleaseNotificationStatus,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.serialization import ParseError

commit_hash_re = re.compile(r"[a-f0-9]{7}([a-f0-9]{33})?")


@disable_default_user
@weight(0)
async def notify_releases(request: AthenianWebRequest, body: list[dict]) -> web.Response:
    """Notify about new releases. The release settings must be set to "notification"."""
    # account is automatically checked at this point
    log = logging.getLogger(f"{metadata.__package__}.notify_releases")
    log.info("%s", body)
    try:
        notifications = [WebReleaseNotification.from_dict(n) for n in body]
    except ParseError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    account = request.account
    sdb, mdb, rdb = request.sdb, request.mdb, request.rdb
    authors = []
    for n in notifications:
        author = n.author
        if author is not None and "/" in author:
            author = author.rsplit("/", 1)[1]  # remove github.com/ or any other prefix
        authors.append(author)
    unique_authors = set(authors) - {None}
    statuses = [None] * len(notifications)

    async def main_flow():
        repos = set()
        full_commits = set()
        prefixed_commits = defaultdict(set)
        unique_notifications = set()
        for i, n in enumerate(notifications):
            try:
                repos.add(repo := n.repository.split("/", 1)[1])
            except IndexError:
                raise ResponseError(
                    InvalidRequestError(
                        "[%d].repository" % i,
                        detail="repository name is invalid: %s" % n.repository,
                    ),
                )
            if not commit_hash_re.fullmatch(n.commit):
                raise ResponseError(
                    InvalidRequestError("[%d].commit" % i, detail="invalid commit hash"),
                )
            if len(n.commit) == 7:
                prefixed_commits[repo].add(n.commit)
            else:
                full_commits.add(n.commit)
            if (key := (n.name or n.commit, n.repository)) in unique_notifications:
                statuses[i] = ReleaseNotificationStatus.IGNORED_DUPLICATE
                continue
            unique_notifications.add(key)
        meta_ids = await get_metadata_account_ids(account, sdb, request.cache)
        checker = await access_classes["github.com"](
            account, meta_ids, sdb, mdb, request.cache,
        ).load()
        if denied := await checker.check(repos):
            raise ResponseError(
                ForbiddenError(
                    detail="the following repositories are access denied for account %d: %s"
                    % (account, denied),
                ),
            )

        # the commit may not exist yet in the metadata, but let's try to resolve what we can
        commit_rows, user_rows, *url_prefixes = await gather(
            mdb.fetch_all(
                sa.union(
                    sa.select(
                        PushCommit.acc_id,
                        PushCommit.sha,
                        PushCommit.node_id,
                        PushCommit.repository_full_name,
                    ).where(PushCommit.acc_id.in_(meta_ids), PushCommit.sha.in_(full_commits)),
                    *(
                        sa.select(
                            PushCommit.acc_id,
                            PushCommit.sha,
                            PushCommit.node_id,
                            PushCommit.repository_full_name,
                        ).where(
                            PushCommit.acc_id.in_(meta_ids),
                            PushCommit.repository_full_name == repo,
                            sa.func.substr(PushCommit.sha, 1, 7).in_(prefixes),
                        )
                        for repo, prefixes in prefixed_commits.items()
                    ),
                ),
            ),
            mdb.fetch_all(
                sa.select(User.login, User.node_id).where(
                    User.acc_id.in_(meta_ids), User.login.in_(unique_authors),
                ),
            ),
            *(get_installation_url_prefix(meta_id, mdb, request.cache) for meta_id in meta_ids),
        )
        url_prefixes = dict(zip(meta_ids, url_prefixes))
        resolved_prefixed_commits = {}
        resolved_full_commits = {}
        for row in commit_rows:
            commit, repo = row[PushCommit.sha.name], row[PushCommit.repository_full_name.name]
            resolved_prefixed_commits[(commit[:7], repo)] = resolved_full_commits[
                (commit, repo)
            ] = row
        resolved_users = {}
        for row in user_rows:
            resolved_users[row[User.login.name]] = row[User.node_id.name]
        return (
            resolved_full_commits,
            resolved_prefixed_commits,
            checker.installed_repos,
            resolved_users,
            url_prefixes,
            meta_ids,
        )

    user, (
        resolved_full_commits,
        resolved_prefixed_commits,
        installed_repos,
        resolved_users,
        url_prefixes,
        meta_ids,
    ) = await gather(request.user(), main_flow())

    if None in authors:
        resolved_users[None] = await mdb.fetch_val(
            sa.select(User.node_id).where(User.acc_id.in_(meta_ids), User.login == user.login),
        )
    inserted = []
    repos = set()
    now = datetime.now(timezone.utc)
    empty_resolved = {
        PushCommit.acc_id.name: None,
        PushCommit.sha.name: None,
        PushCommit.node_id.name: None,
    }
    for i, (n, author, status) in enumerate(zip(notifications, authors, statuses)):
        if status == ReleaseNotificationStatus.IGNORED_DUPLICATE:
            log.warning("ignored %s", n.to_dict())
            continue
        repos.add(repo := n.repository.split("/", 1)[1])
        try:
            resolved_commits = (
                resolved_full_commits if len(n.commit) == 40 else resolved_prefixed_commits
            )[(n.commit, repo)]
            statuses[i] = ReleaseNotificationStatus.ACCEPTED_RESOLVED
        except KeyError:
            resolved_commits = empty_resolved
            statuses[i] = ReleaseNotificationStatus.ACCEPTED_PENDING
        inserted.append(
            ReleaseNotification(
                account_id=account,
                repository_node_id=installed_repos[repo],
                commit_hash_prefix=(prefix := resolved_commits[PushCommit.sha.name] or n.commit),
                resolved_commit_hash=(sha := resolved_commits[PushCommit.sha.name]),
                resolved_commit_node_id=(cid := resolved_commits[PushCommit.node_id.name]),
                resolved_at=now if cid is not None else None,
                name=n.name or f"{repo}@{prefix}",
                author_node_id=resolved_users.get(author),
                url=n.url
                or (
                    compose_commit_url(
                        url_prefixes[resolved_commits[PushCommit.acc_id.name]], repo, sha,
                    )
                    if sha is not None
                    else None
                ),
                published_at=n.published_at,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        )
    sql = (await dialect_specific_insert(rdb))(ReleaseNotification)
    sql = sql.on_conflict_do_update(
        index_elements=ReleaseNotification.__table__.primary_key.columns,
        set_={
            ReleaseNotification.name.name: sql.excluded.name,
            ReleaseNotification.author_node_id.name: sql.excluded.author_node_id,
            ReleaseNotification.url.name: sql.excluded.url,
            ReleaseNotification.published_at.name: sql.excluded.published_at,
            ReleaseNotification.updated_at.name: sql.excluded.updated_at,
        },
    )
    if rdb.url.dialect == "sqlite":
        async with rdb.connection() as perdata_conn:
            async with perdata_conn.transaction():
                await perdata_conn.execute_many(sql, inserted)
    else:
        # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
        await rdb.execute_many(sql, inserted)
    if (slack := request.app["slack"]) is not None:

        async def report_new_release_event_to_slack():
            await slack.post_event("new_release_event.jinja2", account=account, repos=repos)

        await defer(report_new_release_event_to_slack(), "report_new_release_event_to_slack")
    return model_response(statuses)
