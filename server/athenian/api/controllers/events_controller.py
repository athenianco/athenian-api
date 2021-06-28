from datetime import date, datetime, timedelta, timezone
import os
import re
from typing import List

from aiohttp import web
from sqlalchemy import and_, delete, func, insert, select, union
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api.async_utils import gather
from athenian.api.balancing import weight
from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.features.entries import MetricEntriesCalculator
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.release_mine import mine_releases
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.reposet import resolve_repos
from athenian.api.controllers.settings import ReleaseMatch, Settings
from athenian.api.defer import defer, wait_deferred
from athenian.api.models.metadata.github import PushCommit, User
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.web import BadRequestError, DeleteEventsCacheRequest, ForbiddenError, \
    InvalidRequestError, ReleaseNotification as WebReleaseNotification
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.serialization import ParseError
from athenian.precomputer.db.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubReleaseFacts

commit_hash_re = re.compile(r"[a-f0-9]{7}([a-f0-9]{33})?")
SLACK_CHANNEL = os.getenv("ATHENIAN_EVENTS_SLACK_CHANNEL", "")


@weight(0)
async def notify_releases(request: AthenianWebRequest, body: List[dict]) -> web.Response:
    """Notify about new releases. The release settings must be set to "notification"."""
    # account is automatically checked at this point
    try:
        notifications = [WebReleaseNotification.from_dict(n) for n in body]
    except ParseError as e:
        raise ResponseError(BadRequestError("%s: %s" % (type(e).__name__, e)))
    account = request.account
    sdb, mdb, rdb = request.sdb, request.mdb, request.rdb
    authors = []
    for n in notifications:
        author = n.author
        if author is not None and "/" in author:
            author = author.rsplit("/", 1)[1]  # remove github.com/ or any other prefix
        authors.append(author)
    unique_authors = set(authors) - {None}

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
                detail="the following repositories are access denied for account %d: %s" %
                       (account, denied),
            ))

        # the commit may not exist yet in the metadata, but let's try to resolve what we can
        commit_rows, user_rows = await gather(mdb.fetch_all(
            union(
                select([PushCommit.sha, PushCommit.node_id, PushCommit.repository_full_name])
                .where(and_(PushCommit.acc_id.in_(meta_ids),
                            PushCommit.sha.in_(full_commits))),
                select([PushCommit.sha, PushCommit.node_id, PushCommit.repository_full_name])
                .where(and_(PushCommit.acc_id.in_(meta_ids),
                            func.substr(PushCommit.sha, 1, 7).in_(prefixed_commits))),
            )),
            mdb.fetch_all(select([User.login, User.node_id])
                          .where(and_(User.acc_id.in_(meta_ids),
                                      User.login.in_(unique_authors)))),
        )
        resolved_prefixed_commits = {}
        resolved_full_commits = {}
        for row in commit_rows:
            commit, repo = row[PushCommit.sha.key], row[PushCommit.repository_full_name.key]
            resolved_prefixed_commits[(commit[:7], repo)] = \
                resolved_full_commits[(commit, repo)] = row
        resolved_users = {}
        for row in user_rows:
            resolved_users[row[User.login.key]] = row[User.node_id.key]
        return (resolved_full_commits,
                resolved_prefixed_commits,
                checker.installed_repos(),
                resolved_users,
                meta_ids)

    user, (resolved_full_commits,
           resolved_prefixed_commits,
           installed_repos,
           resolved_users,
           meta_ids) = await gather(request.user(), main_flow())

    if None in authors:
        resolved_users[None] = await mdb.fetch_val(
            select([User.node_id])
            .where(and_(User.acc_id.in_(meta_ids),
                        User.login == user.login)))
    inserted = []
    repos = set()
    for n, author in zip(notifications, authors):
        repos.add(repo := n.repository.split("/", 1)[1])
        resolved_commits = (
            resolved_full_commits if len(n.commit) == 40 else resolved_prefixed_commits
        ).get((n.commit, repo), {PushCommit.sha.key: None, PushCommit.node_id.key: None})
        inserted.append(ReleaseNotification(
            account_id=account,
            repository_node_id=installed_repos[repo],
            commit_hash_prefix=resolved_commits[PushCommit.sha.key] or n.commit,
            resolved_commit_hash=resolved_commits[PushCommit.sha.key],
            resolved_commit_node_id=resolved_commits[PushCommit.node_id.key],
            name=n.name,
            author_node_id=resolved_users.get(author),
            url=n.url,
            published_at=n.published_at or datetime.now(timezone.utc),
        ).create_defaults().explode(with_primary_keys=True))
    if rdb.url.dialect == "postgresql":
        sql = postgres_insert(ReleaseNotification)
        sql = sql.on_conflict_do_update(
            constraint=ReleaseNotification.__table__.primary_key,
            set_={
                ReleaseNotification.name.key: sql.excluded.name,
                ReleaseNotification.author_node_id.key: sql.excluded.author_node_id,
                ReleaseNotification.url.key: sql.excluded.url,
                ReleaseNotification.published_at.key: sql.excluded.published_at,
            },
        )
    else:  # sqlite
        sql = insert(ReleaseNotification).prefix_with("OR REPLACE")
    if rdb.url.dialect == "sqlite":
        async with rdb.connection() as perdata_conn:
            async with perdata_conn.transaction():
                await perdata_conn.execute_many(sql, inserted)
    else:
        # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
        await rdb.execute_many(sql, inserted)
    if (slack := request.app["slack"]) is not None:
        async def report_new_release_event_to_slack():
            await slack.post("new_release_event.jinja2",
                             channel=SLACK_CHANNEL, account=account, repos=repos)

        await defer(report_new_release_event_to_slack(), "report_new_release_event_to_slack")
    return web.Response(status=200)


@weight(0)
async def clear_precomputed_events(request: AthenianWebRequest, body: dict) -> web.Response:
    """Reset the precomputed data related to the pushed events."""
    model = DeleteEventsCacheRequest.from_dict(body)

    async def login_loader() -> str:
        return (await request.user()).login

    prefixed_repos, meta_ids = await resolve_repos(
        model.repositories, model.account, request.uid, login_loader,
        request.sdb, request.mdb, request.cache, request.app["slack"], strip_prefix=False)
    prefixer = Prefixer.schedule_load(meta_ids, request.mdb, request.cache)
    repos = [r.split("/", 1)[1] for r in prefixed_repos]
    pdb = request.pdb
    if "release" in model.targets:
        await gather(*(
            pdb.execute(delete(table).where(and_(
                table.release_match == ReleaseMatch.event.name,
                table.repository_full_name.in_(repos),
                table.acc_id == model.account,
            )))
            for table in (GitHubDonePullRequestFacts,
                          GitHubMergedPullRequestFacts,
                          GitHubReleaseFacts)
        ), op="delete precomputed releases")

        # preheat these repos
        sdb, mdb, pdb, rdb, cache = \
            request.sdb, request.mdb, request.pdb, request.rdb, request.cache
        time_to = datetime.combine(date.today() + timedelta(days=1),
                                   datetime.min.time(),
                                   tzinfo=timezone.utc)
        no_time_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
        time_from = time_to - timedelta(days=365 * 2)
        (branches, default_branches), settings = await gather(
            BranchMiner.extract_branches(repos, meta_ids, mdb, cache),
            Settings.from_account(model.account, sdb, mdb, cache, None)
            .list_release_matches(prefixed_repos))
        await mine_releases(
            repos, {}, branches, default_branches, no_time_from, time_to, LabelFilter.empty(),
            JIRAFilter.empty(), settings, prefixer, model.account, meta_ids, mdb, pdb, rdb, None,
            force_fresh=True)
        await wait_deferred()
        await MetricEntriesCalculator(
            model.account, meta_ids, mdb, pdb, rdb, None).calc_pull_request_facts_github(
                time_from, time_to, set(repos), {}, LabelFilter.empty(), JIRAFilter.empty(),
                False, settings, True, False)
        await wait_deferred()
    return web.Response(status=200)


@weight(0)
async def notify_deployments(request: AthenianWebRequest, body: List[dict]) -> web.Response:
    """Notify about new deployments."""
    raise NotImplementedError
