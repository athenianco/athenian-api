import asyncio
from datetime import timezone
from itertools import chain, groupby
import json
import marshal
from operator import itemgetter
from typing import Optional

from aiohttp import web
import aiomcache
import sentry_sdk
from sqlalchemy import and_, or_, select

from athenian.api.cache import cached, max_exptime
from athenian.api.controllers.account import get_user_account_status
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web import FilterJIRAStuff, FoundJIRAStuff, InvalidRequestError, \
    JIRAEpic, JIRALabel, NoSourceDataError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


@sentry_span
@cached(
    exptime=max_exptime,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_jira_installation(account: int,
                                sdb: DatabaseLike,
                                cache: Optional[aiomcache.Client]) -> int:
    """Retrieve the JIRA installation ID belonging to the account or raise an exception."""
    jira_id = await sdb.fetch_val(select([AccountJiraInstallation.id])
                                  .where(AccountJiraInstallation.account_id == account))
    if jira_id is None:
        raise ResponseError(NoSourceDataError(
            detail="JIRA has not been installed to the metadata yet."))
    return jira_id


async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    try:
        filt = FilterJIRAStuff.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    time_from, time_to = filt.resolve_time_from_and_to()
    with sentry_sdk.start_span(op="sdb"):
        async with request.sdb.connection() as conn:
            await get_user_account_status(request.uid, filt.account, conn, request.cache)
            jira_id = await get_jira_installation(filt.account, request.sdb, request.cache)
    mdb = request.mdb

    @sentry_span
    async def epic_flow():
        epic_rows = await mdb.fetch_all(
            select([Issue.id, Issue.key, Issue.title, Issue.updated])
            .where(and_(Issue.acc_id == jira_id,
                        Issue.type == "Epic",
                        Issue.created < time_to,
                        or_(Issue.resolved.is_(None), Issue.resolved >= time_from),
                        )))
        epic_ids = [r[Issue.id.key] for r in epic_rows]
        children_rows = await mdb.fetch_all(
            select([Issue.epic_id, Issue.key])
            .where(Issue.epic_id.in_(epic_ids))
            .order_by(Issue.epic_id))
        children = {k: [i[1] for i in g] for k, g in groupby(
            ((r[0], r[1]) for r in children_rows), key=itemgetter(0))}
        epics = sorted(JIRAEpic(id=r[Issue.key.key],
                                title=r[Issue.title.key],
                                updated=r[Issue.updated.key],
                                children=children.get(r[Issue.id.key], []))
                       for r in epic_rows)
        if mdb.url.dialect == "sqlite":
            for epic in epics:
                epic.updated = epic.updated.replace(tzinfo=timezone.utc)
        return epics

    @sentry_span
    async def labels_flow():
        label_rows = await mdb.fetch_all(
            select([Issue.id, Issue.labels, Issue.updated])
            .where(and_(Issue.acc_id == jira_id,
                        Issue.created < time_to,
                        or_(Issue.resolved.is_(None), Issue.resolved >= time_from),
                        )))
        if mdb.url.dialect == "sqlite":
            for i, row in enumerate(label_rows):
                row = dict(row)
                row[Issue.labels.key] = json.loads(row[Issue.labels.key])
                label_rows[i] = row
        labels = set(chain.from_iterable(r[Issue.labels.key] for r in label_rows))
        labels = {k: JIRALabel(title=k, kind="regular", issues_count=0) for k in labels}
        for row in label_rows:
            updated = row[Issue.updated.key]
            for label in row[Issue.labels.key]:
                label = labels[label]  # type: JIRALabel
                if label.last_used is None or label.last_used < updated:
                    label.last_used = updated
                label.issues_count += 1
        if mdb.url.dialect == "sqlite":
            for label in labels.values():
                label.last_used = label.last_used.replace(tzinfo=timezone.utc)
        return sorted(labels.values())

    with sentry_sdk.start_span(op="mdb"):
        epics, labels = await asyncio.gather(epic_flow(), labels_flow(), return_exceptions=True)
        for r in (epics, labels):
            if isinstance(r, Exception):
                raise r from None
    return model_response(FoundJIRAStuff(epics=epics, labels=labels))
