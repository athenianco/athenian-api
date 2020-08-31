from aiohttp import web

from athenian.api.models.web import InvalidRequestError
from athenian.api.models.web.filter_jira_stuff import FilterJIRAStuff
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError


async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    try:
        filt = FilterJIRAStuff.from_dict(body)
    except ValueError as e:
        # for example, passing a date with day=32
        return ResponseError(InvalidRequestError("?", detail=str(e))).response
    _ = filt
    raise NotImplementedError
