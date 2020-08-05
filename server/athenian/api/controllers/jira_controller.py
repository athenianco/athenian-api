from aiohttp import web

from athenian.api.request import AthenianWebRequest


async def filter_jira_stuff(request: AthenianWebRequest, body: dict) -> web.Response:
    """Find JIRA epics and labels used in the specified date interval."""
    raise NotImplementedError
