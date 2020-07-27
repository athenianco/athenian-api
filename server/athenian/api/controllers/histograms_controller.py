from aiohttp import web

from athenian.api.request import AthenianWebRequest


async def calc_histogram_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Calculate histograms over PR distributions."""
    raise NotImplementedError
