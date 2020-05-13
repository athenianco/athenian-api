from aiohttp import web


async def get_contributors(request: web.Request, id: int) -> web.Response:
    """List all the contributors belonging to the specified account.

    :param id: Numeric identifier of the account.

    """
    return web.Response(status=200)
