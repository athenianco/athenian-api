from typing import Any, Dict, Optional

from aiohttp import web

from athenian.api.request import AthenianWebRequest


def info_from_bearerAuth(token: str) -> Optional[Dict[str, Any]]:
    """
    Check and retrieve authentication information from custom bearer token.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.

    The real work happens in Auth0._set_user().
    """
    return {"token": token}


async def create_token(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a new Personal Access Token for the current user and the specified account."""
    pass


async def delete_token(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a Personal Access Token belonging to the user."""
    pass


async def patch_token(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Change Personal Access Token's details."""
    pass


async def list_tokens(request: AthenianWebRequest, id: int) -> web.Response:
    """List Personal Access Tokens of the user in the account."""
    pass
