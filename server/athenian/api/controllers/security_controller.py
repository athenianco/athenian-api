from typing import Any, Dict, Optional


def info_from_bearerAuth(token: str) -> Optional[Dict[str, Any]]:
    """
    Check and retrieve authentication information from custom bearer token.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.

    The real work happens in Auth0._set_user().
    """
    return {"token": token}
