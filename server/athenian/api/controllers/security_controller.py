from typing import Any, Dict, Optional

from athenian.api import Auth0


def info_from_bearerAuth(auth: Auth0, token: str) -> Optional[Dict[str, Any]]:
    """
    Check and retrieve authentication information from custom bearer token.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.
    """
    return auth.extract_token(token)
