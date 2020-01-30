from typing import Dict

from jose import jwt


def info_from_bearerAuth(token: str) -> Dict[str, str]:
    """
    Check and retrieve authentication information from custom bearer token.

    Returned value will be passed in 'token_info' parameter of your operation function, if there
    is one. 'sub' or 'uid' will be set in 'user' parameter of your operation function, if there
    is one. Should return None if auth is invalid or does not allow access to called API.
    """
    return jwt.decode(
        token,
        "athenian",
        algorithms=["RS256", "HS256"],
        audience="https://api.owl.athenian.co",
        issuer="https://athenian.auth0.com/",
    )
