import asyncio
from datetime import timedelta
import logging
import os
import re
from typing import Iterable, List

import aiohttp.web
from aiohttp.web_runner import GracefulExit
import dateutil.parser
from jose import jwt


class AuthError(Exception):
    """Auth error with an HTTP status code."""

    def __init__(self, error, status_code):
        """Create a new auth error."""
        self.error = error
        self.status_code = status_code


class User:
    """User profile information."""

    def __init__(self, sub: str, email: str, name: str, picture: str, updated_at: str, **kwargs):
        """Create a new User object."""
        self.id = sub
        self.email = email
        self.name = name
        self.picture = picture
        self.updated = dateutil.parser.parse(updated_at)


class Auth0:
    """Class for Auth0 middleware compatible with aiohttp."""

    AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
    AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
    log = logging.getLogger("auth")

    def __init__(self, domain=AUTH0_DOMAIN, audience=AUTH0_AUDIENCE, client_id=AUTH0_CLIENT_ID,
                 client_secret=AUTH0_CLIENT_SECRET, whitelist=tuple(), loop=None):
        """Create a new Auth0 middleware."""
        self._domain = domain
        self._audience = audience
        self._whitelist = whitelist
        self._session = aiohttp.ClientSession()
        self._client_id = client_id
        self._client_secret = client_secret
        self._mgmt_event = asyncio.Event()
        self._mgmt_token = None  # type: str
        if loop is None:
            loop = asyncio.get_event_loop()
        self._loop = loop
        loop.call_soon(asyncio.ensure_future, self._acquire_management_token())

    async def mgmt_token(self) -> str:
        """Return the Auth0 management API token."""
        await self._mgmt_event.wait()
        return self._mgmt_token

    async def close(self):
        """Free resources associated with the object."""
        await self._session.close()

    @classmethod
    def ensure_static_configuration(cls):
        """Check that the authentication is properly configured by the environment variables \
        and raise an exception if it is not."""
        if not (cls.AUTH0_DOMAIN and cls.AUTH0_AUDIENCE
                and cls.AUTH0_CLIENT_ID and cls.AUTH0_CLIENT_SECRET):  # noqa: W503
            cls.log.error("API authentication requires setting AUTH0_DOMAIN, AUTH0_AUDIENCE, "
                          "AUTH0_CLIENT_ID and AUTH0_CLIENT_SECRET")
            raise EnvironmentError("AUTH0_DOMAIN, AUTH0_AUDIENCE, AUTH0_CLIENT_ID, "
                                   "AUTH0_CLIENT_SECRET must be set")

    async def get_users(self, users: Iterable[str]) -> List[User]:
        """Fetch several users by their IDs."""
        token = await self.mgmt_token()

        async def get_user(user: str):
            resp = await self._session.get("https://%s/api/v2/users/%s" % (self._domain, user),
                                           headers={"Authorization": "Bearer " + token})
            user = await resp.json()
            return User(**user)

        return await asyncio.gather([get_user(u) for u in users])

    async def _acquire_management_token(self) -> None:
        try:
            resp = await self._session.post("https://%s/oauth/token" % self._domain, headers={
                "content-type": "application/x-www-form-urlencoded",
            }, data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "audience": self._audience,
            })
            data = await resp.json()
            self._mgmt_token = data["access_token"]
            self._mgmt_event.set()
            expires_in = int(data["expires_in"])
        except Exception as e:
            self.log.exception("Failed to renew the mgmt Auth0 token")
            raise GracefulExit() from e
        self.log.info("Acquired new Auth0 management token %s...%s for the next %s",
                      self._mgmt_token[:12], self._mgmt_token[-12:], timedelta(seconds=expires_in))
        expires_in -= 5 * 60  # 5 minutes earlier
        if expires_in < 0:
            expires_in = 0
        self._loop.call_later(expires_in, asyncio.ensure_future, self._acquire_management_token())

    def _is_whitelisted(self, request: aiohttp.web.Request) -> bool:
        for pattern in self._whitelist:
            if re.match(pattern, request.path):
                return True
        return False

    def _get_token_auth_header(self, request: aiohttp.web.Request) -> str:
        """Obtain the access token from the Authorization Header."""
        try:
            auth = request.headers["Authorization"]
        except KeyError:
            raise AuthError("Authorization header is expected", 401) from None

        parts = auth.split()

        if parts[0].lower() != "bearer":
            raise AuthError("Authorization header must start with Bearer", 401)
        elif len(parts) == 1:
            raise AuthError("Token not found", 401)
        elif len(parts) > 2:
            raise AuthError('Authorization header must be "Bearer <token>"', 401)

        token = parts[1]
        return token

    async def _get_user_info(self, token):
        # TODO: cache based on decoded claims
        resp = await self._session.get("https://%s/userinfo" % self._domain,
                                       headers={"Authorization": "Bearer " + token})
        user = await resp.json()
        return User(**user)

    @aiohttp.web.middleware
    async def middleware(self, request, handler):
        """Middleware function compatible with aiohttp."""
        if self._is_whitelisted(request):
            return await handler(request)

        # FIXME(vmarkovtsev): remove the following short circuit when the frontend is ready
        request.user = User("auth0:vmarkovtsev", "vadim@athenian.co", "Vadim Markovtsev",
                            "https://avatars1.githubusercontent.com/u/2793551",
                            "2020-01-08 11:12:04.985028")
        return await handler(request)

        try:
            token = self._get_token_auth_header(request)
            unverified_header = jwt.get_unverified_header(token)
        except AuthError as e:
            return aiohttp.web.Response(body=e.error, status=e.status_code)
        except jwt.JWTError:
            return aiohttp.web.Response(body="Invalid header."
                                        " Use an RS256 signed JWT Access Token.", status=401)
        if unverified_header["alg"] != "RS256":
            return aiohttp.web.Response(body="Invalid header."
                                        " Use an RS256 signed JWT Access Token.", status=401)

        # TODO(dennwc): update periodically instead of pulling it on each request
        jwks_req = await self._session.get("https://%s/.well-known/jwks.json" % self._domain)
        jwks = await jwks_req.json()

        # There might be multiple keys, find the one used to sign this request
        rsa_key = None
        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = {k: key[k] for k in ("kty", "kid", "use", "n", "e")}

        if rsa_key is None:
            return aiohttp.web.Response(body="Unable to find an appropriate key", status=401)

        try:
            jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer="https://%s/" % self._domain,
            )
        except jwt.ExpiredSignatureError:
            return aiohttp.web.Response(body="token expired", status=401)
        except jwt.JWTClaimsError:
            return aiohttp.web.Response(
                body="incorrect claims, please check the audience and issuer", status=401)
        except Exception:
            return aiohttp.web.Response(
                body="Unable to parse the authentication token.", status=401)

        try:
            user = await self._get_user_info(token)
            self.log.info("User %s", vars(user))
        except Exception:
            return aiohttp.web.Response(body="Your auth token is likely revoked.", status=401)

        request.user = user
        return await handler(request)
