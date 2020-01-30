import asyncio
from datetime import timedelta
from http import HTTPStatus
import logging
import os
from random import random
import re
from typing import Any, Dict, List, Optional, Sequence

import aiohttp.web
from aiohttp.web_runner import GracefulExit
from jose import jwt

from athenian.api.models.web.user import User


class AuthError(Exception):
    """Auth error with an HTTP status code."""

    def __init__(self, error, status_code):
        """Create a new auth error."""
        self.error = error
        self.status_code = status_code


class Auth0:
    """Class for Auth0 middleware compatible with aiohttp."""

    AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
    AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
    log = logging.getLogger("auth")

    def __init__(self, domain=AUTH0_DOMAIN, audience=AUTH0_AUDIENCE, client_id=AUTH0_CLIENT_ID,
                 client_secret=AUTH0_CLIENT_SECRET, whitelist: Sequence[str] = tuple(),
                 lazy=False):
        """
        Create a new Auth0 middleware.

        See:
          - https://auth0.com/docs/tokens/guides/get-access-tokens#control-access-token-audience
          - https://auth0.com/docs/api-auth/tutorials/client-credentials

        :param domain: Auth0 domain.
        :param audience: JWT audience parameter.
        :param client_id: Application's Client ID.
        :param client_secret: Application's Client Secret.
        :param whitelist: Routes that do not need authorization.
        :param lazy: Value that indicates whether Auth0 Management API tokens and JWKS data \
                     must be asynchronously requested at first related method call.
        """
        self._domain = domain
        self._audience = audience
        self._whitelist = whitelist
        self._session = aiohttp.ClientSession()
        self._client_id = client_id
        self._client_secret = client_secret
        self._kids_event = asyncio.Event()
        if not lazy:
            self._jwks_loop = asyncio.ensure_future(self._fetch_jwks_loop())
        else:
            self._jwks_loop = None  # type: Optional[asyncio.Future]
        self._kids: Dict[str, Any] = {}
        self._mgmt_event = asyncio.Event()
        self._mgmt_token = None  # type: Optional[str]
        if not lazy:
            self._mgmt_loop = asyncio.ensure_future(self._acquire_management_token_loop())
        else:
            self._mgmt_loop = None  # type: Optional[asyncio.Future]

    async def kids(self) -> Dict[str, Any]:
        """Return the mapping kid -> Auth0 jwks record with that kid."""
        if self._jwks_loop is None:
            self._jwks_loop = asyncio.ensure_future(self._fetch_jwks_loop())
        await self._kids_event.wait()
        return self._kids

    async def mgmt_token(self) -> str:
        """Return the Auth0 management API token."""
        if self._mgmt_loop is None:
            self._mgmt_loop = asyncio.ensure_future(self._acquire_management_token_loop())
        await self._mgmt_event.wait()
        return self._mgmt_token

    async def close(self):
        """Free resources associated with the object."""
        if self._jwks_loop is not None:
            self._jwks_loop.cancel()
        if self._mgmt_loop is not None:  # this may happen if lazy_mgmt=True
            self._mgmt_loop.cancel()
        session = self._session
        # FIXME(vmarkovtsev): remove this bloody mess when this issue is resolved:
        # https://github.com/aio-libs/aiohttp/issues/1925#issuecomment-575754386
        transports = 0
        all_is_lost = asyncio.Event()
        for conn in session.connector._conns.values():
            for handler, _ in conn:
                proto = getattr(handler.transport, "_ssl_protocol", None)
                if proto is None:
                    continue
                transports += 1
                orig_lost = proto.connection_lost
                orig_eof_received = proto.eof_received

                def connection_lost(exc):
                    orig_lost(exc)
                    nonlocal transports
                    transports -= 1
                    if transports == 0:
                        all_is_lost.set()

                def eof_received():
                    try:
                        orig_eof_received()
                    except AttributeError:
                        # It may happen that eof_received() is called after
                        # _app_protocol and _transport are set to None.
                        # Jeez, asyncio sucks sometimes.
                        pass

                proto.connection_lost = connection_lost
                proto.eof_received = eof_received
        await session.close()
        if transports > 0:
            await all_is_lost.wait()

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

    @aiohttp.web.middleware
    async def middleware(self, request, handler):
        """Middleware function compatible with aiohttp."""
        request.auth = self
        if self._is_whitelisted(request):
            return await handler(request)
        response = await self._set_user(request)
        if response is not None:
            return response
        return await handler(request)

    async def get_users(self, users: Sequence[str]) -> List[Optional[User]]:
        """Fetch several users by their IDs."""
        token = await self.mgmt_token()
        assert len(users) >= 0  # we need __len__

        async def get_user(user: str) -> Optional[User]:
            for retries in range(1, 31):
                try:
                    resp = await self._session.get(
                        "https://%s/api/v2/users/%s" % (self._domain, user),
                        headers={"Authorization": "Bearer " + token})
                    if resp.status == HTTPStatus.TOO_MANY_REQUESTS:
                        self.log.warning(
                            "Auth0 Management API rate limit hit while listing %d users, retry %d",
                            len(users), retries)
                        await asyncio.sleep(0.5 + random())
                    else:
                        break
                except RuntimeError:
                    # our loop is closed and we are doomed
                    return None
            else:
                return None
            user = await resp.json()
            return User.from_auth0(**user)

        return await asyncio.gather(*[get_user(u) for u in users])

    async def _fetch_jwks_loop(self) -> None:
        while True:
            await self._fetch_jwks()
            await asyncio.sleep(3600)  # 1 hour

    async def _acquire_management_token_loop(self) -> None:
        while True:
            expires_in = await self._acquire_management_token()
            await asyncio.sleep(expires_in)

    async def _fetch_jwks(self) -> None:
        req = await self._session.get("https://%s/.well-known/jwks.json" % self._domain)
        jwks = await req.json()
        req.close()
        self.log.info("Fetched %d JWKS records", len(jwks))
        self._kids = {key["kid"]: {k: key[k] for k in ("kty", "kid", "use", "n", "e")}
                      for key in jwks["keys"]}
        self._kids_event.set()

    async def _acquire_management_token(self) -> float:
        try:
            resp = await self._session.post("https://%s/oauth/token" % self._domain, headers={
                "content-type": "application/x-www-form-urlencoded",
            }, data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "audience": "https://%s/api/v2/" % self._domain,
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
        return expires_in

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
        return User.from_auth0(**user)

    async def _set_user(self, request) -> Optional[aiohttp.web.Response]:
        # FIXME(vmarkovtsev): remove the following short circuit when the frontend is ready
        request.user = (await self.get_users(["auth0|5e1f6dfb57bc640ea390557b"]))[0]
        return

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

        kids = await self.kids()
        try:
            rsa_key = kids[unverified_header["kid"]]
        except KeyError:
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
            self.log.info("User %s", user)
        except Exception:
            return aiohttp.web.Response(body="Your auth token is likely revoked.", status=401)

        request.user = user
