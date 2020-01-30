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
from connexion.exceptions import OAuthProblem
from connexion.operations import secure
from jose import jwt

from athenian.api.models.web.user import User


class Auth0:
    """Class for Auth0 middleware compatible with aiohttp."""

    AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
    AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
    DEFAULT_USER = "github|60340680"
    log = logging.getLogger("auth")

    def __init__(self, domain=AUTH0_DOMAIN, audience=AUTH0_AUDIENCE, client_id=AUTH0_CLIENT_ID,
                 client_secret=AUTH0_CLIENT_SECRET, whitelist: Sequence[str] = tuple(),
                 default_user=DEFAULT_USER, lazy=False):
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
        :param default_user: Default user ID - the one that's assigned to public, unauthorized \
                             requests.
        :param lazy: Value that indicates whether Auth0 Management API tokens and JWKS data \
                     must be asynchronously requested at first related method call.
        """
        self._domain = domain
        self._audience = audience
        self._whitelist = whitelist
        self._session = aiohttp.ClientSession()
        self._client_id = client_id
        self._client_secret = client_secret
        self._default_user_id = default_user
        self._default_user = None  # type: Optional[User]
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

    def kids_sync(self) -> Dict[str, Any]:
        """Return the mapping kid -> Auth0 jwks record with that kid."""
        if self._jwks_loop is None:
            self._jwks_loop = asyncio.ensure_future(self._fetch_jwks_loop())
        return self._kids

    async def mgmt_token(self) -> str:
        """Return the Auth0 management API token."""
        if self._mgmt_loop is None:
            self._mgmt_loop = asyncio.ensure_future(self._acquire_management_token_loop())
        await self._mgmt_event.wait()
        return self._mgmt_token

    async def default_user(self) -> User:
        """Return the user of unauthorized, public requests."""
        if self._default_user is not None:
            return self._default_user
        self._default_user = (await self.get_users([self._default_user_id]))[0]
        return self._default_user

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

    def _verify_authorization_token(self, request: aiohttp.web.Request, token_info_func,
                                    ) -> Optional[Dict[str, Any]]:
        authorization = request.headers.get("Authorization")
        if not authorization:
            authorization = "Bearer null"

        try:
            auth_type, token = authorization.split(None, 1)
        except ValueError:
            raise OAuthProblem(description="Invalid authorization header")

        if auth_type.lower() != "bearer":
            raise OAuthProblem(description="Invalid authorization header")

        token_info = token_info_func(self, token)
        if token_info is None:
            raise OAuthProblem(description="Provided token is not valid")

        return token_info

    def __enter__(self):
        """Monkey-patch connexion.operations.secure.verify_bearer()."""
        self._verify_bearer = secure.verify_bearer

        def verify_bearer(bearer_info_func):
            def wrapper(request, required_scopes):
                return self._verify_authorization_token(request, bearer_info_func)
            return wrapper

        secure.verify_bearer = verify_bearer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Revert the monkey-patch."""
        secure.verify_bearer = self._verify_bearer
        del self._verify_bearer

    @aiohttp.web.middleware
    async def middleware(self, request, handler):
        """Middleware function compatible with aiohttp that bookmarks the instance."""
        request.auth = self
        request.uid = lambda: request["user"]

        async def get_user_info():
            token = request["token_info"]["token"]
            return await self._get_user_info(token)

        request.user = get_user_info
        return await handler(request)

    async def get_users(self, users: Sequence[str]) -> List[Optional[User]]:
        """Fetch several users by their IDs."""
        token = await self.mgmt_token()
        assert len(users) >= 0  # we need __len__

        async def get_batch(batch: List[str]) -> Optional[List[User]]:
            for retries in range(1, 31):
                query = "user_id:(%s)" % " ".join('"%s"' % u for u in batch)
                try:
                    resp = await self._session.get(
                        "https://%s/api/v2/users?q=%s" % (self._domain, query),
                        headers={"Authorization": "Bearer " + token})
                except RuntimeError:
                    # our loop is closed and we are doomed
                    return None
                if resp.status == HTTPStatus.TOO_MANY_REQUESTS:
                    self.log.warning("Auth0 Management API rate limit hit while listing "
                                     "%d/%d users, retry %d",
                                     len(batch), len(users), retries)
                    await asyncio.sleep(0.5 + random())
                elif resp.status in (HTTPStatus.REQUEST_URI_TOO_LONG, HTTPStatus.BAD_REQUEST):
                    if len(batch) == 1:
                        return None
                    m = len(batch) // 2
                    self.log.warning("Auth0 Management API /users raised HTTP %d, bisecting "
                                     "%d/%d -> %d, %d",
                                     resp.status, len(batch), len(users), m, len(batch) - m)
                    b1, b2 = await asyncio.gather(get_batch(batch[:m]), get_batch(batch[m:]))
                    if b1 is None or b2 is None:
                        return None
                    return b1 + b2
                else:
                    if resp.status >= 400:
                        self.log.error("Auth0 Management API /users raised HTTP %d", resp.status)
                    break
            else:  # for retries in range
                return None
            if resp.status != HTTPStatus.OK:
                return None
            found = await resp.json()
            return [User.from_auth0(**u) for u in found]

        return sorted(set(await get_batch(list(users))))

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

    async def _get_user_info(self, token: str) -> User:
        # TODO: cache based on decoded claims
        if token == "gkwillie":
            return await self.default_user()
        resp = await self._session.get("https://%s/userinfo" % self._domain,
                                       headers={"Authorization": "Bearer " + token})
        user = await resp.json()
        return User.from_auth0(**user)

    def extract_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode a JWT token and validate it."""
        if token == "null":
            # gkwillie
            return {"sub": self._default_user_id, "token": "gkwillie"}
        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.JWTError:
            raise OAuthProblem(description="Invalid header. Use an RS256 signed JWT Access Token.")
        if unverified_header["alg"] != "RS256":
            raise OAuthProblem(
                description="Invalid algorithm. Use an RS256 signed JWT Access Token.")

        kids = self.kids_sync()
        if kids is None:
            raise OAuthProblem(
                description="Server has not fetched Auth0 RSA keys yet, please try again")
        try:
            rsa_key = kids[unverified_header["kid"]]
        except KeyError:
            raise OAuthProblem(description="Unable to find an appropriate Auth0 RSA key")
        try:
            claims = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer="https://%s/" % self._domain,
            )
        except jwt.ExpiredSignatureError:
            raise OAuthProblem(description="JWT expired")
        except jwt.JWTClaimsError:
            raise OAuthProblem(description="invalid claims, please check the audience and issuer")
        except jwt.JWTError:
            raise OAuthProblem(description="Unable to parse the authentication token.")
        claims["token"] = token
        breakpoint()
        return claims
