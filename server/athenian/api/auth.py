import asyncio
from datetime import timedelta
import functools
from http import HTTPStatus
import logging
import os
from random import random
import re
from typing import Any, Dict, List, Optional, Sequence

import aiohttp.web
from aiohttp.web_runner import GracefulExit
from connexion.decorators.security import get_authorization_info
from connexion.exceptions import OAuthProblem
from connexion.lifecycle import ConnexionRequest
from connexion.operations import secure
from jose import jwt
from multidict import CIMultiDict
from sqlalchemy import select

from athenian.api.models.state.models import God
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

    async def kids(self) -> Dict[str, Any]:
        """Return the mapping kid -> Auth0 jwks record with that kid; wait until fetched."""
        if self._jwks_loop is None:
            self._jwks_loop = asyncio.ensure_future(self._fetch_jwks_loop())
        await self._kids_event.wait()
        return self._kids

    async def mgmt_token(self) -> str:
        """Return the Auth0 management API token; wait until fetched."""
        if self._mgmt_loop is None:
            self._mgmt_loop = asyncio.ensure_future(self._acquire_management_token_loop())
        await self._mgmt_event.wait()
        return self._mgmt_token

    async def default_user(self) -> User:
        """Return the user of unauthorized, public requests."""
        if self._default_user is not None:
            return self._default_user
        self._default_user = await self.get_user(self._default_user_id)
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

    def __enter__(self):
        """Monkey-patch connexion.operations.secure.verify_security()."""
        self._verify_security = secure.verify_security

        def verify_security(auth_funcs, required_scopes, function):
            @functools.wraps(function)
            async def wrapper(request: ConnexionRequest):
                if "Authorization" not in request.headers:
                    # Otherwise we will never reach self.extract_token and self._set_user
                    request.headers = CIMultiDict(request.headers)
                    request.headers["Authorization"] = "Bearer null"
                token_info = get_authorization_info(auth_funcs, request, required_scopes)
                # token_info = {"token": <token>} at this point, now do the real work
                await self._set_user(request.context, token_info["token"])
                return await function(request)
            return wrapper

        secure.verify_security = verify_security
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Revert the monkey-patch."""
        secure.verify_security = self._verify_security
        del self._verify_security

    async def get_user(self, user: str) -> Optional[User]:
        """Retrieve a user using Auth0 mgmt API by ID."""
        users = await self.get_users([user])
        if len(users) == 0:
            return None
        return next(iter(users.values()))

    async def get_users(self, users: Sequence[str]) -> Dict[str, User]:
        """
        Retrieve several users using Auth0 mgmt API by ID.

        :return: Mapping from user ID to the found user details. Some users may be not found, \
                 some users may be duplicates.
        """
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

        return {u.id: u for u in set(await get_batch(list(users))) if u is not None}

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
        # TODO(dennwc): cache based on decoded claims
        if token == "null":
            return await self.default_user()
        resp = await self._session.get("https://%s/userinfo" % self._domain,
                                       headers={"Authorization": "Bearer " + token})
        user = await resp.json()
        return User.from_auth0(**user)

    async def _set_user(self, request, token: str) -> None:
        token_info = await self._extract_token(token)
        request.auth = self
        request.uid = token_info["sub"]
        god = await request.sdb.fetch_one(
            select([God.mapped_id]).where(God.user_id == request.uid))
        if god is not None:
            request.god_id = request.uid
            mapped_id = god[God.mapped_id.key]
            if mapped_id is not None:
                request.uid = mapped_id
                self.log.info("God mode: %s became %s", request.god_id, mapped_id)

        async def get_user_info():
            if god is not None and request.god_id is not None:
                return await self.get_user(request.uid)
            return await self._get_user_info(token)

        request.user = get_user_info

    async def _extract_token(self, token: str) -> Dict[str, Any]:
        if token == "null":
            return {"sub": self._default_user_id}
        try:
            unverified_header = jwt.get_unverified_header(token)
        except jwt.JWTError as e:
            raise OAuthProblem(
                description="Invalid header: %s. Use an RS256 signed JWT Access Token." % e)
        if unverified_header["alg"] != "RS256":
            raise OAuthProblem(
                description="Invalid algorithm %s. Use an RS256 signed JWT Access Token." %
                unverified_header["alg"])

        kids = await self.kids()
        try:
            rsa_key = kids[unverified_header["kid"]]
        except KeyError:
            raise OAuthProblem(description="Unable to find the matching Auth0 RSA public key")
        try:
            return jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer="https://%s/" % self._domain,
            )
        except jwt.ExpiredSignatureError as e:
            raise OAuthProblem(description="JWT expired: %s" % e)
        except jwt.JWTClaimsError as e:
            raise OAuthProblem(description="invalid claims: %s" % e)
        except jwt.JWTError as e:
            raise OAuthProblem(description="Unable to parse the authentication token: %s" % e)
