import asyncio
from datetime import timedelta
import functools
from http import HTTPStatus
import logging
import os
import pickle
from random import random
import re
import socket
import struct
from typing import Any, Dict, List, Optional, Sequence, Tuple
import warnings

import aiohttp.web
from aiohttp.web_runner import GracefulExit
import aiomcache
from connexion.decorators.security import get_authorization_info
from connexion.exceptions import AuthenticationProblem, OAuthProblem, Unauthorized
from connexion.lifecycle import ConnexionRequest
from connexion.operations import secure
from connexion.utils import deep_get
with warnings.catch_warnings():
    # this will suppress all warnings in this block
    warnings.filterwarnings("ignore", message="int_from_bytes is deprecated")
    from jose import jwt
from multidict import CIMultiDict
import sentry_sdk
from sqlalchemy import select

from athenian.api.async_utils import gather
from athenian.api.cache import cached
from athenian.api.controllers.account import get_user_account_status
from athenian.api.kms import AthenianKMS
from athenian.api.models.state.models import God, UserToken
from athenian.api.models.web import ForbiddenError, GenericError
from athenian.api.models.web.user import User
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import wraps


class Auth0:
    """Class for Auth0 middleware compatible with aiohttp."""

    AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")
    AUTH0_AUDIENCE = os.getenv("AUTH0_AUDIENCE")
    AUTH0_CLIENT_ID = os.getenv("AUTH0_CLIENT_ID")
    AUTH0_CLIENT_SECRET = os.getenv("AUTH0_CLIENT_SECRET")
    DEFAULT_USER = os.getenv("ATHENIAN_DEFAULT_USER")
    KEY = os.getenv("ATHENIAN_INVITATION_KEY")
    USERINFO_CACHE_TTL = 60  # seconds
    log = logging.getLogger("auth")

    def __init__(self, domain=AUTH0_DOMAIN, audience=AUTH0_AUDIENCE, client_id=AUTH0_CLIENT_ID,
                 client_secret=AUTH0_CLIENT_SECRET, whitelist: Sequence[str] = tuple(),
                 default_user=DEFAULT_USER, key=KEY,
                 cache: Optional[aiomcache.Client] = None, lazy=False, force_user: str = ""):
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
        :param key: Global secret used to encrypt sensitive personal information.
        :param cache: memcached client to cache the user profiles.
        :param lazy: Value that indicates whether Auth0 Management API tokens and JWKS data \
                     must be asynchronously requested at first related method call.
        :param force_user: Ignore all the incoming bearer tokens and make all requests on behalf \
                           of this user ID.
        """
        for var, env_name in ((domain, "AUTH0_DOMAIN"),
                              (audience, "AUTH0_AUDIENCE"),
                              (client_id, "AUTH0_CLIENT_ID"),
                              (client_secret, "AUTH0_CLIENT_SECRET"),
                              (default_user, "ATHENIAN_DEFAULT_USER"),
                              (key, "ATHENIAN_INVITATION_KEY")):
            if not var:
                raise EnvironmentError("%s environment variable must be set." % env_name)
        self._domain = domain
        self._audience = audience
        self._whitelist = whitelist
        self._cache = cache
        self._client_id = client_id
        self._client_secret = client_secret
        self._default_user_id = default_user
        self._default_user = None  # type: Optional[User]
        self._key = key
        self.force_user = force_user
        if force_user:
            self.log.warning("Forced user authorization mode: %s", force_user)
        self._session = aiohttp.ClientSession()
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
        if not self._mgmt_token:
            raise LookupError("Could not acquire the Auth0 Management token.")
        return self._mgmt_token

    async def default_user(self) -> User:
        """Return the user of unauthorized, public requests."""
        if self._default_user is not None:
            return self._default_user
        self._default_user = await self.get_user(self._default_user_id)
        if self._default_user is None:
            message = "Failed to fetch the default user (%s) details. " \
                      "Try changing ATHENIAN_DEFAULT_USER" % self._default_user_id
            self.log.error(message)
            raise GracefulExit(message)
        return self._default_user

    @property
    def domain(self) -> str:
        """Return the assigned Auth0 domain, e.g. "athenian.auth0.com"."""
        return self._domain

    @property
    def audience(self) -> str:
        """Return the assigned Auth0 audience URL, e.g. "https://api.athenian.co"."""
        return self._audience

    @property
    def key(self) -> str:
        """Return the global secret used to encrypt sensitive personal information."""
        return self._key

    async def close(self):
        """Free resources and close connections associated with the object."""
        if self._jwks_loop is not None:
            self._jwks_loop.cancel()
        if self._mgmt_loop is not None:  # this may happen if lazy_mgmt=True
            self._mgmt_loop.cancel()
        session = self._session
        # FIXME(vmarkovtsev): remove this bloody mess when this issue is resolved:
        # https://github.com/aio-libs/aiohttp/issues/1925#issuecomment-575754386
        transports = 0
        all_is_lost = asyncio.Event()
        if session.connector is not None:
            for conn in session.connector._conns.values():
                for handler, _ in conn:
                    proto = getattr(handler.transport, "_ssl_protocol", None)
                    if proto is None:
                        continue
                    transports += 1

                    def connection_lost(orig_lost, exc):
                        orig_lost(exc)
                        nonlocal transports
                        transports -= 1
                        if transports == 0:
                            all_is_lost.set()

                    def eof_received(orig_eof_received):
                        try:
                            orig_eof_received()
                        except AttributeError:
                            # It may happen that eof_received() is called after
                            # _app_protocol and _transport are set to None.
                            # Jeez, asyncio sucks sometimes.
                            pass

                    proto.connection_lost = functools.partial(
                        connection_lost, proto.connection_lost)
                    proto.eof_received = functools.partial(eof_received, proto.eof_received)
        await session.close()
        if transports > 0:
            await all_is_lost.wait()

    def __enter__(self):
        """Monkey-patch connexion.operations.secure.verify_security()."""
        self._verify_security = secure.verify_security

        def verify_security(auth_funcs, required_scopes, function):
            @functools.wraps(function)
            async def wrapper(request: ConnexionRequest):
                # we support two auth methods: JWT and apiKey
                has_auth = "Authorization" in request.headers or "X-API-Key" in request.headers
                if not has_auth or self.force_user:
                    # to reach self._set_user, we have to hardcode "null" as a "magic" JWT
                    request.headers = CIMultiDict(request.headers)
                    request.headers["Authorization"] = "Bearer null"
                # invoke security_controller.info_from_*Auth
                token_info = get_authorization_info(auth_funcs, request, required_scopes)
                # token_info = {"token": <token>, "method": "bearer" or "apikey"}
                await self._set_user(context := request.context, **token_info)
                # check whether the user may access the specified account
                if isinstance(request.json, dict):
                    if (account := request.json.get("account")) is not None:
                        assert isinstance(account, int)
                        with sentry_sdk.configure_scope() as scope:
                            scope.set_tag("account", account)
                        await get_user_account_status(
                            context.uid, account, context.sdb, context.cache)
                    elif (account := getattr(context, "account", None)) is not None:
                        canonical = context.match_info.route.resource.canonical
                        route_specs = context.app["route_spec"]
                        if (spec := route_specs.get(canonical, None)) is not None:
                            try:
                                required = "account" in deep_get(spec, [
                                    "requestBody", "content", "application/json", "schema",
                                    "required",
                                ])
                            except KeyError:
                                required = False
                            if required:
                                request.json["account"] = account

                # finish the auth processing and chain forward
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

    @sentry_span
    async def get_users(self, users: Sequence[str]) -> Dict[str, User]:
        """
        Retrieve several users using Auth0 mgmt API by ID.

        :return: Mapping from user ID to the found user details. Some users may be not found, \
                 some users may be duplicates.
        """
        token = await self.mgmt_token()
        assert len(users) >= 0  # we need __len__

        async def get_batch(batch: List[str]) -> List[User]:
            nonlocal token
            query = "user_id:(%s)" % " ".join('"%s"' % u for u in batch)
            for retries in range(1, 31):
                try:
                    resp = await self._session.get(
                        "https://%s/api/v2/users?q=%s" % (self._domain, query),
                        headers={"Authorization": "Bearer " + token})
                except aiohttp.ClientConnectorError as e:
                    if isinstance(e.__cause__, socket.gaierror) and e.__cause__.errno == -3:
                        self.log.warning("Auth0 Management API: %s", e)
                        # Temporary failure in name resolution
                        await asyncio.sleep(0.1)
                        continue
                    else:
                        raise e from None
                except RuntimeError:
                    # our loop is closed and we are doomed
                    return []
                if resp.status == HTTPStatus.TOO_MANY_REQUESTS:
                    self.log.warning("Auth0 Management API rate limit hit while listing "
                                     "%d/%d users, retry %d",
                                     len(batch), len(users), retries)
                    await asyncio.sleep(0.5 + random())
                elif resp.status in (HTTPStatus.REQUEST_URI_TOO_LONG, HTTPStatus.BAD_REQUEST):
                    if len(batch) == 1:
                        return []
                    m = len(batch) // 2
                    self.log.warning("Auth0 Management API /users raised HTTP %d, bisecting "
                                     "%d/%d -> %d, %d",
                                     resp.status, len(batch), len(users), m, len(batch) - m)
                    b1, b2 = await gather(get_batch(batch[:m]), get_batch(batch[m:]))
                    return b1 + b2
                elif resp.status == HTTPStatus.UNAUTHORIZED:
                    # force refresh the token
                    self._mgmt_loop.cancel()
                    self._mgmt_loop = None
                    self._mgmt_token = None
                    token = await self.mgmt_token()
                else:
                    if resp.status >= 400:
                        self.log.error("Auth0 Management API /users raised HTTP %d: %s",
                                       resp.status, await resp.json())
                    break
            else:  # for retries in range
                return []
            if resp.status != HTTPStatus.OK:
                return []
            found = await resp.json()
            return [User.from_auth0(**u, encryption_key=self.key) for u in found]

        return {u.id: u for u in await get_batch(list(users))}

    async def _fetch_jwks_loop(self) -> None:
        while True:
            await self._fetch_jwks()
            await asyncio.sleep(3600)  # 1 hour

    async def _acquire_management_token_loop(self) -> None:
        while True:
            expires_in = await self._acquire_management_token(0)
            await asyncio.sleep(expires_in)

    async def _fetch_jwks(self) -> None:
        req = await self._session.get("https://%s/.well-known/jwks.json" % self._domain)
        jwks = await req.json()
        self.log.info("Fetched %d JWKS records", len(jwks))
        self._kids = {key["kid"]: {k: key[k] for k in ("kty", "kid", "use", "n", "e")}
                      for key in jwks["keys"]}
        self._kids_event.set()

    async def _acquire_management_token(self, attempt: int) -> float:
        data = {}
        max_attempts = 10
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
            # do not use %s - Sentry does not display it properly
            if attempt >= max_attempts:
                self.log.exception("Failed to renew the Auth0 management token: " + str(data))
                raise GracefulExit() from e
            self.log.warning("Failed to renew the Auth0 management token %d / %d: %s: %s",
                             attempt + 1, max_attempts, e, data)
            await asyncio.sleep(0.1)
            return await self._acquire_management_token(attempt + 1)
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
        if token == "null":
            return await self.default_user()
        return await self._get_user_info_cached(token)

    @cached(
        exptime=lambda self, **_: self.USERINFO_CACHE_TTL,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda token, **_: (token,),
        cache=lambda self, **_: self._cache,
    )
    async def _get_user_info_cached(self, token: str) -> User:
        resp = await self._session.get("https://%s/userinfo" % self._domain,
                                       headers={"Authorization": "Bearer " + token})
        try:
            user = await resp.json()
        except aiohttp.ContentTypeError:
            raise ResponseError(GenericError(
                "/errors/Auth0", title=resp.reason, status=resp.status,
                detail=await resp.text()))
        if resp.status != 200:
            raise ResponseError(GenericError(
                "/errors/Auth0", title=resp.reason, status=resp.status,
                detail=user.get("description", str(user))))
        return User.from_auth0(**user, encryption_key=self.key)

    async def _set_user(self, request: AthenianWebRequest, token: str, method: str) -> None:
        if method == "bearer":
            token_info = await self._extract_bearer_token(token)
            request.uid, request.account = token_info["sub"], None
        elif method == "apikey":
            request.uid, request.account = await self._extract_api_key(token, request)
        else:
            raise AssertionError("Unsupported auth method: %s" % method)

        god = await request.sdb.fetch_one(
            select([God.mapped_id]).where(God.user_id == request.uid))
        if god is not None:
            request.god_id = request.uid
            if "X-Identity" in request.headers:
                mapped_id = request.headers["X-Identity"]
            else:
                mapped_id = god[God.mapped_id.key]
            if mapped_id is not None:
                request.uid = mapped_id
                self.log.info("God mode: %s became %s", request.god_id, mapped_id)

        request.is_default_user = request.uid == self._default_user_id
        sentry_sdk.set_user({"id": request.uid})

        async def get_user_info():
            if method != "bearer" or (god is not None and request.god_id is not None):
                user_info = await self.get_user(request.uid)
            else:
                user_info = await self._get_user_info(token)
            sentry_sdk.set_user({"username": user_info.login, "email": user_info.email})
            return user_info

        request.user = get_user_info

    async def _extract_bearer_token(self, token: str) -> Dict[str, Any]:
        if token == "null":
            return {"sub": self.force_user or self._default_user_id}
        # People who understand what's going on here:
        # - @dennwc
        # - @vmarkovtsev
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

    async def _extract_api_key(self, token: str, request: AthenianWebRequest) -> Tuple[str, int]:
        kms = request.app["kms"]  # type: AthenianKMS
        if kms is None:
            raise AuthenticationProblem(
                status=HTTPStatus.UNAUTHORIZED,
                title="Unable to authenticate with an API key.",
                detail="The backend was not properly configured and there is no connection with "
                       "Google Key Management Service to decrypt API keys.")
        try:
            plaintext = await kms.decrypt(token)
        except aiohttp.ClientResponseError:
            raise Unauthorized()
        try:
            token_id = struct.unpack("<q", plaintext)[0]
        except (ValueError, struct.error):
            raise Unauthorized() from None
        token_obj = await request.sdb.fetch_one(
            select([UserToken]).where(UserToken.id == token_id))
        if token_obj is None:
            raise Unauthorized()
        uid = token_obj[UserToken.user_id.key]
        account = token_obj[UserToken.account_id.key]
        return uid, account


def disable_default_user(func):
    """Decorate an endpoint handler to raise 403 if the user is the default one."""
    async def wrapped_disable_default_user(request: AthenianWebRequest,
                                           *args, **kwargs) -> aiohttp.web.Response:
        if request.is_default_user:
            raise ResponseError(ForbiddenError("%s is the default user" % request.uid))
        return await func(request, *args, **kwargs)

    wraps(wrapped_disable_default_user, func)
    return wrapped_disable_default_user
