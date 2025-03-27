import asyncio
import base64
import io
import logging
import os
from typing import Union

from gcloud.aio.auth import session
from gcloud.aio.kms import KMS, encode

from athenian.api import metadata

session.log.exception = session.log.warning


class AthenianKMS:
    """Google Cloud Key Management Service for Athenian API."""

    log = logging.getLogger("%s.kms" % metadata.__package__)
    timeout_retries = 10

    def __init__(self):
        """Initialize a new instance of AthenianKMS class."""
        evars = {}
        for var, env_names in (
            ("keyproject", ("GOOGLE_KMS_PROJECT", "GOOGLE_CLOUD_PROJECT")),
            ("keyring", ("GOOGLE_KMS_KEYRING",)),
            ("keyname", ("GOOGLE_KMS_KEYNAME",)),
        ):
            for env_name in env_names:
                try:
                    evars[var] = os.environ[env_name]
                    break
                except KeyError:
                    continue
            else:
                raise EnvironmentError(
                    "%s must be defined, see https://cloud.google.com/kms/docs/reference/rest"
                    % env_names[0],
                )

        self._evars = evars

        service_file_inline = os.getenv(
            "GOOGLE_KMS_SERVICE_ACCOUNT_JSON_INLINE")
        if service_file_inline is not None:
            self._service_file = io.StringIO(service_file_inline)
        else:
            self._service_file = os.getenv("GOOGLE_KMS_SERVICE_ACCOUNT_JSON")

        self._kms = None
        self.log.info(
            "Using Google KMS %(keyproject)s/%(keyring)s/%(keyname)s", evars)

    async def _get_kms(self) -> KMS:
        if self._kms is None:
            self._kms = KMS(**self._evars, service_file=self._service_file)

        return self._kms

    async def encrypt(self, plaintext: Union[bytes, str]) -> str:
        """Encrypt text using Google KMS."""
        kms = await self._get_kms()
        for attempt in range(self.timeout_retries):
            try:
                return await kms.encrypt(encode(plaintext))
            except asyncio.TimeoutError as e:
                self.log.warning("encrypt attempt %d", attempt + 1)
                if attempt == self.timeout_retries - 1:
                    raise e from None

    async def decrypt(self, ciphertext: str) -> bytes:
        """Decrypt text using Google KMS."""
        # we cannot use gcloud.aio.kms.decode because it converts bytes to string with str.decode()
        kms = await self._get_kms()
        for attempt in range(self.timeout_retries):
            try:
                payload = await kms.decrypt(ciphertext)
            except asyncio.TimeoutError as e:
                self.log.warning("decrypt attempt %d", attempt + 1)
                if attempt == self.timeout_retries - 1:
                    raise e from None
            else:
                variant = payload.replace("-", "+").replace("_", "/")
                return base64.b64decode(variant)

    async def close(self):
        """Close the underlying HTTPS session."""
        if self._kms is not None:
            await self._kms.close()
