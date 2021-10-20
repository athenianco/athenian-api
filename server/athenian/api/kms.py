import asyncio
import base64
import io
import logging
import os
from typing import Union

from gcloud.aio.auth import session
from gcloud.aio.kms import encode, KMS

from athenian.api import metadata


session.log.exception = session.log.warning


class AthenianKMS:
    """Google Cloud Key Management Service for Athenian API."""

    log = logging.getLogger("%s.kms" % metadata.__package__)
    timeout_retries = 10

    def __init__(self):
        """Initialize a new instance of AthenianKMS class."""
        evars = {}
        for var, env_name in (("keyproject", "GOOGLE_KMS_PROJECT"),
                              ("keyring", "GOOGLE_KMS_KEYRING"),
                              ("keyname", "GOOGLE_KMS_KEYNAME")):
            evars[var] = x = os.getenv(env_name)
            if x is None:
                raise EnvironmentError(
                    "%s must be defined, see https://cloud.google.com/kms/docs/reference/rest"
                    % env_name)
        service_file_inline = os.getenv("GOOGLE_KMS_SERVICE_ACCOUNT_JSON_INLINE")
        if service_file_inline is not None:
            service_file = io.StringIO(service_file_inline)
        else:
            service_file = os.getenv("GOOGLE_KMS_SERVICE_ACCOUNT_JSON")
        self._kms = KMS(**evars, service_file=service_file)
        self.log.info("Using Google KMS %(keyproject)s/%(keyring)s/%(keyname)s" % evars)

    async def encrypt(self, plaintext: Union[bytes, str]) -> str:
        """Encrypt text using Google KMS."""
        for attempt in range(self.timeout_retries):
            try:
                return await self._kms.encrypt(encode(plaintext))
            except asyncio.TimeoutError as e:
                self.log.warning("encrypt attempt %d", attempt + 1)
                if attempt == self.timeout_retries - 1:
                    raise e from None

    async def decrypt(self, ciphertext: str) -> bytes:
        """Decrypt text using Google KMS."""
        # we cannot use gcloud.aio.kms.decode because it converts bytes to string with str.decode()
        for attempt in range(self.timeout_retries):
            try:
                payload = await self._kms.decrypt(ciphertext)
            except asyncio.TimeoutError as e:
                self.log.warning("decrypt attempt %d", attempt + 1)
                if attempt == self.timeout_retries - 1:
                    raise e from None
            else:
                variant = payload.replace("-", "+").replace("_", "/")
                return base64.b64decode(variant)

    async def close(self):
        """Close the underlying HTTPS session."""
        await self._kms.close()
