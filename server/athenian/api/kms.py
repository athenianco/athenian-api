import logging
import os

from gcloud.aio.kms import decode, encode, KMS

from athenian.api import metadata


class AthenianKMS:
    """Google Cloud Key Management Service for Athenian API."""

    log = logging.getLogger("%s.kms" % metadata.__package__)

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
        self._kms = KMS(**evars, service_file=os.getenv("GOOGLE_KMS_SERVICE_ACCOUNT_JSON"))
        self.log.info("Using Google KMS %(keyproject)s/%(keyring)s/%(keyname)s" % evars)

    async def encrypt(self, plaintext: str) -> str:
        """Encrypt text using Google KMS."""
        return await self._kms.encrypt(encode(plaintext))

    async def decrypt(self, ciphertext: str) -> str:
        """Decrypt text using Google KMS."""
        return decode(await self._kms.decrypt(ciphertext))

    async def close(self):
        """Close the underlying HTTPS session."""
        await self._kms.close()
