import aiohttp
import pytest

from athenian.api import AthenianKMS


async def test_kms_roundtrip_correct():
    kms = AthenianKMS()
    try:
        assert await kms.decrypt(await kms.encrypt("abcxyz")) == b"abcxyz"
    finally:
        await kms.close()


async def test_kms_roundtrip_attack_random():
    kms = AthenianKMS()
    try:
        with pytest.raises(aiohttp.ClientResponseError):
            await kms.decrypt("whatever")
    finally:
        await kms.close()


async def test_kms_roundtrip_attack_mutate():
    kms = AthenianKMS()
    try:
        s = await kms.encrypt("abcxyz")
        test = "M" + s
        with pytest.raises(aiohttp.ClientResponseError):
            await kms.decrypt(test)
        test = s[:5] + chr(ord(s[5]) + 1) + s[6:]
        with pytest.raises(aiohttp.ClientResponseError):
            await kms.decrypt(test)
    finally:
        await kms.close()
