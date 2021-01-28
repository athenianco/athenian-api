import pyffx


def encrypt(data: bytes, key: bytes) -> str:
    """Encrypt bytes with FFX algorithm that preserves the length."""
    data = data.hex()
    return pyffx.String(key, alphabet="0123456789abcdef", length=len(data)).encrypt(data)
