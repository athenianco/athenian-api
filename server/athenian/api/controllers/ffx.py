import pyffx


def encrypt(data: bytes, key: bytes) -> str:
    """Encrypt bytes with FFX algorithm that preserves the length."""
    data = data.hex()
    return pyffx.String(key, alphabet="0123456789abcdef", length=len(data)).encrypt(data)


def decrypt(data: str, key: bytes) -> bytes:
    """Decrypt a FFX string."""
    return bytes.fromhex(pyffx.String(key, alphabet="0123456789abcdef", length=len(data))
                         .decrypt(data))
