from xxhash import xxh64_hexdigest


def gen_cache_key(fmt: str, *args) -> bytes:
    """Compose a memcached-friendly cache key from a printf-like."""
    full_key = (fmt % args).encode()
    first_half = xxh64_hexdigest(full_key[:len(full_key) // 2])
    second_half = xxh64_hexdigest(full_key[len(full_key) // 2:])
    return (first_half + second_half).encode()
