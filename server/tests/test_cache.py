import aiomcache
import pytest

from athenian.api.cache import gen_cache_key


@pytest.mark.parametrize("fmt,args", [("text", []),
                                      ("", []),
                                      ("1", []),
                                      ("%s", [""]),
                                      ("xxx %s %d yyy", ["y", 2]),
                                      ("x" * 100500, [])])
def test_gen_cache_key_formats(fmt, args):
    key = gen_cache_key(fmt, *args)
    assert key
    aiomcache.Client._validate_key(aiomcache.Client, key)
    for _ in range(100):
        # check that there is no randomness
        assert key == gen_cache_key(fmt, *args)


def test_gen_cache_key_distinct():
    key1 = gen_cache_key("a" * 10000)
    key2 = gen_cache_key("a" * 9999 + "b")
    assert key1 != key2
    key2 = gen_cache_key("b" + "a" * 9999)
    assert key1 != key2
