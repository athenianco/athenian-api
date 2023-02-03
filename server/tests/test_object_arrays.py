import numpy as np
import pytest

from athenian.api.object_arrays import objects_to_pyunicode_bytes


def test_objects_to_utf8_bytes_empty():
    arr = np.array([], dtype=object)
    conv = objects_to_pyunicode_bytes(arr)
    assert conv.dtype == "S1"
    assert len(conv) == 0


@pytest.mark.parametrize("limit", [None, 8, 3])
def test_objects_to_utf8_bytes_limit(limit):
    arr = np.array(["test", "тест", "😊😊😊", None], dtype=object)
    conv = objects_to_pyunicode_bytes(arr, limit)
    if limit is None:
        limit = 12
    assert conv.dtype == f"S{limit}"
    assert conv.tolist() == [
        b"test"[:limit],
        b"B\x045\x04A\x04B\x04"[:limit],
        b"\n\xf6\x01\x00\n\xf6\x01\x00\n\xf6\x01"[:limit].rstrip(b"\x00"),
        b"None"[:limit],
    ]
