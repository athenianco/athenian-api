import numpy as np


def int_to_str(*arrs: np.ndarray) -> np.ndarray:
    """
    Convert one or more arrays of 64-bit integers to S(total_bytes + 1).

    We cannot use arr.byteswap().view("S8") because the trailing zeros get discarded \
    in np.char.add. Thus we have to pad with ";".
    """
    assert len(arrs) > 0
    size = len(arrs[0])
    str_len = sum(arr.dtype.itemsize for arr in arrs) + 1
    arena = np.full((size, str_len), ord(";"), dtype=np.uint8)
    pos = 0
    for arr in arrs:
        assert arr.dtype.kind in ("i", "u")
        assert len(arr) == size
        if (itemsize := arr.dtype.itemsize) > 1:
            col = arr.byteswap()
        else:
            col = arr
        arena[:, pos:pos + itemsize] = col.view(np.uint8).reshape(size, itemsize)
        pos += itemsize
    return arena.ravel().view(f"S{str_len}")
