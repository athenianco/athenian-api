import numpy as np


def int_to_str(*arrs: np.ndarray) -> np.ndarray:
    """
    Convert one or more arrays of 64-bit integers to S(8 * len + 1).

    We cannot use arr.byteswap().view("S8") because the trailing zeros are discarded \
    in np.char.add. Thus we have to pad with ";".
    """
    assert (arrs_count := len(arrs)) > 0
    size = len(arrs[0])
    str_len = 8 * arrs_count + 1
    arena = np.full((size, str_len), ord(";"), dtype=np.uint8)
    pos = 0
    for arr in arrs:
        assert arr.dtype == int
        assert len(arr) == size
        arena[:, pos:pos + 8] = arr.byteswap().view(np.uint8).reshape(size, 8)
        pos += 8
    return arena.ravel().view(f"S{str_len}")
