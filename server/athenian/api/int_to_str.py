import numpy as np


def int_to_str(arr: np.ndarray) -> np.ndarray:
    """
    Convert array of 64-bit integers to S9.

    We cannot use arr.byteswap().view("S8") because the trailing zeros are discarded \
    in np.char.add. Thus we have to pad with ";".
    """
    assert arr.dtype == int
    arena = np.full((len(arr), 9), ord(";"), dtype=np.uint8)
    arena[:, :8] = arr.byteswap().view(np.uint8).reshape(len(arr), 8)
    return arena.ravel().view("S9")
