from typing import Tuple

import numpy as np


class SparseMask:
    """Sparse multidimensional boolean mask."""

    def __init__(self, arr: np.ndarray):
        """Initialize a new SparseMask from a dense boolean mask."""
        assert arr.dtype == bool
        self._indexes = np.flatnonzero(arr.ravel())
        self._shape = arr.shape

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the mask."""
        return self._shape

    def dense(self) -> np.ndarray:
        """Reconstruct the dense boolean mask."""
        arr = np.zeros(self._shape, dtype=bool)
        arr.ravel()[self._indexes] = True
        return arr
