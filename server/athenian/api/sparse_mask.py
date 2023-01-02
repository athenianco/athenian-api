from dataclasses import dataclass

import numpy as np
from numpy import typing as npt

from athenian.api.sorted_intersection import sorted_intersect1d, sum_repeated_with_step


@dataclass(frozen=True, slots=True)
class SparseMask:
    """Sparse multidimensional boolean mask."""

    indexes: npt.NDArray[int | np.uint32]
    shape: tuple[int, ...]

    @classmethod
    def from_dense(cls, arr: npt.NDArray[bool]):
        """Initialize a new SparseMask from a dense boolean mask."""
        assert arr.dtype == bool
        return SparseMask(np.flatnonzero(arr.ravel()), arr.shape)

    @classmethod
    def empty(cls, *shape: int):
        """Initialize an empty mask of `shape`."""
        return SparseMask(np.array([], dtype=int), shape)

    def dense(self) -> npt.NDArray[bool]:
        """Reconstruct the dense boolean mask."""
        arr = np.zeros(self.shape, dtype=bool)
        arr.ravel()[self.indexes] = True
        return arr

    def copy(self) -> "SparseMask":
        """Clone the mask."""
        return SparseMask(self.indexes.copy(), self.shape)

    def __getitem__(self, shape_slices: tuple[slice | None]) -> "SparseMask":
        """Insert new axes into the shape."""
        resolved_shape = []
        pos = 0
        for dim in shape_slices:
            if dim is np.newaxis:
                resolved_shape.append(1)
                continue
            if dim != slice(None, None, None):
                raise NotImplementedError("interval, numeric, boolean indexing is not supported")
            resolved_shape.append(self.shape[pos])
            pos += 1
        resolved_shape.extend(self.shape[pos:])
        return SparseMask(self.indexes, tuple(resolved_shape))

    def __and__(self, other: "SparseMask") -> "SparseMask":
        """Calculate logical AND."""
        if not isinstance(other, SparseMask):
            raise TypeError("Both masks are required to be sparse")
        if self.shape != other.shape:
            raise ValueError("Both mask must be of the same shape")
        if len(self.indexes) == 0 or len(other.indexes) == 0:
            return SparseMask.empty(*self.shape)
        max_index = max(self.indexes[-1], other.indexes[-1])
        pos_self = 0
        pos_other = 0
        results = []
        offset = 0
        if max_index >= (1 << 32):
            while pos_self < len(self.indexes) and pos_other < len(other.indexes):
                border_self = np.searchsorted(self.indexes[pos_self:], offset + (1 << 32))
                border_other = np.searchsorted(other.indexes[pos_self:], offset + (1 << 32))
                results.append(
                    sorted_intersect1d(
                        self.indexes[pos_self:border_self], other.indexes[pos_other:border_other],
                    ).astype(int)
                    + offset,
                )
                offset += 1 << 32
                pos_self = border_self
                pos_other = border_other
            indexes = np.concatenate(results)
        else:
            indexes = sorted_intersect1d(
                self.indexes.astype(np.uint32), other.indexes.astype(np.uint32),
            )
        return SparseMask(indexes, self.shape)

    def __or__(self, other: "SparseMask") -> "SparseMask":
        """Calculate logical OR."""
        if not isinstance(other, SparseMask):
            raise TypeError("Both masks are required to be sparse")
        if self.shape != other.shape:
            raise ValueError("Both mask must be of the same shape")
        return SparseMask(np.union1d(self.indexes, other.indexes), self.shape)

    def __sub__(self, other: "SparseMask") -> "SparseMask":
        """Calculate the difference with `other`."""
        if not isinstance(other, SparseMask):
            raise TypeError("Both masks are required to be sparse")
        if self.shape != other.shape:
            raise ValueError("Both mask must be of the same shape")
        return SparseMask(
            np.setdiff1d(self.indexes, other.indexes, assume_unique=True), self.shape,
        )

    def sum_last_axis(self) -> npt.NDArray[np.uint32]:
        """Emulate `np.sum(self, axis=-1, dtype=np.uint32)`."""
        result = np.zeros(self.shape[:-1], dtype=np.uint32)
        nnz, nnz_counts = np.unique(
            np.floor_divide(self.indexes, self.shape[-1]), return_counts=True,
        )
        result.ravel()[nnz] = nnz_counts
        return result

    def repeat(self, n: int, axis: int) -> "SparseMask":
        """Emulate `np.repeat(self, n, axis)`."""  # noqa: D402
        if axis < 0:
            axis = len(self.shape) + axis
        if axis < 0 or axis >= len(self.shape):
            raise IndexError(f"Invalid axis index: {axis}")
        new_shape = (*self.shape[:axis], self.shape[axis] * n, *self.shape[axis + 1 :])
        if axis == 0:
            new_indexes = sum_repeated_with_step(self.indexes, n, np.prod(self.shape))
        else:
            indexes_by_dim = np.unravel_index(self.indexes, self.shape)
            new_indexes_by_dim = []
            for i, level in enumerate(indexes_by_dim):
                if i != axis:
                    level = np.repeat(level[None, :], n, axis=0).ravel()
                else:
                    level = sum_repeated_with_step(np.ascontiguousarray(level), n, self.shape[i])
                new_indexes_by_dim.append(level)
            new_indexes = np.ravel_multi_index(new_indexes_by_dim, new_shape)
            new_indexes.sort()  # maintain the order
        return SparseMask(new_indexes, new_shape)
