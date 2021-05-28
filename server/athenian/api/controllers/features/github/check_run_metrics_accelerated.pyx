# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.undeclared=True, warn.maybe_uninitialized=True
# distutils: language = c++

cimport cython
from libc.stdint cimport int32_t, int64_t
import numpy as np


def calculate_interval_intersections(starts: np.ndarray,
                                     finishes: np.ndarray,
                                     borders: np.ndarray,
                                     ) -> np.ndarray:
    assert len(starts) == len(finishes)
    result = np.zeros_like(starts, dtype=np.int32)
    _calculate_interval_intersections(starts, finishes, borders, result, result.copy())
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _calculate_interval_intersections(const int64_t[:] starts,
                                            const int64_t[:] finishes,
                                            const int64_t[:] borders,
                                            int32_t[:] intersections,
                                            int32_t[:] futures) nogil:
    cdef int64_t i, j, border_index, group_start, group_finish, my_start, my_finish, my_middle, \
        other_start
    for border_index in range(len(borders)):
        group_start = borders[border_index - 1] if border_index > 0 else 0
        group_finish = borders[border_index]
        for i in range(group_start, group_finish):
            my_start = starts[i]
            my_finish = finishes[i]
            my_middle = my_finish + (my_finish - my_start) // 2
            j = i + 1
            while j < group_finish:
                other_start = starts[j]
                if other_start > my_middle:
                    break
                if my_finish - other_start >= (finishes[j] - other_start) // 2:
                    futures[j] += 1
                j += 1
            intersections[i] = futures[i] + j - i
            while j < group_finish:
                other_start = starts[j]
                if other_start >= my_finish:
                    break
                if my_finish - other_start >= (finishes[j] - other_start) // 2:
                    futures[j] += 1
                j += 1
