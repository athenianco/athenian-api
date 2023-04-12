# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17

cimport cython
from cpython cimport PyObject
from cython.operator cimport dereference as deref
from libc.stdint cimport int32_t, int64_t, uint64_t
from numpy cimport PyArray_DATA

from athenian.api.native.cpython cimport PyUnicode_DATA
from athenian.api.native.mi_heap_destroy_stl_allocator cimport (
    mi_heap_allocator_from_capsule,
    mi_heap_destroy_stl_allocator,
    mi_unordered_map,
)
from athenian.api.native.optional cimport optional
from athenian.api.native.string_view cimport string_view

import numpy as np


def split_duplicate_check_runs(
    suite_ids: np.ndarray,
    names: np.ndarray,
    started_ats: np.ndarray,
    alloc_capsule=None,
) -> int:
    assert suite_ids.dtype == int
    assert suite_ids.data.c_contiguous
    assert names.dtype == object
    assert names.data.c_contiguous
    assert started_ats.dtype == "datetime64[s]"
    assert len(suite_ids) == len(names) == len(started_ats)
    started_ats[started_ats != started_ats] = 0
    suite_order = np.argsort(suite_ids, kind="stable")
    cdef:
        optional[mi_heap_destroy_stl_allocator[char]] alloc
        long result
        PyObject **data_obj = <PyObject **> PyArray_DATA(names)
        int64_t[:] suite_ids_view = suite_ids
        const int64_t[:] started_ats_int64 = started_ats.view(np.int64)
        const int64_t[:] suite_order_view = suite_order
    if alloc_capsule is not None:
        alloc.emplace(deref(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
    with nogil:
        result = _split_duplicate_check_runs(
            suite_ids_view,
            data_obj,
            started_ats_int64,
            suite_order_view,
            deref(alloc),
        )
    return result


cdef struct CheckSuiteState:
    int32_t epoch
    optional[mi_unordered_map[string_view, long]] types


@cython.boundscheck(False)
@cython.wraparound(False)
cdef long _split_duplicate_check_runs(
    int64_t[:] suite_ids,
    PyObject **names,
    const int64_t[:] started_ats,
    const int64_t[:] suite_order,
    mi_heap_destroy_stl_allocator[char] &alloc,
) nogil:
    cdef:
        optional[mi_unordered_map[int64_t, CheckSuiteState]] suite_groups
        long i, j, epoch, split = 0
        long *type_pos
        int64_t suite_id, previous_suite_id = -1, started_at, previous_started_at = 0
        uint64_t suite_id_mask = 0xFFFFFFFFFFFFFF
        CheckSuiteState *state
    suite_groups.emplace(alloc)
    # reverse order produces better results
    for i in range(len(suite_ids) - 1, -1, -1):
        suite_id = suite_ids[i]
        state = &deref(deref(suite_groups).try_emplace(suite_id).first).second
        if not state.types.has_value():
            state.types.emplace(alloc)
        type_pos = &deref(deref(state.types).try_emplace(<const char*> PyUnicode_DATA(names[i]), 0).first).second
        epoch = state.epoch
        if type_pos[0] == epoch:
            type_pos[0] += 1
            epoch += 1
            state.epoch = epoch
        else:
            type_pos[0] = epoch
        if epoch > 1:
            suite_ids[i] = ((<uint64_t>(epoch - 1)) << (64 - 8)) | <uint64_t>suite_id
        split += epoch > 1
    # last resort heuristic: if more than 12 hours passed since check suite start, force split
    epoch = 1
    for j in range(len(suite_ids)):
        i = suite_order[j]
        suite_id = suite_ids[i]
        started_at = started_ats[i]
        if suite_id == previous_suite_id:
            if (started_at - previous_started_at) >= 12 * 3600:
                epoch += 1
                previous_started_at = started_at
            if epoch > 1:
                split += 1
                suite_ids[i] = (
                    (<uint64_t>(epoch - 1) + (<uint64_t>suite_id >> (64 - 8))) << (64 - 8)
                ) | (<uint64_t>suite_id & suite_id_mask)
        else:
            epoch = 1
            previous_started_at = started_at
            previous_suite_id = suite_id
    return split
