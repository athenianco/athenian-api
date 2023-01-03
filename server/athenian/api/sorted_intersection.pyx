# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize -std=c++17

from cpython cimport Py_INCREF, PyObject
from libc.stdint cimport int64_t, uint32_t
from numpy cimport (
    NPY_ARRAY_C_CONTIGUOUS,
    NPY_UINT,
    PyArray_DATA,
    PyArray_DescrFromType,
    PyArray_DIM,
    PyArray_IS_C_CONTIGUOUS,
    PyArray_NDIM,
    dtype,
    import_array,
    ndarray,
    npy_intp,
)

from athenian.api.native.cpython cimport PyUnicode_DATA
from athenian.api.native.numpy cimport (
    PyArray_Descr,
    PyArray_DESCR,
    PyArray_DescrNew,
    PyArray_NewFromDescr,
    PyArray_Type,
)


cdef extern from "native/sorted_intersection.h" nogil:
    size_t intersect(
        const char *algorithm,
        const uint32_t *set1,
        const size_t length1,
        const uint32_t *set2,
        const size_t length2,
        uint32_t *out
    )

import_array()

"""
Benchmark results on production metrics:

v1
4.79 ms ± 180 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
v3
6.94 ms ± 110 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
simd
4.92 ms ± 293 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
galloping
2.78 ms ± 30.5 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
mut_part
3.35 ms ± 294 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
scalar
3.21 ms ± 15.1 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
simdgalloping
7.08 ms ± 93.6 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
simd_avx2
4.6 ms ± 336 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
v1_avx2
4.22 ms ± 143 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
v3_avx2
10.5 ms ± 152 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
simdgalloping_avx2
10.7 ms ± 193 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
highlyscalable_intersect_SIMD
4.78 ms ± 340 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
lemire_highlyscalable_intersect_SIMD
3.97 ms ± 387 µs per loop (mean ± std. dev. of 7 runs, 200 loops each)
"""

def sorted_intersect1d(
    ndarray arr1 not None,
    ndarray arr2 not None,
    str algo="galloping",
) -> ndarray:
    assert PyArray_NDIM(arr1) == 1
    assert PyArray_IS_C_CONTIGUOUS(arr1)
    assert PyArray_DESCR(<PyObject *> arr1).kind == b"u"
    assert PyArray_NDIM(arr2) == 1
    assert PyArray_IS_C_CONTIGUOUS(arr2)
    assert PyArray_DESCR(<PyObject *> arr2).kind == b"u"

    cdef:
        uint32_t *arr1_data = <uint32_t *> PyArray_DATA(arr1)
        uint32_t *arr2_data = <uint32_t *> PyArray_DATA(arr2)
        npy_intp len1 = PyArray_DIM(arr1, 0)
        npy_intp len2 = PyArray_DIM(arr2, 0)
        ndarray output
        dtype u32dtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_UINT))

    output = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> u32dtype,
        1,
        &len1 if len1 > len2 else &len2,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(u32dtype)
    output.shape[0] = intersect(
        <char *>PyUnicode_DATA(<PyObject *> algo),
        arr1_data,
        len1,
        arr2_data,
        len2,
        <uint32_t *> PyArray_DATA(output),
    )
    return output


cdef extern from "native/sum_repeated_with_step.h" nogil:
    void sum_repeated_with_step_avx2(
        const int64_t *src,
        int64_t src_len,
        int64_t repeats,
        int64_t step,
        int64_t *dst
    )


def sum_repeated_with_step(ndarray arr not None, long repeats, long step) -> ndarray:
    """
    Calculate fused sum of wide range with repeated vector.

        np.repeat(arr[None, :], repeats, axis=0).ravel() + np.repeat(
            np.arange(repeats, dtype=int) * step, len(arr),
        )
    """
    assert PyArray_NDIM(arr) == 1
    assert PyArray_IS_C_CONTIGUOUS(arr)

    cdef:
        dtype arr_dtype = arr.dtype
        int64_t src_len = PyArray_DIM(arr, 0)
        npy_intp dst_len = src_len * repeats
        int64_t *src_data = <int64_t *> PyArray_DATA(arr)

    assert arr_dtype.kind == b"i"
    output = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> arr_dtype,
        1,
        &dst_len,
        NULL,
        NULL,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    Py_INCREF(arr_dtype)
    sum_repeated_with_step_avx2(src_data, src_len, repeats, step, <int64_t *> PyArray_DATA(output))
    return output
