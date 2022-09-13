# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize

from typing import Any, Sequence

cimport cython
from cpython cimport Py_INCREF, PyObject, PyTypeObject
from cpython.bytearray cimport PyByteArray_AS_STRING, PyByteArray_Check
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_Check
from cpython.memoryview cimport PyMemoryView_Check, PyMemoryView_GET_BUFFER
from cpython.unicode cimport PyUnicode_Check
from numpy cimport (
    NPY_ARRAY_C_CONTIGUOUS,
    PyArray_CheckExact,
    PyArray_DATA,
    PyArray_Descr,
    PyArray_DIM,
    PyArray_ISOBJECT,
    PyArray_ISSTRING,
    PyArray_NDIM,
    PyArray_SetBaseObject,
    dtype as nddtype,
    import_array,
    ndarray,
    npy_bool,
    npy_intp,
)

import asyncpg
import numpy as np

import_array()

cdef extern from "asyncpg_recordobj.h":
    PyObject *ApgRecord_GET_ITEM(PyObject *, int)


cdef extern from "Python.h":
    # added nogil -> from cpython cimport ...
    # these are the macros that read directly from the internal ob_items
    PyObject *PyList_GET_ITEM(PyObject *, Py_ssize_t) nogil
    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t) nogil
    bint PyList_CheckExact(PyObject *) nogil
    Py_ssize_t PyList_GET_SIZE(PyObject *) nogil
    Py_ssize_t PyUnicode_GET_LENGTH(PyObject *) nogil
    Py_ssize_t PyBytes_GET_SIZE(PyObject *) nogil

    PyObject *Py_None
    PyObject *Py_True


cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyArray_Type
    ndarray PyArray_NewFromDescr(
        PyTypeObject *subtype,
        PyArray_Descr *descr,
        int nd,
        const npy_intp *dims,
        const npy_intp *strides,
        void *data,
        int flags,
        PyObject *obj,
    )


@cython.boundscheck(False)
def to_object_arrays_split(rows: list[Sequence[Any]],
                           typed_indexes: Sequence[int],
                           obj_indexes: Sequence[int],
                           ) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of tuples into an object array. Any subclass of
    tuple in `rows` will be casted to tuple.

    Parameters
    ----------
    rows : 2-d array (N, K)
        list of tuples to be converted into an array. Each tuple must be of equal length,
        otherwise, the results are undefined.
    typed_indexes : array of integers
        Sequence of integer indexes in each tuple in `rows` that select the first result.
    obj_indexes : array of integers
        Sequence of integer indexes in each tuple in `rows` that select the second result.

    Returns
    -------
    (np.ndarray[object, ndim=2], np.ndarray[object, ndim=2])
    The first array is the concatenation of columns in `rows` chosen by `typed_indexes`.
    The second array is the concatenation of columns in `rows` chosen by `object_indexes`.
    """
    cdef:
        Py_ssize_t i, j, size, cols_typed, cols_obj
        ndarray[object, ndim=2] result_typed
        ndarray[object, ndim=2] result_obj
        PyObject *record
        long[:] typed_indexes_arr
        long[:] obj_indexes_arr

    assert isinstance(rows, list)
    typed_indexes_arr = np.asarray(typed_indexes, dtype=int)
    obj_indexes_arr = np.asarray(obj_indexes, dtype=int)
    size = len(rows)
    cols_typed = len(typed_indexes_arr)
    cols_obj = len(obj_indexes_arr)

    result_typed = np.empty((cols_typed, size), dtype=object)
    result_obj = np.empty((cols_obj, size), dtype=object)
    if size == 0:
        return result_typed, result_obj

    if isinstance(rows[0], asyncpg.Record):
        for i in range(size):
            record = PyList_GET_ITEM(<PyObject *>rows, i)
            for j in range(cols_typed):
                result_typed[j, i] = <object>ApgRecord_GET_ITEM(record, typed_indexes_arr[j])
            for j in range(cols_obj):
                result_obj[j, i] = <object>ApgRecord_GET_ITEM(record, obj_indexes_arr[j])
    elif isinstance(rows[0], tuple):
        for i in range(size):
            record = PyList_GET_ITEM(<PyObject *>rows, i)
            for j in range(cols_typed):
                result_typed[j, i] = <object>PyTuple_GET_ITEM(record, typed_indexes_arr[j])
            for j in range(cols_obj):
                result_obj[j, i] = <object>PyTuple_GET_ITEM(record, obj_indexes_arr[j])
    else:
        # convert to tuple
        for i in range(size):
            row = tuple(rows[i])
            for j in range(cols_typed):
                result_typed[j, i] = row[typed_indexes_arr[j]]
            for j in range(cols_obj):
                result_obj[j, i] = row[obj_indexes_arr[j]]

    return result_typed, result_obj


def as_bool(ndarray arr not None) -> np.ndarray:
    if arr.dtype == bool:
        return arr
    assert arr.dtype == object
    assert arr.ndim == 1
    new_arr = np.empty(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _as_bool_vec(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _as_bool_vec(const char *obj_arr,
                       const long stride,
                       const long size,
                       npy_bool *out_arr) nogil:
    cdef long i
    for i in range(size):
        # Py_None and Py_False become 0
        out_arr[i] = Py_True == (<const PyObject **> (obj_arr + i * stride))[0]



def is_null(ndarray arr not None) -> np.ndarray:
    if arr.dtype != object:
        return np.zeros(len(arr), dtype=bool)
    assert arr.ndim == 1
    new_arr = np.zeros(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _is_null_vec(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _is_null_vec(const char *obj_arr,
                       const long stride,
                       const long size,
                       npy_bool *out_arr) nogil:
    cdef long i
    for i in range(size):
        out_arr[i] = Py_None == (<const PyObject **> (obj_arr + i * stride))[0]


def is_not_null(ndarray arr not None) -> np.ndarray:
    if arr.dtype != object:
        return np.ones(len(arr), dtype=bool)
    assert arr.ndim == 1
    new_arr = np.zeros(len(arr), dtype=bool)
    cdef:
        const char *arr_obj = <const char *> PyArray_DATA(arr)
        long size = len(arr), stride = arr.strides[0]
        npy_bool *out_bools = <npy_bool *> PyArray_DATA(new_arr)
    with nogil:
        _is_not_null(arr_obj, stride, size, out_bools)
    return new_arr


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _is_not_null(const char *obj_arr,
                       const long stride,
                       const long size,
                       npy_bool *out_arr) nogil:
    cdef long i
    for i in range(size):
        out_arr[i] = Py_None != (<const PyObject **> (obj_arr + i * stride))[0]


def nested_lengths(arr not None, output=None) -> np.ndarray:
    cdef:
        long size
        bint is_array = PyArray_CheckExact(arr)
        ndarray result

    if is_array:
        assert PyArray_ISOBJECT(arr) or PyArray_ISSTRING(arr)
        assert PyArray_NDIM(arr) == 1
        size = PyArray_DIM(arr, 0)
    else:
        assert PyList_CheckExact(<PyObject *> arr)
        size = PyList_GET_SIZE(<PyObject *> arr)

    if output is None:
        result = np.zeros(size, dtype=int)
    else:
        assert PyArray_CheckExact(output)
        assert output.dtype == int
        result = output
    if size == 0:
        return result
    if is_array:
        return _nested_lengths_arr(arr, size, result)
    return _nested_lengths_list(<PyObject *> arr, size, result)


cdef ndarray _nested_lengths_arr(ndarray arr, long size, ndarray result):
    cdef:
        PyObject **elements = <PyObject **>PyArray_DATA(arr)
        PyObject *element
        long i
        long *result_data

    if PyArray_ISSTRING(arr):
        return np.char.str_len(arr)

    result_data = <long *>PyArray_DATA(result)
    element = elements[0]
    if PyArray_CheckExact(<object> element):
        for i in range(size):
            result_data[i] = PyArray_DIM(<ndarray> elements[i], 0)
    elif PyList_CheckExact(element):
        for i in range(size):
            result_data[i] = PyList_GET_SIZE(elements[i])
    elif PyUnicode_Check(<object> element):
        for i in range(size):
            result_data[i] = PyUnicode_GET_LENGTH(elements[i])
    elif PyBytes_Check(<object> element):
        for i in range(size):
            result_data[i] = PyBytes_GET_SIZE(elements[i])
    else:
        raise AssertionError(f"Unsupported nested type: {type(<object> element).__name__}")
    return result


cdef ndarray _nested_lengths_list(PyObject *arr, long size, ndarray result):
    cdef:
        PyObject *element
        long i
        long *result_data

    result_data = <long *>PyArray_DATA(result)
    element = PyList_GET_ITEM(arr, 0)
    if PyArray_CheckExact(<object> element):
        for i in range(size):
            result_data[i] = PyArray_DIM(<ndarray> PyList_GET_ITEM(arr, i), 0)
    elif PyList_CheckExact(element):
        for i in range(size):
            result_data[i] = PyList_GET_SIZE(PyList_GET_ITEM(arr, i))
    elif PyUnicode_Check(<object> element):
        for i in range(size):
            result_data[i] = PyUnicode_GET_LENGTH(PyList_GET_ITEM(arr, i))
    elif PyBytes_Check(<object> element):
        for i in range(size):
            result_data[i] = PyBytes_GET_SIZE(PyList_GET_ITEM(arr, i))
    else:
        raise AssertionError(f"Unsupported nested type: {type(<object> element).__name__}")
    return result


def array_from_buffer(buffer not None, nddtype dtype, npy_intp count, npy_intp offset=0) -> ndarray:
    cdef:
        void *data
    if PyBytes_Check(buffer):
        data = PyBytes_AS_STRING(buffer) + offset
    elif PyByteArray_Check(buffer):
        data = PyByteArray_AS_STRING(buffer) + offset
    elif PyMemoryView_Check(buffer):
        data = (<char *>PyMemoryView_GET_BUFFER(buffer).buf) + offset
    else:
        raise ValueError(f"Unsupported buffer type: {type(buffer).__name__}")
    Py_INCREF(dtype)
    Py_INCREF(buffer)
    arr = PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> dtype,
        1,
        &count,
        NULL,
        data,
        NPY_ARRAY_C_CONTIGUOUS,
        NULL,
    )
    PyArray_SetBaseObject(arr, buffer)
    return arr
