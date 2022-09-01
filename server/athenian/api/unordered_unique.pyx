# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize -std=c++17
import cython

from cpython cimport PyObject
from libc.stddef cimport wchar_t
from libc.stdint cimport int32_t, int64_t
from libc.string cimport memchr, memcpy
from libcpp.unordered_map cimport pair, unordered_map
from libcpp.unordered_set cimport unordered_set
from numpy cimport (
    PyArray_DATA,
    PyArray_DESCR,
    PyArray_DIM,
    PyArray_NDIM,
    PyArray_STRIDE,
    dtype as np_dtype,
    ndarray,
)

import numpy as np


cdef extern from "<string_view>" namespace "std" nogil:
    cppclass string_view:
        string_view() except +
        string_view(const char *, size_t) except +
        const char *data()


cdef extern from "wchar.h" nogil:
    wchar_t *wmemchr(const wchar_t *, wchar_t, size_t)


cdef extern from "Python.h":
    Py_ssize_t PyUnicode_GET_LENGTH(PyObject *) nogil
    void *PyUnicode_DATA(PyObject *) nogil
    unsigned int PyUnicode_KIND(PyObject *) nogil
    void Py_INCREF(PyObject *)
    PyObject *Py_None


def unordered_unique(ndarray arr not None) -> np.ndarray:
    cdef:
        np_dtype dtype = <np_dtype>PyArray_DESCR(arr)
    assert PyArray_NDIM(arr) == 1
    if dtype.kind == b"S" or dtype.kind == b"U":
        return _unordered_unique_str(arr, dtype)
    elif dtype.kind == b"i" or dtype.kind == b"u":
        if dtype.itemsize == 8:
            return _unordered_unique_int[int64_t](arr, dtype, 0)
        elif dtype.itemsize == 4:
            return _unordered_unique_int[int64_t](arr, dtype, 4)
        else:
            raise AssertionError(f"dtype {dtype} is not supported")
    elif dtype.kind == b"O":
        return _unordered_unique_pystr(arr)
    else:
        raise AssertionError(f"dtype {dtype} is not supported")


@cython.cdivision(True)
cdef ndarray _unordered_unique_pystr(ndarray arr):
    cdef:
        PyObject **data_in = <PyObject **>PyArray_DATA(arr)
        PyObject **data_out
        PyObject *str_obj
        char *str_data
        unsigned int str_kind
        Py_ssize_t str_len
        int64_t i, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0)
        unordered_map[string_view, int64_t] hashtable
        pair[string_view, int64_t] it
        ndarray result

    with nogil:
        hashtable.reserve(length // 16)
        for i in range(length):
            str_obj = data_in[i * stride]
            if str_obj == Py_None:
                continue
            str_data = <char *> PyUnicode_DATA(str_obj)
            str_len = PyUnicode_GET_LENGTH(str_obj)
            str_kind = PyUnicode_KIND(str_obj)
            hashtable.insert(pair[string_view, int64_t](string_view(str_data, str_len * str_kind), i))

    result = np.empty(hashtable.size(), dtype=object)
    data_out = <PyObject **>PyArray_DATA(result)
    i = 0
    for it in hashtable:
        str_obj = data_in[it.second]
        data_out[i] = str_obj
        Py_INCREF(str_obj)
        i += 1
    return result


@cython.cdivision(True)
cdef ndarray _unordered_unique_str(ndarray arr, np_dtype dtype):
    cdef:
        char *data = <char *>PyArray_DATA(arr)
        int64_t i, \
            itemsize = dtype.itemsize, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0)
        unordered_set[string_view] hashtable
        string_view it
        ndarray result

    with nogil:
        hashtable.reserve(length // 16)
        for i in range(length):
            hashtable.insert(string_view(data + i * stride, itemsize))

    result = np.empty(hashtable.size(), dtype=dtype)

    with nogil:
        data = <char *>PyArray_DATA(result)
        i = 0
        for it in hashtable:
            memcpy(data + i * itemsize, it.data(), itemsize)
            i += 1
    return result


ctypedef fused varint:
    int64_t
    int32_t


@cython.cdivision(True)
cdef ndarray _unordered_unique_int(ndarray arr, np_dtype dtype, varint _):
    cdef:
        char *data = <char *> PyArray_DATA(arr)
        int64_t i, \
            itemsize = dtype.itemsize, \
            length = PyArray_DIM(arr, 0), \
            stride = PyArray_STRIDE(arr, 0)
        unordered_set[varint] hashtable
        varint it
        ndarray result

    with nogil:
        hashtable.reserve(length // 16)
        for i in range(length):
            hashtable.insert((<varint *>(data + i * stride))[0])

    result = np.empty(hashtable.size(), dtype=dtype)

    with nogil:
        data = <char *> PyArray_DATA(result)
        i = 0
        for it in hashtable:
            (<varint *>(data + i * itemsize))[0] = it
            i += 1
    return result


def in1d_str(
    ndarray trial not None,
    ndarray dictionary not None,
    bint skip_leading_zeros = False,
) -> np.ndarray:
    cdef:
        np_dtype dtype_trial = <np_dtype>PyArray_DESCR(trial)
        np_dtype dtype_dict = <np_dtype>PyArray_DESCR(dictionary)
    assert PyArray_NDIM(trial) == 1
    assert PyArray_NDIM(dictionary) == 1
    assert dtype_trial.kind == b"S" or dtype_trial.kind == b"U"
    assert dtype_trial.kind == dtype_dict.kind
    return _in1d_str(trial, dictionary, dtype_trial.kind == b"S", skip_leading_zeros)


cdef ndarray _in1d_str(ndarray trial, ndarray dictionary, bint is_char, int skip_leading_zeros):
    cdef:
        char *data_trial = <char *>PyArray_DATA(trial)
        char *data_dictionary = <char *> PyArray_DATA(dictionary)
        char *output
        char *s
        char *nullptr
        np_dtype dtype_trial = <np_dtype>PyArray_DESCR(trial)
        np_dtype dtype_dict = <np_dtype>PyArray_DESCR(dictionary)
        int64_t i, size, \
            itemsize = dtype_dict.itemsize, \
            length = PyArray_DIM(dictionary, 0), \
            stride = PyArray_STRIDE(dictionary, 0)
        unordered_set[string_view] hashtable
        unordered_set[string_view].iterator end
        ndarray result

    with nogil:
        hashtable.reserve(length * 4)
        if is_char:
            for i in range(length):
                s = data_dictionary + i * stride
                nullptr = s
                if skip_leading_zeros:
                    while nullptr < (s + itemsize) and nullptr[0] == 0:
                        nullptr += 1
                nullptr = <char *> memchr(nullptr, 0, itemsize + (s - nullptr))
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                hashtable.insert(string_view(s, size))
        else:
            for i in range(length):
                s = data_dictionary + i * stride
                nullptr = <char *> wmemchr(<wchar_t *>s, 0, itemsize >> 2)
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                hashtable.insert(string_view(s, size))

        itemsize = dtype_trial.itemsize
        length = PyArray_DIM(trial, 0)
        stride = PyArray_STRIDE(trial, 0)

    result = np.empty(length, dtype=bool)

    with nogil:
        output = <char *>PyArray_DATA(result)
        end = hashtable.end()
        if is_char:
            for i in range(length):
                s = data_trial + i * stride
                nullptr = s
                if skip_leading_zeros:
                    while nullptr < (s + itemsize) and nullptr[0] == 0:
                        nullptr += 1
                nullptr = <char *> memchr(nullptr, 0, itemsize + (s - nullptr))
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                output[i] = hashtable.find(string_view(s, size)) != end
        else:
            for i in range(length):
                s = data_trial + i * stride
                nullptr = <char *> wmemchr(<wchar_t *> s, 0, itemsize >> 2)
                if nullptr:
                    size = nullptr - s
                else:
                    size = itemsize
                output[i] = hashtable.find(string_view(s, size)) != end
    return result
