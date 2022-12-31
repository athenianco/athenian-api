# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: extra_compile_args = -mavx2 -ftree-vectorize


from cpython cimport PyObject
from libc.stddef cimport wchar_t
from libc.string cimport memchr, memset
from numpy cimport PyArray_DATA, PyArray_DIM, PyArray_ISOBJECT, PyArray_ISSTRING, ndarray

from athenian.api.native.cpython cimport (
    Py_None,
    PyList_GET_ITEM,
    PyList_GET_SIZE,
    PyUnicode_1BYTE_KIND,
    PyUnicode_DATA,
    PyUnicode_FromKindAndData,
    PyUnicode_GET_LENGTH,
    PyUnicode_KIND,
)

import numpy as np


cdef extern from "<wchar.h>" nogil:
    wchar_t *wmemchr(const wchar_t *s, wchar_t c, size_t n)


# performance: do not check that repo is a str, downstream functions will do that anyway
def drop_logical_repo(repo) -> str:
    return _drop_logical_repo(<PyObject *> repo)


cdef inline str _drop_logical_repo(PyObject *repo):
    cdef:
        char *data
        char *head
        wchar_t *wdata
        wchar_t *whead
        Py_ssize_t length
        unsigned int kind
        int i = 0, matches = 0
    # note: the function is too small to release the GIL
    data = <char *> PyUnicode_DATA(repo)
    length = PyUnicode_GET_LENGTH(repo)
    kind = PyUnicode_KIND(repo)
    if kind == PyUnicode_1BYTE_KIND:
        head = <char *> memchr(data, b"/", length)
        if head != NULL:
            head = <char *> memchr(head + 1, b"/", length - (head + 1 - data))
        if head == NULL:
            head = data + length
        return PyUnicode_FromKindAndData(kind, data, head - data)
    if kind == sizeof(wchar_t):
        wdata = <wchar_t *> data
        whead = wmemchr(wdata, b"/", length)
        if whead != NULL:
            whead = wmemchr(whead + 1, b"/", length - (whead + 1 - wdata))
        if whead == NULL:
            whead = wdata + length
        return PyUnicode_FromKindAndData(kind, wdata, whead - wdata)
    for i in range(length):
        # assume little-endian
        if data[i * kind] == b"/":
            matches += 1
            if matches == 2:
                break
    return PyUnicode_FromKindAndData(kind, data, i)


def mark_logical_repos_in_list(list repos) -> tuple[ndarray, int]:
    """Put 0 for each physical repo, 1 for each logical.

    :return: logical mask and the (number of physical repos - 1) \
             (compatible with np.[arg]partition). If there are no physical repos, the value is 0.
    """
    cdef:
        PyObject *repo
        ndarray result
        char *repo_data
        wchar_t *repo_wdata
        char *result_data
        char *head
        wchar_t *whead
        Py_ssize_t i, arr_length, repo_length
        unsigned int kind
        int matches, comp
        long logical_count = 0, last_physical

    arr_length = PyList_GET_SIZE(<PyObject *> repos)
    result = np.zeros(arr_length, dtype=np.uint8)
    result_data = <char *> PyArray_DATA(result)
    with nogil:
        for i in range(arr_length):
            repo = PyList_GET_ITEM(<PyObject *> repos, i)
            repo_data = <char *> PyUnicode_DATA(<PyObject *> repo)
            repo_length = PyUnicode_GET_LENGTH(<PyObject *> repo)
            kind = PyUnicode_KIND(<PyObject *> repo)
            if kind == PyUnicode_1BYTE_KIND:
                head = <char *> memchr(repo_data, b"/", repo_length)
                if head != NULL:
                    head = <char *> memchr(head + 1, b"/", repo_length - (head + 1 - repo_data))
                comp = (head != NULL)
                logical_count += comp
                result_data[i] = comp
            elif kind == sizeof(wchar_t):
                repo_wdata = <wchar_t *> repo_data
                whead = wmemchr(repo_wdata, b"/", repo_length)
                if whead != NULL:
                    whead = wmemchr(whead + 1, b"/", repo_length - (whead + 1 - repo_wdata))
                comp = (whead != NULL)
                logical_count += comp
                result_data[i] = comp
            else:
                matches = 0
                for i in range(repo_length):
                    # assume little-endian
                    if repo_data[i * kind] == b"/":
                        matches += 1
                        if matches == 2:
                            break
                comp = matches > 1
                logical_count += comp
                result_data[i] = comp
    last_physical = arr_length - 1 - logical_count
    if last_physical < 0:
        last_physical = 0
    return result, last_physical


def drop_logical_in_array(ndarray names) -> ndarray:
    cdef:
        long str_len = names.dtype.itemsize, pos = 0, count = 0, i
        ndarray slashes, result
        char *str_data
        PyObject **obj_data
        PyObject *str_obj
        long *slash_data

    if PyArray_ISSTRING(names):
        slashes = np.flatnonzero(names.view(np.uint8) == ord(b"/"))
        slash_data = <long *> PyArray_DATA(slashes)
        result = names.copy()
        str_data = <char *> PyArray_DATA(result)
        for i in range(PyArray_DIM(slashes, 0)):
            i = slash_data[i]
            while i >= (pos + str_len):
                pos += str_len
                count = 0
            count += 1
            if count == 2:
                memset(str_data + i, 0, pos + str_len - i)
    else:
        assert PyArray_ISOBJECT(names)
        result = np.empty_like(names)
        obj_data = <PyObject **> PyArray_DATA(names)
        for i in range(PyArray_DIM(names, 0)):
            str_obj = obj_data[i]
            if str_obj == Py_None:
                result[i] = None
            else:
                result[i] = _drop_logical_repo(str_obj)
    return result
