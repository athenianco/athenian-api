# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize

from libc.stdint cimport int64_t
from libc.stdio cimport printf
from libc.stdlib cimport lldiv, lldiv_t
from libc.string cimport memchr, memcpy
from numpy cimport (
    PyArray_DATA,
    PyArray_DESCR,
    PyArray_DIM,
    PyArray_STRIDE,
    dtype as np_dtype,
    ndarray,
)


cdef extern from "Python.h":
    object PyUnicode_New(Py_ssize_t, Py_UCS4)
    void *PyUnicode_DATA(object)



def in_any_values_inline(ndarray values) -> str:
    assert values.ndim == 1
    assert values.dtype.kind in ("S", "U", "i", "u"), f"unsupported dtype {values.dtype}"
    assert len(values) > 0, "= ANY(VALUES) is invalid syntax"
    cdef:
        np_dtype dtype = <np_dtype> PyArray_DESCR(values)
        int is_s = dtype.kind == b"S"
        int is_str = is_s or dtype.kind == b"U"
        int stride = PyArray_STRIDE(values, 0)
        int itemsize = dtype.itemsize
        int length = PyArray_DIM(values, 0)
        int effective_itemsize = \
            (itemsize if dtype.kind == b"S" else itemsize >> 2) + 2 \
            if is_str \
            else len(str(values.max()))
        Py_ssize_t size = (7 + (effective_itemsize + 3) * length - 1)
        result = PyUnicode_New(size, 255)
        char *buf = <char *> PyUnicode_DATA(result)
        char *data = <char *> PyArray_DATA(values)

    with nogil:
        if is_str:
            if is_s:
                _in_any_values_s(data, stride, itemsize, length, buf)
            else:
                _in_any_values_u(data, stride, itemsize, length, buf)
        elif itemsize == 8:
            _in_any_values_int64(data, stride, length, effective_itemsize, buf)
        else:
            raise AssertionError(f"unsupported dtype {dtype}")
    return result


cdef void _in_any_values_s(const char *data,
                           int stride,
                           int itemsize,
                           int length,
                           char *output) nogil:
    cdef:
        int i, pos = 7
        char *quoteptr
        char *nullptr
    memcpy(output, b"VALUES ", 7)

    for i in range(length):
        output[pos] = b"("
        pos += 1
        output[pos] = b"'"
        pos += 1
        memcpy(output + pos, data + stride * i, itemsize)
        pos += itemsize
        output[pos] = b"'"
        pos += 1
        output[pos] = b")"
        pos += 1
        if i < length - 1:
            output[pos] = b","
            pos += 1

    nullptr = <char *> memchr(output, 0, pos)
    while nullptr:
        quoteptr = <char *> memchr(nullptr + 1, b"'", itemsize)
        nullptr[0] = b"'"
        for i in range(1, (quoteptr - nullptr) + 1):
            nullptr[i] = b" "
        nullptr = <char *> memchr(quoteptr, 0, pos - (quoteptr - output))


cdef void _in_any_values_u(const char *data,
                           int stride,
                           int itemsize,
                           int length,
                           char *output) nogil:
    cdef:
        int i, j, fill, pos = 7
        char c
    memcpy(output, b"VALUES ", 7)

    for i in range(length):
        output[pos] = b"("
        pos += 1
        output[pos] = b"'"
        pos += 1
        fill = False
        for j in range(0, itemsize, 4):
            c = data[stride * i + j]
            if fill:
                c = b" "
            elif c == 0:
                c = b"'"
                fill = True
            output[pos] = c
            pos += 1
        output[pos] = b" " if fill else b"'"
        pos += 1
        output[pos] = b")"
        pos += 1
        if i < length - 1:
            output[pos] = b","
            pos += 1


cdef void _in_any_values_int64(const char *data,
                               int stride,
                               int length,
                               int alignment,
                               char *output) nogil:
    cdef:
        int i, pos = 7, valstart
        lldiv_t qr
    memcpy(output, b"VALUES ", 7)

    for i in range(length):
        output[pos] = b"("
        pos += 1
        valstart = pos
        pos += alignment - 1
        qr.quot = (<const int64_t *>(data + i * stride))[0]
        while True:
            qr = lldiv(qr.quot, 10)
            output[pos] = (<char>b'0') + (<char>qr.rem)
            pos -= 1
            if qr.quot == 0:
                break
        while pos >= valstart:
            output[pos] = b" "
            pos -= 1
        pos = valstart + alignment
        output[pos] = b")"
        pos += 1
        if i < length - 1:
            output[pos] = b","
            pos += 1


def in_inline(ndarray values) -> str:
    assert values.ndim == 1
    assert values.dtype.kind in ("S", "U", "i", "u"), f"unsupported dtype {values.dtype}"

    if len(values) == 0:
        return "null"

    cdef:
        np_dtype dtype = <np_dtype> PyArray_DESCR(values)
        int is_s = dtype.kind == b"S"
        int is_str = is_s or dtype.kind == b"U"
        int stride = PyArray_STRIDE(values, 0)
        int itemsize = dtype.itemsize
        int length = PyArray_DIM(values, 0)
        int effective_itemsize = \
            (itemsize if dtype.kind == b"S" else itemsize >> 2) + 2 \
            if is_str \
            else len(str(values.max()))
        Py_ssize_t size = (effective_itemsize + 1) * length - 1
        result = PyUnicode_New(size, 255)
        char *buf = <char *> PyUnicode_DATA(result)
        char *data = <char *> PyArray_DATA(values)

    with nogil:
        if is_str:
            if is_s:
                _in_s(data, stride, itemsize, length, buf)
            else:
                _in_u(data, stride, itemsize, length, buf)
        elif itemsize == 8:
            _in_int64(data, stride, length, effective_itemsize, buf)
        else:
            raise AssertionError(f"unsupported dtype {dtype}")
    return result


cdef void _in_s(const char *data,
                int stride,
                int itemsize,
                int length,
                char *output) nogil:
    cdef:
        int i, pos = 0
        char *quoteptr
        char *nullptr

    for i in range(length):
        output[pos] = b"'"
        pos += 1
        memcpy(output + pos, data + stride * i, itemsize)
        pos += itemsize
        output[pos] = b"'"
        pos += 1
        if i < length - 1:
            output[pos] = b","
            pos += 1

    nullptr = <char *> memchr(output, 0, pos)
    while nullptr:
        quoteptr = <char *> memchr(nullptr + 1, b"'", itemsize)
        nullptr[0] = b"'"
        for i in range(1, (quoteptr - nullptr) + 1):
            nullptr[i] = b" "
        nullptr = <char *> memchr(quoteptr, 0, pos - (quoteptr - output))


cdef void _in_u(const char *data,
                int stride,
                int itemsize,
                int length,
                char *output) nogil:
    cdef:
        int i, j, fill, pos = 0
        char c

    for i in range(length):
        output[pos] = b"'"
        pos += 1
        fill = False
        for j in range(0, itemsize, 4):
            c = data[stride * i + j]
            if fill:
                c = b" "
            elif c == 0:
                c = b"'"
                fill = True
            output[pos] = c
            pos += 1
        output[pos] = b" " if fill else b"'"
        pos += 1
        if i < length - 1:
            output[pos] = b","
            pos += 1


cdef void _in_int64(const char *data,
                    int stride,
                    int length,
                    int alignment,
                    char *output) nogil:
    cdef:
        int i, pos = 0, valstart
        lldiv_t qr

    for i in range(length):
        valstart = pos
        pos += alignment - 1
        qr.quot = (<const int64_t *>(data + i * stride))[0]
        while True:
            qr = lldiv(qr.quot, 10)
            output[pos] = (<char>b'0') + (<char>qr.rem)
            pos -= 1
            if qr.quot == 0:
                break
        while pos >= valstart:
            output[pos] = b" "
            pos -= 1
        pos = valstart + alignment
        if i < length - 1:
            output[pos] = b","
            pos += 1
