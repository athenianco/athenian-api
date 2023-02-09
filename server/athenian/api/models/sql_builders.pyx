# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize

from cpython cimport PyList_CheckExact, PyObject

import cython

from libc.stdint cimport int64_t
from libc.stdlib cimport lldiv, lldiv_t
from libc.string cimport memchr, memcpy
from numpy cimport (
    PyArray_CheckExact,
    PyArray_DATA,
    PyArray_DESCR,
    PyArray_DIM,
    PyArray_STRIDE,
    dtype as np_dtype,
    import_array,
    ndarray,
)

from athenian.api.native.cpython cimport (
    Py_None,
    PyBytes_AS_STRING,
    PyBytes_Check,
    PyBytes_GET_SIZE,
    PyList_GET_ITEM,
    PyLong_AsLong,
    PyLong_CheckExact,
    PyObject_TypeCheck,
    PyUnicode_Check,
    PyUnicode_DATA,
    PyUnicode_GET_LENGTH,
    PyUnicode_KIND,
    PyUnicode_New,
)
from athenian.api.native.numpy cimport PyArray_ScalarAsCtype, PyIntegerArrType_Type

import_array()


def in_any_values_inline(values) -> str:
    if PyArray_CheckExact(values):
        return in_any_values_inline_array(values)
    if PyList_CheckExact(values):
        return in_any_values_inline_list(values)
    raise ValueError(f"Only numpy arrays and lists are supported, got {type(values)}")


cdef str in_any_values_inline_array(ndarray values):
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
        char *buf = <char *> PyUnicode_DATA(<PyObject *> result)
        char *data = <char *> PyArray_DATA(values)

    with nogil:
        if is_str:
            if is_s:
                _in_any_values_array_s(data, stride, itemsize, length, buf)
            else:
                _in_any_values_array_u(data, stride, itemsize, length, buf)
        elif itemsize == 8:
            _in_any_values_array_int64(data, stride, length, effective_itemsize, buf)
        else:
            raise AssertionError(f"unsupported dtype {dtype}")
    return result


cdef void _in_any_values_array_s(
    const char *data,
    int stride,
    int itemsize,
    int length,
    char *output,
) nogil:
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


cdef void _in_any_values_array_u(
    const char *data,
    int stride,
    int itemsize,
    int length,
    char *output,
) nogil:
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


cdef void _in_any_values_array_int64(
    const char *data,
    int stride,
    int length,
    int alignment,
    char *output,
) nogil:
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


cdef str in_any_values_inline_list(list values):
    cdef:
        Py_ssize_t i, length = len(values)
        PyObject *item
        bint with_null = length == 0

    for i in range(length):
        item = PyList_GET_ITEM(<PyObject *> values, i)
        if item == Py_None:
            with_null = True
            continue
        if PyUnicode_Check(item):
            return _in_any_values_inline_list_u(values)
        elif PyBytes_Check(item):
            return _in_any_values_inline_list_s(values)
        elif PyLong_CheckExact(item):
            return _in_any_values_inline_list_int64(values, False)
        elif PyObject_TypeCheck(item, &PyIntegerArrType_Type):
            return _in_any_values_inline_list_int64(values, True)
        else:
            raise ValueError(f"Unsupported type of list item #{i}: {type(values[i])}")

    assert with_null
    return "VALUES (null)"


cdef str _in_any_values_inline_list_u(list values):
    cdef:
        Py_ssize_t i, j, length = len(values), size1 = 7 - 1, size2 = 0, size4 = 0, str_len, result_len
        int kind, effective_kind
        Py_UCS4 max_char
        str result
        char *output
        char *border
        char *str_data
        PyObject *item

    for i in range(length):
        item = PyList_GET_ITEM(<PyObject *> values, i)
        if item == Py_None:
            continue
        if not PyUnicode_Check(item):
            raise ValueError(f"Mixed types in list: expected str, got {type(values[i])}")
        kind = PyUnicode_KIND(item)
        str_len = PyUnicode_GET_LENGTH(item)
        if kind == 1:
            size1 += str_len
        elif kind == 2:
            size2 += str_len
        else:
            size4 += str_len
        size1 += 5
    if size4 > 0:
        effective_kind = 4
    elif size2 > 0:
        effective_kind = 2
    else:
        effective_kind = 1
    result_len = size1 + size2 + size4
    max_char = (1 << (8 * effective_kind) - 1) if effective_kind != 4 else 1114111
    result = PyUnicode_New(result_len, max_char)
    output = <char *> PyUnicode_DATA(<PyObject *> result)
    with nogil:
        border = output + (result_len - 2) * effective_kind
        if effective_kind == 1:
            memcpy(output, b"VALUES ", 7)
            output += 7
        elif effective_kind == 2:
            memcpy(output, b"V\x00A\x00L\x00U\x00E\x00S\x00 \x00", 14)
            output += 14
        else:
            memcpy(
                output,
                b"V\x00\x00\x00A\x00\x00\x00L\x00\x00\x00U\x00\x00\x00E\x00\x00\x00S\x00\x00\x00 \x00\x00\x00",
                28,
            )
            output += 28

        for i in range(length):
            item = PyList_GET_ITEM(<PyObject *> values, i)
            if item == Py_None:
                continue
            kind = PyUnicode_KIND(item)
            str_len = PyUnicode_GET_LENGTH(item)
            str_data = <char *> PyUnicode_DATA(item)
            if effective_kind == 1:
                memcpy(output, b"('", 2)
                output += 2
                memcpy(output, str_data, str_len)
                output += str_len
                str_len = 3 if output != border else 2
                memcpy(output, b"'),", str_len)
                output += str_len
            elif effective_kind == 2:
                memcpy(output, b"(\x00'\x00", 4)
                output += 4
                if kind == 1:
                    for j in range(str_len):
                        output[j * 2] = str_data[j]
                        output[j * 2 + 1] = 0
                else:
                    memcpy(output, str_data, str_len * 2)
                output += str_len * 2
                str_len = 6 if output != border else 4
                memcpy(output, b"'\x00)\x00,\x00", str_len)
                output += str_len
            else:
                memcpy(output, b"(\x00\x00\x00'\x00\x00\x00", 8)
                output += 8
                if kind == 1:
                    for j in range(str_len):
                        output[j * 4] = str_data[j]
                        output[j * 4 + 1] = 0
                        output[j * 4 + 2] = 0
                        output[j * 4 + 3] = 0
                elif kind == 2:
                    for j in range(str_len):
                        output[j * 4] = str_data[j * 2]
                        output[j * 4 + 1] = str_data[j * 2 + 1]
                        output[j * 4 + 2] = 0
                        output[j * 4 + 3] = 0
                else:
                    memcpy(output, str_data, str_len * 4)
                output += str_len * 4
                str_len = 12 if output != border else 8
                memcpy(output, b"'\x00\x00\x00)\x00\x00\x00,\x00\x00\x00", str_len)
                output += str_len

    return result


cdef str _in_any_values_inline_list_s(list values):
    cdef:
        Py_ssize_t i, length = len(values), str_len, result_len = 0
        str result
        char *output
        char *border
        PyObject *item

    for i in range(length):
        item = PyList_GET_ITEM(<PyObject *> values, i)
        if item == Py_None:
            continue
        if not PyBytes_Check(item):
            raise ValueError(f"Mixed types in list: expected bytes, got {type(values[i])}")
        result_len += PyBytes_GET_SIZE(item) + 5

    result_len += 7 - 1
    result = PyUnicode_New(result_len, 255)
    output = <char *> PyUnicode_DATA(<PyObject *> result)
    with nogil:
        border = output + result_len - 2
        memcpy(output, b"VALUES ", 7)
        output += 7

        for i in range(length):
            item = PyList_GET_ITEM(<PyObject *> values, i)
            if item == Py_None:
                continue
            str_len = PyBytes_GET_SIZE(item)
            str_data = PyBytes_AS_STRING(item)
            memcpy(output, b"('", 2)
            output += 2
            memcpy(output, str_data, str_len)
            output += str_len
            str_len = 3 if output != border else 2
            memcpy(output, b"'),", str_len)
            output += str_len

    return result


@cython.cdivision(True)
cdef str _in_any_values_inline_list_int64(list values, bint numpy):
    cdef:
        Py_ssize_t i, length = len(values), pos = 7, nulls = 0, result_len, valstart
        long val = 0, max_val = 0
        int digits = 0
        str result
        char *output
        lldiv_t qr
        PyObject *item

    for i in range(length):
        item = PyList_GET_ITEM(<PyObject *> values, i)
        if item == Py_None:
            nulls += 1
            continue
        if numpy:
            PyArray_ScalarAsCtype(item, &val)
        else:
            val = PyLong_AsLong(item)
        if val > max_val:
            max_val = val

    while max_val:
        max_val //= 10
        digits += 1

    result_len = 6 + 1 + (3 + digits) * (length - nulls) - 1
    result = PyUnicode_New(result_len, 255)
    output = <char *> PyUnicode_DATA(<PyObject *> result)
    with nogil:
        memcpy(output, b"VALUES ", 7)
        for i in range(length):
            item = PyList_GET_ITEM(<PyObject *> values, i)
            if item == Py_None:
                continue
            if numpy:
                PyArray_ScalarAsCtype(item, &val)
            else:
                val = PyLong_AsLong(item)
            output[pos] = b"("
            pos += 1
            valstart = pos
            pos += digits - 1
            qr.quot = val
            while True:
                qr = lldiv(qr.quot, 10)
                output[pos] = (<char> b'0') + (<char> qr.rem)
                pos -= 1
                if qr.quot == 0:
                    break
            while pos >= valstart:
                output[pos] = b" "
                pos -= 1
            pos = valstart + digits
            output[pos] = b")"
            pos += 1
            if pos < result_len:
                output[pos] = b","
                pos += 1
    return result


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
        char *buf = <char *> PyUnicode_DATA(<PyObject *> result)
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
