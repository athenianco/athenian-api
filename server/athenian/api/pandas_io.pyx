# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize
from datetime import timezone
import pickle
from typing import Any

from cpython cimport (
    Py_INCREF,
    PyBytes_FromStringAndSize,
    PyList_New,
    PyList_SET_ITEM,
    PyLong_FromLong,
    PyObject,
    PyTuple_New,
    PyTuple_SET_ITEM,
)
from libc.stdint cimport uint32_t
from libc.string cimport memcmp, memcpy, memset, strcpy, strncmp
from libcpp.string cimport string
from libcpp.vector cimport vector
from numpy cimport (
    NPY_OBJECT,
    NPY_STRING,
    NPY_UNICODE,
    NPY_VOID,
    PyArray_Descr,
    dtype as npdtype,
    import_array,
    ndarray,
    npy_intp,
)

import numpy as np
from pandas import DataFrame, DatetimeTZDtype
from pandas.core.arrays import DatetimeArray
from pandas.core.internals import BlockManager, DatetimeTZBlock

import_array()


cdef extern from "<string.h>" nogil:
    size_t strnlen(const char *, size_t)


cdef extern from "Python.h":
    # added nogil -> from cpython cimport ...
    char *PyBytes_AS_STRING(PyObject *) nogil
    Py_ssize_t PyBytes_GET_SIZE(PyObject *) nogil
    PyObject *PyList_GET_ITEM(PyObject *, Py_ssize_t) nogil
    Py_ssize_t PyList_GET_SIZE(PyObject *) nogil
    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t) nogil
    bint PyUnicode_Check(PyObject *) nogil
    bint PyList_CheckExact(PyObject *) nogil
    Py_ssize_t PyUnicode_GET_LENGTH(PyObject *) nogil
    char *PyUnicode_DATA(PyObject *) nogil
    unsigned int PyUnicode_KIND(PyObject *) nogil
    bint PyLong_CheckExact(PyObject *) nogil
    long PyLong_AsLong(PyObject *) nogil
    PyObject *Py_None

    str PyUnicode_FromKindAndData(unsigned int kind, void *buffer, Py_ssize_t size)


cdef extern from "numpy/arrayobject.h":
    char *PyArray_BYTES(PyObject *) nogil
    npy_intp PyArray_DIM(PyObject *, size_t) nogil
    int PyArray_NDIM(PyObject *) nogil
    npy_intp PyArray_ITEMSIZE(PyObject *) nogil
    bint PyArray_CheckExact(PyObject *) nogil
    PyArray_Descr *PyArray_DESCR(PyObject *) nogil
    int PyArray_TYPE(PyObject *) nogil


def serialize_args(tuple args) -> bytes:
    cdef:
        bytes result, buffer
        Py_ssize_t size = 4
        list buffers = []
        char *output
        bint is_df

    for arg in args:
        if isinstance(arg, DataFrame):
            is_df = True
            buffer = serialize_df(arg)
        else:
            is_df = False
            buffer = pickle.dumps(arg)
        size += len(buffer) + 5
        buffers.append((is_df, buffer))
    result = PyBytes_FromStringAndSize(NULL, size)
    output = PyBytes_AS_STRING(<PyObject *> result)
    (<uint32_t *> output)[0] = len(buffers)
    output += 4
    for is_df, buffer in buffers:
        output[0] = is_df
        output += 1
        size = len(buffer)
        (<uint32_t *> output)[0] = size
        output += 4
        memcpy(output, PyBytes_AS_STRING(<PyObject *> buffer), size)
        output += size
    return result


def deserialize_args(bytes buffer) -> tuple[Any]:
    cdef:
        uint32_t size, i
        tuple result
        long offset = 4
        object item
        char is_df

    input = PyBytes_AS_STRING(<PyObject *> buffer)
    size = (<uint32_t *> input)[0]
    input += 4
    result = PyTuple_New(size)
    for i in range(size):
        is_df = input[0]
        input += 1
        size = (<uint32_t *> input)[0]
        input += 4
        offset += 5
        if is_df:
            item = deserialize_df(buffer[offset: offset + size])
        else:
            item = pickle.loads(buffer[offset: offset + size])
        offset += size
        input += size
        Py_INCREF(item)
        PyTuple_SET_ITEM(result, i, item)
    return result


DEF nodim = 0xFFFFFFFF


def serialize_df(df not None) -> bytes:
    cdef:
        list blocks = [], arrs = []
        object arr
        PyObject *arr_obj
        PyObject *arr_item
        PyObject *aux_bytes
        long size = 0, aux_size, object_block_size = 0, ndim
        bytes pickled, buffer, obj_block
        char *input
        char *output
        Py_ssize_t i, error_i = -1
        vector[vector[ColumnMeasurement]] measurements

    mgr = df._mgr
    measurements.resize(len(mgr.blocks))
    for i, block in enumerate(mgr.blocks):
        loc, arr = block.__getstate__()
        blocks.append((type(block), loc))
        arr_obj = <PyObject *> arr
        if not PyArray_CheckExact(arr_obj):
            # DatetimeArray
            arr = arr._data
            arr_obj = <PyObject *> arr
        size += 16 + 4 * 2  # common header: dtype and shape
        ndim = PyArray_NDIM(arr_obj)
        assert 0 < ndim <= 2, f"block #{i} dimensions are not supported: {block}"
        if arr.dtype != object:
            size += (
                PyArray_ITEMSIZE(arr_obj)
                * PyArray_DIM(arr_obj, 0)
                * (PyArray_DIM(arr_obj, 1) if ndim > 1 else 1)
            )
            object_block_size = 0
        else:
            assert ndim == 2
            object_block_size = _measure_object_block(arr_obj, &measurements[i])
            if object_block_size <= 0:
                raise AssertionError(
                    f'Unsupported object column "{df.columns[loc[-object_block_size]]}": '
                    f'{arr[-object_block_size, :10]}'
                )
            size += object_block_size
        arrs.append((arr, str(arr.dtype).encode(), object_block_size))
    pickled = pickle.dumps((mgr.axes, blocks))
    aux_bytes = <PyObject *> pickled
    aux_size = PyBytes_GET_SIZE(aux_bytes)
    buffer = PyBytes_FromStringAndSize(NULL, 4 + aux_size + size)
    with nogil:
        input = PyBytes_AS_STRING(aux_bytes)
        output = PyBytes_AS_STRING(<PyObject *> buffer)
        (<uint32_t *> output)[0] = aux_size
        output += 4
        memcpy(output, input, aux_size)
        output += aux_size
        for i in range(PyList_GET_SIZE(<PyObject *> arrs)):
            arr_item = PyList_GET_ITEM(<PyObject *> arrs, i)
            arr_obj = PyTuple_GET_ITEM(arr_item, 0)
            aux_bytes = PyTuple_GET_ITEM(arr_item, 1)
            object_block_size = PyLong_AsLong(PyTuple_GET_ITEM(arr_item, 2))
            aux_size = PyBytes_GET_SIZE(aux_bytes)
            if aux_size > 16:
                error_i = i
                break
            input = PyBytes_AS_STRING(aux_bytes)
            memcpy(output, input, aux_size)
            memset(output + aux_size, 0, 16 - aux_size)
            output += 16
            ndim = PyArray_NDIM(arr_obj)
            (<uint32_t *> output)[0] = PyArray_DIM(arr_obj, 0)
            (<uint32_t *> output)[1] = PyArray_DIM(arr_obj, 1) if ndim > 1 else nodim
            output += 8
            if not strncmp(input, b"object", aux_size):
                _serialize_object_block(arr_obj, output, &measurements[i])
                output += object_block_size
                continue
            aux_size = PyArray_ITEMSIZE(arr_obj) * PyArray_DIM(arr_obj, 0) * (PyArray_DIM(arr_obj, 1) if ndim > 1 else 1)
            memcpy(output, PyArray_BYTES(arr_obj), aux_size)
            output += aux_size
    del pickled
    if error_i >= 0:
        raise AssertionError(f"Unsupported block: {mgr.blocks[error_i]}")
    return buffer


cdef enum ObjectDType:
    ODT_INVALID = 0
    ODT_NDARRAY_FIXED = 1
    ODT_NDARRAY_RAGGED = 2
    ODT_NDARRAY_STR = 3
    ODT_LIST_STR = 4
    ODT_INT = 5
    ODT_STR = 6


cdef struct ColumnMeasurement:
    ObjectDType dtype
    long nulls
    string subdtype


cdef long _measure_object_block(PyObject *block, vector[ColumnMeasurement] *measurements):
    cdef:
        npy_intp columns, rows, y, x, i
        PyObject **data = <PyObject **> PyArray_BYTES(block)
        PyObject *item
        PyObject **subitems
        PyObject *subitem
        ObjectDType dtype
        Py_ssize_t size = 0, item_size, subitem_size
        long nulls
        char *descr = NULL
        str str_dtype
        int int_dtype
        ColumnMeasurement *measurement

    with nogil:
        columns = PyArray_DIM(block, 0)
        rows = PyArray_DIM(block, 1)
        measurements.resize(columns)
        measurement = measurements.data()
        for y in range(columns):
            # 1 byte to encode the column's inferred type
            # uint32_t count of nulls
            # 16 bytes to encode the nested dtype if applicable
            size += 1 + 4 + 16
            nulls = int_dtype = 0
            dtype = ODT_INVALID
            for x in range(rows):
                item = data[x]
                if item == Py_None:
                    size += 4
                    nulls += 1
                    continue
                if dtype == ODT_INVALID:
                    if PyArray_CheckExact(item):
                        descr = (<char *>PyArray_DESCR(item)) + sizeof(PyObject)
                        with gil:
                            str_dtype = str(<object>PyArray_DESCR(item))
                            item_size = PyUnicode_GET_LENGTH(<PyObject *> str_dtype)
                            if item_size > 16:
                                return -y
                            measurement[y].subdtype = string(
                                <const char *>PyUnicode_DATA(<PyObject *> str_dtype),
                                item_size,
                            )
                            del str_dtype
                        int_dtype = PyArray_TYPE(item)
                        if int_dtype != NPY_OBJECT:
                            if int_dtype == NPY_VOID:
                                return -y
                            if int_dtype == NPY_STRING or int_dtype == NPY_UNICODE:
                                dtype = ODT_NDARRAY_RAGGED
                            else:
                                dtype = ODT_NDARRAY_FIXED
                        else:
                            dtype = ODT_NDARRAY_STR
                    elif PyList_CheckExact(item):
                        dtype = ODT_LIST_STR
                    elif PyUnicode_Check(item):
                        dtype = ODT_STR
                    elif PyLong_CheckExact(item):
                        dtype = ODT_INT
                    else:
                        return -y
                    measurement[y].dtype = dtype
                if dtype <= ODT_NDARRAY_STR:
                    if (
                        not PyArray_CheckExact(item)
                        or PyArray_NDIM(item) != 1
                    ):
                        return -y
                    if PyArray_TYPE(item) != int_dtype or (
                        int_dtype != NPY_STRING and int_dtype != NPY_UNICODE and
                        memcmp(
                            descr,
                            (<char *>PyArray_DESCR(item)) + sizeof(PyObject),
                            8 + 4 + sizeof(int) * 3,
                        )
                    ):
                        return -y

                    if dtype == ODT_NDARRAY_STR:
                        subitems = <PyObject **> PyArray_BYTES(item)
                        size += 4
                        for i in range(PyArray_DIM(item, 0)):
                            subitem = subitems[i]
                            if not PyUnicode_Check(subitem):
                                return -y
                            subitem_size = PyUnicode_GET_LENGTH(subitem)
                            if subitem_size >= (1 << (32 - 2)):  # we pack kind in the first two bits
                                return -y
                            size += PyUnicode_GET_LENGTH(subitem) * PyUnicode_KIND(subitem) + 4
                    else:
                        if dtype == ODT_NDARRAY_RAGGED:
                            size += 4
                        size += PyArray_ITEMSIZE(item) * PyArray_DIM(item, 0) + 4
                elif dtype == ODT_LIST_STR:
                    if not PyList_CheckExact(item):
                        return -y
                    size += 4
                    for i in range(PyList_GET_SIZE(item)):
                        subitem = PyList_GET_ITEM(item, i)
                        if not PyUnicode_Check(subitem):
                            return -y
                        subitem_size = PyUnicode_GET_LENGTH(subitem)
                        if subitem_size >= (1 << (32 - 2)):  # we pack kind in the first two bits
                            return -y
                        size += PyUnicode_GET_LENGTH(subitem) * PyUnicode_KIND(subitem) + 4
                elif dtype == ODT_STR:
                    if not PyUnicode_Check(item):
                        return -y
                    item_size = PyUnicode_GET_LENGTH(item)
                    if item_size >= (1 << (32 - 2)):  # we pack kind in the first two bits
                        return -y
                    size += PyUnicode_GET_LENGTH(item) * PyUnicode_KIND(item) + 4
                elif dtype == ODT_INT:
                    if not PyLong_CheckExact(item):
                        return -y
                    size += 8
            measurement[y].nulls = nulls
            data += rows
    return size


cdef void _serialize_object_block(
    PyObject *block,
    char *output,
    vector[ColumnMeasurement] *measurements,
) nogil:
    cdef:
        npy_intp columns, rows, y, x, i
        PyObject **data = <PyObject **> PyArray_BYTES(block)
        PyObject *item
        PyObject **subitems
        ObjectDType dtype
        Py_ssize_t item_size, subitem_size
        uint32_t *bookmark
        ColumnMeasurement measurement

    columns = PyArray_DIM(block, 0)
    rows = PyArray_DIM(block, 1)
    for y in range(columns):
        measurement = measurements.data()[y]
        output[0] = dtype = measurement.dtype
        output += 1
        (<uint32_t *> output)[0] = measurement.nulls
        output += 4
        strcpy(output, measurement.subdtype.c_str())
        item_size = measurement.subdtype.size() + 1
        memset(output + item_size, 0, 16 - item_size)
        output += 16
        bookmark = <uint32_t *> output
        output += measurement.nulls * 4
        for x in range(rows):
            item = data[x]
            if item == Py_None:
                bookmark[0] = x
                bookmark += 1
                continue
            if dtype == ODT_NDARRAY_STR:
                item_size = PyArray_DIM(item, 0)
                (<uint32_t *> output)[0] = item_size
                output += 4
                subitems = <PyObject **> PyArray_BYTES(item)
                for i in range(item_size):
                    _write_str(subitems[i], &output)
            elif dtype == ODT_NDARRAY_FIXED or dtype == ODT_NDARRAY_RAGGED:
                item_size = PyArray_DIM(item, 0)
                (<uint32_t *> output)[0] = item_size
                output += 4
                subitem_size = PyArray_ITEMSIZE(item)
                if dtype == ODT_NDARRAY_RAGGED:
                    (<uint32_t *> output)[0] = subitem_size
                    output += 4
                item_size *= subitem_size
                memcpy(output, PyArray_BYTES(item), item_size)
                output += item_size
            elif dtype == ODT_LIST_STR:
                item_size = PyList_GET_SIZE(item)
                (<uint32_t *> output)[0] = item_size
                output += 4
                for i in range(item_size):
                    _write_str(PyList_GET_ITEM(item, i), &output)
            elif dtype == ODT_STR:
                _write_str(item, &output)
            elif dtype == ODT_INT:
                (<long *> output)[0] = PyLong_AsLong(item)
                output += 8
        data += rows


cdef inline void _write_str(PyObject *obj, char **output) nogil:
    cdef:
        Py_ssize_t length
        uint32_t size
        unsigned int kind
    length = PyUnicode_GET_LENGTH(obj)
    kind = PyUnicode_KIND(obj)
    size = length * kind
    (<uint32_t *> output[0])[0] = length | ((kind - 1) << (32 - 2))
    output[0] += 4
    memcpy(output[0], PyUnicode_DATA(obj), size)
    output[0] += size


class CorruptedBuffer(ValueError):
    """Any buffer format mismatch."""


def deserialize_df(bytes buffer) -> DataFrame:
    cdef:
        char *input = PyBytes_AS_STRING(<PyObject *> buffer)
        char *origin = input
        Py_ssize_t input_size = PyBytes_GET_SIZE(<PyObject *> buffer)
        Py_ssize_t origin_size = input_size
        uint32_t aux, columns, rows, y, x, i, nulls_count, null_index, nullpos, subitem_size
        tuple shape
        char *arr_data
        long arr_size
        ObjectDType obj_dtype
        str subdtype_name, listval
        npdtype subdtype
        ndarray subarr
        list sublist
        uint32_t *nullptr
        object block

    if input_size < 4:
        raise CorruptedBuffer()
    aux = (<uint32_t *> input)[0]
    input += 4
    input_size -= 4
    if input_size < aux:
        raise CorruptedBuffer()

    axes, blocks = pickle.loads(buffer[4:4 + aux])
    input += aux
    input_size -= aux
    mgr_blocks = []
    for block_type, loc in blocks:
        if input_size < 16 + 8:
            raise CorruptedBuffer()
        dtype = npdtype(input[:strnlen(input, 16)].decode())
        input += 16
        input_size -= 16
        columns = (<uint32_t *> input)[0]
        rows = (<uint32_t *> input)[1]
        input += 8
        input_size -= 8
        if rows != nodim:
            shape = (columns, rows)
        else:
            shape = (columns,)
        arr = np.empty(shape, dtype=dtype)
        if dtype != object:
            arr_data = PyArray_BYTES(<PyObject *> arr)
            arr_size = dtype.itemsize * columns * (rows if rows != nodim else 1)
            if input_size < arr_size:
                raise CorruptedBuffer()
            memcpy(arr_data, input, arr_size)
            input += arr_size
            input_size -= arr_size
        else:
            for y in range(columns):
                if input_size < 1 + 4 + 16:
                    raise CorruptedBuffer()
                obj_dtype = <ObjectDType> input[0]
                nulls_count = (<uint32_t *> (input + 1))[0]
                input += 5
                input_size -= 5
                subdtype_name = input[:strnlen(input, 16)].decode()
                if subdtype_name:
                    subdtype = npdtype(subdtype_name)
                else:
                    subdtype = None
                input += 16
                input_size -= 16
                if input_size < nulls_count * 4:
                    raise CorruptedBuffer()
                nullptr = <uint32_t *>input
                input += nulls_count * 4
                input_size -= nulls_count * 4
                nullpos = 0
                if nullpos < nulls_count:
                    null_index = nullptr[0]
                else:
                    null_index = 0xFFFFFFFF
                for x in range(rows):
                    if x == null_index:
                        nullpos += 1
                        if nullpos < nulls_count:
                            null_index = nullptr[nullpos]
                        else:
                            null_index = 0xFFFFFFFF
                        continue  # already None in np.empty()
                    if obj_dtype == ODT_NDARRAY_STR:
                        if input_size < 4:
                            raise CorruptedBuffer()
                        aux = (<uint32_t *> input)[0]
                        input += 4
                        input_size -= 4
                        subarr = np.empty(aux, dtype=object)
                        for i in range(aux):
                            subarr[i] = _read_str(&input, &input_size)
                        arr[y, x] = subarr
                    elif obj_dtype == ODT_NDARRAY_FIXED:
                        if input_size < 4:
                            raise CorruptedBuffer()
                        aux = (<uint32_t *> input)[0]
                        input += 4
                        input_size -= 4
                        subarr = np.empty(aux, dtype=subdtype)
                        aux *= subdtype.itemsize
                        if input_size < aux:
                            raise CorruptedBuffer()
                        memcpy(PyArray_BYTES(<PyObject *> subarr), input, aux)
                        input += aux
                        input_size -= aux
                        arr[y, x] = subarr
                    elif obj_dtype == ODT_NDARRAY_RAGGED:
                        if input_size < 8:
                            raise CorruptedBuffer()
                        aux = (<uint32_t *> input)[0]
                        subitem_size = (<uint32_t *> input)[1]
                        input += 8
                        input_size -= 8
                        subdtype = npdtype((chr(subdtype.kind), subitem_size))
                        subarr = np.empty(aux, dtype=subdtype)
                        aux *= subitem_size
                        if input_size < aux:
                            raise CorruptedBuffer()
                        memcpy(PyArray_BYTES(<PyObject *> subarr), input, aux)
                        input += aux
                        input_size -= aux
                        arr[y, x] = subarr
                    elif obj_dtype == ODT_LIST_STR:
                        if input_size < 4:
                            raise CorruptedBuffer()
                        aux = (<uint32_t *> input)[0]
                        input += 4
                        input_size -= 4
                        sublist = PyList_New(aux)
                        for i in range(aux):
                            listval = _read_str(&input, &input_size)
                            Py_INCREF(listval)
                            PyList_SET_ITEM(sublist, i, listval)
                        arr[y, x] = sublist
                    elif obj_dtype == ODT_STR:
                        try:
                            arr[y, x] = _read_str(&input, &input_size)
                        except CorruptedBuffer as e:
                            raise e from None
                    elif obj_dtype == ODT_INT:
                        if input_size < 8:
                            raise CorruptedBuffer()
                        arr[y, x] = PyLong_FromLong((<long *> input)[0])
                        input += 8
                        input_size -= 8
        if block_type is DatetimeTZBlock:
            block = block_type(DatetimeArray(arr, dtype=DatetimeTZDtype(tz=timezone.utc)), loc)
        else:
            block = block_type(arr, loc)
        mgr_blocks.append(block)
    assert input_size == 0
    assert (input - origin) == origin_size
    mgr = BlockManager(mgr_blocks, axes, do_integrity_check=False)
    df = DataFrame(mgr)
    return df


cdef str _read_str(char **input, Py_ssize_t *input_size):
    cdef:
        uint32_t header
        unsigned int kind
        Py_ssize_t size
        str result

    if input_size[0] < 4:
        raise CorruptedBuffer()
    header = (<uint32_t *> input[0])[0]
    input[0] += 4
    input_size[0] -= 4
    kind = 1 + (header >> (32 - 2))
    header &= 0x3FFFFFFF
    if input_size[0] < header:
        raise CorruptedBuffer()
    result = PyUnicode_FromKindAndData(kind, input[0], header)
    size = header * kind
    input[0] += size
    input_size[0] -= size
    return result
