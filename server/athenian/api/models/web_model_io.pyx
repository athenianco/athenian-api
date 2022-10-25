# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=False
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize -std=c++20
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib

cimport cython

from cython.operator import dereference

from cpython cimport (
    Py_INCREF,
    PyBytes_FromStringAndSize,
    PyDict_New,
    PyDict_SetItem,
    PyFloat_FromDouble,
    PyList_New,
    PyList_SET_ITEM,
    PyLong_FromLong,
    PyObject,
    PyTuple_New,
    PyTuple_SET_ITEM,
)
from cpython.datetime cimport PyDateTimeAPI, import_datetime
from cpython.dict cimport PyDict_GetItemString
from libc.stdint cimport int32_t, uint8_t, uint16_t, uint32_t
from libc.stdio cimport FILE, SEEK_CUR, fclose, fread, fseek, ftell
from libc.string cimport memcpy, strlen
from numpy cimport import_array, npy_int64

from athenian.api.native.cpython cimport (
    Py_False,
    Py_None,
    Py_True,
    Py_TYPE,
    PyBool_Type,
    PyBytes_AS_STRING,
    PyBytes_Check,
    PyBytes_GET_SIZE,
    PyDateTime_CAPI,
    PyDateTime_Check,
    PyDateTime_DATE_GET_HOUR,
    PyDateTime_DATE_GET_MINUTE,
    PyDateTime_DATE_GET_SECOND,
    PyDateTime_DATE_GET_TZINFO,
    PyDateTime_DELTA_GET_DAYS,
    PyDateTime_DELTA_GET_SECONDS,
    PyDateTime_GET_DAY,
    PyDateTime_GET_MONTH,
    PyDateTime_GET_YEAR,
    PyDelta_Check,
    PyDict_CheckExact,
    PyDict_Next,
    PyDict_Size,
    PyDict_Type,
    PyFloat_AS_DOUBLE,
    PyFloat_Check,
    PyFloat_CheckExact,
    PyFloat_Type,
    PyList_CheckExact,
    PyList_GET_ITEM,
    PyList_GET_SIZE,
    PyList_Type,
    PyLong_AsLong,
    PyLong_CheckExact,
    PyLong_Type,
    PyMemberDef,
    PyObject_TypeCheck,
    PyTuple_GET_ITEM,
    PyTypeObject,
    PyUnicode_1BYTE_KIND,
    PyUnicode_2BYTE_KIND,
    PyUnicode_4BYTE_KIND,
    PyUnicode_Check,
    PyUnicode_DATA,
    PyUnicode_FromKindAndData,
    PyUnicode_GET_LENGTH,
    PyUnicode_KIND,
    PyUnicode_Type,
)
from athenian.api.native.mi_heap_stl_allocator cimport (
    ios_in,
    ios_out,
    mi_heap_allocator_from_capsule,
    mi_heap_stl_allocator,
    mi_stringstream,
    mi_vector,
)
from athenian.api.native.numpy cimport (
    PyArray_CheckExact,
    PyArray_DATA,
    PyArray_DIM,
    PyArray_IS_C_CONTIGUOUS,
    PyArray_NDIM,
    PyArray_ScalarAsCtype,
    PyDatetimeArrType_Type,
    PyDoubleArrType_Type,
    PyFloatArrType_Type,
    PyIntegerArrType_Type,
    PyTimedeltaArrType_Type,
)
from athenian.api.native.optional cimport optional
from athenian.api.native.utf8 cimport ucs4_to_utf8_json

import pickle
from types import GenericAlias

from athenian.api.typing_utils import is_generic, is_optional


cdef extern from "stdio.h" nogil:
    FILE *fmemopen(void *buf, size_t size, const char *mode)


cdef extern from "<stdlib.h>" nogil:
    char *gcvt(double number, int ndigit, char *buf)


cdef extern from "numpy/arrayobject.h" nogil:
    ctypedef enum NPY_DATETIMEUNIT:
        NPY_FR_ERROR = -1
        NPY_FR_M = 1
        NPY_FR_W = 2
        NPY_FR_D = 4
        NPY_FR_h = 5
        NPY_FR_m = 6
        NPY_FR_s = 7
        NPY_FR_ms = 8
        NPY_FR_us = 9
        NPY_FR_ns = 10
        NPY_FR_ps = 11
        NPY_FR_fs = 12
        NPY_FR_as = 13
        NPY_FR_GENERIC = 14

    ctypedef struct PyArray_DatetimeMetaData:
        NPY_DATETIMEUNIT base
        int num

    ctypedef struct PyDatetimeScalarObject:
        npy_int64 obval
        PyArray_DatetimeMetaData obmeta


cdef extern from "web_model_io.h" nogil:
    void set_datetimestruct_days(npy_int64 days, int *year, int *month, int *day)


import_datetime()
import_array()


cdef enum DataType:
    DT_INVALID = 0
    DT_MODEL = 1
    DT_LIST = 2
    DT_DICT = 3
    DT_LONG = 4
    DT_FLOAT = 5
    DT_STRING = 6
    DT_DT = 7
    DT_TD = 8
    DT_BOOL = 9


cdef enum DataFlags:
    DF_KEY_UNMAPPED = 1
    DF_OPTIONAL = 2
    DF_VERBATIM = 4


ctypedef struct ModelFields:
    DataType type
    uint8_t flags
    Py_ssize_t offset
    const void *key
    PyTypeObject *model
    optional[mi_vector[ModelFields]] nested


cdef inline DataType _discover_data_type(
    PyTypeObject *obj,
    PyTypeObject **deref,
    bint *optional,
    bint *verbatim,
) except DT_INVALID:
    if is_optional(<object> obj):
        optional[0] = 1
        verbatim[0] = hasattr((<object> obj).__origin__, "__verbatim__")
        args = (<object> obj).__args__
        obj = <PyTypeObject *> PyTuple_GET_ITEM(<PyObject *> args, 0)
    if obj == &PyLong_Type:
        return DT_LONG
    elif obj == &PyFloat_Type:
        return DT_FLOAT
    elif obj == &PyUnicode_Type:
        return DT_STRING
    elif obj == PyDateTimeAPI.DateTimeType:
        return DT_DT
    elif obj == PyDateTimeAPI.DeltaType:
        return DT_TD
    elif obj == &PyBool_Type:
        return DT_BOOL
    elif is_generic(<object> obj):
        origin = (<object> obj).__origin__
        args = (<object> obj).__args__
        deref[0] = obj
        if <PyTypeObject *> origin == &PyList_Type:
            return DT_LIST
        elif <PyTypeObject *> origin == &PyDict_Type:
            return DT_DICT
        else:
            return DT_INVALID
    elif hasattr(<object> obj, "attribute_types"):
        deref[0] = obj
        return DT_MODEL
    else:
        raise AssertionError(f"Field type is not supported: {<object> obj}")


cdef inline void _apply_data_type(
    Py_ssize_t offset,
    PyTypeObject *member_type,
    ModelFields *fields,
    mi_heap_stl_allocator[char] &alloc,
) except *:
    cdef:
        PyTypeObject *deref = NULL
        bint optional = 0, verbatim = 0
        DataType dtype = _discover_data_type(member_type, &deref, &optional, &verbatim)
        ModelFields *back = &dereference(fields.nested).emplace_back()
    back.type = dtype
    back.offset = offset
    if optional:
        back.flags |= DF_OPTIONAL
    if verbatim:
        back.flags |= DF_VERBATIM
    back.nested.emplace(alloc)
    if deref != NULL:
        _discover_fields(deref, back, alloc)


cdef void _discover_fields(
    PyTypeObject *model,
    ModelFields *fields,
    mi_heap_stl_allocator[char] &alloc,
) except *:
    cdef:
        object attribute_types
        object attribute_map
        PyTypeObject *member_type
        PyMemberDef *members
        PyObject *key
        ModelFields *back

    if fields.type == DT_MODEL:
        attribute_types = (<object> model).attribute_types
        attribute_map = (<object> model).attribute_map
        fields.model = model
        members = model.tp_members
        for i in range(len((<object> model).__slots__)):
            member_type = <PyTypeObject *> PyDict_GetItemString(attribute_types, members[i].name + 1)
            _apply_data_type(members[i].offset, member_type, fields, alloc)
            back = &dereference(fields.nested).back()
            key = PyDict_GetItemString(attribute_map, members[i].name + 1)
            if key != NULL:
                back.flags &= ~DF_KEY_UNMAPPED
                back.key = key
            else:
                back.flags |= DF_KEY_UNMAPPED
                back.key = members[i].name + 1
    elif fields.type == DT_LIST:
        attribute_types = (<object> model).__args__
        _apply_data_type(
            0,
            <PyTypeObject *> PyTuple_GET_ITEM(<PyObject *> attribute_types, 0),
            fields,
            alloc,
        )
    elif fields.type == DT_DICT:
        attribute_types = (<object> model).__args__
        _apply_data_type(
            0,
            <PyTypeObject *> PyTuple_GET_ITEM(<PyObject *> attribute_types, 0),
            fields,
            alloc,
        )
        _apply_data_type(
            0,
            <PyTypeObject *> PyTuple_GET_ITEM(<PyObject *> attribute_types, 1),
            fields,
            alloc,
        )
    else:
        raise AssertionError(f"Cannot recurse in dtype {fields.type}")


@cython.cdivision(True)
cdef PyObject *_write_object(PyObject *obj, ModelFields *spec, mi_stringstream &stream) nogil:
    cdef:
        char dtype = spec.type, bool
        long val_long
        double val_double
        float val_float
        uint32_t str_length, val32, i
        uint16_t val16[4]
        int32_t vali32
        PyObject *exc
        bint is_unicode, is_float
        NPY_DATETIMEUNIT npy_unit
        npy_int64 obval
        Py_ssize_t dict_pos = 0
        PyObject *dict_key = NULL
        PyObject *dict_val = NULL
        PyObject **npdata
        ModelFields *field
    if obj == Py_None:
        dtype = 0
        stream.write(<char *> &dtype, 1)
        return NULL
    stream.write(<char *> &dtype, 1)
    if dtype == DT_LONG:
        if PyLong_CheckExact(obj):
            val_long = PyLong_AsLong(obj)
        elif PyObject_TypeCheck(obj, &PyIntegerArrType_Type):
            val_long = 0
            PyArray_ScalarAsCtype(obj, &val_long)
        else:
            return obj
        stream.write(<char *> &val_long, sizeof(long))
    elif dtype == DT_FLOAT:
        if PyFloat_CheckExact(obj):
            val_double = PyFloat_AS_DOUBLE(obj)
        elif PyLong_CheckExact(obj):
            val_double = PyLong_AsLong(obj)
        elif PyObject_TypeCheck(obj, &PyDoubleArrType_Type):
            PyArray_ScalarAsCtype(obj, &val_double)
        elif PyObject_TypeCheck(obj, &PyFloatArrType_Type):
            PyArray_ScalarAsCtype(obj, &val_float)
            val_double = val_float
        elif PyObject_TypeCheck(obj, &PyIntegerArrType_Type):
            val_long = 0
            PyArray_ScalarAsCtype(obj, &val_long)
            val_double = val_long
        else:
            return obj
        stream.write(<char *> &val_double, sizeof(double))
    elif dtype == DT_STRING:
        is_unicode = PyUnicode_Check(obj)
        if not is_unicode and not PyBytes_Check(obj):
            return obj
        if is_unicode:
            str_length = PyUnicode_GET_LENGTH(obj)
            val32 = str_length | ((PyUnicode_KIND(obj) - 1) << 30)
            stream.write(<char *> &val32, 4)
            # each code point in PyUnicode_DATA buffer has PyUnicode_KIND(obj) bytes
            stream.write(<char *> PyUnicode_DATA(obj), PyUnicode_KIND(obj) * str_length)
        else:
            val32 = PyBytes_GET_SIZE(obj)
            stream.write(<char *> &val32, 4)
            stream.write(PyBytes_AS_STRING(obj), val32)
    elif dtype == DT_DT:
        if not PyDateTime_Check(obj):
            if PyObject_TypeCheck(obj, &PyDatetimeArrType_Type):
                npy_unit = (<PyDatetimeScalarObject *> obj).obmeta.base
                obval = (<PyDatetimeScalarObject *> obj).obval
                if npy_unit == NPY_FR_ns:
                    obval //= 1000000000
                elif npy_unit == NPY_FR_us:
                    obval //= 1000000
                elif npy_unit != NPY_FR_s:
                    return obj
                memcpy(val16, &obval, 8)   # little-endian
            else:
                return obj
        else:
            val16[0] = PyDateTime_GET_YEAR(obj) << 4
            val16[0] |= PyDateTime_GET_MONTH(obj)
            val16[1] = PyDateTime_GET_DAY(obj) << 7
            val16[1] |= PyDateTime_DATE_GET_HOUR(obj)
            val16[2] = (PyDateTime_DATE_GET_MINUTE(obj) << 8) | 0x8000
            val16[2] |= PyDateTime_DATE_GET_SECOND(obj)
        stream.write(<char *> val16, 2 * 3)
    elif dtype == DT_TD:
        if not PyDelta_Check(obj):
            if PyObject_TypeCheck(obj, &PyTimedeltaArrType_Type):
                npy_unit = (<PyDatetimeScalarObject *> obj).obmeta.base
                obval = (<PyDatetimeScalarObject *> obj).obval
                if npy_unit == NPY_FR_ns:
                    obval //= 1000000000
                elif npy_unit == NPY_FR_us:
                    obval //= 1000000
                elif npy_unit != NPY_FR_s:
                    return obj
                if obval >= 0:
                    vali32 = obval // (24 * 3600)
                    var_long = obval % (24 * 3600)
                else:
                    vali32 = -1 -(obval // (24 * 3600))
                    var_long = 24 * 3600 + obval % (24 * 3600)
            else:
                return obj
        else:
            vali32 = PyDateTime_DELTA_GET_DAYS(obj)
            var_long = PyDateTime_DELTA_GET_SECONDS(obj)
        vali32 <<= 1
        if var_long >= 1 << 16:
            vali32 |= 1
        val16[0] = var_long & 0xFFFF
        stream.write(<char *> &vali32, 4)
        stream.write(<char *> val16, 2)
    elif dtype == DT_BOOL:
        bool = obj == Py_True
        if not bool and obj != Py_False:
            return obj
        stream.write(<char *> &bool, 1)
    elif dtype == DT_LIST:
        if not PyList_CheckExact(obj):
            if not PyArray_CheckExact(obj) or not PyArray_IS_C_CONTIGUOUS(obj) or PyArray_NDIM(obj) != 1:
                return obj
            val32 = PyArray_DIM(obj, 0)
            stream.write(<char *> &val32, 4)
            npdata = <PyObject **> PyArray_DATA(obj)
            for i in range(val32):
                exc = _write_object(npdata[i], &dereference(spec.nested)[0], stream)
                if exc != NULL:
                    return exc
        else:
            val32 = PyList_GET_SIZE(obj)
            stream.write(<char *> &val32, 4)
            for i in range(val32):
                exc = _write_object(PyList_GET_ITEM(obj, i), &dereference(spec.nested)[0], stream)
                if exc != NULL:
                    return exc
    elif dtype == DT_DICT:
        if not PyDict_CheckExact(obj):
            return obj
        val32 = PyDict_Size(obj)
        stream.write(<char *> &val32, 4)
        while PyDict_Next(obj, &dict_pos, &dict_key, &dict_val):
            exc = _write_object(dict_key, &dereference(spec.nested)[0], stream)
            if exc != NULL:
                return exc
            exc = _write_object(dict_val, &dereference(spec.nested)[1], stream)
            if exc != NULL:
                return exc
    elif dtype == DT_MODEL:
        val32 = dereference(spec.nested).size()
        stream.write(<char *> &val32, 4)
        for i in range(val32):
            field = &dereference(spec.nested)[i]
            exc = _write_object(
                dereference(<PyObject **>((<char *> obj) + field.offset)),
                field,
                stream,
            )
            if exc != NULL:
                return exc
    else:
        return obj
    return NULL


cdef void _serialize_list_of_models(
    list models,
    mi_stringstream &stream,
    mi_heap_stl_allocator[char] &alloc,
) except *:
    cdef:
        uint32_t size
        ModelFields spec
        type item_type
        PyObject *exc

    spec.type = DT_LIST
    spec.nested.emplace(alloc)
    if len(models) == 0:
        size = 0
        stream.write(<char *> &size, 4)
        return
    item_type = type(models[0])
    result = pickle.dumps(GenericAlias(list, (item_type,)))
    _apply_data_type(0, <PyTypeObject *> item_type, &spec, alloc)
    with nogil:
        size = PyBytes_GET_SIZE(<PyObject *> result)
        stream.write(<char *> &size, 4)
        stream.write(PyBytes_AS_STRING(<PyObject *> result), size)
        exc = _write_object(<PyObject *> models, &spec, stream)
    if exc != NULL:
        raise ValueError(f"Could not serialize {<object> exc} of type {type(<object> exc)} in {item_type.__name__}")


cdef void _serialize_generic(model, mi_stringstream &stream) except *:
    cdef:
        bytes buf = pickle.dumps(model)
        uint32_t size = len(buf)
    stream.write(<char *> &size, 4)
    stream.write(PyBytes_AS_STRING(<PyObject *> buf), size)


def serialize_models(tuple models not None, alloc_capsule=None) -> bytes:
    cdef:
        optional[mi_stringstream] stream
        bytes result
        char count
        optional[mi_heap_stl_allocator[char]] alloc
        size_t size
    assert len(models) < 255
    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
        dereference(alloc).disable_free()
    stream.emplace(ios_in | ios_out, dereference(alloc))
    count = len(models)
    dereference(stream).write(&count, 1)
    for model in models:
        if PyList_CheckExact(<PyObject *> model):
            _serialize_list_of_models(model, dereference(stream), dereference(alloc))
        else:
            _serialize_generic(model, dereference(stream))
    size = dereference(stream).tellp()
    result = PyBytes_FromStringAndSize(NULL, size)
    dereference(stream).seekg(0)
    dereference(stream).read(PyBytes_AS_STRING(<PyObject *> result), size)
    return result


def deserialize_models(bytes buffer not None, alloc_capsule=None) -> tuple[list[object], ...]:
    cdef:
        char *input = PyBytes_AS_STRING(<PyObject *> buffer)
        uint32_t aux = 0, tuple_pos
        str corrupted_msg = "Corrupted buffer ar position %d: %s"
        FILE *stream
        tuple result
        long pos
        bytes type_buf
        object model_type
        ModelFields spec
        optional[mi_heap_stl_allocator[char]] alloc

    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
        dereference(alloc).disable_free()
    stream = fmemopen(input, PyBytes_GET_SIZE(<PyObject *> buffer), b"r")
    if fread(&aux, 1, 1, stream) != 1:
        raise ValueError(corrupted_msg % (ftell(stream), "tuple"))
    result = PyTuple_New(aux)
    for tuple_pos in range(aux):
        if fread(&aux, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "pickle/header"))
        if aux == 0:
            model = []
        else:
            pos = ftell(stream)
            if fseek(stream, aux, SEEK_CUR):
                raise ValueError(corrupted_msg % (ftell(stream), "pickle/body"))
            type_buf = PyBytes_FromStringAndSize(input + pos, aux)
            model_type = pickle.loads(type_buf)
            if not isinstance(model_type, (type, GenericAlias)):
                model = model_type
            else:
                spec.type = DT_LIST
                spec.nested.emplace(dereference(alloc))
                _discover_fields(<PyTypeObject *> model_type, &spec, dereference(alloc))
                model = _read_model(&spec, stream, input, corrupted_msg)
        Py_INCREF(model)
        PyTuple_SET_ITEM(result, tuple_pos, model)
    fclose(stream)
    return result


cdef object _read_model(ModelFields *spec, FILE *stream, const char *raw, str corrupted_msg):
    cdef:
        char dtype, bool
        long val_long
        double val_double
        uint32_t aux32, i
        int32_t auxi32
        uint16_t aux16[4]
        unsigned int kind
        int year, month, day, hour, minute, second
        PyObject *utctz = (<PyDateTime_CAPI *> PyDateTimeAPI).TimeZone_UTC
        PyObject *obj_val
        ModelFields *field

    if fread(&dtype, 1, 1, stream) != 1:
        raise ValueError(corrupted_msg % (ftell(stream), "dtype"))
    if dtype == DT_INVALID:
        return None
    if dtype != spec.type:
        raise ValueError(corrupted_msg % (ftell(stream), f"dtype {dtype} != {spec.type}"))
    if dtype == DT_LONG:
        if fread(&val_long, sizeof(long), 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "long"))
        return PyLong_FromLong(val_long)
    elif dtype == DT_FLOAT:
        if fread(&val_double, sizeof(double), 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "float"))
        return PyFloat_FromDouble(val_double)
    elif dtype == DT_STRING:
        if fread(&aux32, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "str/header"))
        kind = (aux32 >> 30) + 1
        aux32 &= 0x3FFFFFFF
        val_long = ftell(stream)
        # move stream forward of the number of bytes we are about to read from raw
        if fseek(stream, aux32 * kind, SEEK_CUR):
            raise ValueError(corrupted_msg % (ftell(stream), "str/body"))
        return PyUnicode_FromKindAndData(kind, raw + val_long, aux32)
    elif dtype == DT_DT:
        if fread(aux16, 2, 3, stream) != 3:
            raise ValueError(corrupted_msg % (ftell(stream), "dt"))
        if aux16[2] & 0x8000:
            year = aux16[0] >> 4
            month = aux16[0] & 0xF
            day = aux16[1] >> 7
            hour = aux16[1] & 0x7F
            minute = (aux16[2] >> 8) & 0x7F
            second = aux16[2] & 0xFF
        else:
            aux16[3] = 0
            obj_val = PyDatetimeArrType_Type.tp_alloc(&PyDatetimeArrType_Type, 0)
            memcpy(&(<PyDatetimeScalarObject *>obj_val).obval, aux16, 8)
            (<PyDatetimeScalarObject *> obj_val).obmeta.base = NPY_FR_s
            (<PyDatetimeScalarObject *> obj_val).obmeta.num = 1
            return <object> obj_val
        return PyDateTimeAPI.DateTime_FromDateAndTime(
            year, month, day, hour, minute, second, 0, <object> utctz, PyDateTimeAPI.DateTimeType,
        )
    elif dtype == DT_TD:
        if fread(&auxi32, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "td"))
        if fread(aux16, 2, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "td"))
        day = auxi32 >> 1
        second = aux16[0] + ((auxi32 & 1) << 16)
        return PyDateTimeAPI.Delta_FromDelta(day, second, 0, 1, PyDateTimeAPI.DeltaType)
    elif dtype == DT_BOOL:
        if fread(&bool, 1, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "bool"))
        if bool:
            return True
        return False
    elif dtype == DT_MODEL:
        obj = <object> spec.model
        if fread(&aux32, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "model"))
        if aux32 != dereference(spec.nested).size():
            raise ValueError(corrupted_msg % (ftell(stream), f"{obj} has changed"))
        obj = obj.__new__(obj)
        for i in range(aux32):
            field = &dereference(spec.nested)[i]
            val = _read_model(field, stream, raw, corrupted_msg)
            Py_INCREF(val)
            (<PyObject **>((<char *><PyObject *> obj) + field.offset))[0] = <PyObject *> val
        return obj
    elif dtype == DT_LIST:
        if fread(&aux32, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "list"))
        obj = PyList_New(aux32)
        for i in range(aux32):
            val = _read_model(&dereference(spec.nested)[0], stream, raw, corrupted_msg)
            Py_INCREF(val)
            PyList_SET_ITEM(obj, i, val)
        return obj
    elif dtype == DT_DICT:
        if fread(&aux32, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "dict"))
        obj = PyDict_New()
        for i in range(aux32):
            key = _read_model(&dereference(spec.nested)[0], stream, raw, corrupted_msg)
            val = _read_model(&dereference(spec.nested)[1], stream, raw, corrupted_msg)
            PyDict_SetItem(obj, key, val)
        return obj
    else:
        raise AssertionError(f"Unsupported dtype: {dtype}")


def model_to_json(model, alloc_capsule=None) -> bytes:
    cdef:
        bytes result
        optional[mi_heap_stl_allocator[char]] alloc
        optional[mi_stringstream] stream
        type root_type
        ModelFields spec
        PyObject *error

    if model is None:
        return b"null"
    if PyList_CheckExact(<PyObject *> model):
        if PyList_GET_SIZE(<PyObject *> model) == 0:
            return b"[]"
        spec.type = DT_LIST
        root_type = type(model[0])
    else:
        spec.type = DT_MODEL
        root_type = type(model)

    if alloc_capsule is not None:
        alloc.emplace(dereference(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
        dereference(alloc).disable_free()
    stream.emplace(ios_in | ios_out, dereference(alloc))

    spec.nested.emplace(dereference(alloc))
    _apply_data_type(0, <PyTypeObject *> root_type, &spec, dereference(alloc))

    with nogil:
        error = _write_json(<PyObject *> model, spec, dereference(stream))
    if error != NULL:
        raise AssertionError(f"failed to serialize to JSON: {<object> error}")

    size = dereference(stream).tellp()
    result = PyBytes_FromStringAndSize(NULL, size)
    dereference(stream).seekg(0)
    dereference(stream).read(PyBytes_AS_STRING(<PyObject *> result), size)
    return result


cdef ModelFields fake_str_model
fake_str_model.type = DT_STRING


@cython.cdivision(True)
cdef PyObject *_write_json(PyObject *obj, ModelFields &spec, mi_stringstream &stream) nogil:
    cdef:
        PyObject *key = NULL
        PyObject *value = NULL
        PyObject *r
        Py_ssize_t pos = 0, size, i, j, item_len, char_len
        unsigned int kind
        char sym
        char *data
        int aux, auxdiv, rem, year, month, day
        long val_long, div
        npy_int64 obval
        double val_double
        float val_float
        char buffer[24]
        ModelFields *nested

    if obj == Py_None:
        stream.write(b"null", 4)
        return NULL
    if spec.type == DT_MODEL:
        if Py_TYPE(obj).tp_members == NULL:
            # this is just a check for __slots__, it's hard to validate better without GIL
            return obj
        stream.write(b"{", 1)
        kind = 0
        for i in range(<Py_ssize_t> dereference(spec.nested).size()):
            nested = &dereference(spec.nested)[i]
            value = dereference(<PyObject **>((<char *> obj) + nested.offset))
            if (nested.flags & DF_OPTIONAL) and not (nested.flags & DF_VERBATIM):
                if value == NULL or value == Py_None:
                    continue
                if PyList_CheckExact(value) and PyList_GET_SIZE(value) == 0:
                    continue
                if PyDict_CheckExact(value) and PyDict_Size(value) == 0:
                    continue
                if PyArray_CheckExact(value) and PyArray_NDIM(value) == 1 and PyArray_DIM(value, 0) == 0:
                    continue
            if kind:
                stream.write(b",", 1)
            else:
                kind = 1
            if nested.flags & DF_KEY_UNMAPPED:
                stream.write(b'"', 1)
                stream.write(<const char *> nested.key, strlen(<const char *> nested.key))
                stream.write(b'"', 1)
            else:
                r = _write_json(<PyObject *> nested.key, fake_str_model, stream)
                if r != NULL:
                    return r
            stream.write(b":", 1)
            r = _write_json(value, dereference(nested), stream)
            if r != NULL:
                return r
        stream.write(b"}", 1)
    elif spec.type == DT_DICT:
        stream.write(b"{", 1)
        if not PyDict_CheckExact(obj):
            return obj
        while PyDict_Next(obj, &pos, &key, &value):
            if pos != 1:
                stream.write(b",", 1)
            r = _write_json(key, dereference(spec.nested)[0], stream)
            if r != NULL:
                return r
            stream.write(b":", 1)
            r = _write_json(value, dereference(spec.nested)[1], stream)
            if r != NULL:
                return r
        stream.write(b"}", 1)
    elif spec.type == DT_LIST:
        stream.write(b"[", 1)
        nested = &dereference(spec.nested)[0]
        if not PyList_CheckExact(obj):
            if not PyArray_CheckExact(obj) or not PyArray_IS_C_CONTIGUOUS(obj) or PyArray_NDIM(obj) != 1:
                return obj
            npdata = <PyObject **> PyArray_DATA(obj)
            for i in range(PyArray_DIM(obj, 0)):
                if i != 0:
                    stream.write(b",", 1)
                r = _write_json(npdata[i], dereference(nested), stream)
                if r != NULL:
                    return r
        else:
            for i in range(PyList_GET_SIZE(obj)):
                if i != 0:
                    stream.write(b",", 1)
                r = _write_json(PyList_GET_ITEM(obj, i), dereference(nested), stream)
                if r != NULL:
                    return r
        stream.write(b"]", 1)
    elif spec.type == DT_STRING:
        stream.write(b'"', 1)

        if PyUnicode_Check(obj):
            data = <char *> PyUnicode_DATA(obj)
            kind = PyUnicode_KIND(obj)
            item_len = PyUnicode_GET_LENGTH(obj)
            if kind == PyUnicode_1BYTE_KIND:
                for i in range(item_len):
                    stream.write(buffer, ucs4_to_utf8_json(data[i], buffer))
            elif kind == PyUnicode_2BYTE_KIND:
                for i in range(item_len):
                    stream.write(buffer, ucs4_to_utf8_json((<uint16_t *> data)[i], buffer))
            elif kind == PyUnicode_4BYTE_KIND:
                for i in range(item_len):
                    stream.write(buffer, ucs4_to_utf8_json((<uint32_t *> data)[i], buffer))
        elif PyBytes_Check(obj):
            data = PyBytes_AS_STRING(obj)
            item_len = PyBytes_GET_SIZE(obj)
            for i in range(item_len):
                stream.write(buffer, ucs4_to_utf8_json(data[i], buffer))
        else:
            return obj

        stream.write(b'"', 1)
    elif spec.type == DT_DT:
        buffer[0] = buffer[21] = b'"'
        if not PyDateTime_Check(obj):
            if PyObject_TypeCheck(obj, &PyDatetimeArrType_Type):
                npy_unit = (<PyDatetimeScalarObject *> obj).obmeta.base
                obval = (<PyDatetimeScalarObject *> obj).obval
                if npy_unit == NPY_FR_ns:
                    obval //= 1000000000
                elif npy_unit == NPY_FR_us:
                    obval //= 1000000
                elif npy_unit != NPY_FR_s:
                    return obj
                val_long = obval // (60 * 60 * 24)
                obval = obval - val_long * 60 * 60 * 24

                year = month = day = 0
                set_datetimestruct_days(val_long, &year, &month, &day)
                aux = year
                pos = 4
                while pos > 0:
                    auxdiv = aux
                    aux = aux // 10
                    buffer[pos] = auxdiv - aux * 10 + ord(b"0")
                    pos -= 1
                buffer[5] = b"-"

                aux = month
                if aux < 10:
                    buffer[6] = b"0"
                    buffer[7] = ord(b"0") + aux
                else:
                    buffer[6] = b"1"
                    buffer[7] = ord(b"0") + aux - 10
                buffer[8] = b"-"

                aux = day
                auxdiv = aux // 10
                buffer[9] = ord(b"0") + auxdiv
                buffer[10] = ord(b"0") + aux - auxdiv * 10
                buffer[11] = b"T"

                auxdiv = obval // 60
                aux = obval - auxdiv * 60
                rem = auxdiv
                auxdiv = aux // 10
                buffer[18] = ord(b"0") + auxdiv
                buffer[19] = ord(b"0") + aux - auxdiv * 10
                buffer[20] = b"Z"

                auxdiv = rem // 60
                aux = rem - auxdiv * 60
                rem = auxdiv
                auxdiv = aux // 10
                buffer[15] = ord(b"0") + auxdiv
                buffer[16] = ord(b"0") + aux - auxdiv * 10
                buffer[17] = b":"

                aux = rem
                auxdiv = aux // 10
                buffer[12] = ord(b"0") + auxdiv
                buffer[13] = ord(b"0") + aux - auxdiv * 10
                buffer[14] = b":"
            else:
                return obj
        else:
            if (<PyDateTime_CAPI *> PyDateTimeAPI).TimeZone_UTC != PyDateTime_DATE_GET_TZINFO(obj):
                return obj
            aux = PyDateTime_GET_YEAR(obj)
            pos = 4
            while pos > 0:
                auxdiv = aux
                aux = aux // 10
                buffer[pos] = auxdiv - aux * 10 + ord(b"0")
                pos -= 1
            buffer[5] = b"-"
            aux = PyDateTime_GET_MONTH(obj)
            if aux < 10:
                buffer[6] = b"0"
                buffer[7] = ord(b"0") + aux
            else:
                buffer[6] = b"1"
                buffer[7] = ord(b"0") + aux - 10
            buffer[8] = b"-"
            aux = PyDateTime_GET_DAY(obj)
            auxdiv = aux // 10
            buffer[9] = ord(b"0") + auxdiv
            buffer[10] = ord(b"0") + aux - auxdiv * 10
            buffer[11] = b"T"
            aux = PyDateTime_DATE_GET_HOUR(obj)
            auxdiv = aux // 10
            buffer[12] = ord(b"0") + auxdiv
            buffer[13] = ord(b"0") + aux - auxdiv * 10
            buffer[14] = b":"
            aux = PyDateTime_DATE_GET_MINUTE(obj)
            auxdiv = aux // 10
            buffer[15] = ord(b"0") + auxdiv
            buffer[16] = ord(b"0") + aux - auxdiv * 10
            buffer[17] = b":"
            aux = PyDateTime_DATE_GET_SECOND(obj)
            auxdiv = aux // 10
            buffer[18] = ord(b"0") + auxdiv
            buffer[19] = ord(b"0") + aux - auxdiv * 10
            buffer[20] = b"Z"
        stream.write(buffer, 22)
    elif spec.type == DT_TD:
        stream.write(b'"', 1)
        if not PyDelta_Check(obj):
            if PyObject_TypeCheck(obj, &PyTimedeltaArrType_Type):
                npy_unit = (<PyDatetimeScalarObject *> obj).obmeta.base
                val_long = (<PyDatetimeScalarObject *> obj).obval
                if npy_unit == NPY_FR_ns:
                    val_long //= 1000000000
                elif npy_unit == NPY_FR_us:
                    val_long //= 1000000
                elif npy_unit != NPY_FR_s:
                    return obj
            else:
                return obj
        else:
            val_long = PyDateTime_DELTA_GET_DAYS(obj)
            val_long *= 24 * 3600
            val_long += PyDateTime_DELTA_GET_SECONDS(obj)
        if val_long < 0:
            stream.write(b"-", 1)
            val_long = -val_long
        if val_long == 0:
            stream.write(b"0", 1)
        else:
            pos = 0
            while val_long:
                div = val_long
                val_long = val_long // 10
                buffer[pos] = div - val_long * 10 + ord(b"0")
                pos += 1
            for i in range(pos // 2):
                sym = buffer[i]
                div = pos - i - 1
                buffer[i] = buffer[div]
                buffer[div] = sym
            stream.write(buffer, pos)
        stream.write(b's"', 2)
    elif spec.type == DT_LONG:
        if PyLong_CheckExact(obj):
            val_long = PyLong_AsLong(obj)
        elif PyObject_TypeCheck(obj, &PyIntegerArrType_Type):
            val_long = 0
            PyArray_ScalarAsCtype(obj, &val_long)
        else:
            return obj
        if val_long < 0:
            stream.write(b"-", 1)
            val_long = -val_long
        if val_long == 0:
            stream.write(b"0", 1)
        else:
            pos = 0
            while val_long:
                div = val_long
                val_long = val_long // 10
                buffer[pos] = div - val_long * 10 + ord(b"0")
                pos += 1
            for i in range(pos // 2):
                sym = buffer[i]
                div = pos - i - 1
                buffer[i] = buffer[div]
                buffer[div] = sym
            stream.write(buffer, pos)
    elif spec.type == DT_BOOL:
        if obj == Py_True:
            stream.write(b"true", 4)
        elif obj == Py_False:
            stream.write(b"false", 5)
        else:
            return obj
    elif spec.type == DT_FLOAT:
        if PyFloat_CheckExact(obj):
            val_double = PyFloat_AS_DOUBLE(obj)
        elif PyLong_CheckExact(obj):
            val_double = PyLong_AsLong(obj)
        elif PyObject_TypeCheck(obj, &PyDoubleArrType_Type):
            PyArray_ScalarAsCtype(obj, &val_double)
        elif PyObject_TypeCheck(obj, &PyFloatArrType_Type):
            PyArray_ScalarAsCtype(obj, &val_float)
            val_double = val_float
        elif PyObject_TypeCheck(obj, &PyIntegerArrType_Type):
            val_long = 0
            PyArray_ScalarAsCtype(obj, &val_long)
            val_double = val_long
        else:
            return obj
        gcvt(val_double, 24, buffer)
        stream.write(buffer, strlen(buffer))
    else:
        return obj
    return NULL
