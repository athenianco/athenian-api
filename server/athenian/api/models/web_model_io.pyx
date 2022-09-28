# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize

cimport cython
from cpython cimport (
    Py_DECREF,
    Py_INCREF,
    PyBytes_FromStringAndSize,
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
from libc.stdint cimport int32_t, uint16_t, uint32_t
from libc.stdio cimport FILE, SEEK_CUR, fclose, fread, fseek, ftell, fwrite
from libc.stdlib cimport free
from libc.string cimport memcpy
from libcpp.vector cimport vector
from numpy cimport import_array, npy_int64

import pickle

from athenian.api.typing_utils import is_generic, is_optional


cdef extern from "stdio.h" nogil:
    FILE *open_memstream(char **ptr, size_t *sizeloc)
    FILE *fmemopen(void *buf, size_t size, const char *mode)


cdef extern from "structmember.h":
    ctypedef struct PyMemberDef:
        const char *name
        int type
        Py_ssize_t offset
        int flags
        const char *doc


cdef extern from "Python.h":
    ctypedef PyObject *(*allocfunc)(PyTypeObject *cls, Py_ssize_t nitems)

    ctypedef struct PyTypeObject:
        allocfunc tp_alloc
        PyMemberDef *tp_members

    char *PyBytes_AS_STRING(PyObject *) nogil
    Py_ssize_t PyBytes_GET_SIZE(PyObject *) nogil
    PyObject *PyList_GET_ITEM(PyObject *, Py_ssize_t) nogil
    Py_ssize_t PyList_GET_SIZE(PyObject *) nogil
    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t) nogil
    bint PyUnicode_Check(PyObject *) nogil
    bint PyBytes_Check(PyObject *) nogil
    bint PyList_CheckExact(PyObject *) nogil
    Py_ssize_t PyUnicode_GET_LENGTH(PyObject *) nogil
    void *PyUnicode_DATA(PyObject *) nogil
    unsigned int PyUnicode_KIND(PyObject *) nogil
    bint PyLong_CheckExact(PyObject *) nogil
    long PyLong_AsLong(PyObject *) nogil
    double PyFloat_AS_DOUBLE(PyObject *) nogil
    bint PyFloat_CheckExact(PyObject *) nogil
    bint PyObject_TypeCheck(PyObject *, PyTypeObject *) nogil

    unsigned int PyUnicode_1BYTE_KIND
    unsigned int PyUnicode_2BYTE_KIND
    unsigned int PyUnicode_4BYTE_KIND

    PyObject *Py_None
    PyObject *Py_True
    PyObject *Py_False
    PyTypeObject PyLong_Type
    PyTypeObject PyFloat_Type
    PyTypeObject PyUnicode_Type
    PyTypeObject PyBool_Type

    str PyUnicode_FromKindAndData(unsigned int kind, void *buffer, Py_ssize_t size)
    PyObject *PyObject_GetAttr(PyObject *o, object attr_name)


cdef extern from "datetime.h" nogil:
    bint PyDateTime_Check(PyObject *)
    bint PyDelta_Check(PyObject *)

    int PyDateTime_GET_YEAR(PyObject *)
    int PyDateTime_GET_MONTH(PyObject *)
    int PyDateTime_GET_DAY(PyObject *)

    int PyDateTime_DATE_GET_HOUR(PyObject *)
    int PyDateTime_DATE_GET_MINUTE(PyObject *)
    int PyDateTime_DATE_GET_SECOND(PyObject *)

    int PyDateTime_DELTA_GET_DAYS(PyObject *)
    int PyDateTime_DELTA_GET_SECONDS(PyObject *)

    ctypedef struct PyDateTime_CAPI:
        PyObject *TimeZone_UTC


cdef extern from "numpy/arrayobject.h" nogil:
    PyTypeObject PyDatetimeArrType_Type
    PyTypeObject PyTimedeltaArrType_Type

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

    void PyArray_ScalarAsCtype(PyObject *scalar, void *ctypeptr)
    bint PyArray_IsIntegerScalar(PyObject *)


import_datetime()
import_array()


cdef enum DataType:
    DT_INVALID = 0
    DT_MODEL = 1
    DT_LIST = 2
    DT_LONG = 3
    DT_FLOAT = 4
    DT_STRING = 5
    DT_DT = 6
    DT_TD = 7
    DT_BOOL = 8


ctypedef struct ModelFields:
    DataType type
    Py_ssize_t offset
    PyTypeObject *model
    vector[ModelFields] nested


cdef inline DataType _discover_data_type(PyObject *obj, PyTypeObject **deref) except DT_INVALID:
    cdef:
        PyTypeObject *as_type = <PyTypeObject *> obj

    if is_optional(<object> obj):
        args = (<object> obj).__args__
        as_type = <PyTypeObject *> PyTuple_GET_ITEM(<PyObject *> args, 0)
    if as_type == &PyLong_Type:
        return DT_LONG
    elif as_type == &PyFloat_Type:
        return DT_FLOAT
    elif as_type == &PyUnicode_Type:
        return DT_STRING
    elif as_type == PyDateTimeAPI.DateTimeType:
        return DT_DT
    elif as_type == PyDateTimeAPI.DeltaType:
        return DT_TD
    elif as_type == &PyBool_Type:
        return DT_BOOL
    elif is_generic(<object> obj):
        args = (<object> obj).__args__
        as_type = <PyTypeObject *> PyTuple_GET_ITEM(<PyObject *> args, 0)
        if is_optional(<object> as_type):
            args = (<object> as_type).__args__
            as_type = <PyTypeObject *> PyTuple_GET_ITEM(<PyObject *> args, 0)
        deref[0] = as_type
        return DT_LIST
    elif hasattr(<object> obj, "attribute_types"):
        deref[0] = as_type
        return DT_MODEL
    else:
        raise AssertionError(f"Field type is not supported: {<object> obj}")


cdef inline void _apply_data_type(
    Py_ssize_t offset,
    PyObject *member_type,
    ModelFields *fields,
) except *:
    cdef:
        PyTypeObject *deref = NULL
        DataType dtype = _discover_data_type(member_type, &deref)
    if deref != NULL:
        fields.nested.push_back(_discover_fields(deref, dtype, offset))
    else:
        fields.nested.push_back(ModelFields(dtype, offset, NULL))


cdef ModelFields _discover_fields(PyTypeObject *model, DataType dtype, Py_ssize_t offset) except *:
    cdef:
        PyObject *attribute_types_attr
        object attribute_types
        PyObject *member_type
        PyMemberDef *members
        ModelFields fields = ModelFields(dtype, offset, NULL)

    attribute_types_attr = PyObject_GetAttr(<PyObject *> model, "attribute_types")
    if attribute_types_attr != NULL:
        attribute_types = <object> attribute_types_attr
        Py_DECREF(<object> attribute_types_attr)
        fields.model = model
        members = model.tp_members
        for i in range(len((<object> model).__slots__)):
            member_type = PyDict_GetItemString(attribute_types, members[i].name + 1)
            _apply_data_type(members[i].offset, member_type, &fields)
    else:
        _apply_data_type(0, <PyObject *> model, &fields)
    return fields


@cython.cdivision(True)
cdef PyObject *_write_object(PyObject *obj, ModelFields *spec, FILE *stream) nogil:
    cdef:
        char dtype = spec.type, bool
        long val_long = 0
        double val_double
        uint32_t str_length, val32 = 0, i
        uint16_t val16[4]
        int32_t vali32
        PyObject *exc
        bint is_unicode, is_float
        NPY_DATETIMEUNIT npy_unit
        npy_int64 obval
    if obj == Py_None:
        dtype = 0
        fwrite(&dtype, 1, 1, stream)
        return NULL
    fwrite(&dtype, 1, 1, stream)
    if dtype == DT_LONG:
        if not PyLong_CheckExact(obj):
            if PyArray_IsIntegerScalar(obj):
                PyArray_ScalarAsCtype(obj, &val_long)
            else:
                return obj
        else:
            val_long = PyLong_AsLong(obj)
        fwrite(&val_long, sizeof(long), 1, stream)
    elif dtype == DT_FLOAT:
        is_float = PyFloat_CheckExact(obj)
        if not is_float and not PyLong_CheckExact(obj):
            return obj
        if is_float:
            val_double = PyFloat_AS_DOUBLE(obj)
        else:
            val_double = PyLong_AsLong(obj)
        fwrite(&val_double, sizeof(double), 1, stream)
    elif dtype == DT_STRING:
        is_unicode = PyUnicode_Check(obj)
        if not is_unicode and not PyBytes_Check(obj):
            return obj
        if is_unicode:
            str_length = PyUnicode_GET_LENGTH(obj)
            val32 = str_length | ((PyUnicode_KIND(obj) - 1) << 30)
            fwrite(&val32, 4, 1, stream)
            fwrite(PyUnicode_DATA(obj), 1, str_length, stream)
        else:
            val32 = PyBytes_GET_SIZE(obj)
            fwrite(&val32, 4, 1, stream)
            fwrite(PyBytes_AS_STRING(obj), 1, val32, stream)
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
        fwrite(val16, 2, 3, stream)
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
        fwrite(&vali32, 4, 1, stream)
        fwrite(val16, 2, 1, stream)
    elif dtype == DT_BOOL:
        bool = obj == Py_True
        if not bool and obj != Py_False:
            return obj
        fwrite(&bool, 1, 1, stream)
    elif dtype == DT_LIST:
        if not PyList_CheckExact(obj):
            return obj
        val32 = PyList_GET_SIZE(obj)
        fwrite(&val32, 4, 1, stream)
        if spec.model != NULL:
            spec.type = DT_MODEL
            for i in range(val32):
                exc = _write_object(PyList_GET_ITEM(obj, i), spec, stream)
                if exc != NULL:
                    return exc
            spec.type = DT_LIST
        else:
            for i in range(val32):
                exc = _write_object(PyList_GET_ITEM(obj, i), &spec.nested[0], stream)
                if exc != NULL:
                    return exc
    elif dtype == DT_MODEL:
        val32 = spec.nested.size()
        fwrite(&val32, 4, 1, stream)
        for field in spec.nested:
            exc = _write_object((<PyObject **>((<char *> obj) + field.offset))[0], &field, stream)
            if exc != NULL:
                return exc
    else:
        return obj
    return NULL


cdef void _serialize_list_of_models(list models, FILE *stream) except *:
    cdef:
        uint32_t size
        ModelFields spec
        type item_type
        PyObject *exc

    if len(models) == 0:
        size = 0
        fwrite(&size, 4, 1, stream)
        return
    item_type = type(models[0])
    result = pickle.dumps(item_type)
    spec = _discover_fields(<PyTypeObject *> item_type, DT_LIST, 0)
    with nogil:
        size = PyBytes_GET_SIZE(<PyObject *> result)
        fwrite(&size, 4, 1, stream)
        fwrite(PyBytes_AS_STRING(<PyObject *> result), 1, size, stream)
        exc = _write_object(<PyObject *> models, &spec, stream)
    if exc != NULL:
        raise ValueError(f"Could not serialize {<object> exc} of type {type(<object> exc)}")


cdef void _serialize_generic(model, FILE *stream) except *:
    cdef:
        bytes buf = pickle.dumps(model)
        uint32_t size = len(buf)
    fwrite(&size, 4, 1, stream)
    fwrite(PyBytes_AS_STRING(<PyObject *> buf), size, 1, stream)


def serialize_models(tuple models not None) -> bytes:
    cdef:
        char *output = NULL
        size_t output_size = 0
        FILE *stream
        bytes result
        char count
    assert len(models) < 255
    stream = open_memstream(&output, &output_size)
    count = len(models)
    fwrite(&count, 1, 1, stream)
    try:
        for model in models:
            if PyList_CheckExact(<PyObject *> model):
                _serialize_list_of_models(model, stream)
            else:
                _serialize_generic(model, stream)
    finally:
        fclose(stream)
        result = PyBytes_FromStringAndSize(output, output_size)
        free(output)
    return result


def deserialize_models(bytes buffer not None) -> tuple[list[object], ...]:
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
            if not isinstance(model_type, type):
                model = model_type
            else:
                spec = _discover_fields(<PyTypeObject *> model_type, DT_LIST, 0)
                model = _read_model(&spec, stream, input, corrupted_msg)
        Py_INCREF(model)
        PyTuple_SET_ITEM(result, tuple_pos, model)
    fclose(stream)
    return result


cdef object _read_model(ModelFields *spec, FILE *stream, const char *raw, str corrupted_msg):
    cdef:
        char dtype = 0, bool = 0
        long long_val = 0
        double double_val = 0
        uint32_t aux32 = 0, i
        int32_t auxi32 = 0
        uint16_t aux16[4]
        unsigned int kind
        int year, month, day, hour, minute, second
        PyObject *utctz = (<PyDateTime_CAPI *> PyDateTimeAPI).TimeZone_UTC
        PyObject *obj_val
        ModelFields field

    if fread(&dtype, 1, 1, stream) != 1:
        raise ValueError(corrupted_msg % (ftell(stream), "dtype"))
    if dtype == DT_INVALID:
        return None
    if dtype != spec.type:
        raise ValueError(corrupted_msg % (ftell(stream), f"dtype {dtype} != {spec.type}"))
    if dtype == DT_LONG:
        if fread(&long_val, sizeof(long), 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "long"))
        return PyLong_FromLong(long_val)
    elif dtype == DT_FLOAT:
        if fread(&double_val, sizeof(double), 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "float"))
        return PyFloat_FromDouble(double_val)
    elif dtype == DT_STRING:
        if fread(&aux32, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "str/header"))
        kind = (aux32 >> 30) + 1
        aux32 &= 0x3FFFFFFF
        long_val = ftell(stream)
        if fseek(stream, aux32, SEEK_CUR):
            raise ValueError(corrupted_msg % (ftell(stream), "str/body"))
        return PyUnicode_FromKindAndData(kind, raw + long_val, aux32)
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
        if aux32 != spec.nested.size():
            raise ValueError(corrupted_msg % (ftell(stream), f"{obj} has changed"))
        obj = obj.__new__(obj)
        for field in spec.nested:
            val = _read_model(&field, stream, raw, corrupted_msg)
            Py_INCREF(val)
            (<PyObject **>((<char *><PyObject *> obj) + field.offset))[0] = <PyObject *> val
        return obj
    elif dtype == DT_LIST:
        if fread(&aux32, 4, 1, stream) != 1:
            raise ValueError(corrupted_msg % (ftell(stream), "list"))
        obj = PyList_New(aux32)
        for i in range(aux32):
            if spec.model != NULL:
                spec.type = DT_MODEL
                val = _read_model(spec, stream, raw, corrupted_msg)
                spec.type = DT_LIST
            else:
                val = _read_model(&spec.nested[0], stream, raw, corrupted_msg)
            Py_INCREF(val)
            PyList_SET_ITEM(obj, i, val)
        return obj
    else:
        raise AssertionError(f"Unsupported dtype: {dtype}")
