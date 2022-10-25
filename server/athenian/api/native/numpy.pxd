from cpython cimport PyObject
from numpy cimport dtype as npdtype, npy_intp

from athenian.api.native.cpython cimport PyTypeObject


cdef extern from "numpy/arrayobject.h":
    PyTypeObject PyArray_Type
    PyTypeObject PyDatetimeArrType_Type
    PyTypeObject PyDoubleArrType_Type
    PyTypeObject PyIntegerArrType_Type
    PyTypeObject PyFloatArrType_Type
    PyTypeObject PyTimedeltaArrType_Type

    ctypedef struct PyArray_Descr:
        char kind
        char type
        char byteorder
        char flags
        int type_num
        int itemsize "elsize"
        int alignment

    PyObject *PyArray_NewFromDescr(
        PyTypeObject *subtype,
        PyArray_Descr *descr,
        int nd,
        const npy_intp *dims,
        const npy_intp *strides,
        void *data,
        int flags,
        PyObject *obj,
    )
    npdtype PyArray_DescrNew(npdtype)

    void *PyArray_DATA(PyObject *) nogil
    char *PyArray_BYTES(PyObject *) nogil
    npy_intp PyArray_DIM(PyObject *, size_t) nogil
    npy_intp PyArray_STRIDE(PyObject *, size_t) nogil
    int PyArray_NDIM(PyObject *) nogil
    npy_intp PyArray_ITEMSIZE(PyObject *) nogil
    bint PyArray_CheckExact(PyObject *) nogil
    PyArray_Descr *PyArray_DESCR(PyObject *) nogil
    int PyArray_TYPE(PyObject *) nogil
    bint PyArray_IS_C_CONTIGUOUS(PyObject *) nogil
    bint PyArray_IS_F_CONTIGUOUS(PyObject *) nogil
    void PyArray_ScalarAsCtype(PyObject *scalar, void *ctypeptr) nogil
