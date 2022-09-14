# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -mavx2 -ftree-vectorize

from cpython cimport Py_INCREF, PyObject, PyTypeObject
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_New
from cython.operator cimport dereference
from libc.stdint cimport uint32_t, uint64_t
from libc.string cimport memcpy, memset
from libcpp.algorithm cimport sort as std_sort
from libcpp.unordered_map cimport pair, unordered_map
from libcpp.vector cimport vector
from numpy cimport (
    NPY_ARRAY_DEFAULT,
    NPY_STRING,
    PyArray_DATA,
    PyArray_Descr,
    PyArray_DescrFromType,
    PyArray_DIM,
    dtype as npdtype,
    import_array,
    ndarray,
    npy_intp,
)

import asyncpg
import numpy as np

from athenian.api.internal.miners.types import PullRequestJIRAIssueItem


cdef extern from "Python.h":
    PyObject *PyList_New(Py_ssize_t len)
    void PyList_SET_ITEM(PyObject *list, Py_ssize_t i, PyObject *o)
    Py_ssize_t PyUnicode_GET_LENGTH(str)
    void *PyUnicode_DATA(str)


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
    npdtype PyArray_DescrNew(npdtype)


cdef extern from "../../../asyncpg_recordobj.h":
    PyObject *ApgRecord_GET_ITEM(PyObject *, int)


cdef extern from "<string_view>" namespace "std" nogil:
    cppclass string_view:
        string_view() except +
        string_view(const char *, size_t) except +
        const char *data()
        size_t size()


ctypedef struct PullRequestAddr:
    uint32_t di
    uint32_t ri
    uint64_t pri


import_array()


cdef void delete_ptr_in_capsule(obj):
    cdef unordered_map[long, vector[PullRequestAddr]] *pr_to_ix = <unordered_map[long, vector[PullRequestAddr]] *> PyCapsule_GetPointer(obj, NULL)
    del pr_to_ix


def calc_pr_to_ix_prs(ndarray prs not None, ndarray prs_offsets not None) -> tuple[ndarray, object]:
    cdef:
        unordered_map[long, vector[PullRequestAddr]] *pr_to_ix = new unordered_map[long, vector[PullRequestAddr]]()
        npdtype objdtype = np.dtype(object)
        npy_intp deps_count = PyArray_DIM(prs, 0), di, ri, j, dep_prs_count, dep_prs_offsets_count
        PyObject **jira_col_data
        PyObject **prs_data = <PyObject **> PyArray_DATA(prs)
        PyObject **prs_offsets_data = <PyObject **> PyArray_DATA(prs_offsets)
        PyObject **dep_jira_col_data
        long *dep_prs_offsets_data
        long *dep_prs_data
        PyObject *dep_prs
        PyObject *dep_prs_offsets
        ndarray empty = np.array([], dtype="S")
        long beg, end
        object capsule

    (<PyObject *> objdtype).ob_refcnt += deps_count + 1
    jira_col = PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &deps_count,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    jira_col_data = <PyObject **> PyArray_DATA(jira_col)
    for di in range(deps_count):
        dep_prs = prs_data[di]
        dep_prs_count = PyArray_DIM(<ndarray> dep_prs, 0)
        dep_prs_data = <long *> PyArray_DATA(<ndarray> dep_prs)
        dep_prs_offsets = prs_offsets_data[di]
        dep_prs_offsets_count = PyArray_DIM(<ndarray> dep_prs_offsets, 0) + 1
        dep_prs_offsets_data = <long *> PyArray_DATA(<ndarray> dep_prs_offsets)
        dep_jira_col = PyArray_NewFromDescr(
            &PyArray_Type,
            <PyArray_Descr *> objdtype,
            1,
            &dep_prs_offsets_count,
            NULL,
            NULL,
            NPY_ARRAY_DEFAULT,
            NULL,
        )
        dep_jira_col_data = <PyObject **> PyArray_DATA(dep_jira_col)
        for j in range(dep_prs_offsets_count):
            dep_jira_col_data[j] = <PyObject *> empty
        (<PyObject *> empty).ob_refcnt += dep_prs_offsets_count
        jira_col_data[di] = <PyObject *> dep_jira_col
        Py_INCREF(dep_jira_col)
        for ri in range(dep_prs_offsets_count):
            if ri == 0:
                beg = 0
            else:
                beg = dep_prs_offsets_data[ri - 1]
            if ri == dep_prs_offsets_count - 1:
                end = dep_prs_count
            else:
                end = dep_prs_offsets_data[ri]
            for j in range(beg, end):
                dereference(pr_to_ix)[dep_prs_data[j]].push_back(PullRequestAddr(di, ri, ~0ull))
    capsule = PyCapsule_New(pr_to_ix, NULL, delete_ptr_in_capsule)
    return jira_col, capsule


def calc_pr_to_ix_releases(ndarray releases not None, pr_to_ix_obj not None) -> None:
    cdef:
        npy_intp di, ri, pri, releases_count = PyArray_DIM(releases, 0), dep_releases_count
        PyObject **releases_data = <PyObject **> PyArray_DATA(releases)
        ndarray prs_jira_arr
        npdtype objdtype = np.dtype(object)
        unordered_map[long, vector[PullRequestAddr]] *pr_to_ix = <unordered_map[long, vector[PullRequestAddr]] *> PyCapsule_GetPointer(pr_to_ix_obj, NULL)
        PyObject **node_ids_data
        long released_prs_count
        PyObject *node_ids
        PyObject *prs_jira_list
        long *release_node_ids_data

    for di in range(releases_count):
        dep_releases = <object> releases_data[di]
        if dep_releases.empty:
            continue
        dep_releases_count = len(dep_releases)
        Py_INCREF(objdtype)
        prs_jira_arr = PyArray_NewFromDescr(
            &PyArray_Type,
            <PyArray_Descr *> objdtype,
            1,
            &dep_releases_count,
            NULL,
            NULL,
            NPY_ARRAY_DEFAULT,
            NULL,
        )
        prs_jira_arr_data = <PyObject **> PyArray_DATA(prs_jira_arr)
        node_ids_data = <PyObject **> PyArray_DATA(dep_releases["prs_node_id"].values)
        for ri in range(dep_releases_count):
            node_ids = node_ids_data[ri]
            release_node_ids_data = <long *> PyArray_DATA(<ndarray> node_ids)
            released_prs_count = PyArray_DIM(<ndarray> node_ids, 0)
            prs_jira_arr_data[ri] = prs_jira_list = PyList_New(released_prs_count)
            for pri in range(released_prs_count):
                PyList_SET_ITEM(prs_jira_list, pri, PyList_New(0))
                dereference(pr_to_ix)[release_node_ids_data[pri]].push_back(PullRequestAddr(di, ri, pri))
        dep_releases["prs_jira"] = prs_jira_arr


def pr_to_ix_to_node_id_array(pr_to_ix_obj not None) -> ndarray:
    cdef:
        unordered_map[long, vector[PullRequestAddr]] *pr_to_ix = <unordered_map[long, vector[PullRequestAddr]] *> PyCapsule_GetPointer(pr_to_ix_obj, NULL)
        pair[long, vector[PullRequestAddr]] it
        ndarray arr
        long *arr_data
        long i = 0

    arr = np.empty(pr_to_ix.size(), dtype=int)
    arr_data = <long *> PyArray_DATA(arr)
    for it in dereference(pr_to_ix):
        arr_data[i] = it.first
        i += 1
    return arr


def apply_jira_rows(list rows not None, deployments not None, pr_to_ix_obj not None) -> dict:
    cdef:
        unordered_map[long, vector[PullRequestAddr]] *pr_to_ix = <unordered_map[long, vector[PullRequestAddr]] *> PyCapsule_GetPointer(pr_to_ix_obj, NULL)
        dict issues = {}
        unordered_map[long, unordered_map[long, unordered_map[long, vector[string_view]]]] release_keys
        pair[long, unordered_map[long, unordered_map[long, vector[string_view]]]] pair3
        pair[long, unordered_map[long, vector[string_view]]] pair2
        pair[long, vector[string_view]] pair1
        unordered_map[long, unordered_map[long, vector[string_view]]] overall_keys
        str key
        string_view key_view
        tuple tr
        ndarray overall_col
        ndarray aux
        list prs

    if rows:
        if isinstance(rows[0], asyncpg.Record):
            for r in rows:
                key = <object> ApgRecord_GET_ITEM(<PyObject *> r, 1)
                key_view = string_view(<char *> PyUnicode_DATA(key), PyUnicode_GET_LENGTH(key))
                issues[key] = PullRequestJIRAIssueItem(
                    id=key,
                    title=<object> ApgRecord_GET_ITEM(<PyObject *> r, 2),
                    epic=<object> ApgRecord_GET_ITEM(<PyObject *> r, 5),
                    labels=<object> ApgRecord_GET_ITEM(<PyObject *> r, 3),
                    type=<object> ApgRecord_GET_ITEM(<PyObject *> r, 4),
                )
                for addr in dereference(pr_to_ix)[<long><object> ApgRecord_GET_ITEM(<PyObject *> r, 0)]:
                    # we know tha the kind is always 1-byte
                    if addr.pri != ~0ull:
                        release_keys[addr.di][addr.ri][addr.pri].push_back(key_view)
                    else:
                        overall_keys[addr.di][addr.ri].push_back(key_view)
        else:
            for r in rows:
                tr = tuple(r)
                key = tr[1]
                key_view = string_view(<char *> PyUnicode_DATA(key), PyUnicode_GET_LENGTH(key))
                issues[key] = PullRequestJIRAIssueItem(
                    id=key, title=tr[2], epic=tr[5], labels=tr[3], type=tr[4],
                )
                for addr in dereference(pr_to_ix)[<long> tr[0]]:
                    # we know tha the kind is always 1-byte
                    if addr.pri != ~0ull:
                        release_keys[addr.di][addr.ri][addr.pri].push_back(key_view)
                    else:
                        overall_keys[addr.di][addr.ri].push_back(key_view)
    overall_col = deployments["jira"].values
    for pair2 in overall_keys:
        aux = overall_col[pair2.first]
        for pair1 in pair2.second:
            aux[pair1.first] = vec_to_arr(&pair1.second)
    releases_col = deployments["releases"].values
    for pair3 in release_keys:
        aux = releases_col[pair3.first]["prs_jira"].values
        for pair2 in pair3.second:
            prs = aux[pair2.first]
            for pair1 in pair2.second:
                prs[pair1.first] = vec_to_arr(&pair1.second)
    return issues


cdef inline ndarray vec_to_arr(vector[string_view] *vec):
    cdef:
        size_t max_size = 0, pos = 0
        npy_intp count = vec.size()
        ndarray arr
        npdtype dtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_STRING))
        char *arr_data

    std_sort(vec.begin(), vec.end())
    for key_view in dereference(vec):
        if key_view.size() > max_size:
            max_size = key_view.size()

    Py_INCREF(dtype)
    dtype.itemsize = max_size
    arr = PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> dtype,
        1,
        &count,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    arr_data = <char *> PyArray_DATA(arr)
    memset(arr_data, 0, count * max_size)
    for key_view in dereference(vec):
        memcpy(arr_data + pos * max_size, key_view.data(), key_view.size())
        pos += 1
    return arr
