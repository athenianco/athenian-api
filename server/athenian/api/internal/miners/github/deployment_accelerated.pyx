# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib

from cpython cimport PyObject
from cython.operator cimport dereference as deref
from libcpp.utility cimport move, pair
from numpy cimport (
    NPY_ARRAY_DEFAULT,
    NPY_LONG,
    NPY_OBJECT,
    NPY_UINT,
    PyArray_DescrFromType,
    PyArray_ISOBJECT,
    dtype as npdtype,
    import_array,
    ndarray,
    npy_intp,
    npy_uint32,
)

from athenian.api.native.cpython cimport (
    Py_INCREF,
    PyList_New,
    PyList_SET_ITEM,
    PyObjectPtr,
    PyUnicode_DATA,
    PyUnicode_GET_LENGTH,
)
from athenian.api.native.mi_heap_stl_allocator cimport (
    mi_heap_stl_allocator,
    mi_unordered_map,
    mi_vector,
)
from athenian.api.native.numpy cimport (
    PyArray_DATA,
    PyArray_Descr,
    PyArray_DESCR,
    PyArray_DescrNew,
    PyArray_DIM,
    PyArray_IS_C_CONTIGUOUS,
    PyArray_NDIM,
    PyArray_NewFromDescr,
    PyArray_Type,
)
from athenian.api.native.optional cimport optional
from athenian.api.native.string_view cimport string_view

import numpy as np

import_array()


cdef extern from "deployment_accelerated.h" nogil:
    # IDK how to painlessly define this in Cython
    void argsort_bodies[T, I](mi_vector[T] &bodies, mi_vector[I] &indexes)


def split_prs_to_jira_ids(
    ndarray pr_node_ids not None,
    ndarray pr_offsets not None,
    ndarray map_prs not None,
    ndarray map_jira not None,
) -> tuple[ndarray, ndarray]:
    cdef ndarray arr
    for arr in (pr_node_ids, pr_offsets, map_prs, map_jira):
        assert PyArray_NDIM(<PyObject *> arr) == 1
        assert PyArray_IS_C_CONTIGUOUS(<PyObject *> arr)
    assert PyArray_DESCR(<PyObject *> map_prs).type_num == NPY_LONG
    assert PyArray_ISOBJECT(pr_node_ids)
    assert PyArray_ISOBJECT(pr_offsets)
    assert PyArray_ISOBJECT(map_jira)

    cdef npy_intp deps_count = PyArray_DIM(<PyObject *> pr_node_ids, 0)
    if deps_count == 0:
        return np.array([], dtype=object), np.array([], dtype=object)

    cdef:
        mi_heap_stl_allocator[char] alloc = mi_heap_stl_allocator[char]()
        optional[mi_unordered_map[long, mi_vector[string_view]]] id_map
        mi_unordered_map[long, mi_vector[string_view]].iterator id_map_iter
        optional[mi_unordered_map[string_view, PyObjectPtr]] origin_map
        long *map_prs_data = <long *> PyArray_DATA(<PyObject *> map_prs)
        PyObject **map_jira_data = <PyObject **> PyArray_DATA(<PyObject *> map_jira)
        PyObject **pr_node_ids_data = <PyObject **> PyArray_DATA(<PyObject *> pr_node_ids)
        PyObject **pr_offsets_data = <PyObject **> PyArray_DATA(<PyObject *> pr_offsets)
        npy_uint32 *local_pr_offsets_data
        long *dep_pr_node_ids_data
        bint invalid_dtype_pr_offsets = False
        bint invalid_dtype_pr_node_ids = False
        npy_intp i, j, k, dep_pos, dep_len, sub_len, repo_idx, local_pr_offsets_count
        long node_id
        size_t v, issues_count
        size_t empty_arr_refs_count = 0, empty_list_refs_count = 0, extra_objdtype_refs_count = 0
        char *strptr
        Py_ssize_t strlength
        PyObject *str_obj
        PyObject *list_obj
        optional[mi_vector[string_view]] strvec
        ndarray result_repo, result_pr
        PyObject *dep_repo_issues
        PyObject *dep_pr_issues
        PyObject *empty_arr
        PyObject *empty_list
        PyObject *repo_issues
        PyObject *aux
        optional[mi_vector[mi_vector[mi_unordered_map[string_view, PyObjectPtr]]]] resolved
        optional[mi_vector[mi_vector[mi_vector[PyObjectPtr]]]] resolved_by_pr
        string_view issue
        string_view *id_map_vec_data
        mi_unordered_map[string_view, PyObjectPtr] *issues = NULL
        mi_unordered_map[string_view, PyObjectPtr] *resolved_data = NULL
        mi_vector[mi_vector[PyObjectPtr]] *resolved_by_pr_data
        mi_vector[mi_vector[PyObjectPtr]] *resolved_by_pr_dep = NULL
        PyObject **resolved_by_pr_dep_issues
        npdtype objdtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_OBJECT))
        PyObject **result_repo_data
        PyObject **result_pr_data
        PyObject **dep_repo_issues_data
        PyObject **repo_issues_data
        PyObject **dep_pr_issues_data
        PyObject **pr_issues
        pair[string_view, PyObjectPtr] origin_pair
        optional[mi_vector[npy_intp]] boilerplate_indexes
        optional[mi_vector[string_view]] boilerplate_bodies
        optional[mi_vector[PyObjectPtr]] boilerplate_ptrs

    with nogil:
        id_map.emplace(alloc)
        origin_map.emplace(alloc)
        for i in range(PyArray_DIM(<PyObject *> map_prs, 0)):
            str_obj = map_jira_data[i]
            node_id = map_prs_data[i]
            strptr = <char *> PyUnicode_DATA(str_obj)
            strlength = PyUnicode_GET_LENGTH(str_obj)
            deref(origin_map)[string_view(strptr, strlength)] = str_obj
            id_map_iter = deref(id_map).find(node_id)
            if id_map_iter == deref(id_map).end():
                strvec.emplace(alloc)
                deref(strvec).emplace_back(strptr, strlength)
                deref(id_map)[node_id] = move(deref(strvec))
            else:
                deref(id_map_iter).second.emplace_back(strptr, strlength)

        resolved.emplace(alloc)
        deref(resolved).resize(deps_count)
        resolved_by_pr.emplace(alloc)
        deref(resolved_by_pr).resize(deps_count)
        resolved_by_pr_data = deref(resolved_by_pr).data()
        for dep_pos in range(deps_count):
            aux = pr_node_ids_data[dep_pos]
            if (
                PyArray_DESCR(aux).type_num != NPY_LONG
                or not PyArray_IS_C_CONTIGUOUS(aux)
                or PyArray_NDIM(aux) != 1
            ):
                invalid_dtype_pr_node_ids = True
                break
            dep_pr_node_ids_data = <long *> PyArray_DATA(aux)
            dep_len = PyArray_DIM(aux, 0)
            repo_idx = 0
            aux = pr_offsets_data[dep_pos]
            if (
                PyArray_DESCR(aux).type_num != NPY_UINT
                or not PyArray_IS_C_CONTIGUOUS(aux)
                or PyArray_NDIM(aux) != 1
            ):
                invalid_dtype_pr_offsets = True
                break
            local_pr_offsets_data = <npy_uint32 *> PyArray_DATA(aux)
            local_pr_offsets_count = PyArray_DIM(aux, 0)
            deref(resolved)[dep_pos].resize(local_pr_offsets_count + (dep_len > 0))
            resolved_data = deref(resolved)[dep_pos].data()
            resolved_by_pr_dep = resolved_by_pr_data + dep_pos
            resolved_by_pr_dep.resize(dep_len)

            for i in range(dep_len):
                while repo_idx < local_pr_offsets_count and i >= local_pr_offsets_data[repo_idx]:
                    repo_idx += 1
                id_map_iter = deref(id_map).find(dep_pr_node_ids_data[i])
                if id_map_iter != deref(id_map).end():
                    issues = &resolved_data[repo_idx]
                    issues_count = deref(id_map_iter).second.size()
                    deref(resolved_by_pr_dep)[i].resize(issues_count)
                    resolved_by_pr_dep_issues = deref(resolved_by_pr_dep)[i].data()
                    id_map_vec_data = deref(id_map_iter).second.data()
                    for v in range(issues_count):
                        issue = id_map_vec_data[v]
                        str_obj = deref(origin_map)[issue]
                        resolved_by_pr_dep_issues[v] = str_obj
                        deref(issues)[issue] = str_obj
        boilerplate_indexes.emplace(alloc)
        boilerplate_bodies.emplace(alloc)
        boilerplate_ptrs.emplace(alloc)

    assert not invalid_dtype_pr_node_ids
    assert not invalid_dtype_pr_offsets
    (<PyObject *> objdtype).ob_refcnt += 2 * deps_count + 3
    result_repo = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &deps_count,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    result_repo_data = <PyObject **> PyArray_DATA(<PyObject *> result_repo)

    result_pr = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &deps_count,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    result_pr_data = <PyObject **> PyArray_DATA(<PyObject *> result_pr)

    dep_len = 0
    empty_arr = PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &dep_len,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    empty_list = PyList_New(0)
    for i in range(deps_count):
        dep_len = deref(resolved_by_pr)[i].size()
        result_pr_data[i] = dep_pr_issues = PyArray_NewFromDescr(
            &PyArray_Type,
            <PyArray_Descr *> objdtype,
            1,
            &dep_len,
            NULL,
            NULL,
            NPY_ARRAY_DEFAULT,
            NULL,
        )
        dep_pr_issues_data = <PyObject **> PyArray_DATA(dep_pr_issues)
        resolved_by_pr_dep = resolved_by_pr_data + i
        for j in range(dep_len):
            sub_len = deref(resolved_by_pr_dep)[j].size()
            if sub_len == 0:
                dep_pr_issues_data[j] = empty_list
                empty_list_refs_count += 1
                continue
            pr_issues = deref(resolved_by_pr_dep)[j].data()
            dep_pr_issues_data[j] = list_obj = PyList_New(sub_len)
            for k in range(sub_len):
                str_obj = pr_issues[k]
                PyList_SET_ITEM(list_obj, k, str_obj)
                Py_INCREF(str_obj)

        dep_len = deref(resolved)[i].size()
        dep_repo_issues = PyArray_NewFromDescr(
            &PyArray_Type,
            <PyArray_Descr *> objdtype,
            1,
            &dep_len,
            NULL,
            NULL,
            NPY_ARRAY_DEFAULT,
            NULL,
        )
        result_repo_data[i] = dep_repo_issues
        dep_repo_issues_data = <PyObject **> PyArray_DATA(dep_repo_issues)

        resolved_data = deref(resolved)[i].data()
        for j in range(dep_len):
            sub_len = resolved_data[j].size()
            if sub_len == 0:
                dep_repo_issues_data[j] = empty_arr
                empty_arr_refs_count += 1
                continue
            dep_repo_issues_data[j] = repo_issues = PyArray_NewFromDescr(
                &PyArray_Type,
                <PyArray_Descr *> objdtype,
                1,
                &sub_len,
                NULL,
                NULL,
                NPY_ARRAY_DEFAULT,
                NULL,
            )
            extra_objdtype_refs_count += 1
            repo_issues_data = <PyObject **> PyArray_DATA(repo_issues)
            deref(boilerplate_indexes).resize(0)
            deref(boilerplate_bodies).resize(0)
            deref(boilerplate_ptrs).resize(0)
            k = 0
            for origin_pair in resolved_data[j]:
                deref(boilerplate_indexes).push_back(k)
                deref(boilerplate_bodies).emplace_back(origin_pair.first)
                deref(boilerplate_ptrs).push_back(origin_pair.second)
                k += 1
            argsort_bodies(deref(boilerplate_bodies), deref(boilerplate_indexes))
            for v in range(deref(boilerplate_indexes).size()):
                str_obj = deref(boilerplate_ptrs)[deref(boilerplate_indexes)[v]]
                repo_issues_data[v] = str_obj
                Py_INCREF(str_obj)

    empty_arr.ob_refcnt += empty_arr_refs_count
    empty_list.ob_refcnt += empty_list_refs_count
    (<PyObject *> objdtype).ob_refcnt += extra_objdtype_refs_count
    return result_repo, result_pr
