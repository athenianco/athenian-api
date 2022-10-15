# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17
# distutils: libraries = mimalloc
# distutils: runtime_library_dirs = /usr/local/lib

from cpython cimport PyObject
from cython.operator cimport dereference as deref
from libc.string cimport memset
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
    Py_DECREF,
    Py_INCREF,
    PyObjectPtr,
    PyUnicode_DATA,
    PyUnicode_GET_LENGTH,
)
from athenian.api.native.mi_heap_stl_allocator cimport (
    mi_heap_allocator_from_capsule,
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
    alloc_capsule=None,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
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
        return (np.array([], dtype=object),) * 4

    cdef:
        optional[mi_heap_stl_allocator[char]] alloc
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
        npy_intp i, j, k, dep_pos, dep_len, sub_len, offset, repo_idx, local_pr_offsets_count
        long node_id
        size_t v, issues_count
        bint adjusted
        size_t empty_obj_arr_refs_count = 0, empty_u32_arr_refs_count = 0
        size_t extra_objdtype_refs_count, extra_u32dtype_refs_count
        char *strptr
        Py_ssize_t strlength
        PyObject *str_obj
        optional[mi_vector[string_view]] strvec
        ndarray result_repo, result_repo_offsets, result_pr, result_pr_offsets
        PyObject *dep_repo_issues
        PyObject *dep_repo_offsets
        PyObject *dep_pr_issues
        PyObject *dep_pr_offsets
        PyObject *empty_obj_arr
        PyObject *empty_u32_arr
        PyObject *repo_issues
        PyObject *aux
        optional[mi_vector[mi_vector[mi_unordered_map[string_view, PyObjectPtr]]]] resolved
        mi_vector[mi_unordered_map[string_view, PyObjectPtr]] *dep_resolved
        optional[mi_vector[mi_vector[mi_vector[PyObjectPtr]]]] resolved_by_pr
        string_view issue
        string_view *id_map_vec_data
        mi_unordered_map[string_view, PyObjectPtr] *issues = NULL
        mi_unordered_map[string_view, PyObjectPtr] *resolved_data = NULL
        mi_vector[mi_vector[PyObjectPtr]] *resolved_by_pr_data
        mi_vector[mi_vector[PyObjectPtr]] *resolved_by_pr_dep = NULL
        mi_vector[PyObjectPtr] *resolved_by_pr_dep_objs
        npdtype objdtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_OBJECT))
        npdtype u32dtype = PyArray_DescrNew(PyArray_DescrFromType(NPY_UINT))
        PyObject **result_repo_data
        PyObject **result_repo_offsets_data
        PyObject **result_pr_data
        PyObject **result_pr_offsets_data
        PyObject **dep_repo_issues_data
        npy_uint32 *dep_repo_offsets_data
        PyObject **dep_pr_issues_data
        npy_uint32 *dep_pr_offsets_data
        PyObject **resolved_by_pr_dep_issues
        pair[string_view, PyObjectPtr] origin_pair
        optional[mi_vector[npy_intp]] boilerplate_indexes
        optional[mi_vector[string_view]] boilerplate_bodies
        optional[mi_vector[PyObjectPtr]] boilerplate_ptrs

    if alloc_capsule is not None:
        alloc.emplace(deref(mi_heap_allocator_from_capsule(alloc_capsule)))
    else:
        alloc.emplace()
        deref(alloc).disable_free()

    with nogil:
        id_map.emplace(deref(alloc))
        origin_map.emplace(deref(alloc))
        for i in range(PyArray_DIM(<PyObject *> map_prs, 0)):
            str_obj = map_jira_data[i]
            node_id = map_prs_data[i]
            strptr = <char *> PyUnicode_DATA(str_obj)
            strlength = PyUnicode_GET_LENGTH(str_obj)
            deref(origin_map)[string_view(strptr, strlength)] = str_obj
            id_map_iter = deref(id_map).find(node_id)
            if id_map_iter == deref(id_map).end():
                strvec.emplace(deref(alloc))
                deref(strvec).emplace_back(strptr, strlength)
                deref(id_map).try_emplace(node_id, move(deref(strvec)))
            else:
                deref(id_map_iter).second.emplace_back(strptr, strlength)

        resolved.emplace(deref(alloc))
        deref(resolved).reserve(deps_count)
        for dep_pos in range(deps_count):
            deref(resolved).emplace_back(deref(alloc))
        resolved_by_pr.emplace(deref(alloc))
        deref(resolved_by_pr).reserve(deps_count)
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
            dep_resolved = &deref(resolved)[dep_pos]
            sub_len = local_pr_offsets_count + (dep_len > 0)
            dep_resolved.reserve(sub_len)
            for i in range(sub_len):
                dep_resolved.emplace_back(deref(alloc))
            resolved_data = dep_resolved.data()
            resolved_by_pr_dep = &deref(resolved_by_pr).emplace_back(deref(alloc))
            resolved_by_pr_dep.reserve(dep_len)

            for i in range(dep_len):
                resolved_by_pr_dep.emplace_back(deref(alloc))
                while repo_idx < local_pr_offsets_count and i >= local_pr_offsets_data[repo_idx]:
                    repo_idx += 1
                id_map_iter = deref(id_map).find(dep_pr_node_ids_data[i])
                if id_map_iter != deref(id_map).end():
                    issues = &resolved_data[repo_idx]
                    issues_count = deref(id_map_iter).second.size()
                    resolved_by_pr_dep_objs = &deref(resolved_by_pr_dep)[i]
                    resolved_by_pr_dep_objs.resize(issues_count)
                    resolved_by_pr_dep_issues = resolved_by_pr_dep_objs.data()
                    id_map_vec_data = deref(id_map_iter).second.data()
                    for v in range(issues_count):
                        issue = id_map_vec_data[v]
                        str_obj = deref(origin_map)[issue]
                        resolved_by_pr_dep_issues[v] = str_obj
                        deref(issues)[issue] = str_obj
        boilerplate_indexes.emplace(deref(alloc))
        boilerplate_bodies.emplace(deref(alloc))
        boilerplate_ptrs.emplace(deref(alloc))

    assert not invalid_dtype_pr_node_ids
    assert not invalid_dtype_pr_offsets

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
    result_repo_offsets = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &deps_count,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    result_repo_offsets_data = <PyObject **> PyArray_DATA(<PyObject *> result_repo_offsets)

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
    result_pr_offsets = <ndarray> PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &deps_count,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    result_pr_offsets_data = <PyObject **> PyArray_DATA(<PyObject *> result_pr_offsets)

    dep_len = 0
    empty_obj_arr = PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> objdtype,
        1,
        &dep_len,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )
    empty_u32_arr = PyArray_NewFromDescr(
        &PyArray_Type,
        <PyArray_Descr *> u32dtype,
        1,
        &dep_len,
        NULL,
        NULL,
        NPY_ARRAY_DEFAULT,
        NULL,
    )

    extra_u32dtype_refs_count = deps_count + 1
    extra_objdtype_refs_count = 5
    resolved_by_pr_data = deref(resolved_by_pr).data()
    
    for i in range(deps_count):
        dep_len = deref(resolved_by_pr)[i].size()

        adjusted = dep_len > 0
        if adjusted:
            dep_len -= 1
        result_pr_offsets_data[i] = dep_pr_offsets = PyArray_NewFromDescr(
            &PyArray_Type,
            <PyArray_Descr *> u32dtype,
            1,
            &dep_len,
            NULL,
            NULL,
            NPY_ARRAY_DEFAULT,
            NULL,
        )
        if adjusted:
            dep_len += 1
        dep_pr_offsets_data = <npy_uint32 *> PyArray_DATA(dep_pr_offsets)

        resolved_by_pr_dep = resolved_by_pr_data + i
        sub_len = 0
        for j in range(dep_len):
            sub_len += deref(resolved_by_pr_dep)[j].size()
        if sub_len == 0:
            result_pr_data[i] = empty_obj_arr
            empty_obj_arr_refs_count += 1
            if dep_len > 1:
                memset(dep_pr_offsets_data, 0, 4 * (dep_len - 1))
        else:
            result_pr_data[i] = dep_pr_issues = PyArray_NewFromDescr(
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
            dep_pr_issues_data = <PyObject **> PyArray_DATA(dep_pr_issues)

            offset = 0
            for j in range(dep_len):
                sub_len = deref(resolved_by_pr_dep)[j].size()
                if j > 0:
                    dep_pr_offsets_data[j - 1] = offset
                if sub_len == 0:
                    continue
                pr_issues = deref(resolved_by_pr_dep)[j].data()
                for k in range(sub_len):
                    str_obj = pr_issues[k]
                    dep_pr_issues_data[offset] = str_obj
                    Py_INCREF(str_obj)
                    offset += 1

        dep_len = deref(resolved)[i].size()
        sub_len = 0
        resolved_data = deref(resolved)[i].data()
        for j in range(dep_len):
            sub_len += resolved_data[j].size()
        if sub_len == 0:
            result_repo_data[i] = empty_obj_arr
            empty_obj_arr_refs_count += 1
            result_repo_offsets_data[i] = empty_u32_arr
            empty_u32_arr_refs_count += 1
        else:
            result_repo_data[i] = dep_repo_issues = PyArray_NewFromDescr(
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
            dep_repo_issues_data = <PyObject **> PyArray_DATA(dep_repo_issues)

            dep_len -= 1
            result_repo_offsets_data[i] = dep_repo_offsets = PyArray_NewFromDescr(
                &PyArray_Type,
                <PyArray_Descr *> u32dtype,
                1,
                &dep_len,
                NULL,
                NULL,
                NPY_ARRAY_DEFAULT,
                NULL,
            )
            dep_len += 1
            extra_u32dtype_refs_count += 1
            dep_repo_offsets_data = <npy_uint32 *> PyArray_DATA(dep_repo_offsets)
    
            offset = 0
            for j in range(dep_len):
                if j > 0:
                    dep_repo_offsets_data[j - 1] = offset
                sub_len = resolved_data[j].size()
                if sub_len == 0:
                    continue
                deref(boilerplate_indexes).clear()
                deref(boilerplate_bodies).clear()
                deref(boilerplate_ptrs).clear()
                k = 0
                for origin_pair in resolved_data[j]:
                    deref(boilerplate_indexes).push_back(k)
                    deref(boilerplate_bodies).emplace_back(origin_pair.first)
                    deref(boilerplate_ptrs).push_back(origin_pair.second)
                    k += 1
                argsort_bodies(deref(boilerplate_bodies), deref(boilerplate_indexes))
                for k in range(sub_len):
                    str_obj = deref(boilerplate_ptrs)[deref(boilerplate_indexes)[k]]
                    dep_repo_issues_data[offset] = str_obj
                    Py_INCREF(str_obj)
                    offset += 1

    (<PyObject *> objdtype).ob_refcnt += extra_objdtype_refs_count
    empty_obj_arr.ob_refcnt += empty_obj_arr_refs_count
    Py_DECREF(empty_obj_arr)
    (<PyObject *> u32dtype).ob_refcnt += extra_u32dtype_refs_count
    empty_u32_arr.ob_refcnt += empty_u32_arr_refs_count
    Py_DECREF(empty_u32_arr)
    return result_repo, result_repo_offsets, result_pr, result_pr_offsets
