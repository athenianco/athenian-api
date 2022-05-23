# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++

from cpython cimport PyBytes_FromString, PyObject, Py_INCREF
cimport cython
from posix.dlfcn cimport dlclose, dlopen, dlsym, RTLD_LAZY
from cython.operator cimport dereference, postincrement
from libc.stdint cimport int8_t, int32_t, uint32_t, int64_t, uint64_t
from libc.string cimport memcpy, memset, strncmp, strlen
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
from numpy cimport PyArray_BYTES

import asyncpg
import numpy as np
from typing import Any, List, Optional, Sequence, Tuple


cdef extern from "../../../asyncpg_recordobj.h":
    PyObject *ApgRecord_GET_ITEM(PyObject *, int) nogil
    void ApgRecord_SET_ITEM(object, int, object) nogil
    PyObject *ApgRecord_GET_DESC(PyObject *) nogil


cdef extern from "Python.h":
    # Cython defines `long PyLong_AsLong(object)` that increases the refcount
    long PyLong_AsLong(PyObject *) nogil
    # likewise, avoid refcounting
    void *PyUnicode_DATA(PyObject *) nogil
    object PyUnicode_FromString(const char *)
    # nogil!
    PyObject *PyList_GET_ITEM(PyObject *, Py_ssize_t) nogil
    PyObject *PyTuple_GET_ITEM(PyObject *, Py_ssize_t) nogil
    PyObject *Py_None


# ApgRecord_New is not exported from the Python interface of asyncpg
ctypedef object (*_ApgRecord_New)(type, PyObject *, Py_ssize_t)
cdef _ApgRecord_New ApgRecord_New
cdef void *_self = dlopen(NULL, RTLD_LAZY)
ApgRecord_New = <_ApgRecord_New>dlsym(_self, "ApgRecord_New")
dlclose(_self)


def searchsorted_inrange(a: np.ndarray, v: Any, side="left", sorter=None):
    r = np.searchsorted(a, np.atleast_1d(v), side=side, sorter=sorter)
    r[r == len(a)] = 0  # whatever index is fine
    return r


def extract_subdag(hashes: np.ndarray,
                   vertexes: np.ndarray,
                   edges: np.ndarray,
                   heads: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(vertexes) == len(hashes) + 1
    assert heads.dtype.char == "S"
    if len(hashes) == 0:
        return hashes, vertexes, edges
    if len(heads):
        heads = np.sort(heads)
        existing_heads = searchsorted_inrange(hashes, heads)
        existing_heads = existing_heads[hashes[existing_heads] == heads].astype(np.uint32)
    else:
        existing_heads = np.array([], dtype=np.uint32)
    left_vertexes_map = np.zeros_like(vertexes)
    left_vertexes = np.zeros_like(vertexes)
    left_edges = np.zeros_like(edges)
    cdef:
        int left_count
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] existing_heads_view = existing_heads
        uint32_t[:] left_vertexes_map_view = left_vertexes_map
        uint32_t[:] left_vertexes_view = left_vertexes
        uint32_t[:] left_edges_view = left_edges
    with nogil:
        left_count = _extract_subdag(
            vertexes_view, edges_view, existing_heads_view, False,
            left_vertexes_map_view, left_vertexes_view, left_edges_view)
    left_hashes = hashes[left_vertexes_map[:left_count]]
    left_vertexes = left_vertexes[:left_count + 1]
    left_edges = left_edges[:left_vertexes[left_count]]
    return left_hashes, left_vertexes, left_edges


@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint32_t _extract_subdag(const uint32_t[:] vertexes,
                              const uint32_t[:] edges,
                              const uint32_t[:] heads,
                              bool only_map,
                              uint32_t[:] left_vertexes_map,
                              uint32_t[:] left_vertexes,
                              uint32_t[:] left_edges) nogil:
    cdef:
        vector[uint32_t] boilerplate
        uint32_t i, j, head, peek, edge
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(len(heads)):
        head = heads[i]
        boilerplate.push_back(head)
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if left_vertexes_map[peek]:
                continue
            left_vertexes_map[peek] = 1
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not left_vertexes_map[edge]:
                    boilerplate.push_back(edge)
    if only_map:
        return 0
    # compress the vertex index mapping
    cdef uint32_t left_count = 0, edge_index, v
    for i in range(len(left_vertexes_map)):
        if left_vertexes_map[i]:
            left_vertexes_map[i] = left_count + 1  # disambiguate 0, become 1-based indexed
            left_count += 1
    # len(left_vertexes) == 0 means we don't care about the extracted edges
    if len(left_vertexes) > 0:
        # rebuild the edges
        edge_index = 0
        for i in range(len(vertexes) - 1):
            v = left_vertexes_map[i]
            if v:
                v -= 1  # become 0-based indexed again
                left_vertexes[v] = edge_index
                for j in range(vertexes[i], vertexes[i + 1]):
                    edge = edges[j]
                    left_edges[edge_index] = left_vertexes_map[edge] - 1
                    edge_index += 1
        left_vertexes[left_count] = edge_index
    # invert the vertex index mapping
    left_count = 0
    for i in range(len(left_vertexes_map)):
        if left_vertexes_map[i]:
            left_vertexes_map[left_count] = i
            left_count += 1
    return left_count


cdef struct Edge:
    uint32_t vertex
    uint32_t position


def join_dags(hashes: np.ndarray,
              vertexes: np.ndarray,
              edges: np.ndarray,
              new_edges: List[Tuple[str, Optional[str], int]],
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cdef:
        Py_ssize_t size
        int i, hpos, parent_index
        const char *parent_oid
        const char *child_oid
        PyObject *record
        PyObject *obj
        char *new_hashes_data
        unordered_map[string, int] new_hashes_map, hashes_map
        unordered_map[string, int].iterator it
        vector[Edge] *found_edges
        Edge edge
        bool exists
    size = len(new_edges)
    if size == 0:
        return hashes, vertexes, edges
    new_hashes_arr = np.empty(size * 2, dtype="S40")
    new_hashes_data = PyArray_BYTES(new_hashes_arr)
    hpos = 0
    if isinstance(new_edges[0], asyncpg.Record):
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                parent_oid = <const char*>PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0))
                memcpy(new_hashes_data + hpos, parent_oid, 40)
                hpos += 40
                child_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1))
                if strncmp(child_oid, "0" * 40, 40):
                    memcpy(new_hashes_data + hpos, child_oid, 40)
                    hpos += 40
    else:
        assert isinstance(new_edges[0], tuple)
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                parent_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0))
                memcpy(new_hashes_data + hpos, parent_oid, 40)
                hpos += 40
                child_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1))
                if strncmp(child_oid, "0" * 40, 40):
                    memcpy(new_hashes_data + hpos, child_oid, 40)
                    hpos += 40
    new_hashes_arr = new_hashes_arr[:hpos // 40]
    if len(hashes) > 0:
        new_hashes = np.unique(np.concatenate([new_hashes_arr, hashes]))
        found_matches = np.searchsorted(hashes, new_hashes)
        found_matches_in_range = found_matches.copy()
        found_matches_in_range[found_matches == len(hashes)] = 0
        distinct_mask = hashes[found_matches_in_range] != new_hashes
        found_matches = found_matches[distinct_mask]
        new_hashes = new_hashes[distinct_mask]
        result_hashes = np.insert(hashes, found_matches, new_hashes)
    else:
        new_hashes = np.unique(new_hashes_arr)
        found_matches = np.array([], dtype=int)
        result_hashes = new_hashes

    size = len(new_hashes)
    new_hashes_data = PyArray_BYTES(new_hashes)
    with nogil:
        for i in range(size):
            new_hashes_map[string(new_hashes_data + i * 40, 40)] = i
    if len(hashes) > 0:
        size = len(result_hashes)
        new_hashes_data = PyArray_BYTES(result_hashes)
        with nogil:
            for i in range(size):
                hashes_map[string(new_hashes_data + i * 40, 40)] = i
    else:
        hashes_map = new_hashes_map
    del new_hashes

    cdef:
        vector[vector[Edge]] new_edges_lists = vector[vector[Edge]](new_hashes_map.size())
        int new_edges_counter = 0
    size = len(new_edges)
    if isinstance(new_edges[0], asyncpg.Record):
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                child_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1))
                if not strncmp(child_oid, "0" * 40, 40):
                    # initial commit
                    continue
                parent_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0))
                it = new_hashes_map.find(parent_oid)
                if it != new_hashes_map.end():
                    parent_index = PyLong_AsLong(ApgRecord_GET_ITEM(record, 2))
                    found_edges = &new_edges_lists[dereference(it).second]
                    exists = False
                    for j in range(<int>found_edges.size()):
                        if <int>dereference(found_edges)[j].position == parent_index:
                            exists = True
                            break
                    if not exists:
                        # https://github.com/cython/cython/issues/1642
                        edge.vertex = hashes_map[child_oid]
                        edge.position = parent_index
                        found_edges.push_back(edge)
                        new_edges_counter += 1
    else:
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>new_edges, i)
                child_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1))
                if not strncmp(child_oid, "0" * 40, 40):
                    # initial commit
                    continue
                parent_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0))
                it = new_hashes_map.find(parent_oid)
                if it != new_hashes_map.end():
                    parent_index = PyLong_AsLong(PyTuple_GET_ITEM(record, 2))
                    found_edges = &new_edges_lists[dereference(it).second]
                    exists = False
                    for j in range(<int>found_edges.size()):
                        if <int>dereference(found_edges)[j].position == parent_index:
                            exists = True
                            break
                    if not exists:
                        # https://github.com/cython/cython/issues/1642
                        edge.vertex = hashes_map[child_oid]
                        edge.position = parent_index
                        found_edges.push_back(edge)
                        new_edges_counter += 1

    old_vertex_map = np.zeros(len(hashes), dtype=np.uint32)
    result_vertexes = np.zeros(len(result_hashes) + 1, dtype=np.uint32)
    result_edges = np.zeros(len(edges) + new_edges_counter, dtype=np.uint32)
    if len(hashes) > 0:
        found_matches += np.arange(len(found_matches))
    cdef:
        const int64_t[:] found_matches_view = found_matches
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        uint32_t[:] old_vertex_map_view = old_vertex_map
        uint32_t[:] result_vertexes_view = result_vertexes
        uint32_t[:] result_edges_view = result_edges
    with nogil:
        _recalculate_vertices_and_edges(
            found_matches_view, vertexes_view, edges_view, &new_edges_lists,
            old_vertex_map_view, result_vertexes_view, result_edges_view)
    return result_hashes, result_vertexes, result_edges


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _recalculate_vertices_and_edges(const int64_t[:] found_matches,
                                          const uint32_t[:] vertexes,
                                          const uint32_t[:] edges,
                                          const vector[vector[Edge]] *new_edges_lists,
                                          uint32_t[:] old_vertex_map,
                                          uint32_t[:] result_vertexes,
                                          uint32_t[:] result_edges) nogil:
    cdef:
        uint32_t j, left, offset = 0, pos = 0, list_size
        uint32_t old_edge_i = 0, new_edge_i = 0, size = len(result_vertexes) - 1, i
        const vector[Edge] *new_edge_list
        bint has_old = len(vertexes) > 1
        Edge edge
    if has_old:
        # populate old_vertex_map
        for i in range(size):
            if offset >= len(found_matches) or i < found_matches[offset]:
                old_vertex_map[i - offset] = i
            else:
                offset += 1
    # write the edges
    for i in range(size):
        result_vertexes[i] = pos
        if (new_edge_i >= len(found_matches) or i < found_matches[new_edge_i]) and has_old:
            # write old edge
            left = vertexes[old_edge_i]
            offset = vertexes[old_edge_i + 1] - left
            for j in range(offset):
                result_edges[pos + j] = old_vertex_map[edges[left + j]]
            pos += offset
            old_edge_i += 1
        else:
            new_edge_list = &dereference(new_edges_lists)[new_edge_i]
            list_size = new_edge_list.size()
            for j in range(list_size):
                edge = dereference(new_edge_list)[j]
                result_edges[pos + edge.position] = edge.vertex
            pos += list_size
            new_edge_i += 1
    result_vertexes[size] = pos


@cython.boundscheck(False)
def append_missing_heads(edges: List[Tuple[str, str, int]],
                         hashes: np.ndarray) -> None:
    cdef:
        unordered_set[string] hashes_set
        unordered_set[string].const_iterator it
        Py_ssize_t size
        int i
        PyObject *record
        PyObject *desc = Py_None
        object new_record, elem
        const char *hashes_data
    size = len(hashes)
    hashes_data = PyArray_BYTES(hashes)
    with nogil:
        for i in range(size):
            hashes_set.insert(string(hashes_data + 40 * i, 40))
    size = len(edges)
    if size > 0:
        if isinstance(edges[0], asyncpg.Record):
            with nogil:
                for i in range(size):
                    record = PyList_GET_ITEM(<PyObject *>edges, i)
                    hashes_set.erase(<const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0)))
                    hashes_set.erase(<const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1)))
                desc = ApgRecord_GET_DESC(PyList_GET_ITEM(<PyObject *>edges, 0))
        else:
            assert isinstance(edges[0], tuple)
            with nogil:
                for i in range(size):
                    record = PyList_GET_ITEM(<PyObject *>edges, i)
                    hashes_set.erase(<const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0)))
                    hashes_set.erase(<const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1)))
    it = hashes_set.const_begin()
    while it != hashes_set.const_end():
        if desc != Py_None:
            new_record = ApgRecord_New(asyncpg.Record, desc, 3)
            elem = PyUnicode_FromString(dereference(it).c_str())
            Py_INCREF(elem)
            ApgRecord_SET_ITEM(new_record, 0, elem)
            elem = PyUnicode_FromString("0" * 40)
            Py_INCREF(elem)
            ApgRecord_SET_ITEM(new_record, 1, elem)
            Py_INCREF(0)  # interned
            ApgRecord_SET_ITEM(new_record, 2, 0)
        else:
            new_record = (PyUnicode_FromString(dereference(it).c_str()), "0" * 40, 0)
        edges.append(new_record)
        postincrement(it)


ctypedef pair[int, const char *] RawEdge


@cython.boundscheck(False)
def validate_edges_integrity(edges: List[Tuple[str, str, int]]) -> bool:
    cdef:
        Py_ssize_t size
        const char *oid
        int indexes_sum, parent_index, i, j
        unordered_map[string, vector[RawEdge]] children_indexes
        unordered_map[string, vector[RawEdge]].iterator it
        PyObject *record
        PyObject *obj
        vector[RawEdge] *children_range
        bool result, exists

    size = len(edges)
    if size == 0:
        return True
    result = True
    if isinstance(edges[0], asyncpg.Record):
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>edges, i)
                obj = ApgRecord_GET_ITEM(record, 0)
                if obj == Py_None:
                    result = False
                    break
                oid = <const char*> PyUnicode_DATA(obj)
                if strlen(oid) != 40:
                    result = False
                    break
                parent_index = PyLong_AsLong(ApgRecord_GET_ITEM(record, 2))
                children_range = &children_indexes[oid]

                obj = ApgRecord_GET_ITEM(record, 1)
                if obj == Py_None:
                    result = False
                    break
                oid = <const char *> PyUnicode_DATA(obj)
                if strlen(oid) != 40:
                    result = False
                    break

                exists = False
                for j in range(<int>children_range.size()):
                    if dereference(children_range)[j].first == parent_index:
                        exists = True
                        if strncmp(dereference(children_range)[j].second, oid, 40):
                            result = False
                        break
                if not exists:
                    children_range.push_back(RawEdge(parent_index, oid))
        if not result:
            return False
    else:
        assert isinstance(edges[0], tuple)
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>edges, i)
                obj = PyTuple_GET_ITEM(record, 0)
                if obj == Py_None:
                    result = False
                    break
                oid = <const char *> PyUnicode_DATA(obj)
                if strlen(oid) != 40:
                    result = False
                    break
                parent_index = PyLong_AsLong(PyTuple_GET_ITEM(record, 2))
                children_range = &children_indexes[oid]

                obj = PyTuple_GET_ITEM(record, 1)
                if obj == Py_None:
                    result = False
                    break
                oid = <const char *> PyUnicode_DATA(obj)
                if strlen(oid) != 40:
                    result = False
                    break

                exists = False
                for j in range(<int>children_range.size()):
                    if dereference(children_range)[j].first == parent_index:
                        exists = True
                        if strncmp(dereference(children_range)[j].second, oid, 40):
                            result = False
                        break
                if not exists:
                    children_range.push_back(RawEdge(parent_index, oid))
        if not result:
            return False
    with nogil:
        it = children_indexes.begin()
        while it != children_indexes.end():
            children_range = &dereference(it).second
            size = children_range.size()
            indexes_sum = 0
            for i in range(size):
                indexes_sum += dereference(children_range)[i].first
            indexes_sum -= ((size - 1) * size) // 2
            if indexes_sum != 0:
                result = False
                break
            postincrement(it)
    return result


@cython.boundscheck(False)
def find_orphans(edges: List[Tuple[str, str, int]],
                 attach_to: np.ndarray) -> np.ndarray:
    cdef:
        Py_ssize_t size
        const char *child_oid
        const char *parent_oid
        int i
        unordered_set[string] parents
        PyObject *record
        PyObject *obj
        vector[const char *] leaves

    size = len(edges)
    if size == 0:
        return np.array([], dtype="S40")
    if isinstance(edges[0], asyncpg.Record):
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *>edges, i)
                parent_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 0))
                child_oid = <const char *> PyUnicode_DATA(ApgRecord_GET_ITEM(record, 1))
                if strncmp(child_oid, "0" * 40, 40):
                    parents.insert(parent_oid)
                    leaves.push_back(child_oid)
                else:
                    leaves.push_back(parent_oid)
    else:
        with nogil:
            for i in range(size):
                record = PyList_GET_ITEM(<PyObject *> edges, i)
                parent_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 0))
                child_oid = <const char *> PyUnicode_DATA(PyTuple_GET_ITEM(record, 1))
                if strncmp(child_oid, "0" * 40, 40):
                    parents.insert(parent_oid)
                    leaves.push_back(child_oid)
                else:
                    leaves.push_back(parent_oid)

    with nogil:
        size = 0
        for i in range(<int>leaves.size()):
            if parents.find(leaves[i]) == parents.end():
                leaves[size] = leaves[i]
                size += 1
    leaves_arr = np.empty(size, dtype="S40")
    for i in range(size):
        leaves_arr[i] = PyBytes_FromString(leaves[i])
    leaves_arr = np.unique(leaves_arr)
    return leaves_arr[attach_to[searchsorted_inrange(attach_to, leaves_arr)] != leaves_arr]


def mark_dag_access(hashes: np.ndarray,
                    vertexes: np.ndarray,
                    edges: np.ndarray,
                    heads: np.ndarray,
                    heads_order_is_significant: bool) -> np.ndarray:
    """
    Find the earliest parent from `heads` for each commit in `hashes`.

    If `heads_order_is_significant`, the `heads` must be sorted by commit timestamp in descending \
    order. Thus `heads[0]` should be the latest commit.

    If not `heads_order_is_significant`, we sort `heads` topologically, but the earlier commits \
    have the priority over the later commits, if they are the same.

    :return: Indexes in `heads`, *not vertexes*.
    """
    if len(hashes) == 0:
        return np.array([], dtype=np.int64)
    size = len(heads)
    access = np.full(len(vertexes), size, np.int32)
    if size == 0:
        return access[:-1]
    assert heads.dtype.char == "S"
    # we cannot sort heads because the order is important - we return the original indexes
    existing_heads = searchsorted_inrange(hashes, heads)
    matched = hashes[existing_heads] == heads
    head_vertexes = np.full(size + 1, len(vertexes), np.uint32)
    head_vertexes[:-1][matched] = existing_heads[matched]
    heads = head_vertexes
    del head_vertexes
    if not matched.any():
        return access[:-1]
    order = np.full(size, size, np.int32)
    cdef:
        bool heads_order_is_significant_native = heads_order_is_significant
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_without_tail = heads[:-1]
        const uint32_t[:] heads_view = heads
        int32_t[:] order_view = order
        int32_t[:] access_view = access
    with nogil:
        _toposort(vertexes_view, edges_view, heads_without_tail,
                  heads_order_is_significant_native, order_view)
        _mark_dag_access(vertexes_view, edges_view, heads_view, order_view, access_view)
    return access[:-1]  # len(vertexes) = len(hashes) + 1


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _toposort(const uint32_t[:] vertexes,
                    const uint32_t[:] edges,
                    const uint32_t[:] heads,
                    bool heads_order_is_significant,
                    int32_t[:] order,
                    ) nogil:
    """Topological sort of `heads`. The order is reversed!"""
    cdef:
        vector[uint32_t] boilerplate
        vector[int32_t] visited = vector[int32_t](len(vertexes))
        uint32_t j, head, peek, edge, missing = len(vertexes)
        int64_t i, order_pos = 0
        int32_t status, size = len(heads), vv = size + 1
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(len(heads)):
        head = heads[i]
        if head == missing:
            continue
        visited[head] = i - size  # fused head marks in `visited`  array
    # heads, unvisited -> -len(heads), ..., -2, -1
    # normal vertexes, unvisited -> 0
    # heads, visited -> 1, 2, ..., len(heads)
    # normal vertexes, visited -> len(heads) + 1
    for i in range(len(heads) - 1, -1, -1):  # reverse order is release-friendly
        # we start from the earliest head and end at the latest
        head = heads[i]
        if head == missing:
            continue
        boilerplate.push_back(head)
        while not boilerplate.empty():
            peek = boilerplate.back()
            status = visited[peek]
            if status > 0:
                boilerplate.pop_back()
                if status < vv:
                    status -= 1  # index of the head
                    if status >= i or not heads_order_is_significant:
                        # status >= i means it comes after => appeared earlier
                        # we must ignore future releases standing in front
                        order[order_pos] = status
                        order_pos += 1
                    visited[peek] = vv
                continue
            visited[peek] += vv
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if visited[edge] <= 0:
                    boilerplate.push_back(edge)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _mark_dag_access(const uint32_t[:] vertexes,
                           const uint32_t[:] edges,
                           const uint32_t[:] heads,
                           const int32_t[:] order,
                           int32_t[:] access) nogil:
    cdef:
        vector[uint32_t] boilerplate
        uint32_t j, head, peek, edge, missing = len(vertexes)
        int64_t i, original_index
        int32_t size = len(order)
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(size):
        head = heads[order[i]]
        if head == missing:
            continue
        original_index = order[i]
        boilerplate.push_back(head)
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if access[peek] < size:
                continue
            access[peek] = original_index
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if access[edge] == size:
                    boilerplate.push_back(edge)


def mark_dag_parents(hashes: np.ndarray,
                     vertexes: np.ndarray,
                     edges: np.ndarray,
                     heads: np.ndarray,
                     timestamps: np.ndarray,
                     ownership: np.ndarray,
                     slay_hydra: bool = True) -> np.ndarray:
    """
    :param slay_hydra: When there is a head that reaches several roots and not all of them have \
                       parents, clear the parents so that the regular check len(parents) == 0 \
                       works.
    """
    result = np.empty(len(heads), dtype=object)
    if len(hashes) == 0:
        result.fill([])
        return result
    if len(heads) == 0:
        return result
    assert heads.dtype.char == "S"
    # we cannot sort heads because the order is important
    found_heads = searchsorted_inrange(hashes, heads)
    found_heads[hashes[found_heads] != heads] = len(vertexes)
    heads = found_heads.astype(np.uint32)
    timestamps = timestamps.view(np.uint64)
    ownership = ownership.astype(np.int32, copy=False)
    cdef:
        vector[vector[uint32_t]] parents = vector[vector[uint32_t]](len(heads))
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_view = heads
        const uint64_t[:] timestamps_view = timestamps
        const int32_t[:] ownership_view = ownership
        bool slay_hydra_native = slay_hydra
    with nogil:
        full_size = _mark_dag_parents(
            vertexes_view, edges_view, heads_view, timestamps_view, ownership_view,
            slay_hydra_native, &parents)
    concat_parents = np.zeros(full_size, dtype=np.uint32)
    split_points = np.zeros(len(parents), dtype=np.int64)
    cdef:
        uint32_t[:] concat_parents_view = concat_parents
        int64_t[:] split_points_view = split_points
    with nogil:
        _copy_parents_to_array(&parents, concat_parents_view, split_points_view)
    result = np.empty(len(parents), dtype=object)
    result[:] = np.split(concat_parents, split_points[:-1])
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _copy_parents_to_array(const vector[vector[uint32_t]] *parents,
                                 uint32_t[:] output,
                                 int64_t[:] splits) nogil:
    cdef int64_t i, offset = 0
    for i in range(<int64_t>parents.size()):
        vec = dereference(parents)[i]  # (*parents)[i]
        memcpy(&output[offset], vec.data(), 4 * vec.size())
        offset += vec.size()
        splits[i] = offset


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _mark_dag_parents(const uint32_t[:] vertexes,
                           const uint32_t[:] edges,
                           const uint32_t[:] heads,
                           const uint64_t[:] timestamps,
                           const int32_t[:] ownership,
                           bool slay_hydra,
                           vector[vector[uint32_t]] *parents) nogil:
    cdef:
        uint32_t not_found = len(vertexes), head, peek, edge, peak_owner, parent, beg, end
        uint64_t timestamp, head_timestamp
        int64_t i, j, p, sum_len = 0
        bool reached_root
        vector[char] visited = vector[char](len(vertexes) - 1)
        vector[uint32_t] boilerplate
        vector[uint32_t] *my_parents
    for i in range(len(heads)):
        head = heads[i]
        if head == not_found:
            continue
        head_timestamp = timestamps[i]
        my_parents = &dereference(parents)[i]
        reached_root = False
        memset(visited.data(), 0, visited.size())
        boilerplate.push_back(head)
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if visited[peek]:
                continue
            visited[peek] = 1
            peak_owner = ownership[peek]
            if peak_owner != i:
                timestamp = timestamps[peak_owner]
                if timestamp < head_timestamp:
                    # we don't expect many parents so scan linear
                    for p in range(<int64_t> my_parents.size()):
                        parent = dereference(my_parents)[p]
                        if parent == peak_owner:
                            break
                        if timestamp > timestamps[parent]:
                            sum_len += 1
                            my_parents.insert(my_parents.begin() + p, peak_owner)
                            break
                    else:
                        sum_len += 1
                        my_parents.push_back(peak_owner)
                    continue
            beg, end = vertexes[peek], vertexes[peek + 1]
            if beg == end:
                reached_root = True
            for j in range(beg, end):
                edge = edges[j]
                if not visited[edge]:
                    boilerplate.push_back(edge)
        if reached_root and slay_hydra:
            # case when there are several different histories merged together
            my_parents.clear()
    return sum_len


def extract_first_parents(hashes: np.ndarray,
                          vertexes: np.ndarray,
                          edges: np.ndarray,
                          heads: np.ndarray,
                          max_depth: int = 0) -> np.ndarray:
    assert heads.dtype.char == "S"
    heads = np.sort(heads)
    if len(hashes):
        found_heads = searchsorted_inrange(hashes, heads)
        heads = found_heads[hashes[found_heads] == heads].astype(np.uint32)
    else:
        heads = np.array([], dtype=np.uint32)
    first_parents = np.zeros_like(hashes, dtype=np.bool_)
    cdef:
        int max_depth_native = max_depth
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_view = heads
        char[:] first_parents_view = first_parents
    with nogil:
        _extract_first_parents(vertexes_view, edges_view, heads_view, max_depth_native,
                               first_parents_view)
    return hashes[first_parents]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_first_parents(const uint32_t[:] vertexes,
                                 const uint32_t[:] edges,
                                 const uint32_t[:] heads,
                                 int max_depth,
                                 char[:] first_parents) nogil:
    cdef:
        uint32_t head
        int i, depth
    for i in range(len(heads)):
        head = heads[i]
        depth = 0
        while not first_parents[head]:
            first_parents[head] = 1
            depth += 1
            if max_depth > 0 and depth >= max_depth:
                break
            if vertexes[head + 1] > vertexes[head]:
                head = edges[vertexes[head]]
            else:
                break


def partition_dag(hashes: np.ndarray,
                  vertexes: np.ndarray,
                  edges: np.ndarray,
                  seeds: np.ndarray) -> np.ndarray:
    seeds = np.sort(seeds)
    if len(hashes):
        found_seeds = searchsorted_inrange(hashes, seeds)
        seeds = found_seeds[hashes[found_seeds] == seeds].astype(np.uint32)
    else:
        seeds = np.array([], dtype=np.uint32)
    borders = np.zeros_like(hashes, dtype=np.bool_)
    cdef:
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] seeds_view = seeds
        char[:] borders_view = borders
    with nogil:
        _partition_dag(vertexes_view, edges_view, seeds_view, borders_view)
    return hashes[borders]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _partition_dag(const uint32_t[:] vertexes,
                         const uint32_t[:] edges,
                         const uint32_t[:] heads,
                         char[:] borders) nogil:
    cdef:
        vector[uint32_t] boilerplate
        vector[char] visited = vector[char](len(vertexes) - 1)
        int i, v
        uint32_t head, edge, peek, j
    for i in range(len(heads)):
        head = heads[i]
        # traverse the DAG from top to bottom, marking the visited nodes
        memset(visited.data(), 0, visited.size())
        boilerplate.push_back(head)
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if visited[peek]:
                continue
            visited[peek] = 1
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not visited[edge]:
                    boilerplate.push_back(edge)
        # include every visited node with back edges from non-visited nodes in the partition_dag
        for v in range(len(vertexes) - 1):
            if visited[v]:
                continue
            for j in range(vertexes[v], vertexes[v + 1]):
                edge = edges[j]
                if visited[edge]:
                    borders[edge] = 1


def extract_pr_commits(hashes: np.ndarray,
                       vertexes: np.ndarray,
                       edges: np.ndarray,
                       pr_merges: np.ndarray) -> Sequence[np.ndarray]:
    if len(hashes) == 0:
        return [np.array([], dtype="S40") for _ in pr_merges]
    order = np.argsort(pr_merges)
    pr_merges = pr_merges[order]
    found_pr_merges = searchsorted_inrange(hashes, pr_merges)
    found_pr_merges[hashes[found_pr_merges] != pr_merges] = len(vertexes)
    pr_merges = found_pr_merges.astype(np.uint32)[np.argsort(order)]
    left_vertexes_map = np.zeros(len(hashes), dtype=np.int8)
    cdef:
        vector[vector[uint32_t]] pr_commits = vector[vector[uint32_t]](len(pr_merges))
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] pr_merges_view = pr_merges
        int8_t[:] left_vertexes_map_view = left_vertexes_map
    with nogil:
        _extract_pr_commits(vertexes_view, edges_view, pr_merges_view, left_vertexes_map_view,
                            &pr_commits)
    result = np.zeros(len(pr_commits), dtype=object)
    for i, pr_vertexes in enumerate(pr_commits):
        result[i] = hashes[list(pr_vertexes)]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_pr_commits(const uint32_t[:] vertexes,
                              const uint32_t[:] edges,
                              const uint32_t[:] pr_merges,
                              int8_t[:] left_vertexes_map,
                              vector[vector[uint32_t]] *pr_commits) nogil:
    cdef:
        int i
        uint32_t first, last, v, j, edge, peek
        uint32_t oob = len(vertexes)
        vector[uint32_t] *my_pr_commits
        vector[uint32_t] boilerplate
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(len(pr_merges)):
        v = pr_merges[i]
        if v == oob:
            continue
        first = vertexes[v]
        last = vertexes[v + 1]
        if last - first != 2:  # we don't support octopus
            continue

        # extract the full sub-DAG of the main branch
        left_vertexes_map[:] = 0
        boilerplate.clear()
        boilerplate.push_back(edges[first])
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if left_vertexes_map[peek]:
                continue
            left_vertexes_map[peek] = 1
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not left_vertexes_map[edge]:
                    boilerplate.push_back(edge)

        # traverse the DAG starting from the side edge, stop on any vertex in the main sub-DAG
        my_pr_commits = &dereference(pr_commits)[i]
        my_pr_commits.push_back(v)  # include the merge commit in the PR
        boilerplate.push_back(edges[last - 1])
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if left_vertexes_map[peek]:
                continue
            left_vertexes_map[peek] = 1
            my_pr_commits.push_back(peek)
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not left_vertexes_map[edge]:
                    boilerplate.push_back(edge)


def extract_independent_ownership(hashes: np.ndarray,
                                  vertexes: np.ndarray,
                                  edges: np.ndarray,
                                  heads: np.ndarray,
                                  stops: np.ndarray) -> np.ndarray:
    if len(hashes) == 0 or len(heads) == 0:
        result = np.empty(len(heads), dtype=object)
        result.fill(np.array([], dtype="S40"))
        return result
    assert heads.dtype.char == "S"
    assert len(heads) == len(stops)
    # we cannot sort heads because the order is important
    found_heads = searchsorted_inrange(hashes, heads)
    found_heads[hashes[found_heads] != heads] = len(vertexes)
    heads = found_heads.astype(np.uint32)
    del found_heads
    all_stops = np.concatenate(stops)
    found_stops = searchsorted_inrange(hashes, all_stops)
    found_stops[hashes[found_stops] != all_stops] = len(vertexes)
    splits = np.zeros(len(stops) + 1, dtype=np.int64)
    np.cumsum([len(arr) for arr in stops], out=splits[1:])
    stops = found_stops.astype(np.uint32)
    left_vertexes_map = np.zeros_like(vertexes)
    left_vertexes = left_edges = np.array([], dtype=np.uint32)
    single_slot = np.zeros(1, dtype=np.uint32)
    cdef:
        vector[vector[uint32_t]] found_commits = vector[vector[uint32_t]](len(heads))
        const uint32_t[:] vertexes_view = vertexes
        const uint32_t[:] edges_view = edges
        const uint32_t[:] heads_view = heads
        const uint32_t[:] stops_view = stops
        const int64_t[:] splits_view = splits
        uint32_t[:] single_slot_view = single_slot
        uint32_t[:] left_vertexes_map_view = left_vertexes_map
        uint32_t[:] left_vertexes_view = left_vertexes
        uint32_t[:] left_edges_view = left_edges
    with nogil:
        _extract_independent_ownership(
            vertexes_view, edges_view, heads_view, stops_view, splits_view,
            single_slot_view, left_vertexes_map_view, left_vertexes_view, left_edges_view,
            &found_commits)
    result = np.zeros(len(found_commits), dtype=object)
    for i, own_vertexes in enumerate(found_commits):
        result[i] = hashes[list(own_vertexes)]
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_independent_ownership(const uint32_t[:] vertexes,
                                         const uint32_t[:] edges,
                                         const uint32_t[:] heads,
                                         const uint32_t[:] stops,
                                         const int64_t[:] splits,
                                         uint32_t[:] single_slot,
                                         uint32_t[:] left_vertexes_map,
                                         uint32_t[:] left_vertexes,
                                         uint32_t[:] left_edges,
                                         vector[vector[uint32_t]] *result) nogil:
    cdef:
        int64_t i, p
        uint32_t j, head, parent, count, peek, edge
        uint32_t oob = len(vertexes)
        vector[uint32_t] *head_result
        vector[uint32_t] boilerplate
        bool has_parent
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    for i in range(len(heads)):
        head = heads[i]
        if head == oob:
            continue
        head_result = &dereference(result)[i]
        left_vertexes_map[:] = 0
        has_parent = False
        for p in range(splits[i], splits[i + 1]):
            parent = stops[p]
            if parent == oob:
                continue
            has_parent = True
            single_slot[0] = parent
            _extract_subdag(
                vertexes, edges, single_slot, True, left_vertexes_map, left_vertexes, left_edges)
        if not has_parent:
            single_slot[0] = head
            count = _extract_subdag(
                vertexes, edges, single_slot, False, left_vertexes_map, left_vertexes, left_edges)
            head_result.reserve(count)
            for j in range(count):
                head_result.push_back(left_vertexes_map[j])
            continue
        boilerplate.push_back(head)
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if left_vertexes_map[peek]:
                continue
            left_vertexes_map[peek] = 1
            head_result.push_back(peek)
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if not left_vertexes_map[edge]:
                    boilerplate.push_back(edge)
