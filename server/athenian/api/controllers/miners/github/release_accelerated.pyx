# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.undeclared=True, warn.maybe_uninitialized=True
# distutils: language = c++

cimport cython
from libc.stdint cimport uint32_t, int64_t
from libcpp.vector cimport vector
import numpy as np
from typing import List, Tuple


def searchsorted_inrange(a, v, side="left", sorter=None):
    r = np.searchsorted(a, v, side=side, sorter=sorter)
    r[r == len(a)] = 0  # whatever index is fine
    return r


def extract_subdag(hashes: np.ndarray,
                   vertexes: np.ndarray,
                   edges: np.ndarray,
                   heads: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert len(vertexes) == len(hashes) + 1
    if len(hashes) == 0:
        return hashes, vertexes, edges
    heads = np.sort(heads)
    existing_heads = searchsorted_inrange(hashes, heads)
    existing_heads = existing_heads[hashes[existing_heads] == heads].astype(np.uint32)
    left_vertexes_map = np.zeros_like(vertexes, shape=len(hashes))
    left_vertexes = np.zeros_like(vertexes)
    left_edges = np.zeros_like(edges)
    left_count = _extract_subdag(
        vertexes, edges, existing_heads, left_vertexes_map, left_vertexes, left_edges)
    left_hashes = hashes[left_vertexes_map[:left_count]]
    left_vertexes = left_vertexes[:left_count + 1]
    left_edges = left_edges[:left_vertexes[left_count]]
    return left_hashes, left_vertexes, left_edges


@cython.boundscheck(False)
@cython.wraparound(False)
cdef uint32_t _extract_subdag(const uint32_t[:] vertexes,
                              const uint32_t[:] edges,
                              const uint32_t[:] heads,
                              uint32_t[:] left_vertexes_map,
                              uint32_t[:] left_vertexes,
                              uint32_t[:] left_edges) nogil:
    cdef vector[uint32_t] boilerplate
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    cdef uint32_t i, j, head, peek, edge
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
    # write the vertex index mapping
    cdef uint32_t left_count = 0
    for i in range(len(left_vertexes_map)):
        if left_vertexes_map[i]:
            left_vertexes_map[i] = left_count + 1  # disambiguate 0, become 1-based indexed
            left_count += 1
    # rebuild the edges
    cdef uint32_t edge_index = 0, v
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


def join_dags(hashes: np.ndarray,
              vertexes: np.ndarray,
              edges: np.ndarray,
              new_edges: List[Tuple[str, str]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    new_hashes = {k for k, v in new_edges}.union(v for k, v in new_edges)
    new_hashes.discard("")  # initial commit virtual parents
    new_hashes = np.sort(np.fromiter(new_hashes, count=len(new_hashes), dtype="U40"))
    if len(hashes) > 0:
        found_matches = np.searchsorted(hashes, new_hashes)
        found_matches_in_range = found_matches.copy()
        found_matches_in_range[found_matches == len(hashes)] = 0
        distinct_mask = hashes[found_matches_in_range] != new_hashes
        found_matches = found_matches[distinct_mask]
        new_hashes = new_hashes[distinct_mask]
        result_hashes = np.insert(hashes, found_matches, new_hashes)
    else:
        found_matches = np.array([], dtype=int)
        result_hashes = new_hashes
    new_hashes_map = {k: i for i, k in enumerate(new_hashes)}
    if len(hashes) > 0:
        hashes_map = {k: i for i, k in enumerate(result_hashes)}
    else:
        hashes_map = new_hashes_map
    del new_hashes
    cdef vector[vector[uint32_t]] new_edges_lists = vector[vector[uint32_t]](len(new_hashes_map))
    new_edges_counter = 0
    for k, v in new_edges:
        if not v:
            # initial commit
            continue
        i = new_hashes_map.get(k, None)
        if i is not None:
            new_edges_lists[i].push_back(hashes_map[v])
            new_edges_counter += 1
    old_vertex_map = np.zeros(len(hashes), dtype=np.uint32)
    result_vertexes = np.zeros(len(result_hashes) + 1, dtype=np.uint32)
    result_edges = np.zeros(len(edges) + new_edges_counter, dtype=np.uint32)
    if len(hashes) > 0:
        found_matches += np.arange(len(found_matches))
    _recalculate_vertices_and_edges(
        found_matches, vertexes, edges, new_edges_lists, old_vertex_map,
        result_vertexes, result_edges)
    return result_hashes, result_vertexes, result_edges


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _recalculate_vertices_and_edges(const int64_t[:] found_matches,
                                          const uint32_t[:] vertexes,
                                          const uint32_t[:] edges,
                                          vector[vector[uint32_t]] new_edges_lists,
                                          uint32_t[:] old_vertex_map,
                                          uint32_t[:] result_vertexes,
                                          uint32_t[:] result_edges) nogil:
    cdef uint32_t j, left, offset = 0, pos = 0, list_size
    cdef uint32_t old_edge_i = 0, new_edge_i = 0, size = len(result_vertexes) - 1, i
    cdef const vector[uint32_t] *new_edge_list
    cdef bint has_old = len(vertexes) > 1
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
            new_edge_list = &new_edges_lists[new_edge_i]
            list_size = new_edge_list.size()
            for j in range(list_size):
                result_edges[pos + j] = new_edge_list.at(j)
            pos += list_size
            new_edge_i += 1
    result_vertexes[size] = pos


def mark_dag_access(hashes: np.ndarray,
                    vertexes: np.ndarray,
                    edges: np.ndarray,
                    heads: np.ndarray):
    if len(hashes) == 0:
        return np.array([], dtype=int)
    # we cannot sort heads because the order is important
    existing_heads = searchsorted_inrange(hashes, heads)
    missing_flag = len(heads)  # mark unmatched commits with this value
    flags = np.where(hashes[existing_heads] == heads)[0]
    heads = existing_heads[flags].astype(np.uint32)
    marked = np.full(len(hashes), missing_flag, int)
    _mark_dag_access(vertexes, edges, heads, flags, missing_flag, marked)
    return marked


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _mark_dag_access(const uint32_t[:] vertexes,
                           const uint32_t[:] edges,
                           const uint32_t[:] heads,
                           const int64_t[:] flags,
                           int64_t missing_flag,
                           int64_t[:] marked) nogil:
    cdef vector[uint32_t] boilerplate
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    cdef uint32_t j, head, peek, edge
    cdef int64_t i, flag
    for i in range(len(heads) - 1, -1, -1):
        head = heads[i]
        flag = flags[i]
        boilerplate.push_back(head)
        while not boilerplate.empty():
            peek = boilerplate.back()
            boilerplate.pop_back()
            if marked[peek] != missing_flag:
                continue
            marked[peek] = flag
            for j in range(vertexes[peek], vertexes[peek + 1]):
                edge = edges[j]
                if marked[edge] == missing_flag:
                    boilerplate.push_back(edge)
