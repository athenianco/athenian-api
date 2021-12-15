# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++

cimport cython
from cython.operator cimport dereference
from libc.stdint cimport int8_t, int32_t, uint32_t, int64_t, uint64_t
from libc.string cimport memcpy, memset
from libcpp cimport bool
from libcpp.vector cimport vector
import numpy as np
from typing import Any, List, Optional, Sequence, Tuple


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
    left_count = _extract_subdag(
        vertexes, edges, existing_heads, left_vertexes_map, left_vertexes, left_edges, False)
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
                              uint32_t[:] left_edges,
                              bool only_map) nogil:
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
    new_hashes = {k for k, v, _ in new_edges}.union(v for k, v, _ in new_edges)
    if "" in new_hashes or None in new_hashes:
        # temporary DB inconsistency, we should retry later
        # we *must* check for both the empty string and None
        return hashes, vertexes, edges
    the_end = "0" * 40
    new_hashes.discard(the_end)  # initial commit virtual parents
    new_hashes = np.sort(np.fromiter(new_hashes, count=len(new_hashes), dtype="S40"))
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
    cdef vector[vector[Edge]] new_edges_lists = vector[vector[Edge]](len(new_hashes_map))
    new_edges_counter = 0
    for k, v, pos in new_edges:
        if v == the_end:
            # initial commit
            continue
        i = new_hashes_map.get(k.encode(), None)
        if i is not None:
            new_edges_lists[i].push_back(Edge(hashes_map[v.encode()], pos))
            new_edges_counter += 1
    old_vertex_map = np.zeros(len(hashes), dtype=np.uint32)
    result_vertexes = np.zeros(len(result_hashes) + 1, dtype=np.uint32)
    result_edges = np.zeros(len(edges) + new_edges_counter, dtype=np.uint32)
    if len(hashes) > 0:
        found_matches += np.arange(len(found_matches))
    _recalculate_vertices_and_edges(
        found_matches, vertexes, edges, &new_edges_lists, old_vertex_map,
        result_vertexes, result_edges)
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
    cdef uint32_t j, left, offset = 0, pos = 0, list_size
    cdef uint32_t old_edge_i = 0, new_edge_i = 0, size = len(result_vertexes) - 1, i
    cdef const vector[Edge] *new_edge_list
    cdef bint has_old = len(vertexes) > 1
    cdef Edge edge
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
    _toposort(vertexes, edges, heads[:-1], heads_order_is_significant, order)
    _mark_dag_access(vertexes, edges, heads[order], order, access)
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
    cdef vector[uint32_t] boilerplate
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    cdef vector[int32_t] visited = vector[int32_t](len(vertexes))
    cdef uint32_t j, head, peek, edge, missing = len(vertexes)
    cdef int64_t i, order_pos = 0
    cdef int32_t status, size = len(heads), vv = size + 1
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
    cdef vector[uint32_t] boilerplate
    boilerplate.reserve(max(1, len(edges) - len(vertexes) + 1))
    cdef uint32_t j, head, peek, edge, missing = len(vertexes)
    cdef int64_t i, original_index
    cdef int32_t size = len(heads)
    for i in range(size):
        head = heads[i]
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
    cdef vector[vector[uint32_t]] parents = vector[vector[uint32_t]](len(heads))
    full_size = _mark_dag_parents(
        vertexes, edges, heads, timestamps, ownership, slay_hydra, &parents)
    concat_parents = np.zeros(full_size, dtype=np.uint32)
    split_points = np.zeros(len(parents), dtype=np.int64)
    _copy_parents_to_array(&parents, concat_parents, split_points)
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
    cdef uint32_t not_found = len(vertexes), head, peek, edge, peak_owner, parent, beg, end
    cdef uint64_t timestamp, head_timestamp
    cdef int64_t i, j, p, sum_len = 0
    cdef bool reached_root
    cdef vector[char] visited = vector[char](len(vertexes) - 1)
    cdef vector[uint32_t] boilerplate
    cdef vector[uint32_t] *my_parents
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
    _extract_first_parents(vertexes, edges, heads, max_depth, first_parents)
    return hashes[first_parents]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _extract_first_parents(const uint32_t[:] vertexes,
                                 const uint32_t[:] edges,
                                 const uint32_t[:] heads,
                                 int max_depth,
                                 char[:] first_parents) nogil:
    cdef uint32_t head
    cdef int i, depth
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
    _partition_dag(vertexes, edges, seeds, borders)
    return hashes[borders]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _partition_dag(const uint32_t[:] vertexes,
                         const uint32_t[:] edges,
                         const uint32_t[:] heads,
                         char[:] borders) nogil:
    cdef vector[uint32_t] boilerplate
    cdef vector[char] visited = vector[char](len(vertexes) - 1)
    cdef int i, v
    cdef uint32_t head, edge, peek, j
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
    cdef vector[vector[uint32_t]] pr_commits = vector[vector[uint32_t]](len(pr_merges))
    left_vertexes_map = np.zeros(len(hashes), dtype=np.int8)
    _extract_pr_commits(vertexes, edges, pr_merges, left_vertexes_map, &pr_commits)
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
    cdef int i
    cdef uint32_t first, last, v, j, edge, peek
    cdef uint32_t oob = len(vertexes)
    cdef vector[uint32_t] *my_pr_commits
    cdef vector[uint32_t] boilerplate
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
    cdef vector[vector[uint32_t]] found_commits = vector[vector[uint32_t]](len(heads))
    left_vertexes_map = np.zeros_like(vertexes)
    left_vertexes = left_edges = np.array([], dtype=np.uint32)
    single_slot = np.zeros(1, dtype=np.uint32)
    _extract_independent_ownership(
        vertexes, edges, heads, stops, splits,
        single_slot, left_vertexes_map, left_vertexes, left_edges,
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
    cdef int64_t i, p
    cdef uint32_t j, head, parent, count, peek, edge
    cdef uint32_t oob = len(vertexes)
    cdef vector[uint32_t] *head_result
    cdef vector[uint32_t] boilerplate
    cdef bool has_parent
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
                vertexes, edges, single_slot, left_vertexes_map, left_vertexes, left_edges, True)
        if not has_parent:
            single_slot[0] = head
            count = _extract_subdag(
                vertexes, edges, single_slot, left_vertexes_map, left_vertexes, left_edges, False)
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
