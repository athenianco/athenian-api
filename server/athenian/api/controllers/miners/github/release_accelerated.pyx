# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.undeclared=True, warn.maybe_uninitialized=True
# distutils: language = c++

from cpython.ref cimport PyObject
from libcpp.vector cimport vector


def traverse_history(history: dict,
                     rel_sha: str,
                     rel_index: int,
                     release_hashes: set):
    _traverse_history(history, rel_sha, rel_index, release_hashes)


cdef _traverse_history(history: dict,
                       rel_sha: str,
                       rel_index: int,
                       release_hashes: set):
    cdef vector[PyObject *] parents
    parents.push_back(<PyObject *>rel_sha)
    cdef str x, h
    while not parents.empty():
        x = <str>parents.back()
        parents.pop_back()
        xh = history[x]
        if xh[0] == rel_index or (x in release_hashes and x != rel_sha):
            continue
        xh[0] = rel_index
        for i from 1 <= i < len(xh) by 1:
            parents.push_back(<PyObject *>xh[i])
