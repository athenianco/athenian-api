# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
from typing import Any, Optional

from athenian.api.internal.settings import ReleaseMatch, ReleaseSettings, default_branch_alias

from cpython cimport PyObject
from cpython.dict cimport PyDict_GetItem
from cpython.unicode cimport PyUnicode_GET_LENGTH
from libc.string cimport memchr, memcmp, memcpy
from libcpp.memory cimport allocator, unique_ptr
from libcpp.unordered_map cimport unordered_map

from athenian.api.native.cpython cimport PyUnicode_DATA
from athenian.api.native.string_view cimport string_view


cdef extern from "string.h" nogil:
    void *memmem(
        const void *haystack, size_t haystacklen,
        const void *needle, size_t needlelen,
    )


cdef:
    const char *default_branch_alias_data = <const char *>PyUnicode_DATA(<PyObject *> default_branch_alias)
    Py_ssize_t default_branch_alias_len = len(default_branch_alias)
    long branch = ReleaseMatch.branch
    long tag = ReleaseMatch.tag
    long tag_or_branch = ReleaseMatch.tag_or_branch
    long event = ReleaseMatch.event
    const char *tag_or_branch_data = <const char *>PyUnicode_DATA(<PyObject *> ReleaseMatch.tag_or_branch.name)
    Py_ssize_t tag_or_branch_name_len = len(ReleaseMatch.tag_or_branch.name)
    const char *rejected_data = <const char *>PyUnicode_DATA(<PyObject *> ReleaseMatch.rejected.name)
    Py_ssize_t rejected_name_len = len(ReleaseMatch.rejected.name)
    const char *force_push_drop_data = <const char *>PyUnicode_DATA(<PyObject *> ReleaseMatch.force_push_drop.name)
    Py_ssize_t force_push_drop_name_len = len(ReleaseMatch.force_push_drop.name)
    unordered_map[string_view, long] release_match_name_to_enum


for obj in ReleaseMatch:
    release_match_name_to_enum[string_view(<char *>PyUnicode_DATA(<PyObject *> obj.name), PyUnicode_GET_LENGTH(obj.name))] = obj


ctypedef void (*free_type)(void *)


def triage_by_release_match(
    repo: str,
    release_match: str,
    release_settings: ReleaseSettings,
    default_branches: dict[str, str],
    result: Any,
    ambiguous: dict[str, Any],
) -> Optional[Any]:
    """Check the release match of the specified `repo` and return `None` if it is not effective \
    according to `release_settings`, or decide between `result` and `ambiguous`."""
    cdef:
        Py_ssize_t release_match_len = PyUnicode_GET_LENGTH(release_match)
        const char *release_match_data = <const char *>PyUnicode_DATA(<PyObject *> release_match)
        PyObject *required_release_match
        const char *match_name
        int match_name_len
        const char *match_by
        int match_by_len
        long match
        long required_release_match_match
        const char *target_data
        Py_ssize_t target_len
        PyObject *default_branch
        const char *default_branch_data
        Py_ssize_t default_branch_len
        const char *found
        unique_ptr[char] resolved_branch
        Py_ssize_t pos
    if (
        (
            release_match_len == rejected_name_len
            and memcmp(release_match_data, rejected_data, rejected_name_len) == 0
        )
        or
        (
            release_match_len == force_push_drop_name_len
            and memcmp(release_match_data, force_push_drop_data, force_push_drop_name_len) == 0
        )
    ):
        return result

    required_release_match = PyDict_GetItem(release_settings.native, repo)
    if required_release_match == NULL:
        # DEV-1451: if we don't have this repository in the release settings, then it is deleted
        raise AssertionError(
            f"You must take care of deleted repositories separately: {repo}",
        ) from None
    match_name = <const char *> memchr(release_match_data, ord(b"|"), release_match_len)
    if match_name == NULL:
        match_name_len = release_match_len
        match_by = release_match_data + release_match_len
        match_by_len = 0
    else:
        match_name_len = match_name - release_match_data
        match_by = match_name + 1
        match_by_len = release_match_len - match_name_len - 1
    match_name = release_match_data
    match = release_match_name_to_enum[string_view(match_name, match_name_len)]
    required_release_match_match = (<object>required_release_match).match
    if required_release_match_match != tag_or_branch:
        if match != required_release_match_match:
            return None
        dump = result
    else:
        if memcmp(match_name, b"event", 5) == 0:
            return None
        dump = ambiguous[release_match[:match_name_len]]
    if match == tag:
        target = (<object>required_release_match).tags
    elif match == branch:
        target = (<object>required_release_match).branches
    elif match == event:
        target = (<object>required_release_match).events
    else:
        raise AssertionError("Precomputed DB may not contain Match.tag_or_branch")
    target_data = <const char *> PyUnicode_DATA(<PyObject *> target)
    target_len = PyUnicode_GET_LENGTH(target)
    if match == branch:
        found = <const char *>memmem(
            target_data, target_len, default_branch_alias_data, default_branch_alias_len,
        )
        if found != NULL:
            default_branch = PyDict_GetItem(default_branches, repo)
            default_branch_len = PyUnicode_GET_LENGTH(<object> default_branch)
            target_len += default_branch_len - default_branch_alias_len
            if target_len != match_by_len:
                return None
            default_branch_data = <const char *> PyUnicode_DATA(default_branch)
            resolved_branch.reset(allocator[char]().allocate(target_len))
            pos = found - target_data
            memcpy(resolved_branch.get(), target_data, pos)
            memcpy(resolved_branch.get() + pos, default_branch_data, default_branch_len)
            pos += default_branch_len
            memcpy(resolved_branch.get() + pos, found + default_branch_alias_len, target_len - pos)
            target_data = resolved_branch.get()

    if target_len != match_by_len or memcmp(target_data, match_by, match_by_len):
        return None
    return dump
