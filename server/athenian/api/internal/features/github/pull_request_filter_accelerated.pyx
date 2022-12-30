# cython: language_level=3, boundscheck=False, nonecheck=False, optimize.unpack_method_calls=True
# cython: warn.maybe_uninitialized=True
# distutils: language = c++
# distutils: extra_compile_args = -std=c++17

from cpython cimport PyObject
from cython.operator cimport dereference
from libc.stdint cimport int64_t

from athenian.api.native.cpython cimport (
    Py_TYPE,
    PyByteArray_AS_STRING,
    PyByteArray_CheckExact,
    PyBytes_AS_STRING,
    PyBytes_Check,
    PyMemberDef,
)
from athenian.api.native.numpy cimport NPY_DATETIME_NAT, NPY_FR_s, PyDatetimeScalarObject

from athenian.api.internal.miners.types import PullRequestEvent, PullRequestFacts, PullRequestStage

PullRequestStage_DEPLOYED = PullRequestStage.DEPLOYED
PullRequestStage_FORCE_PUSH_DROPPED = PullRequestStage.FORCE_PUSH_DROPPED
PullRequestStage_RELEASE_IGNORED = PullRequestStage.RELEASE_IGNORED
PullRequestStage_DONE = PullRequestStage.DONE
PullRequestStage_RELEASING = PullRequestStage.RELEASING
PullRequestStage_MERGING = PullRequestStage.MERGING
PullRequestStage_REVIEWING = PullRequestStage.REVIEWING
PullRequestStage_WIP = PullRequestStage.WIP

PullRequestEvent_CREATED = PullRequestEvent.CREATED
PullRequestEvent_COMMITTED = PullRequestEvent.COMMITTED
PullRequestEvent_REVIEWED = PullRequestEvent.REVIEWED
PullRequestEvent_REVIEW_REQUESTED = PullRequestEvent.REVIEW_REQUESTED
PullRequestEvent_APPROVED = PullRequestEvent.APPROVED
PullRequestEvent_MERGED = PullRequestEvent.MERGED
PullRequestEvent_REJECTED = PullRequestEvent.REJECTED
PullRequestEvent_RELEASED = PullRequestEvent.RELEASED
PullRequestEvent_CHANGES_REQUESTED = PullRequestEvent.CHANGES_REQUESTED
PullRequestEvent_DEPLOYED = PullRequestEvent.DEPLOYED


cdef int PullRequestFactsOffset[128]

cdef enum PullRequestFactsField:
    PullRequestFactsFieldDone
    PullRequestFactsFieldMerged
    PullRequestFactsFieldApproved
    PullRequestFactsFieldForcePushDropped
    PullRequestFactsFieldReleaseIgnored
    PullRequestFactsFieldFirstReviewRequest
    PullRequestFactsFieldFirstReviewRequestExact
    PullRequestFactsFieldCreated
    PullRequestFactsFieldClosed
    PullRequestFactsFieldReleased


PullRequestFactsOffset[<int>PullRequestFactsFieldDone] = PullRequestFacts.dtype.fields[PullRequestFacts.f.done][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldMerged] = PullRequestFacts.dtype.fields[PullRequestFacts.f.merged][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldApproved] = PullRequestFacts.dtype.fields[PullRequestFacts.f.approved][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldForcePushDropped] = PullRequestFacts.dtype.fields[PullRequestFacts.f.force_push_dropped][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldReleaseIgnored] = PullRequestFacts.dtype.fields[PullRequestFacts.f.release_ignored][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldFirstReviewRequest] = PullRequestFacts.dtype.fields[PullRequestFacts.f.first_review_request][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldCreated] = PullRequestFacts.dtype.fields[PullRequestFacts.f.created][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldFirstReviewRequestExact] = PullRequestFacts.dtype.fields[PullRequestFacts.f.first_review_request_exact][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldClosed] = PullRequestFacts.dtype.fields[PullRequestFacts.f.closed][-1]
PullRequestFactsOffset[<int>PullRequestFactsFieldReleased] = PullRequestFacts.dtype.fields[PullRequestFacts.f.released][-1]


def collect_events_and_stages(
    facts,
    dict hard_events,
    time_from,
):
    cdef:
        set events = set()
        set stages = set()
        PyMemberDef *slots = Py_TYPE(<PyObject *> facts).tp_members
        int64_t time_from_i64 = (<PyDatetimeScalarObject *> time_from).obval
        PyObject *data_obj = dereference(
            <PyObject **> ((<char *><PyObject *> facts) + slots[1].offset)
        )
        char *data
    if PyBytes_Check(data_obj):
        data = PyBytes_AS_STRING(data_obj)
    elif PyByteArray_CheckExact(data_obj):
        data = PyByteArray_AS_STRING(data_obj)
    else:
        raise AssertionError(f"unsupported buffer type: {type(facts.data)}")
    cdef:
        int64_t merged = dereference(
            <int64_t *>(data + PullRequestFactsOffset[<int>PullRequestFactsFieldMerged])
        )
        int64_t approved = dereference(
            <int64_t *>(data + PullRequestFactsOffset[<int>PullRequestFactsFieldApproved])
        )

    assert (<PyDatetimeScalarObject *> time_from).obmeta.base == NPY_FR_s

    if dereference(data + PullRequestFactsOffset[<int>PullRequestFactsFieldDone]):
        if dereference(data + PullRequestFactsOffset[<int>PullRequestFactsFieldForcePushDropped]):
            stages.add(PullRequestStage_FORCE_PUSH_DROPPED)
        elif dereference(data + PullRequestFactsOffset[<int>PullRequestFactsFieldReleaseIgnored]):
            stages.add(PullRequestStage_RELEASE_IGNORED)
        stages.add(PullRequestStage_DONE)
    elif merged != NPY_DATETIME_NAT:
        stages.add(PullRequestStage_RELEASING)
    elif approved != NPY_DATETIME_NAT:
        stages.add(PullRequestStage_MERGING)
    elif dereference(<int64_t *>(data + PullRequestFactsOffset[<int>PullRequestFactsFieldFirstReviewRequest])) != NPY_DATETIME_NAT:
        stages.add(PullRequestStage_REVIEWING)
    else:
        stages.add(PullRequestStage_WIP)
    if dereference(<int64_t *>(data + PullRequestFactsOffset[<int>PullRequestFactsFieldCreated])) >= time_from_i64:
        events.add(PullRequestEvent_CREATED)
    if hard_events[PullRequestEvent_COMMITTED]:
        events.add(PullRequestEvent_COMMITTED)
    if hard_events[PullRequestEvent_REVIEWED]:
        events.add(PullRequestEvent_REVIEWED)
    # NaT is < 0
    if dereference(<int64_t *>(data + PullRequestFactsOffset[<int>PullRequestFactsFieldFirstReviewRequestExact])) >= time_from_i64:
        events.add(PullRequestEvent_REVIEW_REQUESTED)
    if approved >= time_from_i64:
        events.add(PullRequestEvent_APPROVED)
    if merged >= time_from_i64:
        events.add(PullRequestEvent_MERGED)
    if merged == NPY_DATETIME_NAT and dereference(<int64_t *>(data + PullRequestFactsOffset[<int>PullRequestFactsFieldClosed])) >= time_from_i64:
        events.add(PullRequestEvent_REJECTED)
    if dereference(<int64_t *>(data + PullRequestFactsOffset[<int>PullRequestFactsFieldReleased])) >= time_from_i64:
        events.add(PullRequestEvent_RELEASED)
    if hard_events[PullRequestEvent_CHANGES_REQUESTED]:
        events.add(PullRequestEvent_CHANGES_REQUESTED)
    if hard_events[PullRequestEvent_DEPLOYED]:
        events.add(PullRequestEvent_DEPLOYED)
        stages.add(PullRequestStage_DEPLOYED)
    return events, stages
