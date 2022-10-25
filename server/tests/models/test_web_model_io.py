from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pytest

from athenian.api.models.web import (
    InvitationCheckResult,
    JIRAEpic,
    JIRAEpicChild,
    JIRAUser,
    MappedJIRAIdentity,
    PullRequest,
    PullRequestNumbers,
    StageTimings,
)
from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web_model_io import deserialize_models, model_to_json, serialize_models
from athenian.api.serialization import FriendlyJson

common_ts = datetime(2022, 10, 25, 11, 23, 45, tzinfo=timezone.utc)
smoke_models = [
    [
        JIRAEpic(
            project="project",
            children=[],
            prs=10,
            id="id",
            title="title",
            created=common_ts,
            updated=common_ts,
            work_began=np.datetime64(common_ts.replace(tzinfo=None), "s"),
            resolved=None,
            lead_time=timedelta(seconds=10),
            life_time=timedelta(seconds=20),
            reporter="reporter",
            assignee=None,
            comments=7,
            priority="priority",
            status="status",
            type="type",
            url="url",
        ),
        JIRAEpic(
            project="other_project",
            children=[
                JIRAEpicChild(
                    id="child_id",
                    title=b"child_title",
                    created=common_ts.replace(year=1999),
                    updated=np.datetime64(common_ts.replace(tzinfo=None), "ns"),
                    work_began=None,
                    resolved=common_ts.replace(month=2),
                    lead_time=None,
                    life_time=timedelta(days=12, seconds=24 * 3600 - 39),
                    reporter="child_reporter",
                    assignee="child_assignee",
                    comments=176,
                    priority=None,
                    status="child_status",
                    type="child_type",
                    url="child_url",
                    subtasks=100,
                    prs=np.int8(123),
                ),
            ],
            prs=np.int64(10),
            id="id",
            title="title",
            created=common_ts.replace(hour=4),
            updated=common_ts.replace(minute=7),
            work_began=common_ts.replace(second=9),
            resolved=None,
            lead_time=np.timedelta64(-20_000_000_000, "ns"),
            life_time=np.timedelta64(20, "s"),
            reporter="reporter",
            assignee=None,
            comments=np.int32(7),
            priority="priority",
            status="status",
            type="type",
            url="url",
        ),
    ],
    [],
    [
        MappedJIRAIdentity(
            developer_id="dev_id",
            developer_name=None,
            jira_name="jira_name_val",
            confidence=0.14159,
        ),
        MappedJIRAIdentity(
            developer_id="222dev_id",
            developer_name="Vadim",
            jira_name="2222jira_name_val",
            confidence=0,
        ),
        MappedJIRAIdentity(
            developer_id="222dev_id",
            developer_name="Vadim",
            jira_name="2222jira_name_val",
            confidence=np.float64(0.14159),
        ),
        MappedJIRAIdentity(
            developer_id="222dev_id",
            developer_name="Vadim",
            jira_name="2222jira_name_val",
            confidence=np.float32(0.14159),
        ),
        MappedJIRAIdentity(
            developer_id="222dev_id",
            developer_name="Vadim",
            jira_name="Podgórski",
            confidence=np.int32(0),
        ),
    ],
    [
        PullRequestNumbers(repository="athenian", numbers=[1, 2, 7, 4]),
    ],
    [
        InvitationCheckResult(active=False, type="admin", valid=True),
    ],
    [
        PullRequest(
            repository="repó_репо",
            number=1234,
            title="title",
            size_added=1,
            size_removed=0,
            files_changed=7,
            created=common_ts,
            updated=common_ts,
            closed=None,
            comments=1,
            commits=2,
            review_requested=common_ts,
            first_review=None,
            approved=None,
            review_comments=45,
            reviews=None,
            merged=None,
            merged_with_failed_check_runs=None,
            released=None,
            release_url=None,
            stage_timings=StageTimings(
                wip=timedelta(days=12), deploy={"prod": timedelta(days=12)},
            ),
            events_time_machine=np.array(["merged"], dtype=object),
            stages_time_machine=None,
            events_now=[],
            stages_now=[],
            participants=[],
            labels=None,
            jira=None,
            deployments=None,
        ),
    ],
    {"a": 111, "b": [None, 1.4], "c": True, "d": False, "e": "Podgórski"},
]


def test_serialize_models_smoke():
    models = tuple(deepcopy(smoke_models))
    models[0][0].created = models[0][0].created.replace(tzinfo=None)
    new_models = deserialize_models(serialize_models(models))
    models[0][0].created = models[0][0].created.replace(tzinfo=timezone.utc)
    models[0][1].children[0].title = models[0][1].children[0].title.decode()
    models[0][1].children[0].updated = np.datetime64(common_ts.replace(tzinfo=None), "s")
    models[0][1].lead_time = np.timedelta64(-20, "s")
    models[2][1].confidence = 0.0
    models[2][3].confidence = np.float64(models[2][3].confidence)
    models[2][4].confidence = 0.0
    assert models == new_models


class TestSerializeModelsUnicode:
    class _M(Model):
        f: str
        g: int

    @pytest.mark.parametrize("f", ["İbZZ KK yök", "Podgórski"])
    def test_not_nested_model(self, f) -> None:
        u = self._M(f=f, g=2)
        res = deserialize_models(serialize_models((u,)))
        assert res == (u,)

    def test_single_model(self) -> None:
        u = self._M(f="İbZZ KK yök", g=2)  # multibyte
        res = deserialize_models(serialize_models(([u],)))
        assert res == ([u],)

        u = self._M(f="KK yök", g=2)  # singlebyte unicode repr
        res = deserialize_models(serialize_models(([u],)))
        assert res == ([u],)

    def test_two_models(self) -> None:
        m0 = self._M(f="İbZZ KK yök", g=3)
        m1 = self._M(f="a~~xèí", g=4)
        res = deserialize_models(serialize_models(([m0], [m1])))
        assert res == ([m0], [m1])

        res = deserialize_models(serialize_models(([m0, m1],)))
        assert res == ([m0, m1],)

    def test_jira_user_model(self) -> None:
        u = JIRAUser(name="İbZZ KK yök", avatar="a", type="atlassian", developer=None)
        u2 = JIRAUser(name="a~~xèí", avatar="a", type="atlassian", developer=None)
        res = deserialize_models(serialize_models((u, u2)))
        assert res == (u, u2)

    def test_jira_user_model_nested(self) -> None:
        u0 = JIRAUser(name="İbZZ KK yök", avatar="𝞈𝝖", type="atlassian", developer=None)
        u1 = JIRAUser(name="İbZàök", avatar="a", type="atlassian", developer="a")
        res = deserialize_models(serialize_models(([], [u0, u1])))
        assert res == ([], [u0, u1])
        assert res[1][0].name == "İbZZ KK yök"
        assert res[1][0].avatar == "𝞈𝝖"


@pytest.mark.parametrize("models", [smoke_models[0][0], *smoke_models])
def test_model_to_json_smoke(models):
    baseline = FriendlyJson.dumps(Model.serialize(models))
    native = model_to_json(models).decode()
    assert json.loads(baseline) == json.loads(native)


def test_model_to_json_unsupported():
    class FooBar:
        pass

    with pytest.raises(AssertionError):
        model_to_json(FooBar())
