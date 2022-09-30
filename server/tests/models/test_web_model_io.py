from datetime import datetime, timedelta, timezone

import numpy as np

from athenian.api.models.web import (
    InvitationCheckResult,
    JIRAEpic,
    JIRAEpicChild,
    MappedJIRAIdentity,
    PullRequest,
    PullRequestNumbers,
    StageTimings,
)
from athenian.api.models.web_model_io import deserialize_models, serialize_models


def test_serialize_models_smoke():
    now = datetime.now(timezone.utc).replace(microsecond=0)
    models = (
        [
            JIRAEpic(
                project="project",
                children=[],
                prs=10,
                id="id",
                title="title",
                created=now.replace(tzinfo=None),
                updated=now,
                work_began=np.datetime64(now, "s"),
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
                        created=now,
                        updated=np.datetime64(now, "ns"),
                        work_began=None,
                        resolved=now,
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
                created=now,
                updated=now,
                work_began=now,
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
        ],
        {"a": 111},
        [
            PullRequestNumbers(repository="athenian", numbers=[1, 2, 7, 4]),
        ],
        [
            InvitationCheckResult(active=False, type="admin", valid=True),
        ],
        [
            PullRequest(
                repository="repo",
                number=1234,
                title="title",
                size_added=1,
                size_removed=0,
                files_changed=7,
                created=now,
                updated=now,
                closed=None,
                comments=1,
                commits=2,
                review_requested=now,
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
                events_time_machine=["merged"],
                stages_time_machine=None,
                events_now=[],
                stages_now=[],
                participants=[],
                labels=None,
                jira=None,
                deployments=None,
            ),
        ],
    )
    new_models = deserialize_models(serialize_models(models))
    models[0][0].created = models[0][0].created.replace(tzinfo=timezone.utc)
    models[0][1].children[0].title = models[0][1].children[0].title.decode()
    models[0][1].children[0].updated = np.datetime64(now, "s")
    models[0][1].lead_time = np.timedelta64(-20, "s")
    models[2][1].confidence = 0.0
    assert models == new_models
