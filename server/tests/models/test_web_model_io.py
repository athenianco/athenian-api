from datetime import datetime, timedelta, timezone

from athenian.api.models.web import JIRAEpic, JIRAEpicChild, MappedJIRAIdentity, PullRequestNumbers
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
                work_began=now,
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
                        updated=now,
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
                        prs=123,
                    ),
                ],
                prs=10,
                id="id",
                title="title",
                created=now,
                updated=now,
                work_began=now,
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
    )
    new_models = deserialize_models(serialize_models(models))
    models[0][0].created = models[0][0].created.replace(tzinfo=timezone.utc)
    models[0][1].children[0].title = models[0][1].children[0].title.decode()
    models[2][1].confidence = 0.0
    assert models == new_models
