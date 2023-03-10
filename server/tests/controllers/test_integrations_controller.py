from datetime import timedelta
import io
from zipfile import ZipFile

import medvedi as md
from pyarrow.parquet import read_table
import pytest
from sqlalchemy import delete

from athenian.api.internal.miners.github import deployment_light
from athenian.api.models.persistentdata.models import DeployedComponent, DeploymentNotification
from athenian.api.models.state.models import UserAccount
from athenian.api.models.web import ContributorIdentity, MatchedIdentity, PullRequestMetricID
from athenian.api.serialization import FriendlyJson


async def test_match_identities_smoke(client, headers):
    body = {
        "account": 1,
        "identities": [
            {
                "names": ["Vadim", "Markovtsv"],
            },
            {
                "emails": ["eiso@athenian.co", "contact@eisokant.com"],
            },
            {
                "names": ["Denys Smyrnov"],
                "emails": ["denys@sourced.tech"],
            },
        ],
    }
    response = await client.request(
        method="POST", path="/v1/match/identities", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + rbody
    model = [MatchedIdentity.from_dict(i) for i in FriendlyJson.loads(rbody)]
    for i in range(len(body["identities"])):
        assert model[i].from_ == ContributorIdentity.from_dict(body["identities"][i])
    assert model[0].to == "github.com/vmarkovtsev"
    assert model[0].confidence < 1
    assert model[1].to == "github.com/eiso"
    assert model[1].confidence == 1
    assert model[2].to == "github.com/dennwc"
    assert model[2].confidence < 1


@pytest.mark.parametrize(
    "body, code",
    [
        ({"account": 1, "identities": [{}]}, 400),
        ({"account": 1, "identities": [{"emails": []}]}, 400),
        ({"account": 1, "identities": [{"names": []}]}, 400),
        ({"account": 1, "identities": [{"names": [], "emails": []}]}, 400),
        ({"account": 1, "identities": [{"emails": ["x@y.co"]}, {"emails": ["x@y.co"]}]}, 400),
        ({"account": 2, "identities": [{"emails": ["x@y.co"]}]}, 422),
        ({"account": 4, "identities": [{"emails": ["x@y.co"]}]}, 404),
    ],
)
async def test_match_identities_nasty_input(client, headers, body, code):
    response = await client.request(
        method="POST", path="/v1/match/identities", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + rbody


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_get_everything_smoke(
    client,
    headers,
    dummy_deployment_label,
    precomputed_deployments,
):
    # preheat
    body = {
        "for": [
            {
                "with": {},
                "repositories": [
                    "github.com/src-d/go-git",
                ],
            },
        ],
        "metrics": [PullRequestMetricID.PR_LEAD_TIME],
        "date_from": "2015-10-13",
        "date_to": "2020-01-23",
        "granularities": ["all"],
        "exclude_inactive": False,
        "account": 1,
    }
    response = await client.request(
        method="POST", path="/v1/metrics/pull_requests", headers=headers, json=body,
    )
    body = (await response.read()).decode("utf-8")
    assert response.status == 200, "Response body is : " + body

    original_threshold = deployment_light.repository_environment_threshold
    deployment_light.repository_environment_threshold = timedelta(days=100500)
    try:
        response = await client.request(
            method="GET", path="/v1/get/export?account=1", headers=headers,
        )
        assert response.status == 200
        body = await response.read()
    finally:
        deployment_light.repository_environment_threshold = original_threshold

    developer_dfs = {
        "jira_mapping": 212,
        "active_active0_commits_pushed_lines_changed": 3135,
        "prs_created": 681,
        "prs_merged": 554,
        "releases": 52,  # 53 releases, one set to null author
        "prs_reviewed_review_approvals_review_neutrals_review_rejections_reviews": 1352,
        "regular_pr_comments": 1035,
        "review_pr_comments": 1604,
        "pr_comments": 2639,
    }

    with ZipFile(io.BytesIO(body)) as zipf:
        with zipf.open("prs.parquet") as prsf:
            prs_df = md.DataFrame.from_arrow(read_table(prsf))
        with zipf.open("releases.parquet") as releasesf:
            releases_df = md.DataFrame.from_arrow(read_table(releasesf))
        for key, size in developer_dfs.items():
            with zipf.open(f"developers_{key}.parquet") as devf:
                df = md.DataFrame.from_arrow(read_table(devf))
                assert len(df) == size
        with zipf.open("check_runs.parquet") as checkf:
            check_runs_df = md.DataFrame.from_arrow(read_table(checkf))
        with zipf.open("jira_issues.parquet") as jiraf:
            jira_issues_df = md.DataFrame.from_arrow(read_table(jiraf))
        with zipf.open("deployments.parquet") as depsf:
            deps_df = md.DataFrame.from_arrow(read_table(depsf))
        with zipf.open("deployments_components.parquet") as depscompsf:
            depscomps_df = md.DataFrame.from_arrow(read_table(depscompsf))
        with zipf.open("deployments_releases.parquet") as depsrelsf:
            depsrels_df = md.DataFrame.from_arrow(read_table(depsrelsf))
        with zipf.open("deployments_labels.parquet") as depslblsf:
            depslbls_df = md.DataFrame.from_arrow(read_table(depslblsf))
    assert len(prs_df) == 678
    assert set(prs_df) == {
        "acc_id",
        "activity_days",
        "additions",
        "approved",
        "author",
        "base_ref",
        "changed_files",
        "closed",
        "closed_at",
        "commits",
        "created",
        "created_at",
        "deletions",
        "deployed",
        "deployment_conclusions",
        "deployments",
        "done",
        "environments",
        "first_comment_on_first_review",
        "first_commit",
        "first_review_request",
        "first_review_request_exact",
        "force_push_dropped",
        "head_ref",
        "jira_ids",
        "jira_labels",
        "jira_priorities",
        "jira_projects",
        "jira_types",
        "last_commit",
        "last_commit_before_first_review",
        "last_review",
        "merge_commit_id",
        "merge_commit_sha",
        "merged",
        "merged_at",
        "merged_by_id",
        "merged_by_login",
        "merged_with_failed_check_runs",
        "merger",
        "node_id",
        "number",
        "participants",
        "regular_comments",
        "release_ignored",
        "release_node_id",
        "release_url",
        "released",
        "releaser",
        "repository_full_name",
        "repository_node_id",
        "review_comments",
        "reviews",
        "size",
        "stage_time_deploy_production",
        "stage_time_merge",
        "stage_time_release",
        "stage_time_review",
        "stage_time_wip",
        "title",
        "updated_at",
        "user_login",
        "user_node_id",
        "work_began",
    }
    assert len(releases_df) == 53
    assert set(releases_df) == {
        "additions",
        "age",
        "commit_authors",
        "commits_count",
        "deletions",
        "deployments",
        "jira_ids",
        "jira_labels",
        "jira_pr_offsets",
        "jira_priorities",
        "jira_projects",
        "jira_types",
        "matched_by",
        "name",
        "node_id",
        "prs_additions",
        "prs_created_at",
        "prs_deletions",
        "prs_node_id",
        "prs_number",
        "prs_title",
        "prs_user_node_id",
        "published",
        "publisher",
        "repository_full_name",
        "sha",
        "url",
    }
    assert len(check_runs_df) == 4614
    assert set(check_runs_df) == {
        "acc_id",
        "additions",
        "author_login",
        "author_user_id",
        "authored_date",
        "changed_files",
        "check_run_node_id",
        "check_suite_completed",
        "check_suite_conclusion",
        "check_suite_node_id",
        "check_suite_started",
        "check_suite_status",
        "commit_node_id",
        "committed_date",
        "completed_at",
        "conclusion",
        "deletions",
        "name",
        "pull_request_closed_at",
        "pull_request_created_at",
        "pull_request_merged",
        "pull_request_node_id",
        "repository_full_name",
        "repository_node_id",
        "sha",
        "started_at",
        "status",
        "url",
    }
    assert len(jira_issues_df) == 1797
    assert set(jira_issues_df) == {
        "assignee",
        "athenian_epic_id",
        "category_name",
        "commenters",
        "created",
        "id",
        "labels",
        "pr_ids",
        "priority_name",
        "prs_began",
        "prs_count",
        "prs_released",
        "reporter",
        "resolved",
        "status",
        "type",
        "updated",
        "work_began",
    }
    assert len(deps_df) == 1
    assert set(deps_df.columns) == {
        "commit_authors",
        "commits_overall",
        "commits_prs",
        "conclusion",
        "environment",
        "finished_at",
        "jira_ids",
        "jira_offsets",
        "lines_overall",
        "lines_prs",
        "name",
        "pr_authors",
        "prs",
        "prs_additions",
        "prs_created_at",
        "prs_deletions",
        "prs_jira_ids",
        "prs_jira_offsets",
        "prs_number",
        "prs_offsets",
        "prs_title",
        "prs_user_node_id",
        "release_authors",
        "repositories",
        "started_at",
        "url",
    }
    assert len(depscomps_df) == 1
    assert set(depscomps_df.columns) == {
        "deployment_name",
        "reference",
        "repository_full_name",
        "repository_node_id",
        "resolved_commit_node_id",
    }
    assert len(depsrels_df) == 51
    assert set(depsrels_df.columns) == {
        "additions",
        "age",
        "author",
        "author_node_id",
        "commit_authors",
        "commit_id",
        "commits_count",
        "deletions",
        "deployment_name",
        "deployments",
        "jira_ids",
        "jira_labels",
        "jira_pr_offsets",
        "jira_priorities",
        "jira_projects",
        "jira_types",
        "matched_by",
        "name",
        "node_id",
        "prs_additions",
        "prs_created_at",
        "prs_deletions",
        "prs_node_id",
        "prs_number",
        "prs_title",
        "prs_user_node_id",
        "published_at",
        "repository_full_name",
        "repository_node_id",
        "sha",
        "tag",
        "url",
    }
    assert len(depslbls_df) == 1
    assert set(depslbls_df.columns) == {"deployment_name", "key", "value"}


@pytest.mark.parametrize(
    "query, code",
    [
        ("?account=2", 422),
        ("?account=3", 404),
        ("?account=1&format=other", 400),
        ("?account=1&format=parquet", 200),
        ("", 400),
        ("<empty>", 200),
    ],
)
async def test_get_everything_nasty_input(client, headers, query, code, sdb):
    if query == "<empty>":
        query = ""
        await sdb.execute(delete(UserAccount).where(UserAccount.account_id == 2))
    response = await client.request(method="GET", path=f"/v1/get/export{query}", headers=headers)
    assert response.status == code


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
async def test_get_everything_no_deployments(client, headers, rdb):
    await rdb.execute(delete(DeployedComponent))
    await rdb.execute(delete(DeploymentNotification))
    response = await client.request(
        method="GET", path="/v1/get/export?account=1", headers=headers,
    )
    assert response.status == 200, (await response.read()).decode()
