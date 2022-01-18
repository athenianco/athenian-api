import io
from zipfile import ZipFile

import pandas as pd
import pytest
from sqlalchemy import delete

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


@pytest.mark.parametrize("body, code", [
    ({"account": 1, "identities": [{}]}, 400),
    ({"account": 1, "identities": [{"emails": []}]}, 400),
    ({"account": 1, "identities": [{"names": []}]}, 400),
    ({"account": 1, "identities": [{"names": [], "emails": []}]}, 400),
    ({"account": 1, "identities": [{"emails": ["x@y.co"]}, {"emails": ["x@y.co"]}]}, 400),
    ({"account": 2, "identities": [{"emails": ["x@y.co"]}]}, 422),
    ({"account": 4, "identities": [{"emails": ["x@y.co"]}]}, 404),
])
async def test_match_identities_nasty_input(client, headers, body, code):
    response = await client.request(
        method="POST", path="/v1/match/identities", headers=headers, json=body,
    )
    rbody = (await response.read()).decode("utf-8")
    assert response.status == code, "Response body is : " + rbody


async def test_get_everything_smoke(client, headers, dummy_deployment_label):
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

    response = await client.request(
        method="GET", path="/v1/get/export?account=1", headers=headers,
    )
    assert response.status == 200
    body = await response.read()

    developer_dfs = {
        "jira_mapping": 206,
        "active_active0_commits_pushed_lines_changed": 3135,
        "prs_created": 681,
        "prs_merged": 554,
        "releases": 53,
        "prs_reviewed_review_approvals_review_neutrals_review_rejections_reviews": 1352,
        "regular_pr_comments": 1035,
        "review_pr_comments": 1604,
        "pr_comments": 2639,
    }

    with ZipFile(io.BytesIO(body)) as zipf:
        with zipf.open("prs.parquet") as prsf:
            prs_df = pd.read_parquet(prsf)
        with zipf.open("releases.parquet") as releasesf:
            releases_df = pd.read_parquet(releasesf)
        for key, size in developer_dfs.items():
            with zipf.open(f"developers_{key}.parquet") as devf:
                df = pd.read_parquet(devf)
                assert len(df) == size
        with zipf.open("check_runs.parquet") as checkf:
            check_runs_df = pd.read_parquet(checkf)
        with zipf.open("jira_issues.parquet") as jiraf:
            jira_issues_df = pd.read_parquet(jiraf)
        with zipf.open("deployments.parquet") as depsf:
            deps_df = pd.read_parquet(depsf)
        with zipf.open("deployments_components.parquet") as depscompsf:
            depscomps_df = pd.read_parquet(depscompsf)
        with zipf.open("deployments_releases.parquet") as depsrelsf:
            depsrels_df = pd.read_parquet(depsrelsf)
        with zipf.open("deployments_labels.parquet") as depslblsf:
            depslbls_df = pd.read_parquet(depslblsf)
    assert len(prs_df) == 679
    assert set(prs_df) == {
        "first_comment_on_first_review", "merged_by_login", "first_commit", "stage_time_review",
        "title", "updated_at", "acc_id", "base_ref", "last_commit", "stage_time_wip", "deletions",
        "repository_node_id", "author", "hidden", "merged", "merged_at", "number", "created",
        "merge_commit_sha", "first_review_request_exact", "stage_time_release", "released",
        "stage_time_merge", "created_at", "user_login", "htmlurl", "approved", "closed_at",
        "changed_files", "last_commit_before_first_review", "force_push_dropped", "additions",
        "work_began", "releaser", "merged_by_id", "user_node_id", "repository_full_name",
        "first_review_request", "last_review", "activity_days", "closed", "merge_commit_id",
        "head_ref", "jira_ids", "merger", "done", "size", "reviews", "release_url",
        "release_node_id", "review_comments", "participants", "deployments", "deployed",
        "environments", "deployment_conclusions",
    }
    assert len(releases_df) == 53
    assert set(releases_df) == {
        "additions", "age", "commit_authors", "commits_count", "deletions", "matched_by", "name",
        "prs_additions", "prs_deletions", "prs_node_id", "prs_number", "prs_title", "prs_jira",
        "prs_user_node_id", "published", "publisher", "repository_full_name", "sha", "url",
        "deployments",
    }
    assert len(check_runs_df) == 4614
    assert set(check_runs_df) == {
        "acc_id", "additions", "author_login", "author_user_id", "authored_date", "changed_files",
        "check_run_node_id", "check_suite_conclusion", "check_suite_node_id", "check_suite_status",
        "commit_node_id", "committed_date", "completed_at", "conclusion", "deletions", "name",
        "pull_request_node_id", "repository_full_name", "repository_node_id", "sha", "started_at",
        "status", "url", "check_suite_started", "check_suite_completed", "pull_request_created_at",
        "pull_request_closed_at", "pull_request_merged",
    }
    assert len(jira_issues_df) == 1797
    assert set(jira_issues_df) == {
        "assignee", "category_name", "commenters", "created", "epic_id", "labels", "pr_ids",
        "priority_name", "prs_began", "prs_count", "prs_released", "reporter", "resolved",
        "status", "type", "updated", "work_began",
    }
    assert len(deps_df) == 1
    assert set(deps_df.columns) == {
        "commit_authors", "commits_overall", "commits_prs", "conclusion", "environment",
        "finished_at", "lines_overall", "lines_prs", "pr_authors", "prs", "prs_offsets",
        "release_authors", "repositories", "started_at", "url",
    }
    assert len(depscomps_df) == 1
    assert set(depscomps_df.columns) == {
        "repository_node_id", "repository_full_name", "reference", "resolved_commit_node_id",
        "deployment_name",
    }
    assert len(depsrels_df) == 51
    assert set(depsrels_df.columns) == {
        "commit_authors", "prs_node_id", "prs_number", "prs_additions",
        "prs_deletions", "prs_user_node_id", "prs_title", "prs_jira",
        "deployments", "age", "additions", "deletions", "commits_count",
        "repository_node_id", "author_node_id", "name",
        "published_at", "tag", "url", "sha", "commit_id", "matched_by",
        "author", "deployment_name",
    }
    assert len(depslbls_df) == 1
    assert set(depslbls_df.columns) == {"deployment_name", "key", "value"}


@pytest.mark.parametrize("query, code", [
    ("?account=2", 422),
    ("?account=3", 404),
    ("?account=1&format=other", 400),
    ("?account=1&format=parquet", 200),
    ("", 400),
    ("<empty>", 200),
])
async def test_get_everything_nasty_input(client, headers, query, code, sdb):
    if query == "<empty>":
        query = ""
        await sdb.execute(delete(UserAccount).where(UserAccount.account_id == 2))
    response = await client.request(
        method="GET", path=f"/v1/get/export{query}", headers=headers,
    )
    assert response.status == code
