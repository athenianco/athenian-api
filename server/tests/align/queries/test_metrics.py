from datetime import date
from typing import Any, Dict, List

from aiohttp.test_utils import TestClient
from morcilla import Database
import pytest
from sqlalchemy import insert

from athenian.api.async_utils import gather
from athenian.api.models.state.models import MappedJIRAIdentity
from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID
from tests.align.utils import align_graphql_request, build_recursive_fields_structure
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import models_insert
from tests.testutils.factory.state import TeamFactory


class BaseMetricsTest:
    def _query(self, depth=4):
        fragment = """
            fragment teamMetricValueFields on TeamMetricValue {
              team {
                id
              }
              value {
                str
                int
                float
              }
            }
        """

        recursive_fields = build_recursive_fields_structure(["...teamMetricValueFields"], depth)

        return (
            fragment
            + f"""
            query ($accountId: Int!,
                   $teamId: Int!,
                   $metrics: [String!]!,
                   $validFrom: Date!,
                   $expiresAt: Date!) {{
              metricsCurrentValues(accountId: $accountId, params: {{
                teamId: $teamId,
                metrics: $metrics,
                validFrom: $validFrom,
                expiresAt: $expiresAt
              }}) {{
                metric
                value {{
                  {recursive_fields}
                }}
              }}
            }}
        """
        )

    async def _request(
        self,
        client: TestClient,
        account_id: int,
        team_id: int,
        metrics: List[str],
        validFrom: date,
        expiresAt: date,
    ) -> dict:
        assert isinstance(metrics, list)
        body = {
            "query": self._query(),
            "variables": {
                "accountId": account_id,
                "teamId": team_id,
                "metrics": metrics,
                "validFrom": str(validFrom),
                "expiresAt": str(expiresAt),
            },
        }
        return await align_graphql_request(client, headers=DEFAULT_HEADERS, json=body)


@pytest.fixture(scope="function")
async def sample_teams(sdb: Database) -> None:
    id_values = [
        MappedJIRAIdentity(
            account_id=1,
            confidence=1,
            github_user_id=github_id,
            jira_user_id=jira_id,
        ).create_defaults().explode(with_primary_keys=True)
        for github_id, jira_id in (
            (40020, "5de5049e2c5dd20d0f9040c1"),
            (39789, "5dd58cb9c7ac480ee5674902"),
            (40191, "5ddec0b9be6c1f0d071ff82d"),
        )
    ]
    await gather(
        models_insert(
            sdb,
            TeamFactory(id=1, members=[40020, 39789, 40070]),
            TeamFactory(id=2, parent_id=1, members=[40191, 39926, 40418]),
        ),
        sdb.execute_many(insert(MappedJIRAIdentity), id_values),
    )


class TestMetricsSmoke(BaseMetricsTest):
    async def test_fetch_all_kinds(self, client: TestClient, sample_teams) -> None:
        res = await self._request(
            client, 1, 1, [
                PullRequestMetricID.PR_LEAD_TIME,
                ReleaseMetricID.RELEASE_PRS,
                JIRAMetricID.JIRA_RESOLVED,
            ],
            date(2016, 1, 1),
            date(2019, 1, 1),
        )
        assert res == {
            "data": {
                "metricsCurrentValues": [{
                    "metric": "pr-lead-time",
                    "value": {
                        "team": {
                            "id": 1,
                        },
                        "value": {"str": "2999016s", "int": None, "float": None},
                        "children": [{
                            "team": {
                                "id": 2,
                            },
                            "value": {"str": "4512690s", "int": None, "float": None},
                            "children": [],
                        }]}}, {
                    "metric": "release-prs",
                    "value": {
                        "team": {
                            "id": 1,
                        },
                        "value": {"str": None, "int": 324, "float": None},
                        "children": [{
                            "team": {
                                "id": 2,
                            },
                            "value": {"str": None, "int": 248, "float": None},
                            "children": [],
                        }]}}, {
                    "metric": "jira-resolved",
                    "value": {
                        "team": {
                            "id": 1,
                        },
                        "value": {"str": None, "int": 0, "float": None},
                        "children": [{
                            "team": {
                                "id": 2,
                            },
                            "value": {"str": None, "int": 0, "float": None},
                            "children": [],
                        }]}}]}}

    @pytest.mark.parametrize("metric, value", [
        ("pr-lead-time", {"str": "2999016s", "children": [{"str": "4512690s"}]}),
        ("release-prs", {"int": 324, "children": [{"int": 248}]}),
        ("jira-resolved", {"int": 0, "children": [{"int": 0}]}),
    ])
    async def test_fetch_one_kind(
            self, client: TestClient, sample_teams, metric: str, value: Dict[str, Any]) -> None:
        res = await self._request(
            client, 1, 1, [metric],
            date(2016, 1, 1),
            date(2019, 1, 1),
        )
        assert len(mv := res["data"]["metricsCurrentValues"]) == 1
        assert mv[0]["metric"] == metric

        def validate_recursively(yours: Dict[str, Any], mine: Dict[str, Any]) -> None:
            for key, val in mine.items():
                if isinstance(val, list):
                    for yours_sub, mine_sub in zip(yours[key], val):
                        validate_recursively(yours_sub, mine_sub)
                else:
                    assert yours["value"][key] == val

        validate_recursively(mv[0]["value"], value)

    async def test_fetch_two(self, client: TestClient, sample_teams):
        res = await self._request(
            client, 1, 1, [JIRAMetricID.JIRA_RESOLVED, JIRAMetricID.JIRA_RESOLUTION_RATE],
            date(2019, 1, 1),
            date(2022, 1, 1),
        )
        assert res == {
            "data": {
                "metricsCurrentValues": [{
                    "metric": "jira-resolved",
                    "value": {
                        "team": {"id": 1}, "value": {"str": None, "int": 738, "float": None},
                        "children": [{
                            "team": {"id": 2}, "value": {"str": None, "int": 163, "float": None},
                            "children": []},
                        ]},
                }, {
                    "metric": "jira-resolution-rate",
                    "value": {
                        "team": {"id": 1},
                        "value": {"str": None, "int": None, "float": 0.9473684430122375},
                        "children": [{
                            "team": {"id": 2},
                            "value": {"str": None, "int": None, "float": 0.8624338507652283},
                            "children": []},
                        ]},
                }],
            }}


class TestMetricsNasty(BaseMetricsTest):
    async def test_fetch_bad_team(self, client: TestClient) -> None:
        res = await self._request(
            client, 1, 1, [JIRAMetricID.JIRA_RESOLVED],
            date(2019, 1, 1),
            date(2022, 1, 1),
        )
        assert res == {
            "errors": [{
                "message": "Team not found",
                "locations": [{"line": 18, "column": 15}],
                "path": ["metricsCurrentValues"],
                "extensions": {
                    "status": 404,
                    "type": "/errors/teams/TeamNotFound",
                    "detail": "Team 1 not found or access denied",
                },
            }]}

    async def test_fetch_bad_account(self, client: TestClient, sample_teams) -> None:
        res = await self._request(
            client, 2, 1, [JIRAMetricID.JIRA_RESOLVED],
            date(2019, 1, 1),
            date(2022, 1, 1),
        )
        assert res == {
            "errors": [{
                "message": "Team not found",
                "locations": [{"line": 18, "column": 15}],
                "path": ["metricsCurrentValues"],
                "extensions": {
                    "status": 404,
                    "type": "/errors/teams/TeamNotFound",
                    "detail": "Team 1 not found or access denied",
                },
            }]}

    async def test_fetch_bad_metric(self, client: TestClient, sample_teams) -> None:
        res = await self._request(
            client, 1, 1, ["whatever"],
            date(2019, 1, 1),
            date(2022, 1, 1),
        )
        assert res == {
            "errors": [{
                "message": "Bad Request",
                "locations": [{"line": 18, "column": 15}],
                "path": ["metricsCurrentValues"],
                "extensions": {
                    "status": 400,
                    "type": "/errors/InvalidRequestError",
                    "detail": "The following metrics are not supported: whatever",
                },
            }]}

    async def test_fetch_bad_dates_order(self, client: TestClient, sample_teams) -> None:
        res = await self._request(
            client, 1, 1, [JIRAMetricID.JIRA_RESOLVED],
            date(2022, 1, 1),
            date(2019, 1, 1),
        )
        assert res == {
            "errors": [{
                "message": "Bad Request",
                "locations": [{"line": 18, "column": 15}],
                "path": ["metricsCurrentValues"],
                "extensions": {
                    "status": 400,
                    "type": "/errors/InvalidRequestError",
                    "detail": "validFrom must be less than or equal to expiresAt",
                },
            }]}

    async def test_fetch_bad_dates_future(self, client: TestClient, sample_teams) -> None:
        res = await self._request(
            client, 1, 1, [JIRAMetricID.JIRA_RESOLVED],
            date(2019, 1, 1),
            date(2129, 1, 1),
        )
        assert res["errors"]
        assert res.get("data") is None
