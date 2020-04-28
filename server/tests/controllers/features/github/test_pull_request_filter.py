from datetime import date, datetime, timezone
from typing import List

from athenian.api.controllers.features.entries import calc_pull_request_metrics_line_github
from athenian.api.controllers.features.github.pull_request_filter import PullRequestListMiner
from athenian.api.controllers.miners.github.pull_request import PullRequestTimesMiner
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, Property, \
    PullRequestListItem
from athenian.api.models.web import PullRequestMetricID


async def test_pr_list_miner_none(mdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestListMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    prs = list(miner)
    assert not prs


async def test_pr_list_miner_match_participants(mdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestListMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    miner.properties = set(Property)
    miner.participants = {ParticipationKind.AUTHOR: ["github.com/mcuadros", "github.com/smola"],
                          ParticipationKind.COMMENTER: ["github.com/mcuadros"]}
    prs = list(miner)  # type: List[PullRequestListItem]
    assert prs
    for pr in prs:
        mcuadros_is_author = "github.com/mcuadros" in pr.participants[ParticipationKind.AUTHOR]
        smola_is_author = "github.com/smola" in pr.participants[ParticipationKind.AUTHOR]
        mcuadros_is_only_commenter = (
            ("github.com/mcuadros" in pr.participants[ParticipationKind.COMMENTER])
            and  # noqa
            (not mcuadros_is_author)
            and  # noqa
            (not smola_is_author)
        )
        assert mcuadros_is_author or smola_is_author or mcuadros_is_only_commenter


async def test_pr_list_miner_no_participants(mdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestListMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    miner.properties = set(Property)
    prs = list(miner)
    assert prs


async def test_pr_list_miner_match_metrics_all_count(mdb, release_match_setting_tag):
    date_from = date(year=2018, month=1, day=1)
    date_to = date(year=2019, month=1, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    miner = await PullRequestListMiner.mine(
        date_from,
        date_to,
        time_from,
        time_to,
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    miner.time_from = time_from
    miner.time_to = time_to
    miner.properties = set(Property)
    prs = list(miner)
    PullRequestTimesMiner.hack = True
    metric = (await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_to]],
        ["src-d/go-git"], release_match_setting_tag, [], mdb, None,
    ))[0][0][0]
    assert len(prs) == metric.value
