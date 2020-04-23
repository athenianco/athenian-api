from datetime import date, datetime, timezone
from typing import List

from athenian.api.controllers.features.github.pull_request_filter import PullRequestListMiner
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, Property, \
    PullRequestListItem


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
