import numpy as np
import pandas as pd

from athenian.api.internal.features.github.pull_request_filter import (
    PullRequestListMiner,
    pr_facts_stages_masks,
    pr_stages_mask,
)
from athenian.api.internal.miners.types import PullRequestEvent, PullRequestFacts, PullRequestStage
from athenian.api.models.web import PullRequestStage as WebPullRequestStage
from tests.testutils.factory.miners import PullRequestFactsFactory
from tests.testutils.time import dt


class TestGetPRFactsStagesMasks:
    """Tests for the function pr_facts_stages_masks."""

    def test_smoke(self) -> None:
        t0 = dt(2001, 1, 1).replace(tzinfo=None)
        prs_facts = self._mk_facts(
            (True, False, False, t0, None, np.datetime64("NAT")),
            (True, False, True, None, None, np.datetime64("NAT")),
            (False, False, False, None, None, np.datetime64("NAT")),
            (False, False, False, None, t0, t0),
            (False, False, False, None, None, t0),
        )

        masks = pr_facts_stages_masks(prs_facts)
        assert len(masks) == len(prs_facts)
        self._assert_mask_stages(masks[0], prs_facts.iloc[0], PullRequestStage.DONE)
        self._assert_mask_stages(
            masks[1], prs_facts.iloc[1], PullRequestStage.DONE, PullRequestStage.RELEASE_IGNORED,
        )
        self._assert_mask_stages(masks[2], prs_facts.iloc[2], PullRequestStage.WIP)
        self._assert_mask_stages(masks[3], prs_facts.iloc[3], PullRequestStage.MERGING)
        self._assert_mask_stages(masks[4], prs_facts.iloc[4], PullRequestStage.REVIEWING)

    @classmethod
    def _mk_facts(cls, *rows: tuple) -> pd.DataFrame:
        columns = [
            PullRequestFacts.f.done,
            PullRequestFacts.f.force_push_dropped,
            PullRequestFacts.f.release_ignored,
            PullRequestFacts.f.merged,
            PullRequestFacts.f.approved,
            PullRequestFacts.f.first_review_request,
        ]
        df = pd.DataFrame.from_records(rows, columns=columns)
        df.merged = df.merged.astype(np.dtype("datetime64[s]"))
        df.approved = df.approved.astype(np.dtype("datetime64[s]"))
        df.first_review_request = df.first_review_request.astype(np.dtype("datetime64[s]"))
        return df

    @classmethod
    def _assert_mask_stages(
        cls,
        mask: int,
        pr_series: pd.Series,
        *stages: PullRequestStage,
    ) -> None:
        expected_mask = 0
        for stage in stages:
            expected_mask |= 1 << (stage.value - 1)
        assert expected_mask == mask

        # compare against PullRequestListMiner._collect_events_and_stages(
        # pr_fact = PullRequestFactsFactory(**pr_series.to_dict())
        pr_facts = PullRequestFactsFactory(
            **{
                f: getattr(pr_series, f) for f in ("done", "force_push_dropped", "release_ignored")
            },
            **{
                f: None if pd.isnull(v := getattr(pr_series, f)) else v
                for f in ("merged", "approved", "first_review_request")
            },
        )
        hard_events = {f: False for f in PullRequestEvent}
        collected_stages = PullRequestListMiner._collect_events_and_stages(
            pr_facts, hard_events, np.datetime64("1970-01-01", "s"),
        )[1]
        assert sorted(stages) == sorted(collected_stages)


class TestPRStagesMask:
    def test_smoke(self) -> None:
        mask = pr_stages_mask((WebPullRequestStage.WIP, WebPullRequestStage.DONE))
        assert mask == 1 | (1 << 5)
        assert pr_stages_mask((WebPullRequestStage.WIP, WebPullRequestStage.DONE)) == mask
