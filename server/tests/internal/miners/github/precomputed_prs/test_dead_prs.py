from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from athenian.api.internal.miners.github.precomputed_prs.dead_prs import (
    load_undead_prs,
    store_undead_prs,
)
from athenian.api.models.precomputed.models import GitHubRebasedPullRequest


async def test_round_trip(pdb):
    ghrpr = GitHubRebasedPullRequest
    prs = pd.DataFrame(
        {
            ghrpr.pr_node_id.name: [1, 2, 1],
            ghrpr.matched_merge_commit_sha.name: np.array([b"0" * 40, b"1" * 40, b"2" * 40]),
            ghrpr.matched_merge_commit_id.name: [4, 5, 6],
            ghrpr.matched_merge_commit_committed_date.name: [
                pd.Timestamp(datetime(2020, 1, 1, tzinfo=timezone.utc)),
                pd.Timestamp(datetime(2020, 2, 1, tzinfo=timezone.utc)),
                pd.Timestamp(datetime(2020, 3, 1, tzinfo=timezone.utc)),
            ],
            ghrpr.matched_merge_commit_pushed_date.name: [
                pd.Timestamp(datetime(2020, 1, 1, tzinfo=timezone.utc)),
                pd.Timestamp(datetime(2020, 2, 1, tzinfo=timezone.utc)),
                pd.Timestamp("NaT"),
            ],
        },
    )
    await store_undead_prs(prs, 1, pdb)
    loaded = await load_undead_prs([1, 2], 1, pdb)
    loaded.sort_values(ghrpr.pr_node_id.name, inplace=True, ignore_index=True)
    prs = prs.iloc[[2, 1]].reset_index(drop=True)
    prs[ghrpr.acc_id.name] = np.ones(2, dtype=np.int32)
    prs[ghrpr.updated_at.name] = loaded[ghrpr.updated_at.name]
    prs = prs.reindex(sorted(prs.columns), axis=1)
    assert_frame_equal(prs, loaded.reindex(sorted(loaded.columns), axis=1))
