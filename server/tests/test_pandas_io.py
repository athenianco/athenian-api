from datetime import datetime, timezone

from pandas.testing import assert_frame_equal

from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.check_run import mine_check_runs
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.pandas_io import deserialize_df, serialize_df


async def test_zero_rows(mdb):
    df = await mine_check_runs(
        datetime(2020, 10, 10, tzinfo=timezone.utc),
        datetime(2020, 10, 23, tzinfo=timezone.utc),
        ["src-d/go-git"],
        [],
        LabelFilter.empty(),
        JIRAFilter.empty(),
        LogicalRepositorySettings.empty(),
        (6366825,),
        mdb,
        None,
    )
    assert df.empty
    assert_frame_equal(df, deserialize_df(serialize_df(df)))
