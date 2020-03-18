from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from athenian.api import FriendlyJson


def test_serialize_datetime():
    obj = [
        pd.Timestamp(0, tzinfo=timezone.utc),
        pd.NaT,
        datetime(year=2020, month=3, day=18, tzinfo=timezone.utc),
        date.today(),
        pd.Timedelta(minutes=1),
        timedelta(seconds=1),
    ]
    s = FriendlyJson.dumps(obj)
    assert s == '["1970-01-01T00:00:00Z", null, "2020-03-18T00:00:00Z", "2020-03-18", "60s", "1s"]'


def test_serialize_datetime_no_utc():
    obj = [pd.Timestamp(0)]
    with pytest.raises(AssertionError):
        FriendlyJson.dumps(obj)
