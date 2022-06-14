from datetime import date, datetime, timedelta, timezone

from freezegun import freeze_time
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
import pytest

from athenian.api.serialization import (
    FriendlyJson,
    deserialize_date,
    deserialize_datetime,
    serialize_timedelta,
)


@freeze_time("2022-04-01")
def test_deserialize_date() -> None:
    dt = deserialize_date("2022-10-23")
    assert dt == date(2022, 10, 23)


@freeze_time("2022-04-01")
def test_deserialize_date_out_of_bounds() -> None:
    for out_of_bounds_val in ("1492-01-12", "2023-08-24"):
        with pytest.raises(OutOfBoundsDatetime):
            deserialize_date(out_of_bounds_val)


@freeze_time("2022-04-01")
def test_deserialize_datetime() -> None:
    dt = deserialize_datetime("2022-10-23T10:54:32Z")
    assert dt == datetime(2022, 10, 23, 10, 54, 32, tzinfo=timezone.utc)


@freeze_time("2022-01-01")
def test_deserialize_datatime_out_of_bounds() -> None:
    for out_of_bounds_val in ("2024-10-10T12:20:20Z", "1984-01-10T12:20:20Z"):
        with pytest.raises(OutOfBoundsDatetime):
            deserialize_datetime(out_of_bounds_val)


@freeze_time("2022-01-01")
def test_deserialize_datatime_custom_bounds() -> None:
    dt = datetime
    assert deserialize_datetime("1901-07-30T05:00:00", min_=dt(1850, 1, 1)) == dt(1901, 7, 30, 5)
    with pytest.raises(OutOfBoundsDatetime):
        deserialize_datetime("1901-07-30T05:00:00", min_=dt(1950, 1, 1))

    assert deserialize_datetime("2070-01-30T05:00:00", max_future_delta=None) == dt(2070, 1, 30, 5)
    assert deserialize_datetime("2022-01-30T05:00:00", max_future_delta=timedelta(days=100)) == dt(
        2022, 1, 30, 5
    )

    with pytest.raises(OutOfBoundsDatetime):
        deserialize_datetime("2022-01-30T05:00:00", max_future_delta=timedelta(days=10))


def test_serialize_date() -> None:
    obj = date(1984, 5, 1)
    assert FriendlyJson.dumps(obj) == '"1984-05-01"'


def test_serialize_timedelta() -> None:
    td = timedelta(hours=2, seconds=2.2)
    assert serialize_timedelta(td) == "7202s"

    td = timedelta(seconds=0.9)
    assert serialize_timedelta(td) == "0s"


def test_serialize_datetime() -> None:
    obj = [
        pd.Timestamp(0, tzinfo=timezone.utc),
        pd.NaT,
        datetime(year=2020, month=3, day=18, tzinfo=timezone.utc),
        date(year=2020, month=3, day=18),
        pd.Timedelta(minutes=1),
        timedelta(seconds=1),
    ]
    s = FriendlyJson.dumps(obj)
    assert s == '["1970-01-01T00:00:00Z", null, "2020-03-18T00:00:00Z", "2020-03-18", "60s", "1s"]'


def test_serialize_datetime_no_utc() -> None:
    obj = [pd.Timestamp(0)]
    with pytest.raises(AssertionError):
        FriendlyJson.dumps(obj)


def test_serialize_numpy() -> None:
    assert FriendlyJson.serialize(np.int64(7)) == 7
    assert FriendlyJson.serialize(np.double(7.5)) == 7.5
