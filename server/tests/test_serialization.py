from datetime import date, datetime, timedelta, timezone
from typing import Optional

from freezegun import freeze_time
import numpy as np
import pandas as pd
from pandas.errors import OutOfBoundsDatetime
import pytest

from athenian.api.models.web.base_model_ import Model
from athenian.api.serialization import (
    FriendlyJson,
    ParseError,
    deserialize_date,
    deserialize_datetime,
    deserialize_model,
    serialize_timedelta,
)
from tests.testutils.time import dt


class TestDeserializeDate:
    @freeze_time("2022-04-01")
    def test_dserialize(self) -> None:
        dt = deserialize_date("2022-10-23")
        assert dt == date(2022, 10, 23)

    @freeze_time("2022-04-01")
    def test_out_of_bounds(self) -> None:
        for out_of_bounds_val in ("1492-01-12", "2025-04-02"):
            with pytest.raises(OutOfBoundsDatetime):
                deserialize_date(out_of_bounds_val)


class TestDeserializeDatetime:
    @freeze_time("2022-04-01")
    def test_deserialize(self) -> None:
        dt = deserialize_datetime("2022-10-23T10:54:32Z")
        assert dt == datetime(2022, 10, 23, 10, 54, 32, tzinfo=timezone.utc)

    @freeze_time("2022-01-01")
    def test_deserialize_datatime_out_of_bounds(self) -> None:
        for out_of_bounds_val in ("2024-10-10T12:20:20Z", "1984-01-10T12:20:20Z"):
            with pytest.raises(OutOfBoundsDatetime):
                deserialize_datetime(out_of_bounds_val)

    @freeze_time("2022-01-01")
    def test_custom_bounds(self) -> None:
        assert deserialize_datetime("1901-07-30T05:00:00Z", min_=dt(1850, 1, 1)) == dt(
            1901, 7, 30, 5,
        )
        with pytest.raises(OutOfBoundsDatetime):
            deserialize_datetime("1901-07-30T05:00:00Z", min_=dt(1950, 1, 1))

        assert deserialize_datetime("2070-01-30T05:00:00Z", max_future_delta=None) == dt(
            2070, 1, 30, 5,
        )
        assert deserialize_datetime(
            "2022-01-30T05:00:00Z", max_future_delta=timedelta(days=100),
        ) == dt(2022, 1, 30, 5)

        with pytest.raises(OutOfBoundsDatetime):
            deserialize_datetime("2022-01-30T05:00:00Z", max_future_delta=timedelta(days=10))

    @freeze_time("2022-01-01")
    def test_with_timezone_name(self) -> None:
        for s in ("2022-07-05 23:14:11 GMT", "2022-07-05 23:14:11 MET", "2022-07-05 23:14:11 WST"):
            with pytest.raises(ValueError):
                deserialize_datetime(s)

    @freeze_time("2022-01-01")
    def test_timezone_offset(self) -> None:
        s = "2022-07-05 23:14:11+03:00"
        assert deserialize_datetime(s) == dt(2022, 7, 5, 20, 14, 11)

    @freeze_time("2022-01-01")
    def test_parse_naive_datetime(self) -> None:
        s = "2022-07-05 23:14:11"
        assert deserialize_datetime(s) == datetime(2022, 7, 5, 23, 14, 11, tzinfo=None)


class TestDeserializeModel:
    def test_attribute_map(self) -> None:
        class T(Model):
            attribute_types = {"a": int, "b": str}
            attribute_map = {"b": "b_mapped"}

        data = {"a": 123, "b_mapped": "foo"}

        t: T = deserialize_model(data, T)
        assert t.a == 123
        assert t.b == "foo"

    def test_union_field(self) -> None:
        class T(Model):
            a: int | str

        t: T = deserialize_model({"a": 42}, T)
        assert t.a == 42

        t = deserialize_model({"a": "foo"}, T)
        assert t.a == "foo"

    def test_optional_field(self) -> None:
        class T(Model):
            a: Optional[date]

        t: T = deserialize_model({"a": "2001-01-01"}, T)
        assert t.a == date(2001, 1, 1)
        with pytest.raises(ParseError):
            deserialize_model({"a": "foo"}, T)


class TestFriendlyJson:
    def test_serialize_date(self) -> None:
        obj = date(1984, 5, 1)
        assert FriendlyJson.dumps(obj) == '"1984-05-01"'

    def test_serialize_timedelta(self) -> None:
        td = timedelta(hours=2, seconds=2.2)
        assert serialize_timedelta(td) == "7202s"

        td = timedelta(seconds=0.9)
        assert serialize_timedelta(td) == "0s"

    def test_serialize_datetime(self) -> None:
        obj = [
            pd.Timestamp(0, tzinfo=timezone.utc),
            pd.NaT,
            datetime(year=2020, month=3, day=18, tzinfo=timezone.utc),
            date(year=2020, month=3, day=18),
            pd.Timedelta(minutes=1),
            timedelta(seconds=1),
        ]
        s = FriendlyJson.dumps(obj)
        assert (
            s
            == '["1970-01-01T00:00:00Z", null, "2020-03-18T00:00:00Z", "2020-03-18", "60s", "1s"]'
        )

    def test_serialize_datetime_no_utc(self) -> None:
        obj = [pd.Timestamp(0)]
        with pytest.raises(AssertionError):
            FriendlyJson.dumps(obj)

    def test_serialize_numpy(self) -> None:
        assert FriendlyJson.serialize(np.int64(7)) == 7
        assert FriendlyJson.serialize(np.double(7.5)) == 7.5
