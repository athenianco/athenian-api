from datetime import date, datetime, timedelta, timezone
import json
import typing
from typing import Optional

import dateutil.parser
from dateutil.tz import tzutc
import numpy as np

from athenian.api import typing_utils

T = typing.TypeVar("T")
Class = typing.Type[T]
_reliable_datetime_types = (date | datetime).__args__  # resist freezegun's monkey-patching


class ParseError(ValueError):
    """Value parsing error that is raised in the functions below."""

    def __init__(self, message: str, path: str):
        """Initialize a new instance of ParseError."""
        super().__init__(message)
        self.path = path


def _deserialize(
    data: dict | list | str,
    klass: Class,
    path: str,
) -> dict | list | Class | int | float | str | bool | date | datetime | timedelta:
    """Deserializes dict, list, str into an object.

    :param data: dict, list or str.
    :param klass: class literal, or string of class name.

    :return: object.
    """
    if data is None:
        return None

    try:
        if klass is int or klass is float or klass is str or klass is bool:
            return _deserialize_primitive(data, klass)
        elif klass is object or klass is dict:
            return _deserialize_object(data)
        elif klass is _reliable_datetime_types[0]:
            return deserialize_date(data)  # type: ignore
        elif klass is _reliable_datetime_types[1]:
            return deserialize_datetime(data)  # type: ignore
        elif klass is timedelta:
            return deserialize_timedelta(data)  # type: ignore
        elif typing_utils.is_generic(klass):
            if typing_utils.is_list(klass):
                return _deserialize_list(data, klass.__args__[0], path)  # type: ignore
            elif typing_utils.is_dict(klass):
                return _deserialize_dict(data, klass.__args__[1], path)  # type: ignore
            # optional is also a union, must stand the first
            elif typing_utils.is_optional(klass):
                return _deserialize(data, klass.__args__[0], path)
            elif typing_utils.is_union(klass):
                return _deserialize_union_type(data, klass, path)
            raise ValueError(f"Invalid generic klass {klass}")
        else:
            return deserialize_model(data, klass, path)  # type: ignore
    except ParseError as e:
        raise e from None
    except Exception as e:
        if klass in (date, datetime):
            klass_str = "RFC 3339 datetime (https://ijmacd.github.io/rfc3339-iso8601)"
        else:
            klass_str = str(klass)
        raise ParseError(f"Failed to parse {data} as {klass_str}: {e}", path) from e


def _deserialize_union_type(
    data,
    klass: Class,
    path: str,
) -> dict | list | Class | int | float | str | bool | date | datetime | timedelta:
    # return the data with no conversion if it already has one of the accepted types;
    # this allows to preserve the received number informal "subtype" from JSON when
    # the union type is `float | int`
    if data.__class__ in klass.__args__:
        return data

    for arg in klass.__args__:
        try:
            return _deserialize(data, arg, path)
        except (ValueError, TypeError):
            continue
    raise ValueError(f"None of the union options fit: {klass.__args__}")


def _deserialize_primitive(data, klass: Class) -> Class | int | float | str | bool:
    """Deserializes to primitive type.

    :param data: data to deserialize.
    :param klass: class literal.

    :return: int, float, str, bool.
    """
    try:
        value = klass(data)
    except (UnicodeEncodeError, TypeError):
        value = data
    return value


def _deserialize_object(value: T) -> T:
    """Return an original value.

    :return: object.
    """
    return value


# default bounds for deserialize_date and deserialize_datetime
_DATETIME_MIN = datetime(2000, 1, 1)
_DATE_MIN = _DATETIME_MIN.date()
_DATETIME_MAX_FUTURE_DELTA = timedelta(days=365) * 2


def deserialize_date(
    string: str,
    *,
    min_: Optional[date] = _DATE_MIN,
    max_future_delta: Optional[timedelta] = _DATETIME_MAX_FUTURE_DELTA,
) -> date:
    """Deserializes string to date.

    Using `min_` and `max_future_delta` parameters the date can be validated
    for inclusion in the given bounds.
    """
    d = dateutil.parser.parse(string, ignoretz=True, yearfirst=True).date()
    if min_ is not None and d < min_:
        raise ValueError(f"{d} is too far in the past")
    if max_future_delta is not None and d > date.today() + max_future_delta:
        raise ValueError(f"{d} is too far in the future")
    return d


def deserialize_datetime(
    string: str,
    *,
    min_: Optional[datetime] = _DATETIME_MIN,
    max_future_delta: Optional[timedelta] = _DATETIME_MAX_FUTURE_DELTA,
) -> datetime:
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.
    Using `min_` and `max_future_delta` parameters the datetime can be validated
    for inclusion in the given bounds.
    """
    dt = dateutil.parser.isoparse(string)

    if min_ is not None and dt < min_.replace(tzinfo=dt.tzinfo):
        raise ValueError(f"{dt} is too far in the past")
    if max_future_delta is not None and dt > datetime.now(dt.tzinfo) + max_future_delta:
        raise ValueError(f"{dt} is too far in the future")
    return dt


def deserialize_timedelta(string: str) -> timedelta:
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :return: datetime.
    """
    if not string.endswith("s"):
        raise ValueError(f"Unsupported timedelta format: {string}")
    seconds = int(string[:-1])
    try:
        np.timedelta64(seconds, "s")
    except OverflowError as e:
        raise ValueError(f"Unsupported timedelta: {seconds}") from e
    return timedelta(seconds=seconds)


def deserialize_model(data: dict, klass: typing.Type[T], path: str = "") -> T | dict:
    """Deserializes dict to model.

    :param data: dict that represents the serialized model.
    :param klass: class literal.
    :param path: request body path.
    :return: model object.
    """
    if not getattr(klass, "attribute_types", False):
        return data

    instance = klass.__new__(klass)
    if data is not None and isinstance(data, dict):
        for attr, attr_type in klass.attribute_types.items():  # type: ignore
            attr_key = klass.attribute_map.get(attr, attr)  # type: ignore
            if attr_key in data:
                value = _deserialize(data[attr_key], attr_type, f"{path}.{attr}")
            else:
                value = klass.default_values.get(attr, None)  # type: ignore
            setattr(instance, attr, value)

    return instance


def _deserialize_list(data: list, boxed_type, path: str) -> list:
    """Deserializes a list and its elements.

    :param data: list to deserialize.
    :param boxed_type: class literal.

    :return: deserialized list.
    """
    return [_deserialize(item, boxed_type, f"{path}[{index}]") for index, item in enumerate(data)]


def _deserialize_dict(data: dict, boxed_type, path: str) -> dict:
    """Deserializes a dict and its elements.

    :param data: dict to deserialize.
    :param boxed_type: class literal.

    :return: deserialized dict.
    """
    return {k: _deserialize(v, boxed_type, f"{path}.{k}") for k, v in data.items()}


def serialize_date(d: date) -> str:
    """Serialize a date object into a string."""
    return d.isoformat()


def serialize_timedelta(td: timedelta) -> str:
    """Serialize a timedelta object into a string."""
    seconds = int(td.total_seconds())
    return f"{seconds}s"


class FriendlyJson:
    """Allows to serialize datetime.datetime and datetime.date to JSON."""

    @classmethod
    def dumps(klass, data, **kwargs):
        """Wrap json.dumps to str() unsupported objects."""
        if "cls" in kwargs:
            del kwargs["cls"]
        try:
            return json.dumps(data, default=klass.serialize, **kwargs)
        except AssertionError as e:
            e.args += (data,)
            raise e from None

    loads = staticmethod(json.loads)

    @classmethod
    def serialize(klass, obj):
        """Format timedeltas and dates according to https://athenianco.atlassian.net/browse/ENG-125
        """  # noqa
        if isinstance(obj, (timedelta, np.timedelta64)):
            if isinstance(obj, np.timedelta64):
                obj = obj.astype("timedelta64[s]").item()
            return serialize_timedelta(obj)
        if isinstance(obj, datetime):
            if obj != obj:
                # NaT
                return None
            tz = obj.tzinfo
            assert tz == timezone.utc or tz == tzutc(), "all timestamps must be UTC: %s" % obj
            return obj.strftime("%Y-%m-%dT%H:%M:%SZ")  # RFC3339
        if isinstance(obj, np.datetime64):
            if obj != obj:
                # NaT
                return None
            return np.datetime_as_string(obj, unit="s", timezone="UTC")
        if isinstance(obj, date):
            return serialize_date(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode()
        try:
            assert obj == obj, "%s: %s" % (type(obj), obj)
        except ValueError as e:
            raise ValueError("%s: %s: %s" % (e, type(obj), obj)) from None
        return str(obj)
