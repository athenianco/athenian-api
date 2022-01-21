import datetime
import json
import typing
from typing import Union

from dateutil.parser import parse as parse_datetime
from dateutil.tz import tzutc
import numpy as np
import pandas as pd

from athenian.api import typing_utils

T = typing.TypeVar("T")
Class = typing.Type[T]


class ParseError(ValueError):
    """Value parsing error that is raised in the functions below."""

    def __init__(self, message: str, path: str):
        """Initialize a new instance of ParseError."""
        super().__init__(message)
        self.path = path


def _deserialize(
    data: Union[dict, list, str], klass: Union[Class, str], path: str,
) -> Union[dict, list, Class, int, float, str, bool, datetime.date, datetime.datetime,
           datetime.timedelta]:
    """Deserializes dict, list, str into an object.

    :param data: dict, list or str.
    :param klass: class literal, or string of class name.

    :return: object.
    """
    if data is None:
        return None

    try:
        if klass in (int, float, str, bool):
            return _deserialize_primitive(data, klass)
        elif klass in (object, dict):
            return _deserialize_object(data)
        elif klass == datetime.date:
            return deserialize_date(data)
        elif klass == datetime.datetime:
            return deserialize_datetime(data)
        elif klass == datetime.timedelta:
            return deserialize_timedelta(data)
        elif typing_utils.is_generic(klass):
            if typing_utils.is_list(klass):
                return _deserialize_list(data, klass.__args__[0], path)
            if typing_utils.is_dict(klass):
                return _deserialize_dict(data, klass.__args__[1], path)
            if typing_utils.is_optional(klass):
                return _deserialize(data, klass.__args__[0], path)
            if typing_utils.is_union(klass):
                for arg in klass.__args__:
                    try:
                        return _deserialize(data, arg, path)
                    except (ValueError, TypeError):
                        continue
        else:
            return deserialize_model(data, klass, path)
    except Exception as e:
        raise ParseError(f"Failed to parse {data} as {klass}: {e}", path) from e


def _deserialize_primitive(data, klass: Class) -> Union[Class, int, float, str, bool]:
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


def deserialize_date(string: str) -> datetime.date:
    """Deserializes string to date.

    :param string: str.
    :return: date.
    """
    dt = parse_datetime(string, ignoretz=True, yearfirst=True).date()
    if dt < datetime.date(2000, 1, 1):
        raise pd.errors.OutOfBoundsDatetime(f"{dt} is too far in the past")
    if dt > datetime.date.today() + datetime.timedelta(days=365):
        raise pd.errors.OutOfBoundsDatetime(f"{dt} is too far in the future")
    return dt


def deserialize_datetime(string: str) -> datetime.datetime:
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :return: datetime.
    """
    dt = parse_datetime(string, yearfirst=True)
    if dt < datetime.datetime(2000, 1, 1, tzinfo=dt.tzinfo):
        raise pd.errors.OutOfBoundsDatetime(f"{dt} is too far in the past")
    if dt > datetime.datetime.now(dt.tzinfo) + datetime.timedelta(days=365):
        raise pd.errors.OutOfBoundsDatetime(f"{dt} is too far in the future")
    return dt


def deserialize_timedelta(string: str) -> datetime.timedelta:
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :return: datetime.
    """
    if not string.endswith("s"):
        raise ValueError("Unsupported timedelta format: " + string)
    pd.Timedelta(td := datetime.timedelta(seconds=int(string[:-1])))
    return td


def deserialize_model(data: dict, klass: Class, path: str = "") -> T:
    """Deserializes dict to model.

    :param data: dict that represents the serialized model.
    :param klass: class literal.
    :param path: request body path.
    :return: model object.
    """
    instance = klass()

    if not instance.openapi_types:
        return data

    if data is not None and isinstance(data, dict):
        for attr, attr_type in instance.openapi_types.items():
            attr_key = instance.attribute_map[attr]
            if attr_key in data:
                value = data[attr_key]
                setattr(instance, attr, _deserialize(value, attr_type, f"{path}.{attr}"))

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
        """Format timedeltas and dates according to https://athenianco.atlassian.net/browse/ENG-125"""  # noqa
        if isinstance(obj, (datetime.timedelta, np.timedelta64)):
            if isinstance(obj, np.timedelta64):
                obj = obj.astype("timedelta64[s]").item()
            return "%ds" % obj.total_seconds()
        if isinstance(obj, datetime.datetime):
            if obj != obj:
                # NaT
                return None
            tz = obj.tzinfo
            assert tz == datetime.timezone.utc or tz == tzutc(), \
                "all timestamps must be UTC: %s" % obj
            return obj.strftime("%Y-%m-%dT%H:%M:%SZ")  # RFC3339
        if isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        try:
            assert obj == obj, "%s: %s" % (type(obj), obj)
        except ValueError as e:
            raise ValueError("%s: %s: %s" % (e, type(obj), obj)) from None
        return str(obj)
