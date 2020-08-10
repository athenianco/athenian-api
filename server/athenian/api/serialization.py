import datetime
import json
import typing
from typing import Union

from dateutil.parser import parse as parse_datetime
from dateutil.tz import tzutc

from athenian.api import typing_utils

T = typing.TypeVar("T")
Class = typing.Type[T]


class ParseError(ValueError):
    """Value parsing error that is raised in the functions below."""


def _deserialize(
    data: Union[dict, list, str], klass: Union[Class, str],
) -> Union[dict, list, Class, int, float, str, bool, datetime.date, datetime.datetime,
           datetime.timedelta]:
    """Deserializes dict, list, str into an object.

    :param data: dict, list or str.
    :param klass: class literal, or string of class name.

    :return: object.
    """
    if data is None:
        return None

    if klass in (int, float, str, bool):
        return _deserialize_primitive(data, klass)
    elif klass == object:
        return _deserialize_object(data)
    elif klass == datetime.date:
        return deserialize_date(data)
    elif klass == datetime.datetime:
        return deserialize_datetime(data)
    elif klass == datetime.timedelta:
        return deserialize_timedelta(data)
    elif typing_utils.is_generic(klass):
        if typing_utils.is_list(klass):
            return _deserialize_list(data, klass.__args__[0])
        if typing_utils.is_dict(klass):
            return _deserialize_dict(data, klass.__args__[1])
        if typing_utils.is_optional(klass):
            return _deserialize(data, klass.__args__[0])
    else:
        return deserialize_model(data, klass)


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
    try:
        return parse_datetime(string, ignoretz=True).date()
    except Exception as e:
        raise ParseError(string) from e


def deserialize_datetime(string: str) -> datetime.datetime:
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :return: datetime.
    """
    try:
        return parse_datetime(string)
    except Exception as e:
        raise ParseError(string) from e


def deserialize_timedelta(string: str) -> datetime.timedelta:
    """Deserializes string to datetime.

    The string should be in iso8601 datetime format.

    :param string: str.
    :return: datetime.
    """
    if not string.endswith("s"):
        raise ParseError("Unsupported timedelta format: " + string)
    try:
        return datetime.timedelta(seconds=int(string[:-1]))
    except Exception as e:
        raise ParseError(string) from e


def deserialize_model(data: Union[dict, list], klass: Class) -> T:
    """Deserializes list or dict to model.

    :param data: dict, list.
    :param klass: class literal.
    :return: model object.
    """
    instance = klass()

    if not instance.openapi_types:
        return data

    if data is not None and isinstance(data, (list, dict)):
        for attr, attr_type in instance.openapi_types.items():
            attr_key = instance.attribute_map[attr]
            if attr_key in data:
                value = data[attr_key]
                setattr(instance, attr, _deserialize(value, attr_type))

    return instance


def _deserialize_list(data: list, boxed_type) -> list:
    """Deserializes a list and its elements.

    :param data: list to deserialize.
    :param boxed_type: class literal.

    :return: deserialized list.
    """
    return [_deserialize(sub_data, boxed_type) for sub_data in data]


def _deserialize_dict(data: dict, boxed_type) -> dict:
    """Deserializes a dict and its elements.

    :param data: dict to deserialize.
    :param boxed_type: class literal.

    :return: deserialized dict.
    """
    return {k: _deserialize(v, boxed_type) for k, v in data.items()}


class FriendlyJson:
    """Allows to serialize datetime.datetime and datetime.date to JSON."""

    @classmethod
    def dumps(klass, data, **kwargs):
        """Wrap json.dumps to str() unsupported objects."""
        if "cls" in kwargs:
            del kwargs["cls"]
        return json.dumps(data, default=klass.serialize, **kwargs)

    loads = staticmethod(json.loads)

    @staticmethod
    def serialize(obj):
        """Format timedeltas and dates according to https://athenianco.atlassian.net/browse/ENG-125"""  # noqa
        if isinstance(obj, datetime.timedelta):
            return "%ds" % obj.total_seconds()
        if isinstance(obj, datetime.datetime):
            if obj != obj:
                # NaT
                return None
            tz = obj.tzinfo
            assert tz == datetime.timezone.utc or tz == tzutc(), "all timestamps must be UTC"
            return obj.strftime("%Y-%m-%dT%H:%M:%SZ")  # RFC3339
        if isinstance(obj, datetime.date):
            return obj.strftime("%Y-%m-%d")
        try:
            assert obj == obj, "%s: %s" % (type(obj), obj)
        except ValueError as e:
            raise ValueError("%s: %s: %s" % (e, type(obj), obj)) from None
        return str(obj)
