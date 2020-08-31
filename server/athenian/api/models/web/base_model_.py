from abc import ABCMeta
from collections.abc import Mapping
import pprint
import typing

from athenian.api import serialization, typing_utils

T = typing.TypeVar("T")


class Slots(ABCMeta):
    """Set __slots__ according to the declared `openapi_types`."""

    def __new__(mcs, name, bases, dikt):
        """Insert __slots__ to the class' __dict__."""
        try:
            openapi_types = dikt["openapi_types"]
        except KeyError:
            for base in bases:
                try:
                    openapi_types = base.openapi_types
                except AttributeError:
                    continue
                else:
                    break
        dikt["__slots__"] = ["_" + k for k in openapi_types]
        return type.__new__(mcs, name, bases, dikt)

    def __instancecheck__(cls, instance):
        """Override for isinstance(instance, cls)."""
        return type.__instancecheck__(cls, instance)

    def __subclasscheck__(cls, subclass):
        """Override for issubclass(subclass, cls)."""
        return type.__subclasscheck__(cls, subclass)


class Model(metaclass=Slots):
    """
    Base API model class. Handles object -> {} and {} -> object transformations.

    Ojo: you should not rename the file to stay compatible with the generated code.
    """

    # openapiTypes: The key is attribute name and the
    # value is attribute type.
    openapi_types = {}

    # attributeMap: The key is attribute name and the
    # value is json key in definition.
    attribute_map = {}

    @classmethod
    def from_dict(cls: typing.Type[T], dikt: dict) -> T:
        """Returns the dict as a model."""
        return serialization.deserialize_model(dikt, cls)

    def to_dict(self) -> dict:
        """Returns the model properties as a dict."""
        result = {}

        for attr_key, json_key in self.attribute_map.items():
            value = getattr(self, attr_key)
            if value is None and typing_utils.is_optional(self.openapi_types[attr_key]):
                continue
            if isinstance(value, list):
                result[json_key] = list(
                    map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value)  # noqa(C812)
                )
            elif hasattr(value, "to_dict"):
                result[json_key] = value.to_dict()
            elif isinstance(value, dict):
                result[json_key] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )  # noqa(C812)
                )
            else:
                result[json_key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model."""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For debugging."""
        return "%s(%s)" % (type(self).__name__, ", ".join(
            "%s=%r" % p for p in self.to_dict().items()))

    def __str__(self):
        """For `print` and `pprint`."""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal."""
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal."""
        return not self == other


class Enum(Slots):
    """Trivial enumeration metaclass."""

    def __new__(mcs, name, bases, dikt):
        """Override Slots.__new__."""
        dikt["__slots__"] = []
        return type.__new__(mcs, name, bases, dikt)

    def __init__(cls, name, bases, dikt):
        """Initialize a new enumeration class type."""
        super().__init__(name, bases, dikt)
        cls.__members = set(v for k, v in dikt.items() if not k.startswith("__"))

    def __iter__(cls) -> typing.Iterable[str]:
        """Iterate the enum members."""
        return iter(cls.__members)

    def __contains__(cls, item: str) -> bool:
        """Check whether the certain string is the enum's member."""
        return item in cls.__members

    def __len__(cls) -> int:
        """Count the number of enumerated values."""
        return len(cls.__members)
