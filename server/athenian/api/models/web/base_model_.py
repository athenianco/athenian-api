from abc import ABCMeta
from itertools import chain
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
        if dikt.get("__enable_slots__", True):
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

    @classmethod
    def serialize(cls, value: typing.Any) -> typing.Any:
        """Convert any value to a JSON-friendly object."""
        if isinstance(value, Model):
            return value.to_dict()
        if isinstance(value, (list, tuple)):
            return [cls.serialize(x) for x in value]
        if isinstance(value, dict):
            return {item[0]: cls.serialize(item[1]) for item in value.items()}
        return value

    def to_dict(self) -> dict:
        """Returns the model properties as a dict."""
        result = {}

        for attr_key, json_key in self.attribute_map.items():
            value = getattr(self, attr_key)
            if value is None and typing_utils.is_optional(self.openapi_types[attr_key]):
                continue
            result[json_key] = self.serialize(value)

        return result

    def copy(self) -> "Model":
        """Clone the object."""
        return type(self)(**{p: getattr(self, p) for p in self.openapi_types})

    def to_str(self) -> str:
        """Returns the string representation of the model."""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For debugging."""
        return "%s(%s)" % (type(self).__name__, ", ".join(
            "%s=%r" % (k, getattr(self, k)) for k in self.openapi_types))

    def __str__(self):
        """For `print` and `pprint`."""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal."""
        slots = self.__slots__
        return {k: getattr(self, k) for k in slots} == {k: getattr(other, k) for k in slots}

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


def AllOf(*mixed: typing.Type[Model], module: str, sealed: bool = True) -> typing.Type[Model]:
    """
    Inherit from multiple Model classes.

    :param sealed: Do not allow inheritance from the resulting class.
    :param module: Name of the calling module.
    """
    for cls in mixed:
        try:
            cls.__dict__
        except AttributeError:
            raise TypeError(
                "%s must have __dict__ (set __enable_slots__ to False)" % cls.__name__) from None

    def __init__(self, **kwargs):
        consumed = set()
        for cls in mixed:
            cls.__init__(self, **{k: kwargs.get(k, None) for k in cls.openapi_types})
            consumed.update(cls.openapi_types)
        if extra := kwargs.keys() - consumed:
            raise TypeError("%s does not support these keyword arguments: %s",
                            type(self).__name__, extra)

    allOf = type("AllOf_" + "_".join(cls.__name__ for cls in mixed), mixed, {
        "openapi_types": dict(chain.from_iterable(cls.openapi_types.items() for cls in mixed)),
        "attribute_map": dict(chain.from_iterable(cls.attribute_map.items() for cls in mixed)),
        "__init__": __init__,
        "__enable_slots__": sealed,
        "__module__": module,
    })
    if len(allOf.openapi_types) < sum(len(cls.openapi_types) for cls in mixed):
        raise TypeError("There are conflicting openapi_types in AllOf classes")
    return allOf
