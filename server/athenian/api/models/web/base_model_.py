from abc import ABCMeta
from itertools import chain
import pprint
import typing

from athenian.api import serialization, typing_utils

T = typing.TypeVar("T")


class Slots(ABCMeta):
    """Set __slots__ according to the declared `attribute_types`."""

    def __new__(mcs, name, bases, dikt, sealed=True):
        """Insert __slots__ to the class' __dict__."""
        attribute_types = {}
        attribute_map = {}
        default_values = {}

        for base in bases:
            try:
                default_values.update(base.default_values)
            except AttributeError:
                pass

            try:
                attribute_types.update(base.attribute_types)
                attribute_map.update(base.attribute_map)
            except AttributeError:
                continue
        try:
            attribute_types.update(dikt["attribute_types"])
            attribute_map.update(dikt["attribute_map"])
        except KeyError:
            for k, v in dikt.get("__annotations__", {}).items():
                try:
                    default_values[k] = dikt[k]
                except KeyError:
                    pass

                if k.upper() == k:
                    continue
                assert not isinstance(v, str), f"Forget about __future__.annotations: {name}"
                if isinstance(v, tuple):
                    attribute_types[k], attribute_map[k] = v
                else:
                    attribute_types[k] = v
        try:
            default_values.update(dikt["default_values"])
        except KeyError:
            pass

        dikt["attribute_types"] = attribute_types
        dikt["attribute_map"] = attribute_map
        dikt["default_values"] = default_values
        if sealed:
            dikt["__slots__"] = tuple("_" + k for k in attribute_types) + dikt.get(
                "__extra_slots__", (),
            )
        else:
            dikt["__slots__"] = ()  # this is required for the magic to work
        for key, type_ in attribute_types.items():
            dikt[key] = property(mcs.__getter(key, type_), mcs.__setter(key, type_))

        return type.__new__(mcs, name, bases, dikt)

    @classmethod
    def __getter(mcs: typing.Type[T], key: str, type_: type) -> typing.Callable[[T], typing.Any]:
        def get(self: T) -> type_:
            return getattr(self, f"_{key}")

        get.__name__ = f"get_{key}"
        return get

    @classmethod
    def __setter(
        mcs: typing.Type[T],
        key: str,
        type_: type,
    ) -> typing.Callable[[T, typing.Any], None]:
        if not typing_utils.is_generic(type_) or not typing_utils.is_optional(type_):

            def set_(self: T, value: type_) -> None:
                try:
                    value = getattr(self, f"validate_{key}")(value)
                except AttributeError:
                    if value is None:
                        raise ValueError(
                            f"Invalid value for `{key}`, must not be `None`",
                        ) from None
                setattr(self, f"_{key}", value)

        else:

            def set_(self: T, value: type_) -> None:
                try:
                    value = getattr(self, f"validate_{key}")(value)
                except AttributeError:
                    pass
                setattr(self, f"_{key}", value)

        set_.__name__ = f"set_{key}"
        return set_

    def __instancecheck__(cls, instance):
        """Override for isinstance(instance, cls)."""
        return type.__instancecheck__(cls, instance)

    def __subclasscheck__(cls, subclass):
        """Override for issubclass(subclass, cls)."""
        return type.__subclasscheck__(cls, subclass)


ModelT = typing.TypeVar("ModelT", bound="Model")


class Model(typing.Mapping, metaclass=Slots):
    """Base API model class. Handles object -> {} and {} -> object transformations."""

    # The key is attribute name and the value is attribute type.
    # We initialize this mapping from the annotations.
    attribute_types: typing.Dict[str, type] = {}

    # Attribute renames. The key is attribute name and the
    # value is json key in definition; attributes using the same name
    # as json property does not need to be defined here.
    attribute_map: typing.Dict[str, str] = {}

    # Additional attributes beyond `attribute_types`, they won't be serialized.
    __extra_slots__ = ()

    def __init__(self, **kwargs: typing.Any):
        """Initialize a new model instance by setting attributes."""
        for key in self.attribute_types:
            setattr(self, key, kwargs.pop(key, None))
        for slot in self.__extra_slots__:
            setattr(self, slot, kwargs.pop(slot, None))
        if kwargs:
            raise TypeError(
                "%s does not support these keyword arguments: %s" % (type(self).__name__, kwargs),
            )

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

        for attr_key in self.attribute_types:
            value = getattr(self, attr_key)
            try:
                if (
                    typing_utils.is_optional(type_ := self.attribute_types[attr_key])
                    and not getattr(type_.__origin__, "__verbatim__", False)
                    and (value is None or len(value) == 0)
                ):
                    continue
            except TypeError:
                pass
            json_key = self.attribute_map.get(attr_key, attr_key)
            result[json_key] = self.serialize(value)

        return result

    def copy(self: ModelT) -> ModelT:
        """Clone the object."""
        return type(self)(**{p: getattr(self, p) for p in self.attribute_types})

    def to_str(self) -> str:
        """Returns the string representation of the model."""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For debugging."""
        return "%s(%s)" % (
            type(self).__name__,
            ", ".join("%s=%r" % (k, getattr(self, k)) for k in self.attribute_types),
        )

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)

    def __str__(self):
        """For `print` and `pprint`."""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal."""
        slots = self.__slots__
        try:
            return {k: getattr(self, k) for k in slots} == {k: getattr(other, k) for k in slots}
        except AttributeError:
            return False

    def __ne__(self, other):
        """Returns true if both objects are not equal."""
        return not self == other

    def __getitem__(self, k: str) -> typing.Optional[typing.List[str]]:
        """Implement []."""
        return getattr(self, k)

    def __len__(self) -> int:
        """Implement len()."""
        return len(self.attribute_types)

    def __iter__(self) -> typing.Iterator[str]:
        """Implement iter()."""
        return iter(self.attribute_types)


class Enum(Slots):
    """Trivial enumeration metaclass."""

    def __new__(mcs, name, bases, dikt):
        """Override Slots.__new__."""
        dikt["__slots__"] = []
        return type.__new__(mcs, name, bases, dikt)

    def __init__(cls, name, bases, dikt):
        """Initialize a new enumeration class type."""
        super().__init__(name, bases, dikt)
        cls.__members = {v for k, v in dikt.items() if not k.startswith("__")}
        cls.__members.update(
            chain.from_iterable(getattr(base, "_Enum__members", ()) for base in bases),
        )

    def __iter__(cls) -> typing.Iterable[str]:
        """Iterate the enum members."""
        return iter(cls.__members)

    def __contains__(cls, item: str) -> bool:
        """Check whether the certain string is the enum's member."""
        return item in cls.__members

    def __len__(cls) -> int:
        """Count the number of enumerated values."""
        return len(cls.__members)

    def __or__(self, other: typing.Union[set, "Enum"]) -> set:
        """Union with another Enum."""
        if isinstance(other, Enum):
            return set(self) | set(other)
        return set(self) | other

    __ror__ = __or__


def AllOf(
    *mixed: typing.Type[Model],
    name: str,
    module: str,
    sealed: bool = True,
) -> typing.Type[Model]:
    """
    Inherit from multiple Model classes.

    :param sealed: Do not allow inheritance from the resulting class.
    :param name: Name of the class, must match the variable name to avoid pickling errors.
    :param module: Name of the calling module.
    """
    for cls in mixed:
        try:
            cls.__dict__
        except AttributeError:
            raise TypeError("%s must have __dict__ (set sealed=False)" % cls.__name__) from None

    allOf = Slots(
        name,
        mixed,
        {
            "attribute_types": dict(
                chain.from_iterable(cls.attribute_types.items() for cls in mixed),
            ),
            "attribute_map": dict(chain.from_iterable(cls.attribute_map.items() for cls in mixed)),
            "__module__": module,
        },
        sealed=sealed,
    )
    if len(allOf.attribute_types) < sum(len(cls.attribute_types) for cls in mixed):
        raise TypeError("There are conflicting attribute_types in AllOf classes")
    return allOf
