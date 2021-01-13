from contextlib import contextmanager
from contextvars import ContextVar
import dataclasses
from typing import Any, Iterator, Mapping, Optional, Tuple, Type, TypeVar, Union

import databases
import numpy as np
import xxhash


def is_generic(klass: type):
    """Determine whether klass is a generic class."""
    return hasattr(klass, "__origin__")


def is_dict(klass: type):
    """Determine whether klass is a Dict."""
    return getattr(klass, "__origin__", None) == dict


def is_list(klass: type):
    """Determine whether klass is a List."""
    return getattr(klass, "__origin__", None) == list


def is_union(klass: type):
    """Determine whether klass is a Union."""
    return getattr(klass, "__origin__", None) == Union


def is_optional(klass: type):
    """Determine whether klass is an Optional."""
    return is_union(klass) and \
        len(klass.__args__) == 2 and issubclass(klass.__args__[1], type(None))


DatabaseLike = Union[databases.Database, databases.core.Connection]


def wraps(wrapper, wrappee):
    """Alternative to functools.wraps() for async functions."""  # noqa: D402
    wrapper.__name__ = wrappee.__name__
    wrapper.__qualname__ = wrappee.__qualname__
    wrapper.__module__ = wrappee.__module__
    wrapper.__doc__ = wrappee.__doc__
    wrapper.__annotations__ = wrappee.__annotations__
    wrapper.__wrapped__ = wrappee
    return wrapper


T = TypeVar("T")


def dataclass(cls: Optional[T] = None,
              /, *,
              slots=False,
              first_mutable: Optional[str] = None,
              **kwargs,
              ) -> Union[T, Type[Mapping[str, Any]]]:
    """
    Generate a dataclasses.dataclass with optional __slots__.

    :param slots: Define __slots__ according to the declared dataclass fields.
    :param first_mutable: First mutable field name. This and all the following fields will be \
                          considered mutable and optional. Such fields are not pickled and can be \
                          changed even though the instance is frozen.
    """
    def wrap(cls):
        cls = dataclasses.dataclass(cls, **kwargs)
        if slots:
            cls = _add_slots_to_dataclass(cls, first_mutable)
        return cls

    # See if we're being called as @dataclass or @dataclass().
    if cls is None:
        # We're called with parens.
        return wrap

    # We're called as @dataclass without parens.
    return wrap(cls)


# Caching context indicator. By default, we don't save the mutable optional fields.
_serialize_mutable_fields_in_dataclasses = ContextVar(
    "serialize_mutable_fields_in_dataclasses", default=False)


@contextmanager
def serialize_mutable_fields_in_dataclasses():
    """Provide a context manager to enable the serialization of mutable optional fields in our \
    dataclasses."""
    _serialize_mutable_fields_in_dataclasses.set(True)
    try:
        yield
    finally:
        _serialize_mutable_fields_in_dataclasses.set(False)


def _add_slots_to_dataclass(cls: T,
                            first_mutable: Optional[str],
                            ) -> Union[T, Type[Mapping[str, Any]]]:
    """Set __slots__ of a dataclass, and make it a Mapping to compensate for a missing __dict__."""
    # Need to create a new class, since we can't set __slots__ after a class has been created.

    # Make sure __slots__ isn't already set.
    if "__slots__" in cls.__dict__:
        raise TypeError(f"{cls.__name__} already specifies __slots__")

    # Create a new dict for our new class.
    cls_dict = dict(cls.__dict__)
    field_names = tuple(f.name for f in dataclasses.fields(cls))
    cls_dict["__slots__"] = field_names
    for field_name in field_names:
        # Remove our attributes, if present. They"ll still be available in _MARKER.
        cls_dict.pop(field_name, None)
    # Remove __dict__ itself.
    cls_dict.pop("__dict__", None)
    # __hash__ cannot be inherited from SlotsMapping, IDK why.
    if (hash_method := cls_dict.pop("__hash__", None)) is not None:
        cls_dict["__hash__"] = hash_method
    else:
        def __hash__(self) -> int:
            """Implement hash() over the immutable fields."""
            return hash(tuple(
                (xxhash.xxh64_intdigest(x.view(np.uint8).data) if isinstance(x, np.ndarray) else x)
                for x in self.__getstate__()
            ))

        cls_dict["__hash__"] = __hash__
    qualname = getattr(cls, "__qualname__", None)
    # Record the mutable fields.
    if first_mutable is None:
        first_mutable_index = len(field_names)
    else:
        first_mutable_index = field_names.index(first_mutable)
    mutable_fields = set(field_names[first_mutable_index:])
    if first_mutable is not None:
        def __setattr__(self, attr: str, val: Any) -> None:
            """Alternative to __setattr__ that works with mutable optional fields."""
            assert attr in mutable_fields, "You can only change mutable optional fields."
            object.__setattr__(self, attr, val)

        def make_with_attr(attr):
            def with_attr(self, value) -> cls:
                """Chain __setattr__ to return `self`."""
                setattr(self, attr, value)
                return self

            return with_attr

        cls_dict["__setattr__"] = __setattr__
        for attr in mutable_fields:
            cls_dict["with_" + attr] = make_with_attr(attr)

    class SlotsMapping(Mapping[str, Any]):
        """Satisfy Mapping abstractions by relying on the __slots__."""

        __slots__ = field_names

        def __getitem__(self, item: str) -> Any:
            """Implement []."""
            return getattr(self, item)

        def __len__(self) -> int:
            """Implement len()."""
            return len(self.__slots__)

        def __iter__(self) -> Iterator[str]:
            """Implement iter()."""
            return iter(self.__slots__)

        def __getstate__(self) -> Any:
            """Support pickling back, we lost it when we deleted __dict__."""
            include_mutable = _serialize_mutable_fields_in_dataclasses.get()
            limit = len(self.__slots__) if include_mutable else first_mutable_index
            return tuple(getattr(self, attr) for attr in self.__slots__[:limit])

        def __setstate__(self, state: Tuple[Any]) -> None:
            """Construct a new class instance from the given `state`."""
            for attr, val in zip(self.__slots__, state):
                object.__setattr__(self, attr, val)
            # Fields with a default value.
            if len(self.__slots__) > len(state):
                for field in dataclasses.fields(self)[len(state):]:
                    object.__setattr__(self, field.name, field.default)

    # And finally create the class.
    cls = type(cls)(cls.__name__, (SlotsMapping, *cls.__bases__), cls_dict)
    if qualname is not None:
        cls.__qualname__ = qualname
    return cls
