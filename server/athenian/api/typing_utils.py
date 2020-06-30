from typing import Union

import databases


def is_generic(klass: type):
    """Determine whether klass is a generic class."""
    return hasattr(klass, "__origin__")


def is_dict(klass: type):
    """Determine whether klass is a Dict."""
    return getattr(klass, "__origin__", None) == dict


def is_list(klass: type):
    """Determine whether klass is a List."""
    return getattr(klass, "__origin__", None) == list


def is_optional(klass: type):
    """Determine whether klass is an Optional."""
    return getattr(klass, "__origin__", None) == Union and \
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
