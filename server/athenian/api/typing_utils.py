import sys
from typing import Union

import databases

if sys.version_info < (3, 7):
    import typing

    def is_generic(klass):
        """Determine whether klass is a generic class."""
        return type(klass) == typing.GenericMeta

    def is_dict(klass):
        """Determine whether klass is a Dict."""
        return klass.__extra__ == dict

    def is_list(klass):
        """Determine whether klass is a List."""
        return klass.__extra__ == list


else:

    def is_generic(klass):
        """Determine whether klass is a generic class."""
        return hasattr(klass, "__origin__")

    def is_dict(klass):
        """Determine whether klass is a Dict."""
        return klass.__origin__ == dict

    def is_list(klass):
        """Determine whether klass is a List."""
        return klass.__origin__ == list


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
