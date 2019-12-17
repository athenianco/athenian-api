# coding: utf-8

import sys

from aiohttp import web
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


class AthenianWebRequest(web.Request):
    """Type hint for any API HTTP request."""

    mdb = None  # type: databases.Database
    sdb = None  # type: databases.Database
