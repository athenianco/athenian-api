from typing import Any
from unittest import mock

from athenian.api.request import AthenianWebRequest


def request_mock(**kwargs: Any) -> AthenianWebRequest:
    """Generate a mock for an `AthenianWebRequest` object."""
    req = mock.MagicMock(spec=AthenianWebRequest)

    defaults = {
        "uid": "XX",
        "sdb": None,
        "mdb": None,
        "cache": None,
        "user": None,
        "is_god": False,
        "app": {"slack": None},
    }
    for field, default in defaults.items():
        val = kwargs.get(field, default)
        setattr(req, field, val)
    return req
