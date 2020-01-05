from typing import List

from aiohttp import web

from athenian.api import serialization
from athenian.api.models.calculated_metrics import CalculatedMetrics
from athenian.api.models.created_identifier import CreatedIdentifier
from athenian.api.models.generic_error import GenericError
from athenian.api.models.invalid_request_error import InvalidRequestError
from athenian.api.models.metrics_request import MetricsRequest
from athenian.api.models.no_source_data_error import NoSourceDataError
from athenian.api.models.repository_set import RepositorySet


async def create_reposet(request: web.Request, id, body=None) -> web.Response:
    """Create a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    :param body: List of repositories to group.
    :type body: List[str]
    """
    return web.Response(status=200)


async def delete_reposet(request: web.Request, id) -> web.Response:
    """Delete a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    return web.Response(status=200)


async def get_reposet(request: web.Request, id) -> web.Response:
    """List a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    """
    return web.Response(status=200)


async def update_reposet(request: web.Request, id, body=None) -> web.Response:
    """Update a repository set.

    :param id: Numeric identifier of the repository set to list.
    :type id: int
    :param body:
    :type body: List[str]
    """
    return web.Response(status=200)
