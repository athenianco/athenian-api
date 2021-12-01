from http import HTTPStatus
import io
import time
from typing import Optional

from aiohttp import web
import aiomcache
import morcilla
import objgraph
import prometheus_client
import pympler.muppy
import pympler.summary
from sqlalchemy import select

from athenian.api.cache import cached
from athenian.api.metadata import __version__
import athenian.api.models.metadata as metadata
from athenian.api.models.web import BadRequestError
from athenian.api.models.web.versions import Versions
from athenian.api.prometheus import PROMETHEUS_REGISTRY_VAR_NAME
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@cached(
    exptime=60 * 60,  # 1 hour
    key=lambda **_: (),
    serialize=lambda s: s.encode(),
    deserialize=lambda b: b.decode(),
)
async def _get_metadata_version(mdb: morcilla.Database, cache: Optional[aiomcache.Client]) -> str:
    return str(await mdb.fetch_val(select([metadata.SchemaMigration.version])))


async def get_versions(request: AthenianWebRequest) -> web.Response:
    """Return the versions of the backend components."""
    metadata_version = await _get_metadata_version(request.mdb, request.cache)
    model = Versions(api=__version__, metadata=metadata_version)
    return model_response(model)


class PrometheusRenderer:
    """Render the status page with Prometheus."""

    var_name = "prometheus_renderer"

    def __init__(self, registry: prometheus_client.CollectorRegistry, cache_ttl=1):
        """Record the registry where the metrics are maintained."""
        self._registry = registry
        self._body = b""
        self._ts = time.time() - cache_ttl
        self._cache_ttl = cache_ttl

    @property
    def body(self):
        """Refresh the metrics from time to time."""
        if time.time() - self._ts > self._cache_ttl:
            self._body = prometheus_client.generate_latest(self._registry)
            self._ts = time.time()
        return self._body

    async def __call__(self, request: web.Request) -> web.Response:
        """Endpoint handler to output the current Prometheus state."""
        resp = web.Response(body=self.body)
        resp.content_type = prometheus_client.CONTENT_TYPE_LATEST
        return resp


async def summarize_memory(request: web.Request) -> web.Response:
    """Return a TXT memory usage summary by object type."""
    limit = request.rel_url.query.get("limit", "20")
    try:
        limit = int(limit)
    except ValueError:
        return ResponseError(
            BadRequestError('"limit" has an invalid integer value "%s"' % limit),
        ).response
    all_objects = pympler.muppy.get_objects()
    summary = pympler.summary.summarize(all_objects)
    body = "\n".join(pympler.summary.format_(summary, limit=limit))
    resp = web.Response(text=body)
    return resp


async def graph_type_memory(request: web.Request) -> web.Response:
    """Generate Graphviz of the objects referencing the objects of the specified type."""
    try:
        typename = request.rel_url.query["type"]
    except KeyError:
        return ResponseError(
            BadRequestError('"type" must be specified in the URL arguments'),
        ).response
    max_depth = request.rel_url.query.get("depth", "5")
    try:
        max_depth = int(max_depth)
    except ValueError:
        return ResponseError(
            BadRequestError('"depth" has an invalid integer value "%s"' % max_depth),
        ).response
    if max_depth > 20:
        return ResponseError(BadRequestError('"depth" cannot be greater than 20')).response
    if max_depth < 1:
        return ResponseError(BadRequestError('"depth" cannot be less than 1')).response
    all_objects = pympler.muppy.get_objects()
    roots = [obj for obj in all_objects if pympler.summary._repr(obj) == typename]
    buf = io.StringIO()
    objgraph.show_backrefs(roots, output=buf, max_depth=max_depth, refcounts=True)
    resp = web.Response(text=buf.getvalue())
    resp.content_type = "text/vnd.graphviz; charset=utf-8"
    return resp


async def render_status(request: web.Request) -> web.Response:
    """Return HTTP 200 to indicate that we are alive."""
    content_type = "text/plain"
    if "health" in request.app:
        return web.Response(content_type=content_type)
    return web.Response(status=HTTPStatus.SERVICE_UNAVAILABLE)


def setup_status(app: web.Application) -> None:
    """
    Register routes of the service endpoints.

    * `/prometheus` serves the runtime metrics.
    * `/memory` and `/objgraph` help to debug memory leaks.
    * `/status` serves the canary health checks.
    """
    app[PrometheusRenderer.var_name] = prometheus_renderer = PrometheusRenderer(
        app[PROMETHEUS_REGISTRY_VAR_NAME])
    # passing prometheus_renderer without __call__ triggers a spurious DeprecationWarning
    # FIXME(vmarkovtsev): https://github.com/aio-libs/aiohttp/issues/4519
    app.router.add_get("/prometheus", prometheus_renderer.__call__)
    app.router.add_get("/memory", summarize_memory)
    app.router.add_get("/objgraph", graph_type_memory)
    app.router.add_get("/status", render_status)
