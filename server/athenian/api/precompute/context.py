from __future__ import annotations

import argparse
from collections import defaultdict
from contextvars import ContextVar
import logging
from typing import Optional

import aiomcache
import sentry_sdk
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient

from athenian.api.__main__ import create_memcached, create_slack
from athenian.api.async_utils import gather
from athenian.api.cache import CACHE_VAR_NAME, setup_cache_metrics
from athenian.api.db import Database, measure_db_overhead_and_retry
from athenian.api.defer import enable_defer
from athenian.api.models.metadata import dereference_schemas as dereference_metadata_schemas
from athenian.api.models.persistentdata import \
    dereference_schemas as dereference_persistentdata_schemas
from athenian.api.prometheus import PROMETHEUS_REGISTRY_VAR_NAME
from athenian.api.typing_utils import dataclass
from athenian.precomputer.db import dereference_schemas as dereference_precomputed_schemas


@dataclass(slots=True)
class PrecomputeContext:
    """Everything initialized for a command to execute."""

    log: logging.Logger
    sdb: Database
    mdb: Database
    pdb: Database
    rdb: Database
    cache: Optional[aiomcache.Client]
    slack: Optional[SlackWebClient]

    @classmethod
    async def create(cls, args: argparse.Namespace, log: logging.Logger) -> PrecomputeContext:
        """Initialize a new precomputer context from the cmdline arguments."""
        enable_defer(False)
        cache = create_memcached(args.memcached, log)
        try:
            setup_cache_metrics({CACHE_VAR_NAME: cache, PROMETHEUS_REGISTRY_VAR_NAME: None})
            for v in cache.metrics["context"].values():
                v.set(defaultdict(int))
            sdb = measure_db_overhead_and_retry(Database(args.state_db))
            try:
                mdb = measure_db_overhead_and_retry(Database(args.metadata_db))
                try:
                    pdb = measure_db_overhead_and_retry(Database(args.precomputed_db))
                    try:
                        rdb = measure_db_overhead_and_retry(Database(args.persistentdata_db))
                        try:
                            await gather(
                                sdb.connect(), mdb.connect(), pdb.connect(), rdb.connect())
                            pdb.metrics = {
                                "hits": ContextVar("pdb_hits", default=defaultdict(int)),
                                "misses": ContextVar("pdb_misses", default=defaultdict(int)),
                            }
                            if mdb.url.dialect == "sqlite":
                                dereference_metadata_schemas()
                            if rdb.url.dialect == "sqlite":
                                dereference_persistentdata_schemas()
                            if pdb.url.dialect == "sqlite":
                                dereference_precomputed_schemas()
                            slack = create_slack(log)
                        except Exception as e:
                            await rdb.disconnect()
                            raise e from None
                    except Exception as e:
                        await pdb.disconnect()
                        raise e from None
                except Exception as e:
                    await mdb.disconnect()
                    raise e from None
            except Exception as e:
                await sdb.disconnect()
                raise e from None
        except Exception as e:
            await cache.close()
            raise e from None
        return cls(
            log=log,
            sdb=sdb,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
            cache=cache,
            slack=slack,
        )

    async def close(self) -> None:
        """Close all the related connections."""
        try:
            await self.cache.close()
            await self.sdb.disconnect()
            await self.mdb.disconnect()
            await self.pdb.disconnect()
            await self.rdb.disconnect()
            if self.slack is not None and self.slack.session is not None:
                await self.slack.session.close()
        except Exception as e:
            self.log.warning("failed to dispose the context: %s: %s", type(e).__name__, e)
            sentry_sdk.capture_exception(e)
