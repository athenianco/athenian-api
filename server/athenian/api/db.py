import logging
import time

import aiohttp.web
import databases
from databases.interfaces import ConnectionBackend, TransactionBackend

from athenian.api import metadata
from athenian.api.typing_utils import wraps


def measure_db_overhead(db: databases.Database,
                        db_id: str,
                        app: aiohttp.web.Application) -> databases.Database:
    """Instrument Database to measure the time spent inside DB i/o."""
    log = logging.getLogger("%s.measure_db_overhead" % metadata.__package__)

    def measure_method_overhead(func) -> callable:
        async def wrapped_measure_method_overhead(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed = app["db_elapsed"].get()
                if elapsed is None:
                    log.warning("Cannot record the %s overhead", db_id)
                else:
                    elapsed[db_id] = elapsed[db_id] + time.time() - start_time

        return wraps(wrapped_measure_method_overhead, func)

    backend_connection = db._backend.connection

    def wrapped_backend_connection() -> ConnectionBackend:
        connection = backend_connection()
        connection.fetch_all = measure_method_overhead(connection.fetch_all)
        connection.fetch_one = measure_method_overhead(connection.fetch_one)
        connection.execute = measure_method_overhead(connection.execute)
        connection.execute_many = measure_method_overhead(connection.execute_many)

        original_transaction = connection.transaction

        def transaction() -> TransactionBackend:
            t = original_transaction()
            t.start = measure_method_overhead(t.start)
            t.commit = measure_method_overhead(t.commit)
            t.rollback = measure_method_overhead(t.rollback)
            return t

        connection.transaction = transaction
        return connection

    db._backend.connection = wrapped_backend_connection
    return db
