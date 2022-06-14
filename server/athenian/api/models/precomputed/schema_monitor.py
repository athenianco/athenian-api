import asyncio
from collections import defaultdict
import logging
from typing import List, Optional

import aiohttp
from alembic import script
from alembic.runtime.migration import MigrationContext
import morcilla

from athenian.api import metadata
from athenian.api.models.precomputed import template


def schedule_pdb_schema_check(
    pdb: morcilla.Database,
    app: aiohttp.web.Application,
    interval: float = 15 * 60,
) -> List[asyncio.Task]:
    """
    Execute the precomputed DB schema version check every `interval` seconds.

    If there is a mismatch, log it as an error.

    :return: List with one element. That element is always the next scheduled asyncio task.
    """
    log = logging.getLogger("%s.scheduled_pdb_schema_check" % metadata.__package__)
    req_rev = script.ScriptDirectory(str(template.parent)).get_current_head()
    sql = MigrationContext.configure(url=str(pdb.url), opts={"as_sql": True})._version.select()
    task_box = [None]  # type: List[Optional[asyncio.Task]]

    async def pdb_schema_check_callback() -> None:
        await asyncio.sleep(interval)
        app["db_elapsed"].set(defaultdict(float))
        try:
            real_rev = await pdb.fetch_val(sql)
        except Exception as e:
            log.exception(e)
        else:
            if real_rev != req_rev:
                log.error("pdb schema unsync: declared %s but actual is %s", req_rev, real_rev)
            else:
                log.debug("Checked")
        task_box[0] = asyncio.create_task(pdb_schema_check_callback(), name="pdb_schema_check")

    task_box[0] = asyncio.create_task(pdb_schema_check_callback(), name="pdb_schema_check")
    log.info("Scheduled regular pdb schema version checks once per %ds" % interval)
    return task_box
