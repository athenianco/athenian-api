from alembic import script
from alembic.runtime.migration import MigrationContext
from sqlalchemy.sql.ddl import CreateTable

from athenian.api.models.precomputed import template
from athenian.api.models.precomputed.schema_monitor import schedule_pdb_schema_check


async def test_schedule_pdb_schema_check(pdb, caplog):
    req_rev = script.ScriptDirectory(str(template.parent)).get_current_head()
    table = MigrationContext.configure(url=str(pdb.url), opts={"as_sql": True})._version
    try:
        await pdb.execute(table.update(values={table.c.version_num: "xxx"}))
    except Exception:
        await pdb.execute(CreateTable(table))
        await pdb.execute(table.insert(values={table.c.version_num: "xxx"}))
    task_box = schedule_pdb_schema_check(pdb, interval=0.001)
    await task_box[0]
    assert caplog.records[-1].levelname == "ERROR"
    task_box[0].cancel()
    await pdb.execute(table.update(values={table.c.version_num: req_rev}))
    task_box = schedule_pdb_schema_check(pdb, interval=0.001)
    await task_box[0]
    assert caplog.records[-1].levelname != "ERROR"
    task_box[0].cancel()
