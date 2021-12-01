#!/usr/bin/env python3
import argparse
import asyncio
import logging
import sys
import textwrap

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.db import check_schema_versions, Database
from athenian.api.preloading.cache import MemoryCachePreloader


def parse_args() -> argparse.Namespace:
    """Parse the command line and return the parsed arguments."""
    parser = argparse.ArgumentParser(metadata.__package__)

    parser.add_argument("--metadata-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/metadata",
                        help="Metadata (GitHub, JIRA, etc.) DB connection string in SQLAlchemy "
                             "format. This DB is readonly.")
    parser.add_argument("--state-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/state",
                        help="Server state (user settings, teams, etc.) DB connection string in "
                             "SQLAlchemy format. This DB is read/write.")
    parser.add_argument("--precomputed-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/precomputed",
                        help="Precomputed objects augmenting the metadata DB and reducing "
                             "the amount of online work. DB connection string in SQLAlchemy "
                             "format. This DB is read/write.")
    parser.add_argument("--persistentdata-db",
                        default="postgresql://postgres:postgres@0.0.0.0:5432/persistentdata",
                        help="Pushed and pulled source data that Athenian owns. DB connection "
                             "string in SQLAlchemy format. This DB is read/write.")
    parser.add_argument("--detailed", required=False, action="store_true",
                        help="Whether to also show the memory usage of each column.")

    return parser.parse_args()


def main():
    """Print memory usage for preloaded DataFrames."""
    args = parse_args()
    log = logging.getLogger(metadata.__package__)
    if not check_schema_versions(args.metadata_db,
                                 args.state_db,
                                 args.precomputed_db,
                                 args.persistentdata_db,
                                 log):
        return None

    db_conns = {
        "sdb": args.state_db,
        "mdb": args.metadata_db,
        "pdb": args.precomputed_db,
        "rdb": args.persistentdata_db,
    }

    return asyncio.run(_main(db_conns, args.detailed))


async def _main(db_conns: dict, detailed: bool):
    tasks, dbs = [], {}
    for db_shortcut, db_conn in db_conns.items():
        db = Database(db_conn)
        tasks.append(db.connect())
        dbs[db_shortcut] = db

    await gather(*tasks)

    await MemoryCachePreloader(None, detailed).preload(**dbs)

    for db_name, db in dbs.items():
        if not (cache := getattr(db, "cache", None)):
            continue

        db_total_mem = cache.memory_usage(True, True)
        print(_summary_line(f"DB: {db_name}", 1, db_total_mem))
        for name, df in cache.dfs.items():
            total_mem = df.memory_usage(True, True)
            print(_summary_line(f"DataFrame: {name}", 2, total_mem))
            if not detailed:
                continue

            for key in ("raw", "processed", "percentage"):
                print(f"{_detail(key, df._mem)}\n")


def _summary_line(title, indent_level, total=None):
    prefix = "#" * indent_level
    s = f"{(prefix + ' ' + title):<50}"
    if total:
        s = f"{s} | [total: {total}]"

    return s + "\n"


def _detail(key, info):
    name = key.title()
    lines = [
        _summary_line(name, 3, info[key]["total"]),
        _summary_line(f"{name} series", 3),
        textwrap.indent(str(info[key]["series"]), " " * 4),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
