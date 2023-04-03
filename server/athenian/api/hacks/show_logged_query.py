#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import subprocess


def _main():
    args = _parse_args()
    body = _get_query_body_multi(args.uuid)
    _show_sql(body)


def _parse_args() -> argparse.Namespace:
    description = (
        "Show a query logged on GCP by its UUID. Query UUID is logged in sentry span description"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("uuid")
    return parser.parse_args()


def _get_query_body(uuid: str) -> str:
    args = [
        "gcloud",
        "logging",
        "read",
        "--freshness=200d",
        "--limit=1",
        "--format=value(jsonPayload.msg)",
        f'"{uuid}"',
    ]
    output = subprocess.check_output(args, text=True)
    # output is like N / M UUID BODY
    uuid_start = output.find(uuid)
    assert uuid_start > -1
    body = output[uuid_start + len(uuid) + 1 :]
    return body.strip()


def _get_query_body_multi(uuid: str) -> str:
    args = [
        "gcloud",
        "logging",
        "read",
        "--freshness=200d",
        "--limit=100",
        "--format=value(jsonPayload.msg)",
        f'"{uuid}"',
    ]
    proc = subprocess.Popen(args, text=True, stdout=subprocess.PIPE)
    assert proc.stdout
    parts = ""
    for line in proc.stdout:
        # line is like N / M UUID BODY
        uuid_start = line.find(uuid)
        header = line[:uuid_start]
        m = re.search(r"(\d+) / (\d+)", header)
        assert m
        part_n = int(m.group(1))

        part = line[uuid_start + len(uuid) + 1 :].strip()
        parts = part + parts
        if part_n <= 1:
            break

    proc.terminate()
    return parts


def _show_sql(body: str) -> None:
    show_sql_script = Path(__file__).parent / "show_sql.py"
    subprocess.run([show_sql_script], input=body, text=True, check=True)


if __name__ == "__main__":
    _main()
