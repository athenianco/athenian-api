import argparse
from datetime import timezone
from io import StringIO
import sys

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account", required=True, type=int, help="Athenian account ID.")
    parser.add_argument("--author", required=True, type=int, help="Release author node ID.")
    parser.add_argument(
        "--repository", required=True, type=int, help="Release repository node ID.",
    )
    return parser.parse_args()


def main() -> None:
    """Go away linter."""
    args = _parse_args()
    df = pd.read_csv(
        StringIO(sys.stdin.read()), parse_dates=["published_at"], infer_datetime_format=True,
    )
    print(
        "insert into athenian.release_notifications(account_id, repository_node_id,"
        " commit_hash_prefix, name, author_node_id, cloned, published_at) values ",
    )
    for i, item in enumerate(df.itertuples()):
        if i > 0:
            print(",")
        sys.stdout.write(
            f"({args.account}, {args.repository}, "
            f"'{item.commit}', '{'/'.join(item.repository.split('/', 1)[1:])}@{item.commit[:7]}', "
            f"{args.author}, true, '{item.published_at.astimezone(timezone.utc).isoformat()}')",
        )
    print(";")


if __name__ == "__main__":
    sys.exit(main())
