import json
import sys

import sqlalchemy as sa
from tqdm import tqdm


def main():
    """Usage: file.py postgresql://state postgresql://metadata"""
    state_uri, metadata_uri = sys.argv[1:]
    state_engine = sa.create_engine(state_uri)
    print("Loading the reposets from sdb", file=sys.stderr)
    with state_engine.begin() as state_conn:
        reposet_rows = state_conn.execute(
            sa.text("SELECT id, owner_id, items FROM repository_sets"),
        )
        print("Loading the accounts from sdb", file=sys.stderr)
        acc_rows = state_conn.execute(
            sa.text("SELECT id, account_id FROM account_github_accounts"),
        )
    meta_ids = {}
    for row in acc_rows:
        meta_ids.setdefault(row[1], []).append(str(row[0]))
    print("Converting to node IDs")
    metadata_engine = sa.create_engine(metadata_uri)
    for row in tqdm(reposet_rows):
        rs_id, owner_id, old_items = row[:3]
        if owner_id not in meta_ids or not old_items or isinstance(old_items[0], list):
            continue
        repo_names = [f"'{r.split('/', 1)[1]}'" for r in old_items]
        with metadata_engine.begin() as metadata_conn:
            node_rows = metadata_conn.execute(
                sa.text(
                    "SELECT graph_id, name_with_owner "
                    "FROM github.node_repository "
                    f"WHERE acc_id IN ({','.join(meta_ids[owner_id])}) "
                    f"  AND name_with_owner IN ({','.join(repo_names)})",
                ),
            )
        name_map = {row[1]: row[0] for row in node_rows}
        new_items = []
        for name in old_items:
            short_name = "/".join(name.split("/", 3)[1:3])
            try:
                new_items.append([name, name_map[short_name]])
            except KeyError:
                continue
        with state_engine.begin() as state_conn:
            stmt = (
                f"UPDATE repository_sets SET items = '{json.dumps(new_items)}' WHERE id = {rs_id}",
            )
            state_conn.execute(sa.text(stmt))


if __name__ == "__main__":
    sys.exit(main())
