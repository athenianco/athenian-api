import json
import sys

from sqlalchemy import create_engine
from tqdm import tqdm


def main():
    """Usage: file.py postgresql://state postgresql://metadata"""
    state_uri, metadata_uri = sys.argv[1:]
    state_engine = create_engine(state_uri)
    print("Loading the reposets from sdb", file=sys.stderr)
    reposet_rows = state_engine.execute("SELECT id, owner_id, items FROM repository_sets")
    print("Loading the accounts from sdb", file=sys.stderr)
    acc_rows = state_engine.execute("SELECT id, account_id FROM account_github_accounts")
    meta_ids = {}
    for row in acc_rows:
        meta_ids.setdefault(row[1], []).append(str(row[0]))
    print("Converting to node IDs")
    metadata_engine = create_engine(metadata_uri)
    for row in tqdm(reposet_rows):
        rs_id, owner_id, old_items = row[:3]
        if owner_id not in meta_ids or not old_items or isinstance(old_items[0], list):
            continue
        repo_names = [f"'{r.split('/', 1)[1]}'" for r in old_items]
        node_rows = metadata_engine.execute(
            f"SELECT graph_id, name_with_owner "
            f"FROM github.node_repository "
            f"WHERE acc_id IN ({','.join(meta_ids[owner_id])}) "
            f"  AND name_with_owner IN ({','.join(repo_names)})")
        name_map = {row[1]: row[0] for row in node_rows}
        new_items = []
        for name in old_items:
            short_name = "/".join(name.split("/", 3)[1:3])
            try:
                new_items.append([name, name_map[short_name]])
            except KeyError:
                continue
        state_engine.execute(
            f"UPDATE repository_sets SET items = '{json.dumps(new_items)}' "
            f"WHERE id = {rs_id}")


if __name__ == "__main__":
    sys.exit(main())
