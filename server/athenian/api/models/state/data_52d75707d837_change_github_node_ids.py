from collections import defaultdict
import sys

from sqlalchemy import create_engine
from tqdm import tqdm


def main():
    """Usage: file.py postgresql://state postgresql://metadata"""
    state_uri, metadata_uri = sys.argv[1:]
    print("Loading the accounts from sdb", file=sys.stderr)
    state_engine = create_engine(state_uri)
    acc_rows = state_engine.execute("SELECT id, account_id FROM account_github_accounts")
    meta_ids = {row[1]: row[0] for row in acc_rows}
    print("Loading the user node IDs from sdb", file=sys.stderr)
    jira_rows = state_engine.execute(
        "SELECT github_user_id, account_id, jira_user_id, confidence, created_at, updated_at "
        "FROM jira_identity_mapping_old").fetchall()
    node_ids = defaultdict(set)
    for row in jira_rows:
        try:
            node_ids[meta_ids[row[1]]].add(row[0])
        except KeyError:
            continue
    print(f"Mapping {sum(len(n) for n in node_ids.values())} IDs to new format "
          f"in {len(node_ids)} batches", file=sys.stderr)
    metadata_engine = create_engine(metadata_uri)
    mapping = defaultdict(dict)
    for acc, acc_node_ids in tqdm(node_ids.items()):
        map_rows = metadata_engine.execute(
            f'SELECT ext_id, node_id FROM github.graph_ids WHERE acc_id = {acc} AND ext_id = '  # noqa
            f'ANY(VALUES {", ".join("(%r)" % n for n in acc_node_ids)})')
        for row in map_rows:
            mapping[acc][row[0]] = row[1]
    print(f"Mapped {sum(len(m) for m in mapping.values())} node IDs", file=sys.stderr)
    values = []
    for row in jira_rows:
        try:
            values.append((mapping[meta_ids[row[1]]][row[0]], *row[1:]))
        except KeyError:
            print(f"Failed to map {row[1]}/{row[0]}")
    sql = ", ".join("(%d, %d, %r, %f, '%s', '%s')" % v for v in values)
    print(f"Inserting {len(values)} records")
    state_engine.execute(
        "INSERT INTO jira_identity_mapping "
        "(github_user_id, account_id, jira_user_id, confidence, created_at, updated_at) "
        "VALUES " + sql)


if __name__ == "__main__":
    sys.exit(main())
