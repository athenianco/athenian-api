import json
import sys

import sqlalchemy as sa
from tqdm import tqdm


def main():
    """Usage: file.py postgresql://state postgresql://metadata"""
    state_uri, metadata_uri = sys.argv[1:]
    state_engine = sa.create_engine(state_uri)
    print("Loading the teams from sdb", file=sys.stderr)
    with state_engine.begin() as state_conn:
        team_rows = list(state_conn.execute(sa.text("SELECT id, owner_id, members FROM teams")))
        print("->", len(team_rows))
        print("Loading the accounts from sdb", file=sys.stderr)
        acc_rows = list(
            state_conn.execute(sa.text("SELECT id, account_id FROM account_github_accounts")),
        )
    print("->", len(acc_rows))
    meta_ids = {}
    for row in acc_rows:
        meta_ids.setdefault(row[1], []).append(str(row[0]))
    print("Converting to node IDs")
    metadata_engine = sa.create_engine(metadata_uri)
    for row in tqdm(sorted(team_rows, key=lambda r: r[0])):
        team_id, owner_id, old_members = row[:3]
        if owner_id not in meta_ids or not old_members or isinstance(old_members[0], int):
            continue
        try:
            github_ids = meta_ids[owner_id]
        except KeyError:
            new_items = []
        else:
            people_logins = [f"'{r.rsplit('/', 1)[1]}'" for r in old_members]
            with metadata_engine.begin() as metadata_conn:
                node_rows = metadata_conn.execute(
                    sa.text(
                        "SELECT node_id, login "
                        "FROM github.api_users "
                        f"WHERE acc_id IN ({','.join(github_ids)}) "
                        f"  AND login IN ({','.join(people_logins)})",
                    ),
                )
            login_map = {row[1]: row[0] for row in node_rows}
            new_items = []
            for login in people_logins:
                try:
                    new_items.append(login_map[login[1:-1]])
                except KeyError:
                    continue
        with state_engine.begin() as state_conn:
            state_conn.execute(
                sa.text(
                    f"UPDATE teams SET members = '{json.dumps(sorted(new_items))}' WHERE id ="
                    f" {team_id}",
                ),
            )


if __name__ == "__main__":
    sys.exit(main())
