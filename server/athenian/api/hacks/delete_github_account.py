from pathlib import Path
from shutil import move
import subprocess
import sys
from tempfile import TemporaryDirectory


def delete_metadata(acc_id: int) -> None:  # noqa: D103
    nodes_file_name = "50_nodes.sql"
    if not (nodes_file := Path(nodes_file_name)).exists():
        with TemporaryDirectory() as tmpdir:
            subprocess.check_call([
                "git", "clone", "--branch", "master", "--single-branch", "--depth", "1",
                "git@github.com:athenianco/metadata.git", tmpdir,
            ])
            move(Path(tmpdir) / "sql" / nodes_file_name, nodes_file_name)
    assert nodes_file.exists()
    print("BEGIN;")
    for line in nodes_file.open().readlines()[::-1]:
        if line.startswith("CREATE TABLE"):
            table = line.split()[-2]
            print(f"DELETE FROM {table} WHERE acc_id = {acc_id};")
    print(f"UPDATE github.accounts SET active = false, suspended = true WHERE id = {acc_id};")
    print("COMMIT;")


def delete_precomputed(acc_id: int) -> None:  # noqa: D103
    print("BEGIN;")
    for table in [
        "commit_deployments",
        "commit_history",
        "deployment_facts",
        "done_pull_request_facts",
        "merged_pull_request_facts",
        "open_pull_request_facts",
        "pull_request_check_runs",
        "pull_request_deployments",
        "release_deployments",
        "release_facts",
        "release_match_spans",
        "releases",
        "repositories",
    ]:
        print(f"DELETE FROM github.{table} WHERE acc_id = {acc_id};")
    print("COMMIT;")


def delete_state(acc_id: int) -> None:  # noqa: D103
    print("BEGIN;")
    for table, acc_col in [
        ("jira_identity_mapping", "account_id"),
        ("repository_sets", "owner_id"),
    ]:
        print(f"DELETE FROM {table} WHERE {acc_col} = {acc_id};")
    print("COMMIT;")


def show_help(_=None) -> int:  # noqa: D103
    print(f"Usage: python3 delete_github_account.py {'|'.join(deleters.keys())} <acc_id>",
          file=sys.stderr)
    return 1


deleters = {
    "metadata": delete_metadata,
    "precomputed": delete_precomputed,
    "state": delete_state,
}

if __name__ == "__main__":
    try:
        sys.exit(deleters.get(sys.argv[1], show_help)(int(sys.argv[2])))
    except (IndexError, ValueError):
        sys.exit(show_help())
