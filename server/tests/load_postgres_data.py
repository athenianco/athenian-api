import argparse
import getpass
import os
import subprocess
import sys

conf = {
    "staging": {
        "port": 5432,
    },
    "production": {
        "port": 5433,
    },
}


def run_pg_cmd_on_guest(guest_cmd, password):
    """Run a postgres command in the guest container."""
    return subprocess.run([
        "docker-compose exec -e PGPASSWORD=%s postgres bash -c '%s'" % (
            password, guest_cmd),
    ], shell=True, check=True)


def parse_args():
    """Parse arguments for the posgres data loader command."""
    parser = argparse.ArgumentParser()

    parser.add_argument("environment", default="staging", choices=conf,
                        help="Name of the environment the data will be taken from")
    parser.add_argument("--remote-postgres-user",
                        default=os.environ.get("CLOUD_SQL_POSTGRES_USER"),
                        help="Name of the user to use for the local db")
    parser.add_argument("--local-postgres-user",
                        default=os.environ.get("POSTGRES_USER", "api"),
                        help="Name of the user to use for the local db")

    parsed_args = parser.parse_args()

    if not parsed_args.remote_postgres_user:
        raise ValueError("Missing remote postgres user")

    if not parsed_args.local_postgres_user:
        raise ValueError("Missing local postgres user")

    return parsed_args


def main():
    """Run the postgres data loader."""
    try:
        args = parse_args()
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    dbs = ["state", "precomputed", "metadata"]
    remote_postgres_password = os.environ.get("POSTGRES_PASSWORD", "")
    if remote_postgres_password == "":
        remote_postgres_password = getpass.getpass(
            prompt=("Password for CloudSQL instance of '%s' environment "
                    "of user '%s': " % (
                        args.environment, args.remote_postgres_user)))

    config = conf[args.environment]
    dumps_base_path = "/db_dumps"
    for db in dbs:
        file_name = "%s.dump" % db
        dump_path = os.path.join(dumps_base_path, file_name)

        guest_cmd = (
            "pg_dump --host cloud_sql_proxy --port=%d --username=%s "
            "-Fc --dbname=%s > %s"
        ) % (config["port"], args.remote_postgres_user, db, dump_path)

        print("Dumping database: %s" % guest_cmd)
        run_pg_cmd_on_guest(guest_cmd, remote_postgres_password)
        print("Done!")

        guest_cmd = "createdb -U %s --lc-collate='C.UTF-8' -T template0 %s" % (
            args.local_postgres_user, db)
        print("Creating database: %s" % guest_cmd)
        run_pg_cmd_on_guest(guest_cmd, remote_postgres_password)

        guest_cmd = "pg_restore -x -Fc -O -U %s -d %s %s" % (
            args.local_postgres_user, db, dump_path)
        print("Loading dump into container: %s" % guest_cmd)
        run_pg_cmd_on_guest(guest_cmd, remote_postgres_password)
        print("Done!")

    return 0


if __name__ == "__main__":
    exit(main())
