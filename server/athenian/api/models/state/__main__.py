import os
import subprocess
import sys

import jinja2


def main(url=None, exec=True):
    """
    Initialize the server state DB.

    This script creates all the tables if they don't exist and migrates the DB to the most
    recent version. It is to simplify the deployment.

    As a bonus, you obtain a functional Alembic INI config for any `alembic` commands.
    """
    root = os.path.dirname(__file__)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(root))
    t = env.get_template("alembic.ini.jinja2")
    path = os.path.relpath(root)
    with open("alembic.ini", "w") as fout:
        fout.write(t.render(url=url, path=path))
    args = ["alembic", "alembic", "upgrade", "head"]
    if os.getenv("OFFLINE"):
        args.append("--sql")
    if exec:
        os.execlp(*args)
    else:
        subprocess.run(" ".join(args[1:]), check=True, shell=True)


if __name__ == "__main__":
    exit(main(sys.argv[1]))
