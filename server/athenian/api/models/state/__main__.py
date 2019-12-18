import os
import sys

import jinja2


def main():
    """
    Initialize the server state DB.

    This script creates all the tables if they don't exist and migrates the DB to the most
    recent version. It is to simplify the deployment.

    As a bonus, you obtain a functional Alembic INI config for any `alembic` commands.
    """
    url = sys.argv[1]
    root = os.path.dirname(__file__)
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(root))
    t = env.get_template("alembic.ini.jinja2")
    path = os.path.relpath(root)
    with open("alembic.ini", "w") as fout:
        fout.write(t.render(url=url, path=path))
    os.execlp("alembic", "alembic", "upgrade", "head")


if __name__ == "__main__":
    exit(main())
