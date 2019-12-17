import os
import sys

import jinja2


def main():
    """Initialize the server state DB."""
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
