from pathlib import Path
import sys
import textwrap
from typing import TypeVar

from athenian.api.models import web


def main():
    """Generate __init__.py with all the exported classes and functions from submodules."""
    root = Path(__file__).parent
    all_imports = []
    for f in root.glob("*.py"):
        if f.name.startswith("_"):
            continue
        ip = "athenian.api.models.web." + f.stem
        print(ip, file=sys.stderr)
        __import__(ip)
        mod = getattr(web, f.stem)
        imports = []
        for k, v in vars(mod).items():
            try:
                if (
                    getattr(v, "__module__", None) == ip
                    and not isinstance(v, TypeVar)
                    and not v.__name__.startswith("_")
                ):
                    imports.append(k)
            except AttributeError:
                continue
        if not imports:
            continue
        imports.sort(key=str.casefold)
        line = "from %s import %s" % (ip, ", ".join(imports))
        lines = textwrap.wrap(line, width=97, subsequent_indent="    ", break_long_words=False)
        all_imports.append(" \\\n".join(lines) + "\n")
    all_imports.sort()
    with open(root / "__init__.py", "w") as fout:
        fout.writelines(all_imports)


if __name__ == "__main__":
    exit(main())
