from pathlib import Path
import sys
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
        all_imports.append("from %s import %s\n" % (ip, ", ".join(imports)))
    all_imports.sort()
    with open(root / "__init__.py", "w") as fout:
        fout.writelines(all_imports)


if __name__ == "__main__":
    exit(main())
