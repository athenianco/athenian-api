#!/usr/bin/env python

import base64
from datetime import datetime
import lzma
import pickle
import sys

import numpy as np
import pandas as pd


def main():
    """Usage: `echo 'string after UUID'| python3 show_sql.py`."""
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", None)
    data = input()
    query, args = pickle.loads(lzma.decompress(base64.b64decode(data)))
    if not query.endswith(";"):
        query = query + ";"
    explain = "EXPLAIN (ANALYZE, BUFFERS)"
    if sys.stdout.isatty():
        print("\033[1;31m%s\033[0m" % explain)
    else:
        print(explain)
    args = list(enumerate(args, start=1))
    for i, arg in reversed(args):
        if isinstance(arg, datetime):
            arg = "'%s'" % arg.isoformat(" ")
        elif isinstance(arg, (list, np.ndarray, pd.Series)):
            if len(arg) == 0:
                dtype = "text"  # the best we can do here
            elif isinstance(arg[0], datetime):
                dtype = "timestamp with time zone"
            elif isinstance(arg[0], (int, np.integer)):
                dtype = "bigint"
            else:
                dtype = "text"
            arg = "ARRAY[" + ",".join("'%s'" % v for v in arg) + f"]::{dtype}[]"
        elif isinstance(arg, str):
            arg = "'%s'" % arg
        else:
            arg = str(arg)
        query = query.replace("$%d" % i, arg)
    print(query)


if __name__ == "__main__":
    exit(main())
