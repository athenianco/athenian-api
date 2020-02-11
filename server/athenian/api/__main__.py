#!/usr/bin/env python3

from athenian.api import main

if __name__ == "__main__":
    exit(main() is None)  # "1" for an error, "0" for a normal return
