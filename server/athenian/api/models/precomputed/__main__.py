import sys

from athenian.api.models import migrate

if __name__ == "__main__":
    exit(migrate("precomputed", sys.argv[1]))
