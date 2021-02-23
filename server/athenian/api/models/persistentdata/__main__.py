import sys

from athenian.api.models import migrate

if __name__ == "__main__":
    exit(migrate("persistentdata", sys.argv[1]))
