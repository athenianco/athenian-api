import os
import sys

from athenian.api.ffx import decrypt


def main():
    """Print decrypted email."""
    if (key := os.getenv("ATHENIAN_INVITATION_KEY")) is None:
        raise EnvironmentError("Must define ATHENIAN_INVITATION_KEY environment variable.")
    if len(sys.argv) != 2:
        print("Usage: show_email.py <string>", file=sys.stderr)
        return 1
    encrypted = sys.argv[1]
    decrypted = decrypt(encrypted, key.encode())
    print(decrypted.split(b"|")[0].decode())


if __name__ == "__main__":
    sys.exit(main())
