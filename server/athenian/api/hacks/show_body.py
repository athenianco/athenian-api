import base64
import json
import lzma
import os


def main():
    """Usage: `echo 'string for UUID'| python3 show_body.py`."""
    data = input()
    body = json.loads(lzma.decompress(base64.b64decode(data)).decode())
    indent = int(os.getenv("INDENT", "0")) or None
    print(json.dumps(body, indent=indent))


if __name__ == "__main__":
    exit(main())
