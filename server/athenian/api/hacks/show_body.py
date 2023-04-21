import base64
import json
import lzma
import os


def main():
    """Usage: `echo 'string for UUID'| python3 show_body.py`."""
    data = input()
    body = json.loads(lzma.decompress(base64.b64decode(data)).decode())
    indent = int(os.getenv("INDENT", "0")) or None
    if not indent:
        print(
            "curl -H 'content-type: application/json' http://0.0.0.0:8081/ --data "
            f"'{json.dumps(body)}'",
        )
    else:
        print(json.dumps(body, indent=indent))


if __name__ == "__main__":
    exit(main())
