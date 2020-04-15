import json
import sys
from urllib.request import Request, urlopen


def main():
    """Become the specified user by Auth0 ID or login."""
    if len(sys.argv) < 2:
        print("Usage: become.py <bearer> [<user login or Auth0 ID>]", file=sys.stderr)
        return 1
    bearer = sys.argv[1]
    user = sys.argv[2] if len(sys.argv) > 2 else ""
    if not user.startswith("github|") and user:
        with urlopen("https://api.github.com/users/" + user) as http:
            user = "github|%d" % json.loads(http.read().decode())["id"]
    url = "https://api.athenian.co/v1/become"
    if user:
        url += "?id=" + user
    req = Request(url)
    req.add_header("Authorization", "Bearer " + bearer)
    with urlopen(req) as http:
        print(http.read().decode())


if __name__ == "__main__":
    exit(main())
