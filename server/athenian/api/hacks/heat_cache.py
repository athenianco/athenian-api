import json
import subprocess
import sys
import tempfile


def _main():
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w+") as tmp:
        prolog = [
            subprocess.Popen([
                sys.executable, "-m", "athenian.api.precompute", "--xcom", tmp.name, *sys.argv[1:],
                cmd])
            for cmd in ("sync-labels",
                        "resolve-deployments",
                        "notify-almost-expired-accounts",
                        "discover-accounts")
        ]
        for p in prolog:
            try:
                p.wait()
            except BaseException as e:  # Including KeyboardInterrupt, wait handled that.
                p.kill()
                p.wait()
                raise e from None
        accounts = json.load(tmp)
    return subprocess.call([
        sys.executable, "-m", "athenian.api.precompute", *sys.argv[1:], "accounts",
    ] + [str(i) for i in accounts])


if __name__ == "__main__":
    exit(_main())
