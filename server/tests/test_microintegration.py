import os
import subprocess
import sys
from threading import Thread, Condition

from athenian.api.models.state.__main__ import main as migrate
from tests.conftest import db_dir


def test_integration_micro(metadata_db, aiohttp_unused_port):
    state_db_path = db_dir / "sdb.sqlite"
    state_db = "sqlite:///%s" % state_db_path
    if state_db_path.exists():
        state_db_path.unlink()
    migrate(state_db, exec=False)
    unused_port = str(aiohttp_unused_port())
    env = os.environ.copy()
    env["ATHENIAN_INVITATION_URL_PREFIX"] = "https://app.athenian.co/i/"
    env["ATHENIAN_INVITATION_KEY"] = "secret"
    env["ATHENIAN_DEFAULT_USER"] = "github|60340680"
    proc = subprocess.Popen(
        [sys.executable, "-m", "athenian.api", "--ui", "--metadata-db=" + metadata_db,
         "--state-db=" + state_db, "--memcached=localhost:11211", "--port=" + unused_port],
        encoding="utf-8", text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    kill_cond = Condition()

    def kill():
        with kill_cond:
            kill_cond.wait(10)
        if proc.poll() is None:
            proc.kill()

    killer = Thread(target=kill)
    killer.start()
    wins = 0
    n_checks = 5
    for line in proc.stderr:
        print(line.rstrip())
        if "Connected to the server state DB" in line:
            wins += 1
        if "Connected to the metadata DB" in line:
            wins += 1
        if "Acquired new Auth0 management token" in line:
            wins += 1
        if "JWKS records" in line:
            wins += 1
        if ("Listening on 0.0.0.0:" + unused_port) in line:
            wins += 1
        if wins == n_checks:
            break
    with kill_cond:
        kill_cond.notify_all()
    killer.join()
    assert proc.poll() is None, "Server crashed"
    proc.terminate()
