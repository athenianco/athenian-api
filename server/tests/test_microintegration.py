import os
import subprocess
import sys
from threading import Condition, Thread

import pytest

from athenian.api.models import migrate
from tests.conftest import db_dir


@pytest.mark.parametrize("gunicorn", [False, True])
def test_integration_micro(
    metadata_db,
    unused_tcp_port_factory,
    worker_id,
    locked_migrations,
    gunicorn,
):
    with locked_migrations:
        _test_integration_micro(metadata_db, unused_tcp_port_factory, worker_id, gunicorn)


def _test_integration_micro(metadata_db, aiohttp_unused_port, worker_id, gunicorn):
    state_db_path = db_dir / ("sdb-%s.sqlite" % worker_id)
    state_db = "sqlite:///%s" % state_db_path
    if state_db_path.exists():
        state_db_path.unlink()
    migrate("state", state_db, exec=False)
    precomputed_db_path = db_dir / ("pdb-%s.sqlite" % worker_id)
    precomputed_db = "sqlite:///%s" % precomputed_db_path
    if precomputed_db_path.exists():
        precomputed_db_path.unlink()
    migrate("precomputed", precomputed_db, exec=False)
    persistentdata_db_path = db_dir / ("rdb-%s.sqlite" % worker_id)
    persistentdata_db = "sqlite:///%s" % persistentdata_db_path
    if persistentdata_db_path.exists():
        persistentdata_db_path.unlink()
    migrate("persistentdata", persistentdata_db, exec=False)
    unused_port = str(aiohttp_unused_port())
    env = os.environ.copy()
    env["ATHENIAN_INVITATION_URL_PREFIX"] = "https://app.athenian.co/i/"
    env[
        "ATHENIAN_JIRA_INSTALLATION_URL_TEMPLATE"
    ] = "https://installation.athenian.co/jira/%s/atlassian-connect.json"
    env["ATHENIAN_INVITATION_KEY"] = "vadim"
    env["ATHENIAN_DEFAULT_USER"] = "github|60340680"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "athenian.api",
            "--ui",
            "--no-google-kms",
            "--metadata-db=" + metadata_db,
            "--state-db=" + state_db,
            "--precomputed-db=" + precomputed_db,
            "--persistentdata-db=" + persistentdata_db,
            "--memcached=localhost:11211",
            "--port=" + unused_port,
            "-n=" + ["1", "2"][gunicorn],
        ],
        encoding="utf-8",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    kill_cond = Condition()

    def kill():
        with kill_cond:
            kill_cond.wait([20, 40][gunicorn])
        if proc.poll() is None:
            proc.terminate()

    killer = Thread(target=kill)
    killer.start()
    wins = 0
    n_checks = 6 * [1, 2][gunicorn]
    for line in proc.stderr:
        print(line.rstrip())
        if "Connected to the server state DB" in line:
            wins += 1
        if "Connected to the metadata DB" in line:
            wins += 1
        if "Connected to the precomputed DB" in line:
            wins += 1
        if "Acquired new Auth0 management token" in line:
            wins += 1
        if "JWKS records" in line:
            wins += 1
        if gunicorn:
            if ("constructed and running" + unused_port) in line:
                wins += 1
        else:
            if ("Listening on 0.0.0.0:" + unused_port) in line:
                wins += 1
        if wins == n_checks:
            with kill_cond:
                kill_cond.notify_all()
    killer.join()
    assert not proc.poll(), "Server crashed"
    proc.terminate()
