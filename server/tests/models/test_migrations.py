import subprocess
import sys
import tempfile


def test_migrations():
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run([sys.executable, "-m", "athenian.api.models.state", "sqlite://"],
                       check=True, cwd=tmpdir)
