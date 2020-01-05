try:
    import pytest
    pytest.fixture = lambda fn: fn
except ImportError:
    pass

from tests.controllers.conftest import metadata_db, state_db

if __name__ == "__main__":
    metadata_db()
    state_db()
