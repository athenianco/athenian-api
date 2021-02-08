import json
import logging

from athenian.api import slogging


def test_structured_logging(capsys):
    slogging.setup("INFO", structured=True)
    logging.getLogger().info("%s", "test")
    out = capsys.readouterr().out
    assert out
    json.loads(out)
