import logging

from athenian.api import slogging


def test_structured_logging():
    slogging.setup("INFO", True)
    logging.getLogger().info("%s", "test")


def test_regular_logging():
    slogging.setup("INFO", False)
    logging.getLogger().info("%s", "test")
