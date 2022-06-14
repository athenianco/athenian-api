import logging
import os

import pytest
import sentry_sdk

from athenian.api.__main__ import setup_context


@pytest.mark.skipif(
    not os.getenv("SENTRY_PROJECT") or not os.getenv("SENTRY_KEY"),
    reason="Sentry is not configured.",
)
def test_sentry_report():
    setup_context(logging.getLogger("test-api"))
    sentry_sdk.capture_message("unit testing Athenian API")
