import os

import pytest
import sentry_sdk


@pytest.mark.skipif(
    not os.getenv("SENTRY_PROJECT") or not os.getenv("SENTRY_KEY"),
    reason="Sentry is not configured.",
)
def test_sentry_report():
    sentry_sdk.capture_message("unit testing Athenian API")
