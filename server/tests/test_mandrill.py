import os

import pytest


@pytest.mark.skipif(not os.getenv("MANDRILL_API_KEY"), reason="MANDRILL_TOKEN is not defined")
async def test_mandrill_template_send(mandrill):
    result = await mandrill.messages_send_template(
        "test", [("vadim@athenian.co",)],
        name="Vadim", user_id="1247608", account_id=1, topic="[CI]")
    assert result == [True]
