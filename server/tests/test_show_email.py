from unittest.mock import patch

from athenian.api.hacks.show_email import main


def test_show_email(capsys):
    with patch(
        "sys.argv",
        [
            "show_sql.py",
            "af253b50a4d7b2c9841f436fbe4c635f270f4388653649b0971f2751a441a556fe63a9dabfa150a444dd",
        ],
    ):  # noqa
        main()
    assert capsys.readouterr().out == "vadim@athenian.co\n"
