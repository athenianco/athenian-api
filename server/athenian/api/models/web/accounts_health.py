from datetime import datetime

from athenian.api.models.web.account_health import AccountHealth
from athenian.api.models.web.base_model_ import Model


class AccountsHealth(Model):
    """The metric measurement timestamps and the measured metric values for each requested \
    account."""

    accounts: dict[int, AccountHealth]
    datetimes: list[datetime]
