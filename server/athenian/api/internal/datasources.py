from __future__ import annotations

from typing import Iterable

from athenian.api.db import Database
from athenian.api.internal.jira import get_jira_id_or_none


class AccountDatasources(frozenset[str]):
    """The collection of datasources configured for an account."""

    GITHUB = "github"
    JIRA = "jira"

    _ALL = {GITHUB, JIRA}

    def __new__(cls, *args: Iterable[str]) -> AccountDatasources:
        """Build a new instance of `AccountDatasources`."""
        inst = super().__new__(cls, *args)
        if wrong := inst - cls._ALL:
            raise ValueError(f"invalid datasources: {sorted(wrong)}")
        return inst

    @classmethod
    async def build_for_account(cls, account: int, sdb: Database) -> AccountDatasources:
        """Build the datasources for the given account."""
        datasources = [cls.GITHUB]
        jira_id = await get_jira_id_or_none(account, sdb)
        if jira_id is not None:
            datasources.append(cls.JIRA)
        return cls(datasources)
