import logging
from typing import Optional

import aiomcache
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient

from athenian.api.db import Database
from athenian.api.typing_utils import dataclass


@dataclass(slots=True)
class PrecomputeContext:
    """Everything initialized for a command to execute."""

    log: logging.Logger
    sdb: Database
    mdb: Database
    pdb: Database
    rdb: Database
    cache: Optional[aiomcache.Client]
    slack: Optional[SlackWebClient]
