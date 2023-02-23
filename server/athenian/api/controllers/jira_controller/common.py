from dataclasses import dataclass

import pandas as pd

from athenian.api.async_utils import gather
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.jira import JIRAConfig, get_jira_installation
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import get_account_repositories
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.request import AthenianWebRequest


@dataclass(frozen=True, slots=True)
class AccountInfo:
    """The information and settings for the account used by multiple Jira controllers."""

    account: int
    meta_ids: tuple[int, ...]
    jira_conf: JIRAConfig
    branches: pd.DataFrame | None
    default_branches: dict[str, str] | None
    release_settings: ReleaseSettings | None
    logical_settings: LogicalRepositorySettings | None
    prefixer: Prefixer


async def collect_account_info(
    account: int,
    request: AthenianWebRequest,
    with_branches_and_settings: bool,
) -> AccountInfo:
    """Collect the AccountInfo from the request."""
    sdb, mdb, pdb, cache = request.sdb, request.mdb, request.pdb, request.cache
    meta_ids = await get_metadata_account_ids(account, sdb, cache)
    prefixer = await Prefixer.load(meta_ids, mdb, cache)
    repos, jira_ids = await gather(
        get_account_repositories(account, prefixer, sdb),
        get_jira_installation(account, sdb, mdb, cache),
        op="sdb/ids",
    )
    repos = [str(r) for r in repos]
    if with_branches_and_settings:
        settings = Settings.from_request(request, account, prefixer)
        (branches, default_branches), logical_settings = await gather(
            BranchMiner.load_branches(
                repos, prefixer, account, meta_ids, mdb, pdb, cache, strip=True,
            ),
            settings.list_logical_repositories(repos),
            op="sdb/branches and releases",
        )
        repos = logical_settings.append_logical_prs(repos)
        release_settings = await settings.list_release_matches(repos)
    else:
        branches = release_settings = logical_settings = None
        default_branches = {}
    return AccountInfo(
        account,
        meta_ids,
        jira_ids,
        branches,
        default_branches,
        release_settings,
        logical_settings,
        prefixer,
    )
