import asyncio
from datetime import datetime
from typing import Collection, Dict, Optional, Sequence, Tuple

import aiomcache
import morcilla
import numpy as np
import pandas as pd
from sqlalchemy import and_, func, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.controllers.jira import JIRAConfig
from athenian.api.controllers.miners.filters import LabelFilter
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues
from athenian.api.controllers.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.jira import Issue
from athenian.api.tracing import sentry_span


@sentry_span
async def filter_epics(jira_ids: JIRAConfig,
                       time_from: Optional[datetime],
                       time_to: Optional[datetime],
                       exclude_inactive: bool,
                       labels: LabelFilter,
                       priorities: Collection[str],
                       reporters: Collection[str],
                       assignees: Collection[Optional[str]],
                       commenters: Collection[str],
                       default_branches: Dict[str, str],
                       release_settings: ReleaseSettings,
                       logical_settings: LogicalRepositorySettings,
                       account: int,
                       meta_ids: Tuple[int, ...],
                       mdb: morcilla.Database,
                       pdb: morcilla.Database,
                       cache: Optional[aiomcache.Client],
                       extra_columns: Collection[InstrumentedAttribute] = (),
                       ) -> Tuple[pd.DataFrame,
                                  pd.DataFrame,
                                  asyncio.Task,  # -> List[Mapping[str, Union[str, int]]]
                                  Dict[str, Sequence[int]]]:
    """
    Fetch JIRA epics and their children issues according to the given filters.

    :return: 1. epics \
             2. children \
             3. asyncio Task to fetch the subtask counts \
             4. map from epic_id to the indexes of the corresponding children in (2)
    """
    # filter the epics according to the passed filters
    epics = await fetch_jira_issues(
        jira_ids, time_from, time_to, exclude_inactive, labels,
        priorities, ["epic"], [], reporters, assignees, commenters, True,
        default_branches, release_settings, logical_settings, account, meta_ids, mdb, pdb, cache,
        extra_columns=extra_columns)
    if epics.empty:
        async def noop():
            return []

        return (epics,
                pd.DataFrame(columns=[
                    Issue.priority_id.name, Issue.status_id.name, Issue.project_id.name]),
                asyncio.create_task(noop()),
                {})
    # discover the issues belonging to those epics
    extra_columns = list(extra_columns)
    if Issue.parent_id not in extra_columns:
        extra_columns.append(Issue.parent_id)
    children = await fetch_jira_issues(
        jira_ids, None, None, False, LabelFilter.empty(),
        [], [], epics[Issue.key.name].values, [], [], [], False,
        default_branches, release_settings, logical_settings, account, meta_ids, mdb, pdb, cache,
        extra_columns=extra_columns)
    # plan to fetch the subtask counts, but not await it now
    subtasks = asyncio.create_task(
        mdb.fetch_all(select([Issue.parent_id, func.count(Issue.id).label("subtasks")])
                      .where(and_(Issue.acc_id == jira_ids[0],
                                  Issue.project_id.in_(jira_ids[1]),
                                  Issue.is_deleted.is_(False),
                                  Issue.parent_id.in_(children.index)))
                      .group_by(Issue.parent_id)),
        name="fetch JIRA subtask counts",
    )
    await asyncio.sleep(0)
    empty_epic_ids_mask = children[Issue.epic_id.name].isnull()
    children.loc[empty_epic_ids_mask, Issue.epic_id.name] = \
        children[Issue.parent_id.name][empty_epic_ids_mask]
    children_epic_ids = children[Issue.epic_id.name].values
    order = np.argsort(children_epic_ids)
    children_epic_ids = children_epic_ids[order]
    unique_children_epic_ids, counts = np.unique(children_epic_ids, return_counts=True)
    children_indexes = np.split(np.arange(len(order))[order], np.cumsum(counts)[:-1])
    epic_id_to_children_indexes = dict(zip(unique_children_epic_ids, children_indexes))
    return epics, children, subtasks, epic_id_to_children_indexes
