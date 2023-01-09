from datetime import datetime
from typing import Collection, Dict, Iterable, Optional, Sequence, Tuple

import aiomcache
import morcilla
import numpy as np
import pandas as pd
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.jira.issue import fetch_jira_issues
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings
from athenian.api.models.metadata.jira import Issue
from athenian.api.tracing import sentry_span


@sentry_span
async def filter_epics(
    jira_ids: JIRAConfig,
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
) -> Tuple[pd.DataFrame, pd.DataFrame, Iterable[Tuple[str, int]], Dict[bytes, Sequence[int]]]:
    """
    Fetch JIRA epics and their children issues according to the given filters.

    :return: 1. epics \
             2. children \
             3. subtask counts \
             4. map from epic_id to the indexes of the corresponding children in (2)
    """
    # filter the epics according to the passed filters
    candidate_types = jira_ids.epic_candidate_types()
    if candidate_types != {"epic"}:
        for col in (Issue.project_id, Issue.type):
            if col not in extra_columns:
                extra_columns = (*extra_columns, col)
    if Issue.epic_id not in extra_columns:
        extra_columns = (*extra_columns, Issue.epic_id)
    jira_filter = JIRAFilter.from_jira_config(jira_ids).replace(
        labels=labels, issue_types=candidate_types, priorities=priorities,
    )

    async def fetch_epic_children(issues: pd.DataFrame) -> pd.DataFrame:
        # discover the issues belonging to those epics
        if issues.empty:
            return pd.DataFrame()
        nonlocal extra_columns
        extra_columns = list(extra_columns)
        if Issue.parent_id not in extra_columns:
            extra_columns.append(Issue.parent_id)
        return await fetch_jira_issues(
            None,
            None,
            JIRAFilter.from_jira_config(jira_ids).replace(epics=issues[Issue.key.name].values),
            False,
            [],
            [],
            [],
            False,
            default_branches,
            release_settings,
            logical_settings,
            account,
            meta_ids,
            mdb,
            pdb,
            cache,
            extra_columns=extra_columns,
        )

    epics, children = await fetch_jira_issues(
        time_from,
        time_to,
        jira_filter,
        exclude_inactive,
        reporters,
        assignees,
        commenters,
        True,
        default_branches,
        release_settings,
        logical_settings,
        account,
        meta_ids,
        mdb,
        pdb,
        cache,
        extra_columns=extra_columns,
        on_raw_fetch_complete=fetch_epic_children,
    )
    if epics.empty:
        return (
            epics,
            pd.DataFrame(
                {
                    Issue.priority_id.name: np.array([], dtype="S8"),
                    Issue.status_id.name: np.array([], dtype="S8"),
                    Issue.project_id.name: np.array([], dtype="S8"),
                },
            ),
            [],
            {},
        )
    if candidate_types != {"epic"}:
        projects = epics[Issue.project_id.name].values.astype("S")
        types = epics[Issue.type.name].values.astype("S")
        df_pairs = np.char.add(np.char.add(projects, b"/"), types)
        indexes = np.flatnonzero(np.in1d(df_pairs, jira_ids.project_epic_pairs()))
        if len(indexes) < len(epics):
            epics.disable_consolidate()
            epics = epics.take(indexes)

    children_parent_ids = children[Issue.parent_id.name].values
    nnz_parent_mask = children_parent_ids != b""
    subtask_mask = (children[Issue.epic_id.name].values != children_parent_ids) & nnz_parent_mask
    unique_parent_ids, subtask_counts = np.unique(
        children_parent_ids[subtask_mask], return_counts=True,
    )
    subtask_counts = zip(unique_parent_ids, subtask_counts)
    children.disable_consolidate()
    children = children.take(np.flatnonzero(~subtask_mask))
    children_epic_ids = children[Issue.epic_id.name].values
    order = np.argsort(children_epic_ids)
    children_epic_ids = children_epic_ids[order]
    unique_children_epic_ids, counts = np.unique(children_epic_ids, return_counts=True)
    children_indexes = np.split(np.arange(len(order))[order], np.cumsum(counts)[:-1])
    epic_id_to_children_indexes = dict(zip(unique_children_epic_ids, children_indexes))
    return epics, children, subtask_counts, epic_id_to_children_indexes
