from datetime import datetime
from typing import Optional, Set, Tuple

import aiomcache
import databases
import pandas as pd
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api.controllers.features.entries import (
    MetricEntriesCalculator as OriginalMetricEntriesCalculator,
)
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.pull_request import (
    PullRequestMiner as OriginalPullRequestMiner,
)
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind
from athenian.api.models.metadata.github import PullRequest
from athenian.api.tracing import sentry_span


class PullRequestMiner(OriginalPullRequestMiner):
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    to access individual PR objects."""

    @classmethod
    def _prs_from_binary_expression(cls, bin_exp: BinaryExpression):
        return [bp.value for bp in bin_exp.get_children()[1].get_children()[0].get_children()]

    @classmethod
    @sentry_span
    async def _fetch_prs_by_filters(
        cls,
        time_from: datetime,
        time_to: datetime,
        repositories: Set[str],
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        pr_blacklist: Optional[BinaryExpression],
        pr_whitelist: Optional[BinaryExpression],
        meta_ids: Tuple[int, ...],
        mdb: databases.Database,
        cache: Optional[aiomcache.Client],
        columns=PullRequest,
        updated_min: Optional[datetime] = None,
        updated_max: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if jira or labels:
            # TODO: not supported yet, call original implementation
            return await super(PullRequestMiner, cls)._fetch_prs_by_filters(
                time_from,
                time_to,
                repositories,
                participants,
                labels,
                jira,
                exclude_inactive,
                pr_blacklist,
                pr_whitelist,
                meta_ids,
                mdb,
                cache,
                columns=columns,
                updated_min=updated_min,
                updated_max=updated_max,
            )

        assert (updated_min is None) == (updated_max is None)

        df = mdb.cache.dfs["prs"].df

        mask = (
            (~df["closed"] | (df["closed_at"] >= time_from))
            & (df["created_at"] < time_to)
            & (df["acc_id"].isin(meta_ids))
            & (~df["hidden"])
            & (df["repository_full_name"].isin(repositories))
        )

        if pr_blacklist is not None:
            pr_blacklist = cls._prs_from_binary_expression(pr_blacklist)
            mask &= ~df["node_id"].isin(pr_blacklist)
        if pr_whitelist is not None:
            pr_whitelist = cls._prs_from_binary_expression(pr_blacklist)
            mask &= df["node_id"].isin(pr_whitelist)

        if exclude_inactive and updated_min is None:
            # this does not provide 100% guarantee because it can be after time_to,
            # we need to properly filter later
            mask &= df["updated_at"] >= time_from
        if updated_min is not None:
            mask &= (df["updated_at"] >= updated_min) & (df["updated_at"] < updated_max)

        if len(participants) == 1:
            if PRParticipationKind.AUTHOR in participants:
                mask &= df["user_login"].isin(participants[PRParticipationKind.AUTHOR])
            elif PRParticipationKind.MERGER in participants:
                mask &= df["merged_by_login"].isin(
                    participants[PRParticipationKind.MERGER],
                )
        elif (
            len(participants) == 2
            and PRParticipationKind.AUTHOR in participants
            and PRParticipationKind.MERGER in participants
        ):
            mask &= df["user_login"].isin(
                participants[PRParticipationKind.AUTHOR],
            ) | df["merged_by_login"].isin(participants[PRParticipationKind.MERGER])

        if columns is PullRequest:
            selected_columns = []
            remove_acc_id = False
        else:
            selected_columns = columns = list(columns)
            if remove_acc_id := (PullRequest.acc_id not in selected_columns):
                selected_columns.append(PullRequest.acc_id)
            if (
                PullRequest.merge_commit_id in columns
                or PullRequest.merge_commit_sha in columns
            ):
                # needed to resolve rebased merge commits
                if PullRequest.number not in selected_columns:
                    selected_columns.append(PullRequest.number)

            selected_columns = [c.key for c in selected_columns]

        prs = mdb.cache.dfs["prs"].filter(mask, columns=selected_columns,
                                          index=PullRequest.node_id.key)

        if remove_acc_id:
            del prs[PullRequest.acc_id.key]
        if PullRequest.closed.key in df:
            cls.adjust_pr_closed_merged_timestamps(prs)

        return prs


class MetricEntriesCalculator(OriginalMetricEntriesCalculator):
    """Calculator for different metrics using preloaded DataFrames."""

    PRMiner = PullRequestMiner
