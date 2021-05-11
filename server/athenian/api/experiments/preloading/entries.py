from datetime import datetime
import functools
import operator
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api.controllers.features.entries import MetricEntriesCalculator
from athenian.api.controllers.features.github.unfresh_pull_request_metrics import \
    UnfreshPullRequestFactsFetcher
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.github.release_load import \
    match_groups_to_conditions, ReleaseLoader
from athenian.api.controllers.miners.github.release_load import \
    remove_ambigous_precomputed_releases
from athenian.api.controllers.miners.github.release_match import ReleaseToPullRequestMapper
from athenian.api.controllers.miners.jira.issue import PullRequestJiraMapper
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind
from athenian.api.controllers.settings import ReleaseMatch
from athenian.api.models.metadata.github import PullRequest
from athenian.api.preloading.cache import MCID, PCID
from athenian.api.tracing import sentry_span


class PreloadedReleaseLoader(ReleaseLoader):
    """Loader for preloaded releases."""

    @classmethod
    @sentry_span
    async def _fetch_precomputed_releases(cls,
                                          match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
                                          time_from: datetime,
                                          time_to: datetime,
                                          account: int,
                                          pdb: databases.Database,
                                          index: Optional[Union[str, Sequence[str]]] = None,
                                          ) -> pd.DataFrame:
        cached_df = pdb.cache.dfs[PCID.releases]
        df = cached_df.df
        mask = cls._match_groups_to_mask(df, match_groups)
        releases = cached_df.filter(mask)
        releases.sort_values("published_at", ascending=False, inplace=True)
        releases = remove_ambigous_precomputed_releases(releases, "repository_full_name")
        if index is not None:
            releases.set_index(index, inplace=True)
        else:
            releases.reset_index(drop=True, inplace=True)
        return releases

    @classmethod
    def _match_groups_to_mask(
            cls,
            df: pd.DataFrame,
            match_groups: Dict[ReleaseMatch, Dict[str, Iterable[str]]]) -> pd.Series:
        or_conditions, _ = match_groups_to_conditions(match_groups)
        or_masks = [
            (df["release_match"] == cond["release_match"]) &
            (df["repository_full_name"].isin(cond["repository_full_name"]))
            for cond in or_conditions
        ]

        return functools.reduce(operator.or_, or_masks)


class PreloadedReleaseToPullRequestMapper(ReleaseToPullRequestMapper):
    """Mapper from preloaded releases to pull requests."""

    release_loader = PreloadedReleaseLoader


class PreloadedUnfreshPullRequestFactsFetcher(UnfreshPullRequestFactsFetcher):
    """Fetcher for preloaded unfresh pull requests facts."""

    release_loader = PreloadedReleaseLoader


class PreloadedBranchMiner(BranchMiner):
    """Load information related to preloaded branches."""

    @classmethod
    async def _extract_branches(cls,
                                repos: Iterable[str],
                                meta_ids: Tuple[int, ...],
                                mdb: databases.Database,
                                ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        df = mdb.cache.dfs[MCID.branches].df

        mask = (
            df["repository_full_name"].isin(repos)
            & df["acc_id"].isin(meta_ids)
            & df["commit_sha"].notna()
        )

        return mdb.cache.dfs[MCID.branches].filter(mask)


class PreloadedPullRequestMiner(PullRequestMiner):
    """Load all the information related to PRS from the metadata DB with some preloaded methods. \
    Iterate over it to access individual PR objects."""

    mappers = PullRequestMiner.AuxiliaryMappers(
        releases_to_prs=PreloadedReleaseToPullRequestMapper.map_releases_to_prs,
        prs_to_releases=PullRequestMiner.mappers.prs_to_releases,
    )

    @classmethod
    def _prs_from_binary_expression(cls, bin_exp: BinaryExpression):
        return [bp.value.encode()
                for bp in bin_exp.get_children()[1].get_children()[0].get_children()]

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
        # FIXME: prefer numpy operations rather than pandas for better performance
        # See:
        #   - https://github.com/athenianco/athenian-api/pull/1337#discussion_r621071935
        #   - https://github.com/athenianco/athenian-api/pull/1337#discussion_r621073088
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

        df = mdb.cache.dfs[MCID.prs].df

        mask = (
            (~df["closed"] | (df["closed_at"] >= time_from))
            & (df["created_at"] < time_to)
            & (df["acc_id"].isin(meta_ids))
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

        prs = mdb.cache.dfs[MCID.prs].filter(mask, columns=selected_columns,
                                             index=PullRequest.node_id.key)

        if remove_acc_id:
            del prs[PullRequest.acc_id.key]
        if PullRequest.closed.key in prs:
            cls.adjust_pr_closed_merged_timestamps(prs)

        return prs


class PreloadedPullRequestJiraMapper(PullRequestJiraMapper):
    """Mapper of pull requests to JIRA tickets."""

    @classmethod
    @sentry_span
    async def load_pr_jira_mapping(cls,
                                   prs: Collection[str],
                                   meta_ids: Tuple[int, ...],
                                   mdb: databases.Database) -> Dict[str, str]:
        """Fetch the mapping from PR node IDs to JIRA issue IDs."""
        cached_df = mdb.cache.dfs[MCID.jira_mapping]
        df = cached_df.df
        mask = df["node_id"].isin([v.encode() for v in prs]) & df["node_acc"].isin(meta_ids)
        mapping = cached_df.filter(mask)
        return dict(zip(mapping["node_id"].values, mapping["jira_id"].values))


class MetricEntriesCalculator(MetricEntriesCalculator):
    """Calculator for different metrics using preloaded DataFrames."""

    pr_miner = PullRequestMiner
    branch_miner = PreloadedBranchMiner
    unfresh_pr_facts_fetcher = UnfreshPullRequestFactsFetcher
    pr_jira_mapper = PullRequestJiraMapper
