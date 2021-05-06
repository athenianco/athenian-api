from datetime import datetime
import functools
import operator
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import aiomcache
import databases
import pandas as pd
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api.controllers.features.entries import \
    MetricEntriesCalculator as OriginalMetricEntriesCalculator
from athenian.api.controllers.features.github.unfresh_pull_request_metrics import \
    UnfreshPullRequestFactsFetcher as OriginalUnfreshPullRequestFactsFetcher
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.pull_request import \
    PullRequestMiner as OriginalPullRequestMiner
from athenian.api.controllers.miners.github.release_load import \
    ReleaseLoader as OriginalReleaseLoader
from athenian.api.controllers.miners.github.release_load import \
    remove_ambigous_precomputed_releases
from athenian.api.controllers.miners.github.release_match import \
    ReleaseToPullRequestMapper as OriginalReleaseToPullRequestMapper
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind
from athenian.api.controllers.settings import ReleaseMatch
from athenian.api.models.metadata.github import PullRequest
from athenian.api.tracing import sentry_span


class ReleaseLoader(OriginalReleaseLoader):
    """Loader for releases."""

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
        cached_df = pdb.cache.dfs["releases"]
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
            df: pd.DataFrame,
            match_groups: Dict[ReleaseMatch, Dict[str, Iterable[str]]]) -> pd.Series:
        or_masks = []
        for match, suffix in [
            (ReleaseMatch.tag, "|"),
            (ReleaseMatch.branch, "|"),
            (ReleaseMatch.rejected, ""),
            (ReleaseMatch.force_push_drop, ""),
            (ReleaseMatch.event, ""),
        ]:
            if not (match_group := match_groups.get(match)):
                continue

            and_masks = [(
                df["release_match"] == "".join([match.name, suffix, v]) &
                df["repository_full_name"].isin(r)
            ) for v, r in match_group.items()]

            or_masks.append(functools.reduce(operator.and_, and_masks))

        return functools.reduce(operator.or_, or_masks)


class ReleaseToPullRequestMapper(OriginalReleaseToPullRequestMapper):
    """Mapper from releases to pull requests."""

    release_loader = ReleaseLoader


class UnfreshPullRequestFactsFetcher(OriginalUnfreshPullRequestFactsFetcher):
    """Fetcher for unfreshed pull requests facts."""

    release_loader = ReleaseLoader


class PullRequestMiner(OriginalPullRequestMiner):
    """Load all the information related to Pull Requests from the metadata DB. Iterate over it \
    to access individual PR objects."""

    mappers = OriginalPullRequestMiner.AuxiliaryMappers(
        releases_to_prs=ReleaseToPullRequestMapper.map_releases_to_prs,
        prs_to_releases=OriginalPullRequestMiner.mappers.prs_to_releases,
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

        df = mdb.cache.dfs["prs"].df

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

        prs = mdb.cache.dfs["prs"].filter(mask, columns=selected_columns,
                                          index=PullRequest.node_id.key)

        if remove_acc_id:
            del prs[PullRequest.acc_id.key]
        if PullRequest.closed.key in df:
            cls.adjust_pr_closed_merged_timestamps(prs)

        return prs


class MetricEntriesCalculator(OriginalMetricEntriesCalculator):
    """Calculator for different metrics using preloaded DataFrames."""

    miner = PullRequestMiner
    unfresh_fetcher = UnfreshPullRequestFactsFetcher
