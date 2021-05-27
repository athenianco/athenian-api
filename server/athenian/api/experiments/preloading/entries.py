from collections import defaultdict
from datetime import datetime, timezone
import functools
import logging
import operator
from typing import Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import aiomcache
import databases
import numpy as np
import pandas as pd
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.controllers.features.entries import \
    MetricEntriesCalculator as OriginalMetricEntriesCalculator
from athenian.api.controllers.features.github.unfresh_pull_request_metrics import \
    UnfreshPullRequestFactsFetcher
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.precomputed_prs import \
    MergedPRFactsLoader, OpenPRFactsLoader
from athenian.api.controllers.miners.github.precomputed_prs.utils import extract_release_match
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.github.release_load import \
    match_groups_to_conditions, ReleaseLoader
from athenian.api.controllers.miners.github.release_load import \
    remove_ambigous_precomputed_releases
from athenian.api.controllers.miners.github.release_match import ReleaseToPullRequestMapper
from athenian.api.controllers.miners.jira.issue import PullRequestJiraMapper
from athenian.api.controllers.miners.types import PRParticipants, PRParticipationKind, \
    PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseSettings
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
        mask = (
            cls._match_groups_to_mask(df, match_groups)
            & (df["acc_id"] == account)
            & (df["published_at"] >= time_from)
            & (df["published_at"] < time_to)
        )
        releases = cached_df.filter(mask)
        releases.sort_values("published_at", ascending=False, inplace=True)
        releases = remove_ambigous_precomputed_releases(releases, "repository_full_name")
        if index is not None:
            releases.set_index(index, inplace=True)
        else:
            releases.reset_index(drop=True, inplace=True)
        return releases

    @classmethod
    @sentry_span
    async def fetch_precomputed_release_match_spans(
            cls,
            match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
            account: int,
            pdb: databases.Database) -> Dict[str, Dict[str, Tuple[datetime, datetime]]]:
        """Find out the precomputed time intervals for each release match group of repositories."""
        cached_df = pdb.cache.dfs[PCID.releases_match_timespan]
        df = cached_df.df
        mask = cls._match_groups_to_mask(df, match_groups) & (df["acc_id"] == account)
        release_match_spans = cached_df.filter(mask)
        spans = {}
        for time_from, time_to, release_match, repository_full_name in zip(
                release_match_spans["time_from"],
                release_match_spans["time_to"],
                release_match_spans["release_match"].values,
                release_match_spans["repository_full_name"].values,
        ):
            if release_match.startswith("tag|"):
                release_match = ReleaseMatch.tag
            else:
                release_match = ReleaseMatch.branch

            times = time_from, time_to
            spans.setdefault(repository_full_name, {})[release_match] = times

        return spans

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

        return (functools.reduce(operator.or_, or_masks) if or_masks
                else np.ones(len(df), dtype=bool))


class PreloadedReleaseToPullRequestMapper(ReleaseToPullRequestMapper):
    """Mapper from preloaded releases to pull requests."""

    release_loader = PreloadedReleaseLoader


class PreloadedMergedPRFactsLoader(MergedPRFactsLoader):
    """Loader for preloaded merged PRs facts."""

    @classmethod
    @sentry_span
    async def load_merged_unreleased_pull_request_facts(
            cls,
            prs: pd.DataFrame,
            time_to: datetime,
            labels: LabelFilter,
            matched_bys: Dict[str, ReleaseMatch],
            default_branches: Dict[str, str],
            release_settings: ReleaseSettings,
            account: int,
            pdb: databases.Database,
            time_from: Optional[datetime] = None,
            exclude_inactive: bool = False,
    ) -> Dict[str, PullRequestFacts]:
        """
        Load the mapping from PR node identifiers which we are sure are not released in one of \
        `releases` to the serialized facts.

        For each merged PR we maintain the set of releases that do include that PR.

        :return: Map from PR node IDs to their facts.
        """
        if labels:
            # TODO: not supported yet, call original implementation
            return await super(PreloadedMergedPRFactsLoader, cls)\
                .load_merged_unreleased_pull_request_facts(
                    prs,
                    time_to,
                    labels,
                    matched_bys,
                    default_branches,
                    release_settings,
                    account,
                    pdb,
                    time_from=time_from,
                    exclude_inactive=exclude_inactive,
            )

        if time_to != time_to:
            return {}
        assert time_to.tzinfo is not None
        if exclude_inactive:
            assert time_from is not None
        log = logging.getLogger("%s.load_merged_unreleased_pull_request_facts" %
                                metadata.__package__)

        cached_df = pdb.cache.dfs[PCID.merged_pr_facts]
        df = cached_df.df

        common_mask = (
            (df["checked_until"] >= time_to) &
            (df["acc_id"] == account)
        )

        if exclude_inactive:
            activity_days = np.concatenate(df["activity_days"])
            activity_mask = np.full(len(df["activity_days"]), False)
            activity_days_in_range = (
                (time_from.replace(tzinfo=timezone.utc) <= activity_days) &
                (activity_days < time_to.replace(tzinfo=timezone.utc))
            )
            activity_offsets = np.cumsum(df["activity_days"].apply(len).values)
            indexes = np.searchsorted(activity_offsets, np.nonzero(activity_days_in_range)[0],
                                      side="right")
            activity_mask[indexes] = 1

            common_mask &= activity_mask

        repos_by_match = defaultdict(list)
        for repo in prs[PullRequest.repository_full_name.key].unique():
            if (release_match := extract_release_match(
                    repo, matched_bys, default_branches, release_settings)) is None:
                # no new releases
                continue
            repos_by_match[release_match].append(repo)

        or_masks = []
        pr_repos = prs[PullRequest.repository_full_name.key].values.astype("S")
        pr_ids = prs.index.values.astype("S")
        for release_match, repos in repos_by_match.items():
            or_masks.append(
                common_mask &
                df["pr_node_id"].isin(pr_ids[np.in1d(pr_repos, np.array(repos, dtype="S"))]) &
                df["repository_full_name"].isin(repos) &
                (df["release_match"] == release_match),
            )
        if not or_masks:
            return {}

        mask = functools.reduce(operator.or_, or_masks)
        merged_pr_facts = cached_df.filter(mask, columns=[
            "pr_node_id", "repository_full_name", "data", "author", "merger",
        ])

        facts = {}
        for node_id, data, repository_full_name, author, merger in zip(
                merged_pr_facts["pr_node_id"].values, merged_pr_facts["data"].values,
                merged_pr_facts["repository_full_name"].values,
                merged_pr_facts["author"].values, merged_pr_facts["merger"].values):
            if data is None:
                # There are two known cases:
                # 1. When we load all PRs without a blacklist (/filter/pull_requests) so some
                #    merged PR is matched to releases but exists in
                #    `github_done_pull_request_facts`.
                # 2. "Impossible" PRs that are merged.
                log.warning("No precomputed facts for merged %s", node_id)
                continue
            facts[node_id] = PullRequestFacts(
                data=data, repository_full_name=repository_full_name,
                author=author, merger=merger)
        return facts


class PreloadedOpenPRFactsLoader(OpenPRFactsLoader):
    """Loader for preloaded open PRs facts."""

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts_unfresh(cls,
                                                   prs: Iterable[str],
                                                   time_from: datetime,
                                                   time_to: datetime,
                                                   exclude_inactive: bool,
                                                   authors: Mapping[str, str],
                                                   account: int,
                                                   pdb: databases.Database,
                                                   ) -> Dict[str, PullRequestFacts]:
        """
        Fetch preloaded precomputed facts about the open PRs from the DataFrame.

        We don't filter PRs by the last update here.

        :param authors: Map from PR node IDs to their author logins.
        :return: Map from PR node IDs to their facts.
        """
        cached_df = pdb.cache.dfs[PCID.open_pr_facts]
        df = cached_df.df
        mask = (
            (df["acc_id"] == account)
            & df["pr_node_id"].isin(pr.encode() for pr in prs)
        )

        if exclude_inactive:
            activity_days = np.concatenate(df["activity_days"])
            activity_mask = np.full(len(df["activity_days"]), False)
            activity_days_in_range = (
                (time_from.replace(tzinfo=timezone.utc) <= activity_days) &
                (activity_days < time_to.replace(tzinfo=timezone.utc))
            )
            activity_offsets = np.cumsum(df["activity_days"].apply(len).values)
            indexes = np.searchsorted(activity_offsets, np.nonzero(activity_days_in_range)[0],
                                      side="right")
            activity_mask[indexes] = 1

            mask &= activity_mask

        open_prs_facts = cached_df.filter(mask, columns=[
            "pr_node_id", "repository_full_name", "data"])
        if open_prs_facts.empty:
            return {}

        facts = {
            pr_node_id: PullRequestFacts(
                data=data,
                repository_full_name=repository_full_name,
                author=authors[pr_node_id],
            )
            for pr_node_id, repository_full_name, data in zip(
                open_prs_facts["pr_node_id"].str.rstrip().values,
                open_prs_facts["repository_full_name"].values,
                open_prs_facts["data"].values,
            )
        }
        return facts


class PreloadedUnfreshPullRequestFactsFetcher(UnfreshPullRequestFactsFetcher):
    """Fetcher for preloaded unfresh pull requests facts."""

    release_loader = PreloadedReleaseLoader
    open_prs_facts_loader = PreloadedOpenPRFactsLoader
    merged_prs_facts_loader = PreloadedMergedPRFactsLoader


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
            return await super(PreloadedPullRequestMiner, cls)._fetch_prs_by_filters(
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


class MetricEntriesCalculator(OriginalMetricEntriesCalculator):
    """Calculator for different metrics using preloaded DataFrames."""

    pr_miner = PreloadedPullRequestMiner
    branch_miner = PreloadedBranchMiner
    unfresh_pr_facts_fetcher = PreloadedUnfreshPullRequestFactsFetcher
    pr_jira_mapper = PreloadedPullRequestJiraMapper
    load_delta = 10
