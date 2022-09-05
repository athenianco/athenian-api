from collections import defaultdict
from datetime import datetime, timezone
import functools
import logging
import operator
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import aiomcache
import morcilla
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlalchemy.sql.elements import BinaryExpression

from athenian.api import metadata
from athenian.api.internal.features.entries import (
    MetricEntriesCalculator as OriginalMetricEntriesCalculator,
)
from athenian.api.internal.features.github.unfresh_pull_request_metrics import (
    UnfreshPullRequestFactsFetcher,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.precomputed_prs import (
    DonePRFactsLoader,
    MergedPRFactsLoader,
    OpenPRFactsLoader,
    triage_by_release_match,
)
from athenian.api.internal.miners.github.precomputed_prs.utils import extract_release_match
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.github.release_load import (
    ReleaseLoader,
    group_repos_by_release_match,
    match_groups_to_conditions,
    set_matched_by_from_release_match,
)
from athenian.api.internal.miners.github.release_match import ReleaseToPullRequestMapper
from athenian.api.internal.miners.types import (
    PRParticipants,
    PRParticipationKind,
    PullRequestFacts,
    PullRequestFactsMap,
    PullRequestID,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import (
    Base as MetadataGitHubBase,
    Branch,
    PullRequest,
    Release,
)
from athenian.api.models.precomputed.models import (
    GitHubBase as PrecomputedGitHubBase,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubOpenPullRequestFacts,
    GitHubRelease as PrecomputedRelease,
    GitHubReleaseMatchTimespan as PrecomputedGitHubReleaseMatchTimespan,
)
from athenian.api.preloading.cache import MCID, PCID
from athenian.api.tracing import sentry_span


def _build_activity_mask(
    model: Union[MetadataGitHubBase, PrecomputedGitHubBase],
    df: pd.DataFrame,
    time_from: datetime,
    time_to: datetime,
):
    activity_mask = np.zeros(len(df), bool)
    if df.empty:
        return activity_mask
    activity_days = np.concatenate(df[model.activity_days.name].values)
    activity_days_in_range = (time_from.replace(tzinfo=timezone.utc) <= activity_days) & (
        activity_days < time_to.replace(tzinfo=timezone.utc)
    )
    activity_offsets = np.cumsum(df[model.activity_days.name].apply(len).values)
    indexes = np.searchsorted(
        activity_offsets, np.nonzero(activity_days_in_range)[0], side="right",
    )
    activity_mask[indexes] = 1

    return activity_mask


def _match_groups_to_mask(
    model: Union[MetadataGitHubBase, PrecomputedGitHubBase],
    df: pd.DataFrame,
    match_groups: Dict[ReleaseMatch, Dict[str, Iterable[str]]],
) -> pd.Series:
    or_conditions, _ = match_groups_to_conditions(match_groups, model)
    or_masks = [
        (df[model.release_match.name] == cond["release_match"])
        & (df[model.repository_full_name.name].isin(cond["repository_full_name"]))
        for cond in or_conditions
    ]

    return functools.reduce(operator.or_, or_masks) if or_masks else np.ones(len(df), dtype=bool)


class PreloadedReleaseLoader(ReleaseLoader):
    """Loader for preloaded releases."""

    @classmethod
    @sentry_span
    async def _fetch_precomputed_releases(
        cls,
        match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
        time_from: datetime,
        time_to: datetime,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
        index: Optional[Union[str, Sequence[str]]] = None,
    ) -> pd.DataFrame:
        model = PrecomputedRelease
        cached_df = pdb.cache.dfs[PCID.releases]
        df = cached_df.get_dfs((account,))
        mask = (
            _match_groups_to_mask(model, df, match_groups)
            & (df[model.acc_id.name] == account)
            & (df[model.published_at.name] >= time_from)
            & (df[model.published_at.name] < time_to)
        )
        releases = cached_df.filter((account,), mask)
        df[Release.author_node_id.name].fillna(0, inplace=True)
        df[Release.author_node_id.name] = df[Release.author_node_id.name].astype(int)
        releases.sort_values(model.published_at.name, ascending=False, inplace=True)
        releases = set_matched_by_from_release_match(
            releases, True, model.repository_full_name.name,
        )
        if index is not None:
            releases.set_index(index, inplace=True)
        else:
            releases.reset_index(drop=True, inplace=True)
        user_node_to_login_get = prefixer.user_node_to_login.get
        releases[Release.author.name] = [
            user_node_to_login_get(u) for u in releases[model.author_node_id.name].values
        ]
        releases[Release.repository_node_id.name] = releases[
            Release.repository_node_id.name
        ].astype(int, copy=False)
        return releases

    @classmethod
    @sentry_span
    async def fetch_precomputed_release_match_spans(
        cls,
        match_groups: Dict[ReleaseMatch, Dict[str, List[str]]],
        account: int,
        pdb: morcilla.Database,
    ) -> Dict[str, Dict[str, Tuple[datetime, datetime]]]:
        """Find out the precomputed time intervals for each release match group of repositories."""
        model = PrecomputedGitHubReleaseMatchTimespan
        cached_df = pdb.cache.dfs[PCID.releases_match_timespan]
        df = cached_df.get_dfs((account,))
        mask = _match_groups_to_mask(model, df, match_groups) & (df[model.acc_id.name] == account)
        release_match_spans = cached_df.filter((account,), mask)
        spans = {}
        for time_from, time_to, release_match, repository_full_name in zip(
            release_match_spans[model.time_from.name],
            release_match_spans[model.time_to.name],
            release_match_spans[model.release_match.name].values,
            release_match_spans[model.repository_full_name.name].values,
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
        model: Union[MetadataGitHubBase, PrecomputedGitHubBase],
        df: pd.DataFrame,
        match_groups: Dict[ReleaseMatch, Dict[str, Iterable[str]]],
    ) -> pd.Series:
        or_conditions, _ = match_groups_to_conditions(match_groups, model)
        or_masks = [
            (df[model.release_match.name] == cond[PrecomputedRelease.release_match.name])
            & (
                df[model.repository_full_name.name].isin(
                    cond[PrecomputedRelease.repository_full_name.name],
                )
            )
            for cond in or_conditions
        ]

        return (
            functools.reduce(operator.or_, or_masks) if or_masks else np.ones(len(df), dtype=bool)
        )


class PreloadedReleaseToPullRequestMapper(ReleaseToPullRequestMapper):
    """Mapper from preloaded releases to pull requests."""

    release_loader = PreloadedReleaseLoader


class PreloadedDonePRFactsLoader(DonePRFactsLoader):
    """Loader for preloaded done PRs facts."""

    @classmethod
    @sentry_span
    async def _load_precomputed_done_filters(
        cls,
        columns: List[InstrumentedAttribute],
        time_from: Optional[datetime],
        time_to: Optional[datetime],
        repos: Collection[str],
        participants: PRParticipants,
        labels: LabelFilter,
        default_branches: Dict[str, str],
        exclude_inactive: bool,
        release_settings: ReleaseSettings,
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
    ) -> Tuple[Dict[PullRequestID, Mapping[str, Any]], Dict[str, List[int]]]:
        """
        Load some data belonging to released or rejected PRs from the preloaded precomputed DB.

        Query version. JIRA must be filtered separately.
        :return: 1. Map (PR node ID, repository name) -> specified column value. \
                 2. Map from repository name to ambiguous PR node IDs which are released by \
                 branch with tag_or_branch strategy and without tags on the time interval.
        """
        if labels:
            # TODO: not supported yet, call original implementation
            return await super(PreloadedDonePRFactsLoader, cls)._load_precomputed_done_filters(
                columns,
                time_from,
                time_to,
                repos,
                participants,
                labels,
                default_branches,
                exclude_inactive,
                release_settings,
                prefixer,
                account,
                pdb,
            )

        model = GitHubDonePullRequestFacts
        cached_df = pdb.cache.dfs[PCID.done_pr_facts]
        df = cached_df.get_dfs((account,))

        with sentry_sdk.start_span(op="_load_precomputed_done_filters/mask_creation"):
            mask = df[model.acc_id.name] == account
            if time_from is not None:
                mask &= df[model.pr_done_at.name] >= time_from
            if time_to is not None:
                mask &= df[model.pr_created_at.name] < time_to
            if repos is not None:
                mask &= df[model.repository_full_name.name].isin(repos)

            if len(participants) > 0:
                mask &= await cls._build_participants_mask(df, participants, prefixer)

            if exclude_inactive:
                with sentry_sdk.start_span(
                    op="_load_precomputed_done_filters/mask_creation/activity_days",
                ):
                    mask &= _build_activity_mask(model, df, time_from, time_to)

            match_groups, _ = group_repos_by_release_match(
                repos, default_branches, release_settings,
            )
            match_groups[ReleaseMatch.rejected] = match_groups[ReleaseMatch.force_push_drop] = {
                "": repos,
            }

            mask &= _match_groups_to_mask(model, df, match_groups)

        done_pr_facts = cached_df.filter(
            (account,),
            mask,
            columns=[
                model.pr_node_id.name,
                model.repository_full_name.name,
                model.release_match.name,
            ]
            + [c.name for c in columns],
        )

        with sentry_sdk.start_span(op="_load_precomputed_done_filters/triage"):
            result = {}
            ambiguous = {ReleaseMatch.tag.name: {}, ReleaseMatch.branch.name: {}}
            # we cannot iterate with zip(column1, column2, ...) because the rest of the code
            # expects dicts and to_dict() is faster than making individual dicts.
            for row in done_pr_facts.to_dict(orient="records"):
                dump = triage_by_release_match(
                    row[model.repository_full_name.name],
                    row[model.release_match.name],
                    release_settings,
                    default_branches,
                    result,
                    ambiguous,
                )
                if dump is None:
                    continue
                dump[row[model.pr_node_id.name]] = row

        return cls._post_process_ambiguous_done_prs(result, ambiguous)

    @classmethod
    async def _build_participants_mask(
        cls,
        df: pd.DataFrame,
        participants: PRParticipants,
        prefixer: Prefixer,
    ) -> pd.Series:
        dev_conds_single, dev_conds_multiple = await cls._build_participants_conditions(
            participants, prefixer,
        )
        or_masks = []
        for col, value in dev_conds_single:
            or_masks.append(df[col.name].isin(value))
        for col, values in dev_conds_multiple:
            or_masks.append(
                df[col.name].apply(
                    lambda actual_values: bool(
                        {int(v) for v in actual_values}.intersection(set(values)),
                    ),
                ),
            )
        return (
            functools.reduce(operator.or_, or_masks) if or_masks else np.ones(len(df), dtype=bool)
        )


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
        prefixer: Prefixer,
        account: int,
        pdb: morcilla.Database,
        time_from: Optional[datetime] = None,
        exclude_inactive: bool = False,
    ) -> PullRequestFactsMap:
        """
        Load the mapping from PR node identifiers which we are sure are not released in one of \
        `releases` to the serialized facts.

        For each merged PR we maintain the set of releases that do include that PR.

        :return: Map from PR node IDs to their facts.
        """
        if labels:
            # TODO: not supported yet, call original implementation
            return await super(
                PreloadedMergedPRFactsLoader, cls,
            ).load_merged_unreleased_pull_request_facts(
                prs,
                time_to,
                labels,
                matched_bys,
                default_branches,
                release_settings,
                prefixer,
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
        log = logging.getLogger(
            "%s.load_merged_unreleased_pull_request_facts" % metadata.__package__,
        )

        model = GitHubMergedPullRequestFacts
        cached_df = pdb.cache.dfs[PCID.merged_pr_facts]
        df = cached_df.get_dfs((account,))

        common_mask = (df[model.checked_until.name] >= time_to) & (
            df[model.acc_id.name] == account
        )

        if exclude_inactive:
            common_mask &= _build_activity_mask(model, df, time_from, time_to)

        repos_by_match = defaultdict(list)
        for repo in prs[model.repository_full_name.name].unique():
            if (
                release_match := extract_release_match(
                    repo, matched_bys, default_branches, release_settings,
                )
            ) is None:
                # no new releases
                continue
            repos_by_match[release_match].append(repo)

        or_masks = []
        pr_repos = prs[model.repository_full_name.name].values.astype("S")
        pr_ids = prs.index.values.astype("S")
        for release_match, repos in repos_by_match.items():
            or_masks.append(
                common_mask
                & df[model.pr_node_id.name].isin(
                    pr_ids[np.in1d(pr_repos, np.array(repos, dtype="S"))],
                )
                & df[model.repository_full_name.name].isin(repos)
                & (df[model.release_match.name] == release_match),
            )
        if not or_masks:
            return {}

        mask = functools.reduce(operator.or_, or_masks)
        merged_pr_facts = cached_df.filter(
            (account,),
            mask,
            columns=[
                model.pr_node_id.name,
                model.repository_full_name.name,
                model.data.name,
                model.author.name,
                model.merger.name,
            ],
        )

        facts = {}
        user_node_map_get = prefixer.user_node_to_prefixed_login.get
        for node_id, data, repository_full_name, author, merger in zip(
            merged_pr_facts[model.pr_node_id.name].values,
            merged_pr_facts[model.data.name].values,
            merged_pr_facts[model.repository_full_name.name].values,
            merged_pr_facts[model.author.name].values,
            merged_pr_facts[model.merger.name].values,
        ):
            if data is None:
                # There are two known cases:
                # 1. When we load all PRs without a blacklist (/filter/pull_requests) so some
                #    merged PR is matched to releases but exists in
                #    `github_done_pull_request_facts`.
                # 2. "Impossible" PRs that are merged.
                log.warning("No precomputed facts for merged %s", node_id)
                continue
            facts[node_id] = PullRequestFacts(
                data=data,
                repository_full_name=repository_full_name,
                author=user_node_map_get(author),
                merger=user_node_map_get(merger),
            )
        return facts


class PreloadedOpenPRFactsLoader(OpenPRFactsLoader):
    """Loader for preloaded open PRs facts."""

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts_unfresh(
        cls,
        prs: Iterable[str],
        time_from: datetime,
        time_to: datetime,
        exclude_inactive: bool,
        authors: Mapping[str, str],
        account: int,
        pdb: morcilla.Database,
    ) -> Dict[str, PullRequestFacts]:
        """
        Fetch preloaded precomputed facts about the open PRs from the DataFrame.

        We don't filter PRs by the last update here.

        :param authors: Map from PR node IDs to their author logins.
        :return: Map from PR node IDs to their facts.
        """
        model = GitHubOpenPullRequestFacts
        cached_df = pdb.cache.dfs[PCID.open_pr_facts]
        df = cached_df.get_dfs((account,))
        mask = (df[model.acc_id.name] == account) & df[model.pr_node_id.name].isin(prs)

        if exclude_inactive:
            mask &= _build_activity_mask(model, df, time_from, time_to)

        open_prs_facts = cached_df.filter(
            (account,),
            mask,
            columns=[model.pr_node_id.name, model.repository_full_name.name, model.data.name],
        )
        if open_prs_facts.empty:
            return {}

        facts = {
            pr_node_id: PullRequestFacts(
                data=data,
                repository_full_name=repository_full_name,
                author=authors[pr_node_id],
            )
            for pr_node_id, repository_full_name, data in zip(
                open_prs_facts[model.pr_node_id.name].values,
                open_prs_facts[model.repository_full_name.name].values,
                open_prs_facts[model.data.name].values,
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
    async def _extract_branches(
        cls,
        repos: Iterable[str],
        meta_ids: Tuple[int, ...],
        mdb: morcilla.Database,
    ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        model = Branch
        df = mdb.cache.dfs[MCID.branches].get_dfs(meta_ids)

        mask = (
            df[model.repository_full_name.name].isin(repos)
            & df[model.acc_id.name].isin(meta_ids)
            & df[model.commit_sha.name].notna()
        )

        return mdb.cache.dfs[MCID.branches].filter(meta_ids, mask)


class PreloadedPullRequestMiner(PullRequestMiner):
    """Load all the information related to PRS from the metadata DB with some preloaded methods. \
    Iterate over it to access individual PR objects."""

    mappers = PullRequestMiner.ReleaseMappers(
        map_releases_to_prs=PreloadedReleaseToPullRequestMapper.map_releases_to_prs,
        map_prs_to_releases=PullRequestMiner.mappers.map_prs_to_releases,
        load_releases=PreloadedReleaseLoader.load_releases,
    )

    @classmethod
    def _prs_from_binary_expression(cls, bin_exp: BinaryExpression):
        return [bp.value for bp in bin_exp.get_children()[1].get_children()[0].get_children()]

    @classmethod
    @sentry_span
    async def _fetch_prs_by_filters(
        cls,
        time_from: Optional[datetime],
        time_to: datetime,
        repositories: Set[str],
        participants: PRParticipants,
        labels: LabelFilter,
        jira: JIRAFilter,
        exclude_inactive: bool,
        pr_blacklist: Optional[BinaryExpression],
        pr_whitelist: Optional[BinaryExpression],
        meta_ids: Tuple[int, ...],
        mdb: morcilla.Database,
        cache: Optional[aiomcache.Client],
        columns=PullRequest,
        updated_min: Optional[datetime] = None,
        updated_max: Optional[datetime] = None,
    ) -> pd.DataFrame:
        # FIXME(se7entyse7en): prefer numpy operations rather than pandas for better performance
        # See:
        #   - https://github.com/athenianco/athenian-api/pull/1337#discussion_r621071935
        #   - https://github.com/athenianco/athenian-api/pull/1337#discussion_r621073088
        if jira or labels:
            # TODO(se7entyse7en): not supported yet, call the original implementation
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

        model = PullRequest
        df = mdb.cache.dfs[MCID.prs].get_dfs(meta_ids)

        mask = (
            (df[model.created_at.name] < time_to)
            & (df[model.acc_id.name].isin(meta_ids))
            & (df[model.repository_full_name.name].isin(repositories))
        )
        if time_from is not None:
            mask &= ~df[model.closed.name] | (df[model.closed_at.name] >= time_from)

        if pr_blacklist is not None:
            pr_blacklist = cls._prs_from_binary_expression(pr_blacklist)
            mask &= ~df[model.node_id.name].isin(pr_blacklist)
        if pr_whitelist is not None:
            pr_whitelist = cls._prs_from_binary_expression(pr_blacklist)
            mask &= df[model.node_id.name].isin(pr_whitelist)

        if exclude_inactive and updated_min is None:
            # this does not provide 100% guarantee because it can be after time_to,
            # we need to properly filter later
            mask &= df[model.updated_at.name] >= time_from
        if updated_min is not None:
            mask &= (df[model.updated_at.name] >= updated_min) & (
                df[model.updated_at.name] < updated_max
            )

        if len(participants) == 1:
            if PRParticipationKind.AUTHOR in participants:
                mask &= df[model.user_login.name].isin(participants[PRParticipationKind.AUTHOR])
            elif PRParticipationKind.MERGER in participants:
                mask &= df[model.merged_by_login.name].isin(
                    participants[PRParticipationKind.MERGER],
                )
        elif (
            len(participants) == 2
            and PRParticipationKind.AUTHOR in participants
            and PRParticipationKind.MERGER in participants
        ):
            mask &= df[model.user_login.name].isin(participants[PRParticipationKind.AUTHOR]) | df[
                model.merged_by_login.name
            ].isin(participants[PRParticipationKind.MERGER])

        if columns is PullRequest:
            selected_columns = []
            remove_acc_id = False
        else:
            selected_columns = columns = list(columns)
            if remove_acc_id := (PullRequest.acc_id not in selected_columns):
                selected_columns.append(PullRequest.acc_id)
            if PullRequest.merge_commit_id in columns or PullRequest.merge_commit_sha in columns:
                # needed to resolve rebased merge commits
                if PullRequest.number not in selected_columns:
                    selected_columns.append(PullRequest.number)

            selected_columns = [c.name for c in selected_columns]

        prs = mdb.cache.dfs[MCID.prs].filter(
            meta_ids, mask, columns=selected_columns, index=PullRequest.node_id.name,
        )

        if remove_acc_id:
            del prs[PullRequest.acc_id.name]
        if PullRequest.closed.name in prs:
            cls.adjust_pr_closed_merged_timestamps(prs)

        return prs


class MetricEntriesCalculator(OriginalMetricEntriesCalculator):
    """Calculator for different metrics using preloaded DataFrames."""

    pr_miner = PreloadedPullRequestMiner
    branch_miner = PreloadedBranchMiner
    unfresh_pr_facts_fetcher = PreloadedUnfreshPullRequestFactsFetcher
    done_prs_facts_loader = PreloadedDonePRFactsLoader
    load_delta = 10

    def is_ready_for(self, account: int, meta_ids: Tuple[int, ...]) -> bool:
        """Check whether the calculator is ready for the given account and meta ids."""
        return self._mdb.cache.is_account_loaded(account) and self._pdb.cache.is_account_loaded(
            account,
        )
