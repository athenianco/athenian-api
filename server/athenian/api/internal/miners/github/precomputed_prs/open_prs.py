from datetime import datetime, timezone
from itertools import repeat
from typing import Collection, Iterable, Mapping, Set, Tuple

import morcilla
import numpy as np
import pandas as pd
import sentry_sdk
from sqlalchemy import case, select

from athenian.api.async_utils import read_sql_query
from athenian.api.db import dialect_specific_insert
from athenian.api.int_to_str import int_to_str
from athenian.api.internal.miners.github.precomputed_prs.utils import (
    append_activity_days_filter,
    collect_activity_days,
)
from athenian.api.internal.miners.types import (
    MinedPullRequest,
    PullRequestFacts,
    PullRequestFactsMap,
)
from athenian.api.models.metadata.github import PullRequest
from athenian.api.models.precomputed.models import GitHubOpenPullRequestFacts
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import in1d_str, unordered_unique


class OpenPRFactsLoader:
    """Loader for open PRs facts."""

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts(
        cls,
        prs: pd.DataFrame,
        repositories: Set[str],
        account: int,
        pdb: morcilla.Database,
    ) -> PullRequestFactsMap:
        """
        Fetch precomputed facts about the open PRs from the DataFrame.

        We filter open PRs inplace so the user does not have to worry about that.
        """
        open_indexes = np.flatnonzero(prs[PullRequest.closed_at.name].isnull().values)
        node_ids = prs.index.get_level_values(0).values[open_indexes]
        authors = dict(zip(node_ids, prs[PullRequest.user_login.name].values[open_indexes]))
        ghoprf = GitHubOpenPullRequestFacts
        default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
        selected = [
            ghoprf.pr_node_id,
            ghoprf.pr_updated_at,
            ghoprf.repository_full_name,
            ghoprf.data,
        ]
        rows = await pdb.fetch_all(
            select(*selected).where(
                ghoprf.acc_id == account,
                ghoprf.pr_node_id.in_(node_ids),
                ghoprf.repository_full_name.in_(repositories),
                ghoprf.format_version == default_version,
            ),
        )
        if not rows:
            return {}
        updated_ats = prs[PullRequest.updated_at.name].values[open_indexes]
        found_node_ids = np.fromiter((r[ghoprf.pr_node_id.name] for r in rows), int, len(rows))
        found_updated_ats = np.fromiter(
            (r[ghoprf.pr_updated_at.name] for r in rows), updated_ats.dtype, len(rows),
        )
        indexes = np.searchsorted(node_ids, found_node_ids)
        passed = np.flatnonzero(updated_ats[indexes] <= found_updated_ats)
        facts = {}
        for i in passed:
            row = rows[i]
            node_id = row[ghoprf.pr_node_id.name]
            repo = row[ghoprf.repository_full_name.name]
            facts[(node_id, repo)] = PullRequestFacts(
                data=row[ghoprf.data.name],
                node_id=node_id,
                repository_full_name=repo,
                author=authors.get(node_id, ""),
                merger="",
                releaser="",
            )
        return facts

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts_unfresh(
        cls,
        prs: pd.Index,
        time_from: datetime,
        time_to: datetime,
        exclude_inactive: bool,
        authors: Mapping[int, str],
        account: int,
        pdb: morcilla.Database,
    ) -> PullRequestFactsMap:
        """
        Fetch precomputed facts about the open PRs from the DataFrame.

        We don't filter PRs by the last update here.

        :param authors: Map from PR node IDs to their author logins.
        :return: Map from PR node IDs to their facts.
        """
        postgres = pdb.url.dialect == "postgresql"
        ghoprf = GitHubOpenPullRequestFacts
        selected = {ghoprf.pr_node_id, ghoprf.repository_full_name, ghoprf.data}
        default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
        prs_repos_bytes = prs.get_level_values(1).values.astype("S")
        filters = [
            ghoprf.acc_id == account,
            ghoprf.pr_node_id.in_(prs.get_level_values(0).unique()),
            ghoprf.repository_full_name.in_(unordered_unique(prs_repos_bytes)),
            ghoprf.format_version == default_version,
        ]
        if exclude_inactive:
            date_range = append_activity_days_filter(
                time_from, time_to, selected, filters, ghoprf.activity_days, postgres,
            )
        selected = sorted(selected, key=lambda i: i.key)
        df = await read_sql_query(select(*selected).where(*filters), pdb, selected)
        if df.empty:
            return {}
        haystack = np.char.add(
            int_to_str(df[ghoprf.pr_node_id.name].values),
            df[ghoprf.repository_full_name.name].values.astype("S"),
        )
        needle = np.char.add(int_to_str(prs.get_level_values(0).values), prs_repos_bytes)
        matched_mask = in1d_str(haystack, needle, skip_leading_zeros=True)
        fetched_node_ids = df[ghoprf.pr_node_id.name].values
        fetched_repos = df[ghoprf.repository_full_name.name].values
        fetched_datas = df[ghoprf.data.name].values
        if exclude_inactive and not postgres:
            fetched_activity_days = df[ghoprf.activity_days.name].values
        else:
            fetched_activity_days = repeat(True)
        if not matched_mask.all():
            fetched_node_ids = fetched_node_ids[matched_mask]
            fetched_repos = fetched_repos[matched_mask]
            fetched_datas = fetched_datas[matched_mask]
            if not postgres:
                fetched_activity_days = fetched_activity_days[matched_mask]
        facts = {}
        for node_id, repo, data, activity_days in zip(
            fetched_node_ids, fetched_repos, fetched_datas, fetched_activity_days,
        ):
            if exclude_inactive and not postgres:
                activity_days = {
                    datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    for d in activity_days
                }
                if not activity_days.intersection(date_range):
                    continue
            facts[(node_id, repo)] = PullRequestFacts(
                data=data,
                node_id=node_id,
                repository_full_name=repo,
                author=authors.get(node_id, ""),
                merger="",
                releaser="",
            )
        return facts

    @classmethod
    @sentry_span
    async def load_open_pull_request_facts_all(
        cls,
        repos: Collection[str],
        pr_node_id_blacklist: Collection[int],
        account: int,
        pdb: morcilla.Database,
    ) -> PullRequestFactsMap:
        """
        Load the precomputed open PR facts through all the time.

        We do not load the repository and the author!

        :return: Map from PR node IDs to their facts.
        """
        ghoprf = GitHubOpenPullRequestFacts
        selected = [
            ghoprf.pr_node_id,
            ghoprf.repository_full_name,
            ghoprf.data,
        ]
        default_version = ghoprf.__table__.columns[ghoprf.format_version.key].default.arg
        filters = [
            ghoprf.acc_id == account,
            ghoprf.pr_node_id.notin_(pr_node_id_blacklist),
            ghoprf.repository_full_name.in_(repos),
            ghoprf.format_version == default_version,
        ]
        query = select(*selected).where(*filters)
        with sentry_sdk.start_span(op="load_open_pull_request_facts_all/fetch"):
            rows = await pdb.fetch_all(query)
        facts = {
            (
                (node_id := row[ghoprf.pr_node_id.name]),
                (repo := row[ghoprf.repository_full_name.name]),
            ): PullRequestFacts(row[ghoprf.data.name], node_id=node_id, repository_full_name=repo)
            for row in rows
        }
        return facts


@sentry_span
async def store_open_pull_request_facts(
    open_prs_and_facts: Iterable[Tuple[MinedPullRequest, PullRequestFacts]],
    account: int,
    pdb: morcilla.Database,
) -> None:
    """
    Persist the facts about open pull requests to the database.

    Each passed PR must be open, we raise an assertion otherwise.
    """
    postgres = pdb.url.dialect == "postgresql"
    if not postgres:
        assert pdb.url.dialect == "sqlite"
    values = []
    for pr, facts in open_prs_and_facts:
        assert not facts.closed
        updated_at = pr.pr[PullRequest.updated_at.name]
        if updated_at != updated_at:
            continue
        values.append(
            GitHubOpenPullRequestFacts(
                acc_id=account,
                pr_node_id=pr.pr[PullRequest.node_id.name],
                repository_full_name=pr.pr[PullRequest.repository_full_name.name],
                pr_created_at=pr.pr[PullRequest.created_at.name],
                number=pr.pr[PullRequest.number.name],
                pr_updated_at=updated_at,
                activity_days=collect_activity_days(pr, facts, not postgres),
                data=facts.data,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        )
    sql = (await dialect_specific_insert(pdb))(GitHubOpenPullRequestFacts)
    sql = sql.on_conflict_do_update(
        index_elements=GitHubOpenPullRequestFacts.__table__.primary_key.columns,
        set_={
            GitHubOpenPullRequestFacts.pr_updated_at.name: sql.excluded.pr_updated_at,
            GitHubOpenPullRequestFacts.updated_at.name: sql.excluded.updated_at,
            GitHubOpenPullRequestFacts.data.name: case(
                (
                    sql.excluded.pr_updated_at >= GitHubOpenPullRequestFacts.pr_updated_at,
                    sql.excluded.data,
                ),
                else_=GitHubOpenPullRequestFacts.data,
            ),
        },
    )
    with sentry_sdk.start_span(op="store_open_pull_request_facts/execute"):
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, values)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, values)
