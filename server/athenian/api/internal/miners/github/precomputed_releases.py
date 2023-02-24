from collections import defaultdict
from datetime import timezone
from itertools import chain
import logging
from typing import Iterable

import morcilla
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, or_, select, union_all

from athenian.api import metadata
from athenian.api.async_utils import gather, read_sql_query
from athenian.api.db import dialect_specific_insert
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.types import ReleaseFacts
from athenian.api.internal.settings import ReleaseMatch, ReleaseSettings, default_branch_alias
from athenian.api.models.metadata.github import Release
from athenian.api.models.precomputed.models import (
    GitHubRelease as PrecomputedRelease,
    GitHubReleaseFacts,
)
from athenian.api.tracing import sentry_span


def reverse_release_settings(
    repos: Iterable[str],
    default_branches: dict[str, str],
    settings: ReleaseSettings,
) -> dict[tuple[ReleaseMatch, str], list[str]]:
    """Map distinct pairs (release match, tag/branch name) to the aggregated repositories."""
    reverse_settings = defaultdict(list)
    for repo in repos:
        setting = settings.native[repo]
        if setting.match == ReleaseMatch.tag:
            value = setting.tags
        elif setting.match == ReleaseMatch.branch:
            value = setting.branches.replace(default_branch_alias, default_branches[repo])
        elif setting.match == ReleaseMatch.event:
            value = ""
        else:
            raise AssertionError("Ambiguous release settings for %s: %s" % (repo, setting))
        reverse_settings[(setting.match, value)].append(repo)
    return reverse_settings


@sentry_span
async def load_precomputed_release_facts(
    releases: pd.DataFrame,
    default_branches: dict[str, str],
    settings: ReleaseSettings,
    account: int,
    pdb: morcilla.Database,
) -> dict[tuple[int, str], ReleaseFacts]:
    """
    Fetch precomputed facts about releases.

    :return: Mapping (Release.id, Release.repository_full_name) -> facts.
    """
    if releases.empty:
        return {}
    reverse_settings = defaultdict(list)
    release_repos = releases[Release.repository_full_name.name].unique()
    for repo in release_repos:
        setting = settings.native[repo]
        if setting.match == ReleaseMatch.tag:
            value = setting.tags
        elif setting.match == ReleaseMatch.branch:
            value = setting.branches.replace(default_branch_alias, default_branches[repo])
        elif setting.match == ReleaseMatch.event:
            value = ""
        else:
            raise AssertionError("Ambiguous release settings for %s: %s" % (repo, setting))
        reverse_settings[(setting.match, value)].append(repo)
    grouped_releases = defaultdict(set)
    for rid, repo in zip(
        releases[Release.node_id.name].values, releases[Release.repository_full_name.name].values,
    ):
        grouped_releases[repo].add(rid)
    default_version = GitHubReleaseFacts.__table__.columns[
        GitHubReleaseFacts.format_version.key
    ].default.arg
    queries = []
    total_ids = 0
    threshold_in_values = 1000
    threshold_total = 1000
    for (m, v), r in reverse_settings.items():
        ids = list(chain.from_iterable(grouped_releases[i] for i in r if i in grouped_releases))
        filters = [
            GitHubReleaseFacts.format_version == default_version,
            GitHubReleaseFacts.acc_id == account,
            GitHubReleaseFacts.repository_full_name.in_(release_repos),
            GitHubReleaseFacts.release_match == compose_release_match(m, v),
        ]
        if len(ids) >= threshold_in_values:
            filters.append(GitHubReleaseFacts.id.in_any_values(ids))
        else:
            filters.append(GitHubReleaseFacts.id.in_(ids))
        total_ids += len(ids)
        queries.append(
            select(
                [
                    GitHubReleaseFacts.id,
                    GitHubReleaseFacts.repository_full_name,
                    GitHubReleaseFacts.data,
                ],
            )
            .where(and_(*filters))
            .with_statement_hint("HashJoin(release_facts *VALUES*)")
            .with_statement_hint(f"Rows(release_facts *VALUES* #{len(ids)})"),
        )

    with sentry_sdk.start_span(
        op="load_precomputed_release_facts/fetch", description=str(len(releases)),
    ):
        if total_ids < threshold_total:
            rows = await pdb.fetch_all(union_all(*queries))
        else:
            rows = chain.from_iterable(await gather(*(pdb.fetch_all(q) for q in queries)))
    result = {
        (
            (node_id := row[GitHubReleaseFacts.id.name]),
            (repo := row[GitHubReleaseFacts.repository_full_name.name]),
        ): ReleaseFacts(
            row[GitHubReleaseFacts.data.name], node_id=node_id, repository_full_name=repo,
        )
        for row in rows
    }
    return result


def compose_release_match(match: ReleaseMatch, value: str) -> str:
    """Render DB `release_match` value."""
    if match == ReleaseMatch.tag:
        return "tag|" + value
    if match == ReleaseMatch.branch:
        return "branch|" + value
    if match == ReleaseMatch.event:
        return ReleaseMatch.event.name
    raise AssertionError("Impossible release match: %s" % match)


@sentry_span
async def store_precomputed_release_facts(
    releases: Iterable[ReleaseFacts],
    default_branches: dict[str, str],
    settings: ReleaseSettings,
    account: int,
    pdb: morcilla.Database,
    on_conflict_replace: bool = False,
) -> None:
    """Put the new release facts to the pdb."""
    values = []
    skipped = defaultdict(int)
    for facts in releases:
        repo = facts.repository_full_name
        if facts.repository_full_name is None:
            # could not prefix => gone
            skipped[repo] += 1
            continue
        setting = settings.native[repo]
        if setting.match == ReleaseMatch.tag:
            value = setting.tags
        elif setting.match == ReleaseMatch.branch:
            value = setting.branches.replace(default_branch_alias, default_branches[repo])
        elif setting.match == ReleaseMatch.event:
            value = ""
        else:
            raise AssertionError("Ambiguous release settings for %s: %s" % (repo, setting))
        values.append(
            GitHubReleaseFacts(
                id=facts.node_id,
                acc_id=account,
                release_match=compose_release_match(setting.match, value),
                repository_full_name=repo,
                published_at=facts.published.item().replace(tzinfo=timezone.utc),
                data=facts.data,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        )
    if not values:
        return
    log = logging.getLogger(f"{metadata.__package__}.store_precomputed_release_facts")
    if skipped:
        log.warning("ignored mined releases: %s", dict(skipped))

    log.info("storing %d release facts", len(values))
    sql = (await dialect_specific_insert(pdb))(GitHubReleaseFacts)
    if on_conflict_replace:
        sql = sql.on_conflict_do_update(
            index_elements=GitHubReleaseFacts.__table__.primary_key.columns,
            set_={GitHubReleaseFacts.data.name: sql.excluded.data},
        )
    else:
        sql = sql.on_conflict_do_nothing()
    with sentry_sdk.start_span(op="store_precomputed_release_facts/execute_many"):
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, values)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, values)


@sentry_span
async def fetch_precomputed_releases_by_name(
    names: dict[str, Iterable[str]],
    account: int,
    pdb: morcilla.Database,
) -> pd.DataFrame:
    """Load precomputed release facts given the mapping from repository names to release names."""
    prel = PrecomputedRelease
    if pdb.url.dialect == "sqlite":
        query = (
            select([prel])
            .where(
                or_(
                    *(
                        and_(
                            prel.acc_id == account,
                            prel.repository_full_name == k,
                            prel.name.in_(v),
                        )
                        for k, v in names.items()
                    ),
                ),
            )
            .order_by(desc(prel.published_at))
        )
    else:
        query = union_all(
            *(
                select(prel)
                .where(prel.acc_id == account, prel.repository_full_name == k, prel.name.in_(v))
                .order_by(desc(prel.published_at))
                for k, v in names.items()
            ),
        )
    df = await read_sql_query(query, pdb, prel)
    df[matched_by_column] = None
    df.loc[
        df[prel.release_match.name].str.startswith("branch|"),
        matched_by_column,
    ] = ReleaseMatch.branch
    df.loc[
        df[prel.release_match.name].str.startswith("tag|"),
        matched_by_column,
    ] = ReleaseMatch.tag
    df.drop(PrecomputedRelease.release_match.name, inplace=True, axis=1)
    return df
