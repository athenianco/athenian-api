from collections import defaultdict
from datetime import timezone
from itertools import chain
from typing import Any, Dict, Iterable, List, Tuple

import databases
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, desc, insert, or_, select, union_all
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api.async_utils import read_sql_query
from athenian.api.controllers.miners.github.released_pr import matched_by_column
from athenian.api.controllers.miners.types import ReleaseFacts
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, ReleaseSettings
from athenian.api.models.metadata.github import Release
from athenian.api.models.precomputed.models import GitHubReleaseFacts
from athenian.api.tracing import sentry_span
from athenian.precomputer.db.models import GitHubRelease as PrecomputedRelease


def reverse_release_settings(repos: Iterable[str],
                             default_branches: Dict[str, str],
                             settings: ReleaseSettings,
                             ) -> Dict[Tuple[ReleaseMatch, str], List[str]]:
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
async def load_precomputed_release_facts(releases: pd.DataFrame,
                                         default_branches: Dict[str, str],
                                         settings: ReleaseSettings,
                                         account: int,
                                         pdb: databases.Database,
                                         ) -> Dict[int, ReleaseFacts]:
    """
    Fetch precomputed facts about releases.

    :return: Mapping Release.id -> facts.
    """
    if releases.empty:
        return {}
    reverse_settings = defaultdict(list)
    for repo in releases[Release.repository_full_name.name].unique():
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
    for rid, repo in zip(releases[Release.node_id.name].values,
                         releases[Release.repository_full_name.name].values):
        grouped_releases[repo].add(rid)
    default_version = \
        GitHubReleaseFacts.__table__.columns[GitHubReleaseFacts.format_version.key].default.arg

    queries = [
        select([GitHubReleaseFacts.id, GitHubReleaseFacts.data])
        .where(and_(GitHubReleaseFacts.format_version == default_version,
                    GitHubReleaseFacts.acc_id == account,
                    GitHubReleaseFacts.id.in_(chain.from_iterable(
                        grouped_releases[i] for i in r if i in grouped_releases)),
                    GitHubReleaseFacts.release_match == compose_release_match(m, v)))
        for (m, v), r in reverse_settings.items()
    ]
    query = union_all(*queries)
    with sentry_sdk.start_span(op="load_precomputed_release_facts/fetch",
                               description=str(len(releases))):
        rows = await pdb.fetch_all(query)
    return {r[0]: ReleaseFacts(r[1]) for r in rows}


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
async def store_precomputed_release_facts(releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                                          default_branches: Dict[str, str],
                                          settings: ReleaseSettings,
                                          account: int,
                                          pdb: databases.Database) -> None:
    """Put the new release facts to the pdb."""
    values = []
    for dikt, facts in releases:
        repo = dikt[Release.repository_full_name.name]
        setting = settings.prefixed[repo]
        repo = repo.split("/", 1)[1]
        if setting.match == ReleaseMatch.tag:
            value = setting.tags
        elif setting.match == ReleaseMatch.branch:
            value = setting.branches.replace(default_branch_alias, default_branches[repo])
        elif setting.match == ReleaseMatch.event:
            value = ""
        else:
            raise AssertionError("Ambiguous release settings for %s: %s" % (repo, setting))
        values.append(GitHubReleaseFacts(
            id=dikt[Release.node_id.name],
            acc_id=account,
            release_match=compose_release_match(setting.match, value),
            repository_full_name=repo,
            published_at=facts.published.item().replace(tzinfo=timezone.utc),
            data=facts.data,
        ).create_defaults().explode(with_primary_keys=True))
    if pdb.url.dialect == "postgresql":
        sql = postgres_insert(GitHubReleaseFacts).on_conflict_do_nothing()
    else:
        sql = insert(GitHubReleaseFacts).prefix_with("OR IGNORE")
    with sentry_sdk.start_span(op="store_precomputed_release_facts/execute_many"):
        if pdb.url.dialect == "sqlite":
            async with pdb.connection() as pdb_conn:
                async with pdb_conn.transaction():
                    await pdb_conn.execute_many(sql, values)
        else:
            # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
            await pdb.execute_many(sql, values)


@sentry_span
async def fetch_precomputed_releases_by_name(names: Dict[str, Iterable[str]],
                                             account: int,
                                             pdb: databases.Database,
                                             ) -> pd.DataFrame:
    """Load precomputed release facts given the mapping from repository names to release names."""
    prel = PrecomputedRelease
    if pdb.url.dialect == "sqlite":
        query = (
            select([prel])
            .where(or_(*(and_(prel.repository_full_name == k,
                              prel.acc_id == account,
                              prel.name.in_(v)) for k, v in names.items())))
            .order_by(desc(prel.published_at))
        )
    else:
        query = union_all(*(
            select([prel])
            .where(and_(prel.repository_full_name == k,
                        prel.name.in_(v),
                        prel.acc_id == account))
            .order_by(desc(prel.published_at))
            for k, v in names.items()))
    df = await read_sql_query(query, pdb, prel)
    df[matched_by_column] = None
    df.loc[df[prel.release_match.name].str.startswith("branch|"), matched_by_column] = \
        ReleaseMatch.branch
    df.loc[df[prel.release_match.name].str.startswith("tag|"), matched_by_column] = \
        ReleaseMatch.tag
    df.drop(PrecomputedRelease.release_match.name, inplace=True, axis=1)
    return df
