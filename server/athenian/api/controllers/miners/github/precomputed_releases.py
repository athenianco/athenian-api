from collections import defaultdict
from itertools import chain
import pickle
from typing import Any, Dict, List, Tuple

import databases
import pandas as pd
import sentry_sdk
from sqlalchemy import and_, insert, or_, select
from sqlalchemy.dialects.postgresql import insert as postgres_insert

from athenian.api.controllers.miners.types import ReleaseFacts
from athenian.api.controllers.settings import default_branch_alias, ReleaseMatch, \
    ReleaseMatchSetting
from athenian.api.models.metadata.github import Release
from athenian.api.models.precomputed.models import GitHubReleaseFacts
from athenian.api.tracing import sentry_span


@sentry_span
async def load_precomputed_release_facts(releases: pd.DataFrame,
                                         default_branches: Dict[str, str],
                                         settings: Dict[str, ReleaseMatchSetting],
                                         pdb: databases.Database) -> Dict[str, ReleaseFacts]:
    """
    Fetch precomputed facts about releases.

    :return: Mapping Release.id -> facts.
    """
    reverse_settings = defaultdict(list)
    for repo, setting in settings.items():
        repo = repo.split("/", 1)[1]
        if setting.match == ReleaseMatch.tag:
            value = setting.tags
        elif setting.match == ReleaseMatch.branch:
            value = setting.branches.replace(default_branch_alias, default_branches[repo])
        else:
            raise AssertionError("Ambiguous release settings for %s: %s" % (repo, setting))
        reverse_settings[(setting.match, value)].append(repo)
    grouped_releases = defaultdict(list)
    for rid, repo in zip(releases[Release.id.key].values,
                         releases[Release.repository_full_name.key].values):
        grouped_releases[repo].append(rid)
    default_version = \
        GitHubReleaseFacts.__table__.columns[GitHubReleaseFacts.format_version.key].default.arg
    query = select([GitHubReleaseFacts.id, GitHubReleaseFacts.data]).where(and_(
        GitHubReleaseFacts.format_version == default_version,
        or_(*(and_(
            GitHubReleaseFacts.id.in_(chain.from_iterable(
                grouped_releases[i] for i in r if i in grouped_releases)),
            GitHubReleaseFacts.release_match == _compose_release_match(m, v),
        ) for (m, v), r in reverse_settings.items()))))
    with sentry_sdk.start_span(op="load_precomputed_release_facts/fetch",
                               description=str(len(releases))):
        rows = await pdb.fetch_all(query)
    return {r[0]: pickle.loads(r[1]) for r in rows}


def _compose_release_match(match: ReleaseMatch, value: str) -> str:
    if match == ReleaseMatch.tag:
        return "tag|" + value
    if match == ReleaseMatch.branch:
        return "branch|" + value
    raise AssertionError("Impossible release match: %s" % match)


@sentry_span
async def store_precomputed_release_facts(releases: List[Tuple[Dict[str, Any], ReleaseFacts]],
                                          default_branches: Dict[str, str],
                                          settings: Dict[str, ReleaseMatchSetting],
                                          pdb: databases.Database) -> None:
    """Put the new release facts to the pdb."""
    values = []
    for dikt, facts in releases:
        repo = dikt[Release.repository_full_name.key]
        setting = settings[repo]
        repo = repo.split("/", 1)[1]
        if setting.match == ReleaseMatch.tag:
            value = setting.tags
        elif setting.match == ReleaseMatch.branch:
            value = setting.branches.replace(default_branch_alias, default_branches[repo])
        else:
            raise AssertionError("Ambiguous release settings for %s: %s" % (repo, setting))
        values.append(GitHubReleaseFacts(
            id=dikt[Release.id.key],
            release_match=_compose_release_match(setting.match, value),
            repository_full_name=repo,
            published_at=facts.published,
            data=pickle.dumps(facts),
        ).create_defaults().explode(with_primary_keys=True))
    if pdb.url.dialect in ("postgres", "postgresql"):
        sql = postgres_insert(GitHubReleaseFacts).on_conflict_do_nothing()
    else:
        sql = insert(GitHubReleaseFacts).prefix_with("OR IGNORE")
    with sentry_sdk.start_span(op="store_precomputed_release_facts/execute_many"):
        await pdb.execute_many(sql, values)
