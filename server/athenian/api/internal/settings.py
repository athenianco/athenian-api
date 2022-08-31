from collections import defaultdict
from dataclasses import dataclass
from enum import IntEnum
from itertools import chain
import re
from typing import (
    Any,
    Callable,
    Collection,
    Coroutine,
    Dict,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import aiomcache
import numpy as np
import pandas as pd
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, select

from athenian.api.db import Database, DatabaseLike, dialect_specific_insert
from athenian.api.internal.account import get_account_repositories, get_metadata_account_ids
from athenian.api.internal.logical_repos import (
    coerce_logical_repos,
    coerce_prefixed_logical_repos,
    drop_logical_repo,
)
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.internal.reposet import resolve_repos
from athenian.api.models.metadata.github import PullRequestLabel
from athenian.api.models.persistentdata.models import DeployedLabel, DeploymentNotification
from athenian.api.models.state.models import LogicalRepository, ReleaseSetting
from athenian.api.models.web import InvalidRequestError, MissingSettingsError, ReleaseMatchStrategy
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError

# rejected: PR was closed without merging.
# force_push_drop: commit history was overwritten and the PR's merge commit no longer exists.
ReleaseMatch = IntEnum(
    "ReleaseMatch",
    {
        "rejected": -2,
        "force_push_drop": -1,
        ReleaseMatchStrategy.BRANCH: 0,
        ReleaseMatchStrategy.TAG: 1,
        ReleaseMatchStrategy.TAG_OR_BRANCH: 2,
        ReleaseMatchStrategy.EVENT: 3,
    },
)
ReleaseMatch.__doc__ = """Enumeration of supported release matching strategies."""

default_branch_alias = "{{default}}"


@dataclass(slots=True, repr=False, frozen=True)
class ReleaseMatchSetting:
    """Internal representation of the repository release match setting."""

    branches: str
    tags: str
    events: str
    match: ReleaseMatch

    def __str__(self) -> str:
        """Return the compact string representation of the object."""
        return '{"branches": "%s", "tags": "%s", "events": "%s", "match": "%s"}' % (
            self.branches,
            self.tags,
            self.events,
            self.match.name,
        )

    def __repr__(self) -> str:
        """Return the Python representation of the object."""
        return 'ReleaseMatchSetting(branches="%s", tags="%s", events="%s", match=Match["%s"])' % (
            self.branches,
            self.tags,
            self.events,
            self.match.name,
        )

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)

    def with_match(self, match: ReleaseMatch) -> "ReleaseMatchSetting":
        """Clone self and override the effective `match`."""
        return ReleaseMatchSetting(
            tags=self.tags,
            branches=self.branches,
            events=self.events,
            match=match,
        )

    def as_db(self, default_branch: str) -> str:
        """Render database `release_match` value. Works only for specific match types."""
        if self.match == ReleaseMatch.tag:
            return "|".join([ReleaseMatch.tag.name, self.tags])
        if self.match == ReleaseMatch.branch:
            return "|".join(
                [
                    ReleaseMatch.branch.name,
                    self.branches.replace(default_branch_alias, default_branch),
                ],
            )
        if self.match == ReleaseMatch.event:
            return "|".join([ReleaseMatch.event.name, self.events])
        if self.match == ReleaseMatch.force_push_drop:
            return ReleaseMatch.force_push_drop.name
        raise AssertionError(f"Impossible release match: {self.match}")

    @classmethod
    def from_db(cls, db: str) -> "ReleaseMatchSetting":
        """Parse the `release_match` value from the database."""
        match_name, match_by = db.split("|", 1)
        release_match = ReleaseMatch[match_name]
        return ReleaseMatchSetting(
            branches=match_by,
            tags=match_by,
            events=match_by,
            match=release_match,
        )

    def compatible_with_db(self, db: str, default_branch: str):
        """Check if the release setting is compatible with the database value."""
        db_match = self.from_db(db)
        if self.match != db_match.match and self.match != ReleaseMatch.tag_or_branch:
            return False
        if db_match.match == ReleaseMatch.branch:
            return self.branches.replace(default_branch_alias, default_branch) == db_match.branches
        elif db_match.match == ReleaseMatch.tag:
            return self.tags == db_match.tags
        elif db_match.match == ReleaseMatch.event:
            return self.events == db_match.events
        raise AssertionError(f"Impossible release match: {db}")

    def __lt__(self, other: "ReleaseMatchSetting") -> bool:
        """Implement self < other to become sortable."""
        if self.match != other.match:
            return self.match < other.match
        if self.tags != other.tags:
            return self.tags < other.tags
        if self.branches != other.branches:
            return self.branches < other.branches
        if self.events != other.events:
            return self.events < other.events
        return False


class ReleaseSettings:
    """Mapping from prefixed repository full names to their release settings."""

    def __init__(self, map_prefixed: Dict[str, ReleaseMatchSetting]):
        """Initialize a new instance of ReleaseSettings class."""
        self._map_prefixed = map_prefixed
        self._map_native = {k.split("/", 1)[1]: v for k, v in map_prefixed.items()}
        self._coherence = dict(zip(self._map_native, self._map_prefixed))

    def __repr__(self) -> str:
        """Implement repr(). Sentry requires very short summaries."""
        return f"<ReleaseSettings(...{len(self)} repositories...)>"

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)

    def __str__(self) -> str:
        """Implement str()."""
        return f"ReleaseSettings({repr(self._map_prefixed)})"

    def __len__(self) -> int:
        """Implement len()."""
        return len(self._map_prefixed)

    def __eq__(self, other: "ReleaseSettings") -> bool:
        """Implement ==."""
        return self._map_prefixed == other._map_prefixed

    def copy(self) -> "ReleaseSettings":
        """Shallow copy the settings."""
        return ReleaseSettings(self._map_prefixed.copy())

    @property
    def prefixed(self) -> Dict[str, ReleaseMatchSetting]:
        """View the release settings with repository name prefixes."""
        return self._map_prefixed

    @property
    def native(self) -> Dict[str, ReleaseMatchSetting]:
        """View the release settings without repository name prefixes."""
        return self._map_native

    def prefixed_for_native(self, name_without_prefix: str) -> str:
        """Return the prefixed repository name for an unprefixed name."""
        return self._coherence[name_without_prefix]

    def set_by_native(self, name_without_prefix: str, value: ReleaseMatchSetting) -> None:
        """Update release settings given a repository name without prefix."""
        self._map_prefixed[self.prefixed_for_native(name_without_prefix)] = self._map_native[
            name_without_prefix
        ] = value

    def set_by_prefixed(self, name_with_prefix: str, value: ReleaseMatchSetting) -> None:
        """Update release settings given a repository name with prefix."""
        self._map_prefixed[name_with_prefix] = self._map_native[
            name_with_prefix.split("/", 1)[1]
        ] = value

    def select(self, repos: Collection[str]) -> "ReleaseSettings":
        """Reduce the settings to the specified repositories only."""
        return ReleaseSettings({self._coherence[r]: self._map_native[r] for r in repos})


class CommonLogicalSettingsMixin:
    """Common methods belonging to both LogicalPRSettings and LogicalDeploymentSettings."""

    def __repr__(self) -> str:
        """Implement repr()."""
        return f"{type(self).__name__}({str(self)})"

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)

    def __bool__(self) -> bool:
        """Return True if there is at least one logical repository, otherwise, False."""
        return bool(self._labels) or bool(self._title_regexps)

    def title(self, repo: str) -> re.Pattern:
        """Return the title regexp for the given logical repository."""
        return self._title_regexps[repo]

    @property
    def logical_repositories(self) -> FrozenSet[str]:
        """Return all known logical repositories."""
        return self._repos

    @property
    def has_titles(self) -> bool:
        """Return value indicating whether there is at least one title filter."""
        return bool(self._title_regexps)

    @property
    def has_labels(self) -> bool:
        """Return value indicating whether there is at least one label filter."""
        return bool(self._labels)

    @staticmethod
    def group_by_repo(
        repos: Sequence[str],
        whitelist: Optional[Collection[str]] = None,
    ) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        Iterate over the indexes of repository groups.

        :param whitelist: Ignore the rest of the repositories.
        """
        if whitelist is not None:
            if not len(whitelist):
                return
            if not isinstance(whitelist, (list, tuple, np.ndarray)):
                whitelist = list(whitelist)
        unique_repos, index_map, counts = np.unique(repos, return_inverse=True, return_counts=True)
        repo_indexes = np.arange(len(repos))[np.argsort(index_map, kind="stable")]
        if whitelist is not None:
            allowed_repos = np.flatnonzero(np.in1d(unique_repos, whitelist, assume_unique=True))
        else:
            allowed_repos = np.arange(len(unique_repos))
        offsets = np.cumsum(counts)
        for repo_index in allowed_repos:
            repo = unique_repos[repo_index]
            end = offsets[repo_index]
            begin = end - counts[repo_index]
            indexes = repo_indexes[begin:end]
            yield repo, indexes


class LogicalPRSettings(CommonLogicalSettingsMixin):
    """Matching rules for PRs in a logical repository."""

    __slots__ = ("_repos", "_title_regexps", "_labels", "_origin")

    def __init__(self, prs: Mapping[str, Dict[str, Any]], origin: str):
        """Initialize a new instance of LogicalPRSettings class."""
        self._origin = origin
        repos = {origin}
        self._title_regexps = title_regexps = {}
        labels = {}
        for repo, obj in prs.items():
            if pattern := obj.get("title"):
                repos.add(repo)
                title_regexps[repo] = re.compile(f"^({pattern})", re.MULTILINE)
            if obj_labels := obj.get("labels", []):
                repos.add(repo)
                for label in obj_labels:
                    labels.setdefault(label, []).append(repo)
        self._repos = frozenset(repos)
        label_keys = list(labels.keys())
        label_values = list(labels.values())
        self._labels = {label_keys[i]: label_values[i] for i in np.argsort(label_keys)}

    def __str__(self) -> str:
        """Implement str()."""
        labels = defaultdict(list)
        for label, repos in self._labels.items():
            for repo in repos:
                labels[repo].append(label)
        return str(
            dict(
                (
                    r,
                    {
                        **(
                            {"title": pattern.pattern[2:-1]}
                            if (pattern := self._title_regexps.get(r))
                            else {}
                        ),
                        **(
                            {"labels": sorted(repo_labels)}
                            if (repo_labels := labels.get(r))
                            else {}
                        ),
                    },
                )
                for r in sorted(self._repos - {self._origin})
            ),
        )

    def match(
        self,
        prs: pd.DataFrame,
        labels: pd.DataFrame,
        pr_indexes: Sequence[int],
        id_column: str,
        title_column: str,
    ) -> Dict[str, List[int]]:
        """
        Map PRs to logical repositories.

        :param prs: index with PR ids + `repository_full_name` column.
        :param labels: index with PR ids + `name` column.
        :param pr_indexes: only consider PRs indexed in `prs`.
        :return: mapping from logical repository names to indexes in `prs`.
        """
        assert isinstance(prs, pd.DataFrame)
        assert isinstance(labels, pd.DataFrame)
        matched = {}
        titles = prs[title_column].values
        if len(pr_indexes):
            titles = titles[pr_indexes]
        else:
            pr_indexes = np.arange(len(prs))
        if self.has_titles:
            lengths = np.fromiter((len(s) for s in titles), int, len(titles)) + 1
            offsets = np.zeros(len(lengths), dtype=int)
            np.cumsum(lengths[:-1], out=offsets[1:])
            concat_titles = "\n".join(titles)  # PR titles are guaranteed to not contain \n
            for repo, regexp in self._title_regexps.items():
                found = [m.start() for m in regexp.finditer(concat_titles)]
                found = pr_indexes[np.in1d(offsets, found, assume_unique=True)]
                matched[repo] = found
        if not labels.empty and self.has_labels:
            assert not isinstance(labels.index, pd.RangeIndex)
            matched_by_label = defaultdict(list)
            pr_ids = prs[id_column].values[pr_indexes]
            order = np.argsort(pr_ids)
            pr_ids = pr_ids[order]
            label_pr_ids = labels.index.get_level_values(0).values
            found_indexes = np.searchsorted(pr_ids, label_pr_ids)
            found_indexes[found_indexes == len(pr_ids)] = 0
            label_pr_indexes = np.flatnonzero(pr_ids[found_indexes] == label_pr_ids)
            if len(label_pr_indexes):
                reverse_indexes = pr_indexes[order][found_indexes[label_pr_indexes]]
                names = labels[PullRequestLabel.name.name].values[label_pr_indexes]
                unique_names, name_map, counts = np.unique(
                    names, return_inverse=True, return_counts=True,
                )
                grouped_pr_indexes = np.split(
                    reverse_indexes[np.argsort(name_map)], np.cumsum(counts[:-1]),
                )
                label_keys = np.array(list(self._labels.keys()))
                label_values = np.array(list(self._labels.values()))
                found_indexes = np.searchsorted(unique_names, label_keys)
                found_indexes[found_indexes == len(unique_names)] = 0
                mask = unique_names[found_indexes] == label_keys
                for label_index, repos in zip(found_indexes[mask], label_values[mask]):
                    label_pr_indexes = grouped_pr_indexes[label_index]
                    for repo in repos:
                        matched_by_label[repo].append(label_pr_indexes)
                for repo, label_matches in matched_by_label.items():
                    matched[repo] = np.unique(
                        np.concatenate([matched.get(repo, []), *label_matches]),
                    )
        try:
            concat_logical = np.unique(np.concatenate(list(matched.values())))
            generic = np.setdiff1d(pr_indexes, concat_logical, assume_unique=True)
        except ValueError:
            generic = pr_indexes
        matched[self._origin] = generic
        return matched


class LogicalDeploymentSettings(CommonLogicalSettingsMixin):
    """Matching rules for deployments in a logical repository."""

    __slots__ = ("_repos", "_title_regexps", "_labels", "_labels_inv", "_origin")

    def __init__(self, deps: Mapping[str, Dict[str, Any]], origin: str):
        """Initialize a new instance of LogicalDeploymentSettings class."""
        self._origin = origin
        repos = {origin}
        self._title_regexps = title_regexps = {}
        self._labels = labels = {}
        self._labels_inv = {}
        for repo, obj in deps.items():
            if pattern := obj.get("title"):
                repos.add(repo)
                title_regexps[repo] = re.compile(f"^({pattern})", re.MULTILINE)
            if obj_labels := obj.get("labels", {}):
                self._labels_inv[repo] = obj_labels
                repos.add(repo)
                for label, values in obj_labels.items():
                    label_values = labels.setdefault(label, {})
                    for value in values:
                        assert isinstance(value, (str, int))
                        label_values.setdefault(value, []).append(repo)
        self._repos = frozenset(repos)

    def __str__(self) -> str:
        """Implement str()."""
        labels = defaultdict(dict)
        for label, label_values in self._labels.items():
            for value, repos in label_values.items():
                for repo in repos:
                    labels[repo].setdefault(label, []).append(value)
        return str(
            dict(
                (
                    r,
                    {
                        **(
                            {"title": pattern.pattern[2:-1]}
                            if (pattern := self._title_regexps.get(r))
                            else {}
                        ),
                        **({"labels": repo_labels} if (repo_labels := labels.get(r)) else {}),
                    },
                )
                for r in sorted(self._repos - {self._origin})
            ),
        )

    def title(self, repo: str) -> re.Pattern:
        """Return the title regexp for the given logical repository."""
        return self._title_regexps[repo]

    def labels(self, repo: str) -> Dict[str, List[str]]:
        """Return the label key values for the given logical repository."""
        return self._labels_inv[repo]

    def match(
        self,
        notifications: pd.DataFrame,
        labels: pd.DataFrame,
    ) -> Dict[str, Set[str]]:
        """
        Split deployed components into logical parts.

        :param notifications: deployment notifications.
        :param labels: labels of the deployment notifications.
        :return: logical repository names mapped to the deployment names.
        """
        assert isinstance(notifications, pd.DataFrame)
        assert isinstance(labels, pd.DataFrame)
        assert notifications.index.name == DeploymentNotification.name.name
        matched = {}
        if self.has_titles:
            titles = notifications.index.values
            lengths = np.fromiter((len(s) for s in titles), int, len(titles)) + 1
            offsets = np.zeros(len(lengths), dtype=int)
            np.cumsum(lengths[:-1], out=offsets[1:])
            concat_titles = "\n".join(titles)  # deployment names are guaranteed to not contain \n
            for repo, regexp in self._title_regexps.items():
                found = [m.start() for m in regexp.finditer(concat_titles)]
                found = titles[np.in1d(offsets, found, assume_unique=True)]
                matched[repo] = set(found)
        if not labels.empty and self.has_labels:
            logical_labels = self._labels
            label_order = np.argsort(labels.index.values)
            unique_label_deployment_names, label_group_counts = np.unique(
                labels.index.values[label_order], return_counts=True,
            )
            label_pos = 0
            for deployment_name, group_size in zip(
                unique_label_deployment_names, label_group_counts,
            ):
                indexes = label_order[label_pos : label_pos + group_size]
                label_pos += group_size
                for label, value in zip(
                    labels[DeployedLabel.key.name].values[indexes],
                    labels[DeployedLabel.value.name].values[indexes],
                ):
                    try:
                        repo_labels = logical_labels[label]
                    except KeyError:
                        continue
                    try:
                        for repo in repo_labels[value]:
                            matched.setdefault(repo, set()).add(deployment_name)
                    except (KeyError, TypeError):
                        continue
        return matched


class LogicalRepositorySettings:
    """Rules for matching PRs, releases, deployments to account-defined sub-repositories."""

    def __init__(self, prs: Mapping[str, Any], deployments: Mapping[str, Any]):
        """Initialize a new instance of LogicalRepositorySettings class."""
        pr_settings = defaultdict(dict)
        for repo, val in prs.items():
            pr_settings[drop_logical_repo(repo)][repo] = val
        self._prs = {repo: LogicalPRSettings(val, repo) for repo, val in pr_settings.items()}
        dep_settings = defaultdict(dict)
        for repo, val in deployments.items():
            dep_settings[drop_logical_repo(repo)][repo] = val
        self._deployments = {
            repo: LogicalDeploymentSettings(val, repo) for repo, val in dep_settings.items()
        }

    def __str__(self) -> str:
        """Implement str()."""
        prs = [f"{repo}: {val}" for repo, val in sorted(self._prs.items())]
        deps = [f"{repo}: {val}" for repo, val in sorted(self._deployments.items())]
        return f"prs: {prs}; deployments: {deps}"

    def __repr__(self) -> str:
        """Implement repr()."""
        return (
            f"LogicalRepositorySettings(prs={repr(self._prs)}, "
            f"deployments={repr(self._deployments)})"
        )

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)

    def has_logical_prs(self) -> bool:
        """Return value indicating whether there are any logical PR settings."""
        return any(self._prs.values())

    def has_logical_deployments(self) -> bool:
        """Return value indicating whether there are any logical PR settings."""
        return any(self._deployments.values())

    @classmethod
    def empty(cls) -> "LogicalRepositorySettings":
        """Initialize clear settings without logical repositories."""
        return LogicalRepositorySettings({}, {})

    def with_logical_prs(self, repos: Iterable[str]) -> Set[str]:
        """Collect all mentioned logical repositories."""
        return set(
            chain.from_iterable(prs.logical_repositories for prs in self._prs.values() if prs),
        ).union(repos)

    def with_logical_deployments(self, repos: Iterable[str]) -> Set[str]:
        """Collect all mentioned logical repositories."""
        return set(
            chain.from_iterable(
                dep.logical_repositories for dep in self._deployments.values() if dep
            ),
        ).union(repos)

    def prs(self, repo: str) -> Optional[LogicalPRSettings]:
        """Return PR match rules for the given repository native name."""
        return self._prs[repo]

    def deployments(self, repo: str) -> Optional[LogicalDeploymentSettings]:
        """Return deployment match rules for the given repository native name."""
        return self._deployments[repo]

    def has_prs_by_label(self, repos: Optional[Iterable[str]] = None) -> bool:
        """
        Return value indicating whether there is at least one logical repo identified by a label.

        :param repos: Physical repositories.
        """
        if repos is None:
            repos = self._prs.keys()
        for repo in repos:
            try:
                if self.prs(repo).has_labels:
                    return True
            except KeyError:
                continue
        return False

    def append_logical_prs(self, items: Collection[str]) -> Collection[str]:
        """
        Extend the physical repositories with their configured logical components.

        Repository names are not prefixed!
        """
        if not self.has_logical_prs():
            return items
        repos = set(items)
        for repo in items:
            try:
                repos.update(self.prs(repo).logical_repositories)
            except KeyError:
                continue
        return repos


class Settings:
    """
    Account settings.

    - Release match rules.
    - Logical repository rules.
    """

    def __init__(
        self,
        do_not_call_me_directly: Any,
        *,
        account: int,
        user_id: Optional[str],
        login: Optional[Callable[[], Coroutine[None, None, str]]],
        sdb: DatabaseLike,
        mdb: DatabaseLike,
        cache: Optional[aiomcache.Client],
        slack: Optional[SlackWebClient],
    ):
        """Initialize a new instance of Settings class."""
        self._account = account
        self._user_id = user_id
        self._login = login
        self._sdb = sdb
        self._mdb = mdb
        self._cache = cache
        self._slack = slack

    @classmethod
    def from_account(
        cls,
        account: int,
        sdb: Database,
        mdb: Database,
        cache: Optional[aiomcache.Client],
        slack: Optional[SlackWebClient],
    ):
        """Create a new Settings class instance in readonly mode given the account ID."""
        return Settings(
            None,
            account=account,
            user_id=None,
            login=None,
            sdb=sdb,
            mdb=mdb,
            cache=cache,
            slack=slack,
        )

    @classmethod
    def from_request(cls, request: AthenianWebRequest, account: int) -> "Settings":
        """Create a new Settings class instance in readwrite mode from the request object and \
        the account ID."""

        async def login_loader() -> str:
            return (await request.user()).login

        return Settings(
            None,
            account=account,
            user_id=request.uid,
            login=login_loader,
            sdb=request.sdb,
            mdb=request.mdb,
            cache=request.cache,
            slack=request.app["slack"],
        )

    async def list_release_matches(
        self,
        repos: Optional[Collection[str]] = None,
    ) -> ReleaseSettings:
        """
        List the current release matching settings for the specified repositories.

        :param repos: *Prefixed* repository names. If is None, load all the repositories \
                      belonging to the account.
        """
        if repos is None:
            repos = await get_account_repositories(self._account, True, self._sdb)
        repos = set(repos).union(coerce_prefixed_logical_repos(repos).keys())
        rows = await self._sdb.fetch_all(
            select([ReleaseSetting]).where(
                and_(
                    ReleaseSetting.account_id == self._account,
                    ReleaseSetting.repository.in_(repos),
                ),
            ),
        )
        settings = []
        loaded = set()
        for row in rows:
            repo = row[ReleaseSetting.repository.name]
            loaded.add(repo)
            settings.append(
                (
                    repo,
                    ReleaseMatchSetting(
                        branches=row[ReleaseSetting.branches.name],
                        tags=row[ReleaseSetting.tags.name],
                        events=row[ReleaseSetting.events.name],
                        match=ReleaseMatch(row[ReleaseSetting.match.name]),
                    ),
                ),
            )
        missing_logical = []
        for repo in repos:
            if repo not in loaded:
                if RepositoryName.from_prefixed(repo).is_logical:
                    missing_logical.append(repo)
                    continue
                settings.append(
                    (
                        repo,
                        ReleaseMatchSetting(
                            branches=default_branch_alias,
                            tags=".*",
                            events=".*",
                            match=ReleaseMatch.tag_or_branch,
                        ),
                    ),
                )
        if missing_logical:
            raise ResponseError(
                MissingSettingsError(
                    detail=f"Logical repositories must have release settings: {missing_logical}",
                ),
            )
        settings.sort()
        settings = dict(settings)
        return ReleaseSettings(settings)

    async def set_release_matches(
        self,
        repos: List[str],
        branches: str,
        tags: str,
        events: str,
        match: ReleaseMatch,
        meta_ids: Optional[Tuple[int, ...]] = None,
        prefixer: Optional[Prefixer] = None,
        dereference: bool = True,
        pointer_root: str = "",
    ) -> Set[str]:
        """Set the release matching rule for a list of repositories."""
        for propname, s in (
            ("branches", ReleaseMatch.branch),
            ("tags", ReleaseMatch.tag),
            ("events", ReleaseMatch.event),
        ):
            propval = locals()[propname]
            if match in (s, ReleaseMatch.tag_or_branch) and not propval:
                raise ResponseError(
                    InvalidRequestError(
                        f"{pointer_root}.{propname}",
                        detail='Value may not be empty given "match" = "%s"' % match.name,
                    ),
                )
            if propname == "events" and propval is None:
                continue
            try:
                re.compile(propval)
            except (re.error, TypeError) as e:
                raise ResponseError(
                    InvalidRequestError(
                        f"{pointer_root}.{propname}", detail=f"Invalid regular expression: {e}",
                    ),
                ) from None
        if not branches:
            branches = default_branch_alias
        if not tags:
            tags = ".*"

        if dereference:
            try:
                if meta_ids is None:
                    meta_ids = await get_metadata_account_ids(
                        self._account, self._sdb, self._cache,
                    )
            except ResponseError as e:
                if repos:
                    raise e from None
                meta_ids = None
            else:
                if prefixer is None:
                    prefixer = await Prefixer.load(meta_ids, self._mdb, self._cache)
                settings = Settings.from_account(
                    self._account, self._sdb, self._mdb, self._cache, self._slack,
                )
            repos, _ = await resolve_repos(
                repos,
                self._account,
                self._user_id,
                self._login,
                meta_ids,
                self._sdb,
                self._mdb,
                self._cache,
                self._slack,
                strip_prefix=False,
            )
        values = []
        for repo in repos:
            if RepositoryName.from_prefixed(repo).is_logical:
                if match not in (ReleaseMatch.tag, ReleaseMatch.event):
                    raise ResponseError(
                        InvalidRequestError(
                            f"{pointer_root}.{match}",
                            detail=(
                                f"Logical repository {repo} must be released either by tag or by "
                                "submitted event."
                            ),
                        ),
                    )
            values.append(
                ReleaseSetting(
                    repository=repo,
                    account_id=self._account,
                    branches=branches,
                    tags=tags,
                    events=events,
                    match=match,
                )
                .create_defaults()
                .explode(with_primary_keys=True),
            )
        query = (await dialect_specific_insert(self._sdb))(ReleaseSetting)
        query = query.on_conflict_do_update(
            index_elements=ReleaseSetting.__table__.primary_key.columns,
            set_={
                ReleaseSetting.match.name: query.excluded.match,
                ReleaseSetting.branches.name: query.excluded.branches,
                ReleaseSetting.tags.name: query.excluded.tags,
                ReleaseSetting.events.name: query.excluded.events,
                ReleaseSetting.updated_at.name: query.excluded.updated_at,
            },
        )

        if isinstance(self._sdb, Database):
            async with self._sdb.connection() as sdb_conn:
                async with sdb_conn.transaction():
                    await sdb_conn.execute_many(query, values)
        else:
            await self._sdb.execute_many(query, values)
        return repos

    async def list_logical_repositories(
        self,
        prefixer: Prefixer,
        repos: Optional[Collection[str]] = None,
        pointer: Optional[str] = None,
    ) -> LogicalRepositorySettings:
        """
        List the current logical repository settings for the specified repositories.

        :param repos: *Prefixed* repository names. If is None load all the repositories \
                      belonging to the account.
        :param pointer: Optional pointer to the web model for error reporting.
        """
        if repos is None:
            try:
                repos = await get_account_repositories(self._account, False, self._sdb)
            except ResponseError:
                repos = []
            diff_repos = set()
        else:
            repos = {r.split("/", 1)[1] for r in repos}
            logical_repos = coerce_logical_repos(repos)
            diff_repos = repos - logical_repos.keys()
            repos = logical_repos.keys()
        repo_name_to_node = prefixer.repo_name_to_node.get
        repo_ids = [n for r in repos if (n := repo_name_to_node(r))]
        rows = await self._sdb.fetch_all(
            select([LogicalRepository]).where(
                and_(
                    LogicalRepository.account_id == self._account,
                    LogicalRepository.repository_id.in_(repo_ids),
                ),
            ),
        )
        prs = {}
        deployments = {}
        repo_node_to_name = prefixer.repo_node_to_name.__getitem__
        for row in rows:
            physical_name = repo_node_to_name(row[LogicalRepository.repository_id.name])
            logical_name = "/".join((physical_name, row[LogicalRepository.name.name]))
            diff_repos.discard(logical_name)
            if prs_setting := row[LogicalRepository.prs.name]:
                prs[logical_name] = prs_setting
            if deployments_setting := row[LogicalRepository.deployments.name]:
                deployments[logical_name] = deployments_setting
        if diff_repos:
            pointer = pointer or "?"
            raise ResponseError(
                InvalidRequestError(
                    pointer, detail="Some logical repositories do not exist: %s" % diff_repos,
                ),
            )
        return LogicalRepositorySettings(prs, deployments)
