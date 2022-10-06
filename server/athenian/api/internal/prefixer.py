from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import pickle
from typing import Dict, Iterable, List, Optional

import aiomcache
from sqlalchemy import select

from athenian.api.async_utils import gather
from athenian.api.cache import cached, short_term_exptime
from athenian.api.db import DatabaseLike
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.models.metadata.github import Repository, User
from athenian.api.request import AthenianWebRequest


@dataclasses.dataclass(frozen=True, slots=True)
class RepositoryName:
    """Parsed name of a repository.

    Possible formats:

    - repository name: "owner/reponame"
    - logical repository name: "owner/reponame/logicalname".
    - prefixed repository name: "provider.com/owner/reponame"
    - prefixed logical repository name: "provider.com/owner/reponame/logicalname"
    """

    prefix: Optional[str]
    owner: str
    physical: str
    logical: Optional[str]

    @classmethod
    def from_prefixed(cls, prefixed_name: str) -> RepositoryName:
        """Build the name one of the "prefixed" formats."""
        if prefixed_name.count("/") < 2:
            raise ValueError(f"Invalid  prefixed repo name {prefixed_name}")

        prefix, rest = prefixed_name.split("/", 1)

        if "." not in prefix:
            raise ValueError(f"Invalid  prefixed repo name {prefixed_name}")

        org, rest = rest.split("/", 1)

        if "/" in rest:
            physical, logical = rest.split("/", 1)
        else:
            physical = rest
            logical = None
        return RepositoryName(prefix, org, physical, logical)

    @property
    def is_logical(self) -> bool:
        """Whether the name refers to a logical repository."""
        return self.logical is not None

    def with_logical(self, logical: Optional[str]) -> RepositoryName:
        """Return a new RepositoryName for the same physical repository with an optional logical \
        name."""
        return dataclasses.replace(self, logical=logical)

    @property
    def unprefixed_physical(self) -> str:
        """Return the unprefixed physical name of the repository."""
        return f"{self.owner}/{self.physical}"

    @property
    def unprefixed(self) -> str:
        """Return the unprefixed name of the repository."""
        name = self.unprefixed_physical
        if self.logical is not None:
            name = f"{name}/{self.logical}"
        return name

    def __str__(self) -> str:
        """Return the canonical full repository name."""
        return (
            f"{(self.prefix + '/') if self.prefix else ''}"
            f"{self.owner}/{self.physical}"
            f"{('/' + self.logical) if self.logical else ''}"
        )

    def __sentry_repr__(self) -> str:
        """Format for Sentry the same way as regular str()."""
        return str(self)


@dataclasses.dataclass(frozen=True, slots=True)
class RepositoryReference:
    """The identity of a repository, physical or logical."""

    node_id: int
    """
    The identifier of the physical repository

    It's a reference to the mdb's `Repository.node_id` column.
    """
    logical_name: Optional[str]
    """
    The logical name of repository, if this is a logical repository.
    """


def strip_proto(url: str) -> str:
    """Remove https:// string prefix."""
    return url.split("://", 1)[1]


@dataclass(slots=True, frozen=True, repr=False)
class Prefixer:
    """
    Prepend service prefixes to repository and user names.

    For example, turn athenianco/athenian-api to github.com/athenianco/athenian-api.
    """

    do_not_construct_me_directly: None
    repo_node_to_prefixed_name: Dict[int, str]
    repo_name_to_prefixed_name: Dict[str, str]
    repo_node_to_name: Dict[int, str]
    repo_name_to_node: Dict[str, int]
    user_node_to_prefixed_login: Dict[int, str]
    user_login_to_prefixed_login: Dict[str, str]
    user_node_to_login: Dict[int, str]
    user_login_to_node: Dict[str, List[int]]  # same users in different accounts

    @staticmethod
    @cached(
        exptime=short_term_exptime,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda meta_ids, **_: (",".join(str(i) for i in meta_ids),),
    )
    async def load(
        meta_ids: Iterable[int],
        mdb: DatabaseLike,
        cache: Optional[aiomcache.Client],
    ) -> "Prefixer":
        """Load node IDs and prefixes for all known repositories and users."""
        repo_rows, user_rows = await gather(
            mdb.fetch_all(
                select(Repository.node_id, Repository.full_name, Repository.html_url).where(
                    Repository.acc_id.in_(meta_ids), Repository.full_name.isnot(None),
                ),
            ),
            mdb.fetch_all(
                select(User.node_id, User.login, User.html_url).where(
                    User.acc_id.in_(meta_ids), User.login.isnot(None),
                ),
            ),
            op="Prefixer",
        )

        repo_node_to_prefixed_name = {
            r[Repository.node_id.name]: strip_proto(r[Repository.html_url.name]) for r in repo_rows
        }
        repo_node_to_name = {
            r[Repository.node_id.name]: r[Repository.full_name.name] for r in repo_rows
        }
        repo_name_to_node = {
            r[Repository.full_name.name]: r[Repository.node_id.name] for r in repo_rows
        }
        repo_name_to_prefixed_name = {
            r[Repository.full_name.name]: strip_proto(r[Repository.html_url.name])
            for r in repo_rows
        }
        user_node_to_prefixed_login = {
            r[User.node_id.name]: strip_proto(r[User.html_url.name]) for r in user_rows
        }
        user_login_to_prefixed_login = {
            r[User.login.name]: strip_proto(r[User.html_url.name]) for r in user_rows
        }
        user_node_to_login = {r[User.node_id.name]: r[User.login.name] for r in user_rows}
        user_login_to_node = {}
        for r in user_rows:
            user_login_to_node.setdefault(r[User.login.name], []).append(r[User.node_id.name])
        return Prefixer(
            None,
            repo_node_to_prefixed_name=repo_node_to_prefixed_name,
            repo_name_to_prefixed_name=repo_name_to_prefixed_name,
            repo_node_to_name=repo_node_to_name,
            repo_name_to_node=repo_name_to_node,
            user_node_to_prefixed_login=user_node_to_prefixed_login,
            user_login_to_prefixed_login=user_login_to_prefixed_login,
            user_node_to_login=user_node_to_login,
            user_login_to_node=user_login_to_node,
        )

    @staticmethod
    async def from_request(request: AthenianWebRequest, account: int) -> Prefixer:
        """
        Initialize a new Prefixer from the account ID and request.

        Use this method with caution! It swallows `meta_ids`.
        """
        meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
        return await Prefixer.load(meta_ids, request.mdb, request.cache)

    def resolve_repo_nodes(self, repo_node_ids: Iterable[int]) -> List[str]:
        """Lookup each repository node ID in repo_node_map."""
        return [self.repo_node_to_prefixed_name[node_id] for node_id in repo_node_ids]

    def prefix_repo_names(self, repo_names: Iterable[str]) -> List[str]:
        """Lookup each repository full name in repo_name_map."""
        return [self.prefix_logical_repo(name) for name in repo_names]

    def resolve_user_nodes(self, user_node_ids: Iterable[int]) -> List[str]:
        """Lookup each user node ID in user_node_to_prefixed_login."""
        return [self.user_node_to_prefixed_login[node_id] for node_id in user_node_ids]

    def prefix_user_logins(self, user_logins: Iterable[str]) -> List[str]:
        """Lookup each user login in user_login_to_prefixed_login."""
        return [
            pl
            for name in user_logins
            if (pl := self.user_login_to_prefixed_login.get(name)) is not None
        ]

    def prefix_logical_repo(self, repo: str) -> Optional[str]:
        """Lookup the repository name prefix for the given logical repository."""
        *physical_repo, logical_name = repo.split("/", 2)
        try:
            if len(physical_repo) == 1:
                return self.repo_name_to_prefixed_name[repo]
            physical_repo = self.repo_name_to_prefixed_name["/".join(physical_repo)]
            return "/".join([physical_repo, logical_name])
        except KeyError:
            return None

    def prefixed_repo_names_to_identities(
        self,
        repo_names: Iterable[str],
    ) -> list[RepositoryReference]:
        """Convert a list of prefixed repository names to RepositoryReference-s."""
        repos = []
        repo_name_to_node = self.repo_name_to_node.__getitem__
        for repo_name in repo_names:
            name = RepositoryName.from_prefixed(repo_name)
            try:
                physical_id = repo_name_to_node(name.unprefixed_physical)
            except KeyError:
                raise ValueError(f"Unknown repository {repo_name}") from None
            repos.append(RepositoryReference(physical_id, name.logical))
        return repos

    def repo_identities_to_prefixed_names(
        self,
        repo_identities: Iterable[RepositoryReference],
    ) -> list[str]:
        """Convert a list of RepositoryReference to a list of prefixed repository names."""
        repo_names = []
        repo_node_to_prefixed_name = self.repo_node_to_prefixed_name.__getitem__
        for repo in repo_identities:
            try:
                physical_name = repo_node_to_prefixed_name(repo.node_id)
            except KeyError:
                raise ValueError(f"Invalid repo_id {repo.node_id}") from None
            name = RepositoryName.from_prefixed(physical_name).with_logical(repo.logical_name)
            repo_names.append(str(name))
        return repo_names

    def __str__(self) -> str:
        """Implement str()."""
        return repr(self)

    def __repr__(self) -> str:
        """Avoid spamming Sentry stacks."""
        return (
            f"<Prefixer with {len(self.repo_node_to_name)} repos, "
            f"{len(self.user_node_to_login)} users>"
        )

    def __sentry_repr__(self) -> str:
        """Override {}.__repr__() in Sentry."""
        return repr(self)
