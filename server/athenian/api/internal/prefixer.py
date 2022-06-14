import pickle
from typing import Dict, Iterable, List, Optional

import aiomcache
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.cache import cached, short_term_exptime
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import Repository, User
from athenian.api.typing_utils import dataclass


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
                select([Repository.node_id, Repository.full_name, Repository.html_url]).where(
                    and_(Repository.acc_id.in_(meta_ids), Repository.full_name.isnot(None)),
                ),
            ),
            mdb.fetch_all(
                select([User.node_id, User.login, User.html_url]).where(
                    and_(User.acc_id.in_(meta_ids), User.login.isnot(None)),
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
