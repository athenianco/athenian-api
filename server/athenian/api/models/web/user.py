from typing import Any, Optional

import dateutil.parser

from athenian.api.models.web.account_status import AccountStatus
from athenian.api.models.web.base_model_ import Model
from athenian.api.typing_utils import VerbatimOptional


class User(Model):
    """User profile from Auth0."""

    __extra_slots__ = ("account",)

    id: str
    native_id: str
    login: str
    name: Optional[str]
    email: Optional[str]
    picture: Optional[str]
    updated: Optional[str]
    accounts: VerbatimOptional[dict[int, AccountStatus]]
    impersonated_by: Optional[str]

    @classmethod
    def from_auth0(
        cls,
        name: str,
        nickname: str,
        picture: str,
        updated_at: str,
        email: Optional[str] = None,
        sub: Optional[str] = None,
        user_id: Optional[str] = None,
        identities: Optional[list[dict]] = None,
        account: Optional[int] = None,
        user_metadata: Optional[dict[str, Any]] = None,
        **_,
    ):
        """Create a new User object from Auth0 /userinfo."""
        if sub is None and user_id is None:
            raise TypeError('Either "sub" or "user_id" must be set to create a User.')
        id_ = sub or user_id
        if identities:
            native_id = identities[0]["user_id"]
        else:
            native_id = id_.rsplit("|", 1)[1]
        if user_metadata is not None:
            email = user_metadata.get("email", email)
        if email is None:
            email = ""
        user = cls(
            id=id_,
            native_id=native_id,
            login=nickname,
            email=email,
            name=name,
            picture=picture,
            updated=dateutil.parser.parse(updated_at),
            accounts={},
        )
        user.account = account
        return user

    def __hash__(self) -> int:
        """Hash the object."""
        return hash(self.id)

    def __eq__(self, other: "User") -> bool:
        """Check objects for equality."""
        return self.id == other.id

    def __lt__(self, other: "User") -> bool:
        """Check whether the object is less than the other."""
        return self.id < other.id
