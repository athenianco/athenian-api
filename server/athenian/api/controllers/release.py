from typing import Optional

from athenian.api.controllers.miners.types import ReleaseParticipants, ReleaseParticipationKind
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.models.web import InvalidRequestError, ReleaseWith
from athenian.api.response import ResponseError


async def extract_release_participants(filt_with: Optional[ReleaseWith],
                                       prefixer: Prefixer,
                                       position: int = None,
                                       ) -> ReleaseParticipants:
    """Resolve and deduplicate people mentioned in releases."""
    if filt_with is None:
        return {}
    if position is None:
        position = ""
    else:
        position = f"[{position}]"
    user_login_to_node = prefixer.user_login_to_node.__getitem__
    participants = {}
    for attr, rpk in (("releaser", ReleaseParticipationKind.RELEASER),
                      ("pr_author", ReleaseParticipationKind.PR_AUTHOR),
                      ("commit_author", ReleaseParticipationKind.COMMIT_AUTHOR)):
        people = []

        def pointer(i: int) -> str:
            return f".with{position}.{attr}[{i}]"

        for i, u in enumerate(getattr(filt_with, attr) or []):
            if "/" not in u:
                raise ResponseError(InvalidRequestError(
                    detail=f'Invalid developer ID: "{u}". Are you missing a "github.com/" prefix?',
                    pointer=pointer(i),
                ))
            try:
                people.extend(user_login_to_node(u.rsplit("/", 1)[1]))
            except KeyError:
                raise ResponseError(InvalidRequestError(
                    pointer=pointer(i),
                    detail=f'User "{u}" does not exist.',
                )) from None
        participants[rpk] = sorted(set(people))
    return participants
