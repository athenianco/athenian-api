from itertools import chain
from typing import Optional, Tuple

from sqlalchemy import and_, select

from athenian.api.controllers.miners.types import ReleaseParticipants, ReleaseParticipationKind
from athenian.api.db import ParallelDatabase
from athenian.api.models.metadata.github import User
from athenian.api.models.web import ReleaseWith


async def extract_release_participants(filt_with: Optional[ReleaseWith],
                                       meta_ids: Tuple[int, ...],
                                       mdb: ParallelDatabase,
                                       ) -> ReleaseParticipants:
    """Resolve and deduplicate people mentioned in releases."""
    if filt_with is None:
        return {}
    everybody = [
        u.rsplit("/", 1)[1]
        for u in set(chain(filt_with.releaser or [],
                           filt_with.pr_author or [],
                           filt_with.commit_author or []))
    ]
    node_rows = await mdb.fetch_all(select([User.node_id, User.html_url])
                                    .where(and_(User.acc_id.in_(meta_ids),
                                                User.login.in_(everybody))))
    nodes_map = {u[User.html_url.name].split("://", 1)[1]: u[User.node_id.name]
                 for u in node_rows}
    participants = {}
    for attr, rpk in (("releaser", ReleaseParticipationKind.RELEASER),
                      ("pr_author", ReleaseParticipationKind.PR_AUTHOR),
                      ("commit_author", ReleaseParticipationKind.COMMIT_AUTHOR)):
        if nodes := list({nodes_map.get(u) for u in (getattr(filt_with, attr) or [])} - {None}):
            participants[rpk] = nodes
    return participants
