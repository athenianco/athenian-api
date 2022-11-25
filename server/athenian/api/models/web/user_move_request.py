from typing import Optional

from athenian.api.models.web.base_model_ import Model


class UserMoveRequest(Model):
    """Request body of `/user/move`.

    Definition of the operation to move the user to a new account.
    """

    new_account_admin: Optional[int]
    new_account_regular: Optional[int]
    old_account: int
    user: str
