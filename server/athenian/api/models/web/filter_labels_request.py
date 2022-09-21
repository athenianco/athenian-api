from athenian.api.models.web.base_model_ import Model


class FilterLabelsRequest(Model):
    """
    Request body of `/filter/labels`.

    Defines the account and the repositories where to look for the labels.
    """

    account: int
    repositories: list[str]
