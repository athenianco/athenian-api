from athenian.api.models.web.base_model_ import Model


class WorkTypeGetRequest(Model):
    """Identifier of a work type - a set of rules to group PRs, releases, etc. together."""

    account: int
    name: str

    def validate_name(self, name: str) -> str:
        """Sets the name of this WorkTypeGetRequest.

        Work type name. It is unique within the account scope.

        :param name: The name of this WorkTypeGetRequest.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `3`",
            )

        return name
