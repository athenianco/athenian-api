from athenian.api.models.web.base_model_ import Model


class DeployedComponent(Model):
    """Definition of the deployed software unit."""

    repository: str
    reference: str

    def validate_reference(self, reference: str) -> str:
        """Sets the reference of this DeployedComponent.

        We accept three ways to define a Git reference: 1. Tag name. 2. Full commit hash
        (40 characters). 3. Short commit hash (7 characters).  We ignore the reference while we
        cannot find it in our database. There can be two reasons: - There is a mistake or a typo
        in the provided data. - We are temporarily unable to synchronize with GitHub.

        :param reference: The reference of this DeployedComponent.
        """
        if reference is None:
            raise ValueError("Invalid value for `reference`, must not be `None`")
        if not reference:
            raise ValueError(
                "Invalid value for `reference`, length must be greater than or equal to `1`",
            )

        return reference
