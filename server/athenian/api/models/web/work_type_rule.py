from athenian.api.models.web.base_model_ import Model


class WorkTypeRule(Model):
    """Specific rule details: name and parameters."""

    name: str
    body: object

    def validate_name(self, name: str) -> str:
        """Sets the name of this WorkTypeRule.

        Rule name - must be unique within the parent work type.

        :param name: The name of this WorkTypeRule.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `1`",
            )

        return name
