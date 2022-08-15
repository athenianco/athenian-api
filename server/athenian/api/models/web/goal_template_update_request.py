from athenian.api.models.web.base_model_ import Model


class GoalTemplateUpdateRequest(Model):
    """Goal Template update request."""

    attribute_types = {
        "name": str,
    }

    def __init__(self, name: str = None):
        """GoalTemplateUpdateRequest - a model defined in OpenAPI

        :param name: The name of this GoalTemplateUpdateRequest.
        """
        self._name = name

    @property
    def name(self):
        """Gets the name of this GoalTemplateUpdateRequest."""
        return self._name

    @name.setter
    def name(self, name):
        """Sets the name of this GoalTemplateUpdateRequest."""
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if name is not None and len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `1`",
            )

        self._name = name
