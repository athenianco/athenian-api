from typing import Optional

from athenian.api.models.web.base_model_ import Model


class WorkTypeRule(Model):
    """Specific rule details: name and parameters."""

    attribute_types = {"name": str, "body": object}

    attribute_map = {"name": "name", "body": "body"}

    def __init__(self, name: Optional[str] = None, body: Optional[object] = None):
        """WorkTypeRule - a model defined in OpenAPI

        :param name: The name of this WorkTypeRule.
        :param body: The body of this WorkTypeRule.
        """
        self._name = name
        self._body = body

    @property
    def name(self) -> str:
        """Gets the name of this WorkTypeRule.

        Rule name - must be unique within the parent work type.

        :return: The name of this WorkTypeRule.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this WorkTypeRule.

        Rule name - must be unique within the parent work type.

        :param name: The name of this WorkTypeRule.
        """
        if name is None:
            raise ValueError("Invalid value for `name`, must not be `None`")
        if len(name) < 1:
            raise ValueError(
                "Invalid value for `name`, length must be greater than or equal to `1`"
            )

        self._name = name

    @property
    def body(self) -> object:
        """Gets the body of this WorkTypeRule.

        Freeform parameters of the rule.

        :return: The body of this WorkTypeRule.
        """
        return self._body

    @body.setter
    def body(self, body: object):
        """Sets the body of this WorkTypeRule.

        Freeform parameters of the rule.

        :param body: The body of this WorkTypeRule.
        """
        if body is None:
            raise ValueError("Invalid value for `body`, must not be `None`")

        self._body = body
