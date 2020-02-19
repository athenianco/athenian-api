import pprint
import typing

from athenian.api import serialization

T = typing.TypeVar("T")


class Model:
    """Base API model class. Handles object -> {} and {} -> object transformations."""

    # openapiTypes: The key is attribute name and the
    # value is attribute type.
    openapi_types = {}

    # attributeMap: The key is attribute name and the
    # value is json key in definition.
    attribute_map = {}

    @classmethod
    def from_dict(cls, dikt: dict) -> T:
        """Returns the dict as a model."""
        return serialization.deserialize_model(dikt, cls)

    def to_dict(self) -> dict:
        """Returns the model properties as a dict."""
        result = {}

        for attr_key, json_key in self.attribute_map.items():
            value = getattr(self, attr_key)
            if value is None:
                continue
            if isinstance(value, list):
                result[json_key] = list(
                    map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value)  # noqa(C812)
                )
            elif hasattr(value, "to_dict"):
                result[json_key] = value.to_dict()
            elif isinstance(value, dict):
                result[json_key] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )  # noqa(C812)
                )
            else:
                result[json_key] = value

        return result

    def to_str(self) -> str:
        """Returns the string representation of the model."""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For debugging."""
        return "%s(%s)" % (type(self).__name__, ", ".join("%s=%r" % p for p in vars(self).items()))

    def __str__(self):
        """For `print` and `pprint`."""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal."""
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal."""
        return not self == other
