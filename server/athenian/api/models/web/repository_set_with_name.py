from typing import Optional

from athenian.api.models.web.base_model_ import Model


class RepositorySetWithName(Model):
    """NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).

    Do not edit the class manually.
    """

    name: Optional[str]
    items: Optional[list[str]]
    precomputed: Optional[bool]
