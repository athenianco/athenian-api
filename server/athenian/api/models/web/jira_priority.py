from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAPriority(Model):
    """JIRA issue priority details."""

    attribute_types = {
        "name": str,
        "image": str,
        "rank": int,
        "color": str,
    }
    attribute_map = {
        "name": "name",
        "image": "image",
        "rank": "rank",
        "color": "color",
    }

    def __init__(
        self,
        name: Optional[str] = None,
        image: Optional[str] = None,
        rank: Optional[int] = None,
        color: Optional[str] = None,
    ):
        """JIRAPriority - a model defined in OpenAPI

        :param name: The name of this JIRAPriority.
        :param image: The image of this JIRAPriority.
        :param rank: The rank of this JIRAPriority.
        :param color: The color of this JIRAPriority.
        """
        self._name = name
        self._image = image
        self._rank = rank
        self._color = color

    def __lt__(self, other: "JIRAPriority") -> bool:
        """Support sorting."""
        return (self._rank, self._name) < (other._rank, other._name)

    def __hash__(self) -> int:
        """Support dict-s."""
        return hash(self._name)

    @property
    def name(self) -> str:
        """Gets the name of this JIRAPriority.

        Name of the priority.

        :return: The name of this JIRAPriority.
        """
        return self._name

    @name.setter
    def name(self, name: str):
        """Sets the name of this JIRAPriority.

        Name of the priority.

        :param name: The name of this JIRAPriority.
        """
        if name is None:
            raise ValueError("Invalid rank for `name`, must not be `None`")

        self._name = name

    @property
    def image(self) -> str:
        """Gets the image of this JIRAPriority.

        URL of the picture that indicates the priority.

        :return: The image of this JIRAPriority.
        """
        return self._image

    @image.setter
    def image(self, image: str):
        """Sets the image of this JIRAPriority.

        URL of the picture that indicates the priority.

        :param image: The image of this JIRAPriority.
        """
        if image is None:
            raise ValueError("Invalid rank for `image`, must not be `None`")

        self._image = image

    @property
    def rank(self) -> int:
        """Gets the rank of this JIRAPriority.

        Measure of importance (bigger is more important).

        :return: The rank of this JIRAPriority.
        """
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        """Sets the rank of this JIRAPriority.

        Measure of importance (smaller is more important).

        :param rank: The rank of this JIRAPriority.
        """
        if rank is None:
            raise ValueError("Invalid rank for `rank`, must not be `None`")
        if rank is not None and rank < 1:
            raise ValueError(
                "Invalid rank for `rank`, must be a rank greater than or equal to `1`"
            )

        self._rank = rank

    @property
    def color(self) -> str:
        """Gets the color of this JIRAPriority.

        24-bit hex RGB.

        :return: The color of this JIRAPriority.
        """
        return self._color

    @color.setter
    def color(self, color: str):
        """Sets the color of this JIRAPriority.

        24-bit hex RGB.

        :param color: The color of this JIRAPriority.
        """
        if color is None:
            raise ValueError("Invalid rank for `color`, must not be `None`")

        self._color = color
