from athenian.api.models.web.base_model_ import Model


class ProductFeature(Model):
    """Client-side product feature definition."""

    name: str
    parameters: object
