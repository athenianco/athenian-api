from typing import Optional

from athenian.api.models.web.base_model_ import Model


class JIRAFilter(Model):
    """Filters related to JIRA traits. The aggregation between each filter type is *AND*."""

    epics: Optional[list[str]]
    labels_include: Optional[list[str]]
    labels_exclude: Optional[list[str]]
    issue_types: Optional[list[str]]
    priorities: Optional[list[str]]
    projects: Optional[list[str]]
    status_categories: Optional[list[str]]
    unmapped: Optional[bool]

    def validate_epics(self, epics: Optional[list[str]]) -> Optional[list[str]]:
        """Validate the epics field."""
        if epics and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return epics

    def validate_labels_include(self, labels_include: Optional[list[str]]) -> Optional[list[str]]:
        """Validate the `labels_include` field."""
        if labels_include and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return labels_include

    def validate_labels_exclude(self, labels_exclude: Optional[list[str]]) -> Optional[list[str]]:
        """Validate `labels_exclude` field."""
        if labels_exclude and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return labels_exclude

    def validate_issue_types(self, issue_types: Optional[list[str]]) -> Optional[list[str]]:
        """Validate the `issue_types` field."""
        if issue_types and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return issue_types

    def validate_priorities(self, priorities: Optional[list[str]]) -> Optional[list[str]]:
        """Validate the `priorities`."""
        if priorities and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return priorities

    def validate_status_categories(
        self,
        status_categories: Optional[list[str]],
    ) -> Optional[list[str]]:
        """Validate the `status_categories` field."""
        if status_categories and getattr(self, "_unmapped", False):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return status_categories

    def validate_unmapped(self, unmapped: Optional[bool]) -> Optional[bool]:
        """Validate the `unmapped" field."""
        if unmapped and (
            getattr(self, "_epics", False)
            or getattr(self, "_labels_include", False)
            or getattr(self, "_labels_exclude", False)
            or getattr(self, "_issue_types", False)
            or getattr(self, "_priorities", False)
            or getattr(self, "_status_categories", False)
        ):
            raise ValueError("`unmapped` may not be mixed with anything else")
        return unmapped

    def __bool__(self):
        """Return True if the filter has any field defined."""
        return any(getattr(self, field) for field in self.attribute_types)
