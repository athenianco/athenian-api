from datetime import date
from typing import List, Optional

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.common_filter_properties import CommonFilterPropertiesMixin
from athenian.api.models.web.jira_filter_with import JIRAFilterWith
from athenian.api.models.web.jira_histogram_definition import JIRAHistogramDefinition


class JIRAHistogramsRequest(Model, CommonFilterPropertiesMixin):
    """Request of `/histograms/jira`."""

    openapi_types = {
        "priorities": Optional[List[str]],
        "types": Optional[List[str]],
        "epics": Optional[List[str]],
        "labels_include": Optional[List[str]],
        "labels_exclude": Optional[List[str]],
        "with_": Optional[List[JIRAFilterWith]],
        "histograms": List[JIRAHistogramDefinition],
        "date_from": date,
        "date_to": date,
        "timezone": int,
        "exclude_inactive": bool,
        "quantiles": Optional[List[float]],
        "account": int,
    }

    attribute_map = {
        "priorities": "priorities",
        "types": "types",
        "epics": "epics",
        "labels_include": "labels_include",
        "labels_exclude": "labels_exclude",
        "with_": "with",
        "histograms": "histograms",
        "date_from": "date_from",
        "date_to": "date_to",
        "timezone": "timezone",
        "exclude_inactive": "exclude_inactive",
        "quantiles": "quantiles",
        "account": "account",
    }

    def __init__(
        self,
        priorities: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        epics: Optional[List[str]] = None,
        labels_include: Optional[List[str]] = None,
        labels_exclude: Optional[List[str]] = None,
        with_: Optional[List[JIRAFilterWith]] = None,
        histograms: Optional[List[JIRAHistogramDefinition]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        timezone: Optional[int] = None,
        exclude_inactive: Optional[bool] = None,
        quantiles: Optional[List[float]] = None,
        account: Optional[int] = None,
    ):
        """JIRAHistogramsRequest - a model defined in OpenAPI

        :param priorities: The priorities of this JIRAHistogramsRequest.
        :param types: The types of this JIRAHistogramsRequest.
        :param epics: The epics of this JIRAHistogramsRequest.
        :param labels_include: The labels_include of this JIRAHistogramsRequest.
        :param labels_exclude: The labels_exclude of this JIRAHistogramsRequest.
        :param with_: The with of this JIRAHistogramsRequest.
        :param histograms: The histograms of this JIRAHistogramsRequest.
        :param date_from: The date_from of this JIRAHistogramsRequest.
        :param date_to: The date_to of this JIRAHistogramsRequest.
        :param timezone: The timezone of this JIRAHistogramsRequest.
        :param exclude_inactive: The exclude_inactive of this JIRAHistogramsRequest.
        :param quantiles: The quantiles of this JIRAHistogramsRequest.
        :param account: The account of this JIRAHistogramsRequest.
        """
        self._priorities = priorities
        self._types = types
        self._epics = epics
        self._labels_include = labels_include
        self._labels_exclude = labels_exclude
        self._with_ = with_
        self._histograms = histograms
        self._date_from = date_from
        self._date_to = date_to
        self._timezone = timezone
        self._exclude_inactive = exclude_inactive
        self._quantiles = quantiles
        self._account = account

    @property
    def priorities(self) -> Optional[List[str]]:
        """Gets the priorities of this JIRAHistogramsRequest.

        Selected issue priorities.

        :return: The priorities of this JIRAHistogramsRequest.
        """
        return self._priorities

    @priorities.setter
    def priorities(self, priorities: Optional[List[str]]):
        """Sets the priorities of this JIRAHistogramsRequest.

        Selected issue priorities.

        :param priorities: The priorities of this JIRAHistogramsRequest.
        """
        self._priorities = priorities

    @property
    def types(self) -> Optional[List[str]]:
        """Gets the types of this JIRAHistogramsRequest.

        Selected issue types.

        :return: The types of this JIRAHistogramsRequest.
        """
        return self._types

    @types.setter
    def types(self, types: Optional[List[str]]):
        """Sets the types of this JIRAHistogramsRequest.

        Selected issue types.

        :param types: The types of this JIRAHistogramsRequest.
        """
        self._types = types

    @property
    def epics(self) -> Optional[List[str]]:
        """Gets the epics of this JIRAHistogramsRequest.

        Selected issue epics.

        :return: The epics of this JIRAHistogramsRequest.
        """
        return self._epics

    @epics.setter
    def epics(self, epics: Optional[List[str]]):
        """Sets the epics of this JIRAHistogramsRequest.

        Selected issue epics.

        :param epics: The epics of this JIRAHistogramsRequest.
        """
        self._epics = epics

    @property
    def labels_include(self) -> Optional[List[str]]:
        """Gets the labels_include of this JIRAHistogramsRequest.

        PRs must relate to at least one JIRA issue label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :return: The labels_include of this JIRAHistogramsRequest.
        """
        return self._labels_include

    @labels_include.setter
    def labels_include(self, labels_include: Optional[List[str]]):
        """Sets the labels_include of this JIRAHistogramsRequest.

        PRs must relate to at least one JIRA issue label from the list. Several labels may be
        concatenated by a comma `,` so that all of them are required.

        :param labels_include: The labels_include of this JIRAHistogramsRequest.
        """
        self._labels_include = labels_include

    @property
    def labels_exclude(self) -> Optional[List[str]]:
        """Gets the labels_exclude of this JIRAHistogramsRequest.

        PRs cannot relate to JIRA issue labels from the list.

        :return: The labels_exclude of this JIRAHistogramsRequest.
        """
        return self._labels_exclude

    @labels_exclude.setter
    def labels_exclude(self, labels_exclude: Optional[List[str]]):
        """Sets the labels_exclude of this JIRAHistogramsRequest.

        PRs cannot relate to JIRA issue labels from the list.

        :param labels_exclude: The labels_exclude of this JIRAHistogramsRequest.
        """
        self._labels_exclude = labels_exclude

    @property
    def with_(self) -> Optional[List[JIRAFilterWith]]:
        """Gets the with_ of this JIRAHistogramsRequest.

        Groups of issue participants. The histograms will be calculated for each group.

        :return: The with_ of this JIRAHistogramsRequest.
        """
        return self._with_

    @with_.setter
    def with_(self, with_: Optional[List[JIRAFilterWith]]):
        """Sets the with_ of this JIRAHistogramsRequest.

        Groups of issue participants. The histograms will be calculated for each group.

        :param with_: The with_ of this JIRAHistogramsRequest.
        """
        self._with_ = with_

    @property
    def histograms(self) -> List[JIRAHistogramDefinition]:
        """Gets the histograms of this JIRAHistogramsRequest.

        Histogram parameters for each wanted topic.

        :return: The histograms of this JIRAHistogramsRequest.
        """
        return self._histograms

    @histograms.setter
    def histograms(self, histograms: List[JIRAHistogramDefinition]):
        """Sets the histograms of this JIRAHistogramsRequest.

        Histogram parameters for each wanted topic.

        :param histograms: The histograms of this JIRAHistogramsRequest.
        """
        if histograms is None:
            raise ValueError("Invalid value for `histograms`, must not be `None`")
        self._histograms = histograms

    @property
    def exclude_inactive(self) -> bool:
        """Gets the exclude_inactive of this JIRAHistogramsRequest.

        Value indicating whether issues with the last update older than `date_from` should be
        ignored.

        :return: The exclude_inactive of this JIRAHistogramsRequest.
        """
        return self._exclude_inactive

    @exclude_inactive.setter
    def exclude_inactive(self, exclude_inactive: bool):
        """Sets the exclude_inactive of this JIRAHistogramsRequest.

        Value indicating whether issues with the last update older than `date_from` should be
        ignored.

        :param exclude_inactive: The exclude_inactive of this JIRAHistogramsRequest.
        """
        if exclude_inactive is None:
            raise ValueError("Invalid value for `exclude_inactive`, must not be `None`")

        self._exclude_inactive = exclude_inactive

    @property
    def quantiles(self) -> Optional[List[float]]:
        """Gets the quantiles of this JIRAHistogramsRequest.

        Cut the distributions at certain quantiles. The default values are [0, 1] which means no
        cutting.

        :return: The quantiles of this JIRAHistogramsRequest.
        """
        return self._quantiles

    @quantiles.setter
    def quantiles(self, quantiles: Optional[List[float]]):
        """Sets the quantiles of this JIRAHistogramsRequest.

        Cut the distributions at certain quantiles. The default values are [0, 1] which means no
        cutting.

        :param quantiles: The quantiles of this JIRAHistogramsRequest.
        """
        self._quantiles = quantiles
