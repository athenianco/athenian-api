from datetime import date, datetime, timedelta, timezone
from typing import Optional, Tuple

from athenian.api.models.web.base_model_ import Model
from athenian.api.models.web.invalid_request_error import InvalidRequestError
from athenian.api.response import ResponseError


class TimeFilterProperties(Model, sealed=False):
    """Define `date_from`, `date_to`, and `timezone` properties."""

    date_from: date
    date_to: date
    timezone: Optional[int]

    def validate_timezone(self, timezone: Optional[int]) -> Optional[int]:
        """Sets the timezone of this Model.

        Local time zone offset in minutes, used to adjust `date_from` and `date_to`.

        :param timezone: The timezone of this Model.
        """
        if timezone is not None and timezone > 780:
            raise ValueError(
                "Invalid value for `timezone`, must be a value less than or equal to `720`",
            )
        if timezone is not None and timezone < -720:
            raise ValueError(
                "Invalid value for `timezone`, must be a value greater than or equal to `-720`",
            )

        return timezone


class CommonFilterProperties(TimeFilterProperties, sealed=False):
    """Define `account`, `date_from`, `date_to`, and `timezone` properties."""

    account: int

    def resolve_time_from_and_to(self) -> Tuple[datetime, datetime]:
        """Extract the time window from the request model: the timestamps of `from` and `to`."""
        if self.date_from > self.date_to:
            raise ResponseError(
                InvalidRequestError(
                    "`date_to` may not be less than `date_from`",
                    detail="from:%s > to:%s" % (self.date_from, self.date_to),
                ),
            )
        time_from = datetime.combine(self.date_from, datetime.min.time(), tzinfo=timezone.utc)
        time_to = datetime.combine(self.date_to, datetime.max.time(), tzinfo=timezone.utc)
        if self.timezone is not None:
            tzoffset = timedelta(minutes=-self.timezone)
            time_from += tzoffset
            time_to += tzoffset
        return time_from, time_to
