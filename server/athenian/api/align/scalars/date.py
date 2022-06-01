from ariadne import ScalarType

from athenian.api.serialization import deserialize_date, serialize_date

date_scalar = ScalarType(
    "Date",
    serializer=serialize_date,
    value_parser=deserialize_date,
)
