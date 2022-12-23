from athenian.api.models.web import DashboardUpdateRequest
from athenian.api.models.web.dashboard import _DashboardUpdateChart


class TestDashboardUpdateRequest:
    def test_from_dict(self) -> None:
        dct: dict = {"charts": [{"id": 1}]}
        model = DashboardUpdateRequest.from_dict(dct)
        assert model.charts == [_DashboardUpdateChart(id=1)]
