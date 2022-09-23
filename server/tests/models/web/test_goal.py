from athenian.api.models.web.goal import GoalValue


class TestGoalValue:
    def test_target_optional(self) -> None:
        gv = GoalValue(current=1, initial=1, target=None)
        assert gv.target is None

        assert gv.to_dict() == {"initial": 1, "current": 1}

    def test_current_optional(self) -> None:
        gv = GoalValue(current=None, initial=1, target=None)
        assert gv.current is None

        assert gv.to_dict() == {"initial": 1, "current": None}
