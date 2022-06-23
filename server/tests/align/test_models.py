from athenian.api.align.models import TeamTree


class TestTeamTree:
    def test_copy(self) -> None:
        tree = TeamTree(42, "foo", [TeamTree(1, "bar", [], [])], [10, 20])
        copy = tree.copy()

        assert copy is not tree
        assert copy == tree
