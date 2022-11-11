from athenian.api.models.web.team_tree import TeamTree


class TestTeamTree:
    def test_copy(self) -> None:
        tree = TeamTree(42, "foo", [TeamTree(1, "bar", [], [])], [10, 20])
        copy = tree.copy()

        assert copy is not tree
        assert copy == tree

    def test_with_children(self) -> None:
        tree = TeamTree(42, "foo", [], [10, 20])
        new_children = [
            TeamTree(43, "bar", [], [10]),
            TeamTree(44, "baz", [], [10]),
        ]
        new_tree = tree.with_children(new_children)

        assert new_tree is not tree
        assert new_tree.id == tree.id

        for i in range(len(new_children)):
            assert new_tree.children[i] == new_tree.children[i]

        assert tree.children == []

    def test_members_not_exported(self) -> None:
        tree = TeamTree(1, "t", [], [10])
        tree_dict = tree.to_dict()
        assert "members" not in tree_dict
