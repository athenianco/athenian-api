from typing import Any, Optional

import pytest

from athenian.api.internal.miners.github.precomputed_prs import triage_by_release_match
from athenian.api.internal.miners.github.precomputed_prs.utils_accelerated import interleave_expand
from athenian.api.internal.settings import (
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    default_branch_alias,
)


def triage_by_release_match_ground_truth(
    repo: str,
    release_match: str,
    release_settings: ReleaseSettings,
    default_branches: dict[str, str],
    result: Any,
    ambiguous: dict[str, Any],
) -> Optional[Any]:
    """Check the release match of the specified `repo` and return `None` if it is not effective \
    according to `release_settings`, or decide between `result` and `ambiguous`."""
    # faster than `release_match in (rejected_name, force_push_drop_name)`
    if (
        release_match == ReleaseMatch.rejected.name
        or release_match == ReleaseMatch.force_push_drop.name
    ):
        return result
    try:
        required_release_match = release_settings.native[repo]
    except KeyError:
        # DEV-1451: if we don't have this repository in the release settings, then it is deleted
        raise AssertionError(
            f"You must take care of deleted repositories separately: {repo}",
        ) from None
    match_name, match_by = release_match.split("|", 1)
    match = ReleaseMatch[match_name]
    if required_release_match.match != ReleaseMatch.tag_or_branch:
        if match != required_release_match.match:
            return None
        dump = result
    else:
        try:
            dump = ambiguous[match_name]
        except KeyError:
            # event
            return None
    if match == ReleaseMatch.tag:
        target = required_release_match.tags
    elif match == ReleaseMatch.branch:
        target = required_release_match.branches.replace(
            default_branch_alias, default_branches.get(repo, default_branch_alias),
        )
    elif match == ReleaseMatch.event:
        target = required_release_match.events
    else:
        raise AssertionError("Precomputed DB may not contain Match.tag_or_branch")
    if target != match_by:
        return None
    return dump


@pytest.mark.parametrize(
    "release_match, release_settings",
    [
        ("rejected", "release_match_setting_tag_or_branch"),
        ("force_push_drop", "release_match_setting_tag_or_branch"),
        ("branch|main", "release_match_setting_branch"),
        ("branch|master", "release_match_setting_branch"),
        ("branch|тест", "release_match_setting_branch"),
        ("branch|мейн", "release_match_setting_branch"),
        ("branch|👍", "release_match_setting_branch"),
        ("branch|👎", "release_match_setting_branch"),
        ("tag|main", "release_match_setting_branch"),
        ("tag|мейн", "release_match_setting_branch"),
        ("tag|👍", "release_match_setting_branch"),
        ("event|main", "release_match_setting_branch"),
        ("tag|.*", "release_match_setting_tag"),
        ("tag|v1.0", "release_match_setting_tag"),
        ("tag|тест", "release_match_setting_tag"),
        ("tag|.*(?!тест)", "release_match_setting_tag"),
        ("tag|👍", "release_match_setting_tag"),
        ("tag|.*(?!👍)", "release_match_setting_tag"),
        ("branch|.*", "release_match_setting_tag"),
        ("branch|мейн", "release_match_setting_tag"),
        ("event|", "release_match_setting_tag"),
        ("tag|.*", "release_match_setting_tag_or_branch"),
        ("branch|main", "release_match_setting_tag_or_branch"),
        ("branch|master", "release_match_setting_tag_or_branch"),
        ("branch|main", "release_match_setting_event"),
        ("tag|.*", "release_match_setting_event"),
        ("event|.*", "release_match_setting_event"),
        ("event|.*(?!тест)", "release_match_setting_event"),
        (
            "branch|main",
            ReleaseSettings(
                {
                    "github.com/src-d/go-git": ReleaseMatchSetting(
                        branches="xx" + default_branch_alias + "yy",
                        tags=".*",
                        events=".*",
                        match=ReleaseMatch.branch,
                    ),
                },
            ),
        ),
        (
            "branch|xxmainyy",
            ReleaseSettings(
                {
                    "github.com/src-d/go-git": ReleaseMatchSetting(
                        branches="xx" + default_branch_alias + "yy",
                        tags=".*",
                        events=".*",
                        match=ReleaseMatch.branch,
                    ),
                },
            ),
        ),
    ],
)
@pytest.mark.parametrize("default_branch", ["main", "мейн", "👍", "мейнаб", "👍й"])
@pytest.mark.parametrize("unicode_mutate_release_settings", [1, 2, 4])
def test_triage_by_release_match(
    release_match,
    release_settings,
    release_match_setting_tag,
    release_match_setting_branch,
    release_match_setting_event,
    release_match_setting_tag_or_branch,
    default_branch,
    unicode_mutate_release_settings,
):
    if isinstance(release_settings, str):
        release_settings = locals()[release_settings]
    if unicode_mutate_release_settings > 1:
        release_settings = release_settings.copy()
        rs = release_settings.native["src-d/go-git"]
        suffix = "ф" if unicode_mutate_release_settings == 2 else "👍"
        release_settings.native["src-d/go-git"] = ReleaseMatchSetting(
            rs.branches + suffix,
            rs.tags + suffix,
            rs.events + suffix,
            rs.match,
        )
    args = (
        "src-d/go-git",
        release_match,
        release_settings,
        {"src-d/go-git": default_branch},
        "foo",
        {ReleaseMatch.tag.name: "bar", ReleaseMatch.branch.name: "qux"},
    )
    assert triage_by_release_match(*args) == triage_by_release_match_ground_truth(*args)


@pytest.mark.parametrize(
    "bytes_in",
    [b"".join(str(i % 10).encode() for i in range(n)) for n in range(100)],
)
@pytest.mark.parametrize("kind", [2, 4])
def test_interleave_expand1(bytes_in, kind):
    bytes_out = b"".join(
        (bytes_in[b : b + 1] + b"\x00" * (kind - 1)) for b in range(len(bytes_in))
    )
    assert interleave_expand(bytes_in, 1, kind) == bytes_out


@pytest.mark.parametrize(
    "bytes_in",
    [b"".join(str(i % 10).encode() for i in range(n)) for n in range(0, 100, 2)],
)
def test_interleave_expand24(bytes_in):
    bytes_out = b"".join((bytes_in[b : b + 2] + b"\x00" * 2) for b in range(0, len(bytes_in), 2))
    assert interleave_expand(bytes_in, 2, 4) == bytes_out
