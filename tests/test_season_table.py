"""The league table and its ordering rule.

Arithmetic that everything downstream trusts, so it is checked against hand-computed tables rather
than against another implementation of itself. The ordering tests exist because the Premier
League's rule is *narrower* than the one most leagues use, and the difference is invisible until a
season lands on it: encoding a head-to-head tiebreak here would change simulated final tables
across the whole corpus and no aggregate score would obviously move.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.season import table as tbl
from plmodel.season.table import Question, SeasonError


_COLUMNS = ["date", "season", "home_team", "away_team", "home_goals", "away_goals"]


def _matches(rows: list[tuple[str, str, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i),
          "season": "2019-20", "home_team": h, "away_team": a,
          "home_goals": hg, "away_goals": ag}
         for i, (h, a, hg, ag) in enumerate(rows)],
        columns=_COLUMNS,
    )


def test_a_hand_computed_table() -> None:
    frame = _matches([("A", "B", 3, 0), ("B", "C", 1, 1), ("C", "A", 2, 2)])
    got = tbl.standings(frame).set_index("team")
    assert got.loc["A", "points"] == 4 and got.loc["A", "goal_difference"] == 3
    assert got.loc["B", "points"] == 1 and got.loc["B", "goal_difference"] == -3
    assert got.loc["C", "points"] == 2 and got.loc["C", "goal_difference"] == 0
    assert list(got.index) == ["A", "C", "B"]
    assert got["played"].tolist() == [2, 2, 2]


def test_a_club_that_has_not_played_is_on_nothing_rather_than_absent() -> None:
    """A preseason table is twenty clubs on zero points, not an empty frame."""
    got = tbl.standings(_matches([("A", "B", 1, 0)]), teams=("A", "B", "C"))
    assert set(got["team"]) == {"A", "B", "C"}
    assert int(got.set_index("team").loc["C", "points"]) == 0
    assert int(got.set_index("team").loc["C", "played"]) == 0


def test_an_empty_fixture_list_still_gives_a_full_table() -> None:
    got = tbl.standings(_matches([]), teams=("A", "B", "C"))
    assert len(got) == 3
    assert got["points"].sum() == 0 and got["played"].sum() == 0


def test_goals_scored_breaks_a_tie_that_goal_difference_cannot() -> None:
    # A wins 4-1 and B wins 3-0: three points and a goal difference of +3 each, and A scored more.
    frame = _matches([("A", "C", 4, 1), ("B", "D", 3, 0)])
    assert list(tbl.standings(frame)["team"])[:2] == ["A", "B"]


def test_head_to_head_is_not_a_premier_league_tiebreak() -> None:
    """B beat A, and still finishes below on the three keys the competition actually uses.

    La Liga and Serie A would put B first. Encoding their rule here would silently rewrite every
    simulated season, which is why this is asserted rather than assumed.
    """
    frame = _matches([
        ("B", "A", 1, 0),   # B wins the head-to-head
        ("A", "C", 4, 0), ("A", "D", 5, 1),      # A: 6 points, +7, 9 scored
        ("B", "C", 7, 0), ("D", "B", 1, 0),      # B: 6 points, +7, 8 scored
    ])
    got = tbl.standings(frame)
    a, b = got.set_index("team").loc["A"], got.set_index("team").loc["B"]
    assert a["points"] == b["points"] and a["goal_difference"] == b["goal_difference"]
    assert a["goals_for"] > b["goals_for"]
    assert list(got["team"])[:2] == ["A", "B"]


def test_a_deduction_is_subtracted_and_reported() -> None:
    frame = _matches([("A", "B", 1, 0), ("B", "A", 1, 0)])
    got = tbl.standings(frame, deductions={"A": 3}).set_index("team")
    assert got.loc["A", "points"] == 0 and got.loc["A", "deduction"] == 3
    assert got.loc["B", "points"] == 3 and got.loc["B", "deduction"] == 0
    assert list(tbl.standings(frame, deductions={"A": 3})["team"]) == ["B", "A"]


def test_an_unplayed_match_is_refused_rather_than_counted_as_a_draw() -> None:
    frame = _matches([("A", "B", 1, 0)])
    frame.loc[0, "home_goals"] = np.nan
    with pytest.raises(SeasonError, match="no score"):
        tbl.standings(frame)


# --- ordering a batch of simulated tables -----------------------------------------------------

def _batch(points, gd, gf):
    return (np.asarray(points), np.asarray(gd), np.asarray(gf))


def test_rank_orders_by_points_then_difference_then_goals() -> None:
    points, gd, gf = _batch([[10, 10, 12]], [[5, 9, 0]], [[20, 20, 20]])
    positions, _ = tbl.rank(points, gd, gf, rng=np.random.default_rng(0))
    assert positions.tolist() == [[2, 1, 0]]


def test_every_position_is_taken_exactly_once() -> None:
    """The structural invariant a probability table rests on: one champion per replicate."""
    rng = np.random.default_rng(1)
    points = rng.integers(0, 40, size=(200, 20))
    gd = rng.integers(-20, 20, size=(200, 20))
    gf = rng.integers(0, 60, size=(200, 20))
    positions, _ = tbl.rank(points, gd, gf, rng=rng)
    for row in positions:
        assert sorted(row.tolist()) == list(range(20))


def test_a_dead_level_pair_splits_the_position_about_evenly() -> None:
    """The competition sends them to a playoff on neutral ground; this is that coin."""
    n = 4000
    points, gd, gf = _batch(np.full((n, 2), 10), np.zeros((n, 2), int), np.full((n, 2), 20))
    positions, level = tbl.rank(points, gd, gf, rng=np.random.default_rng(3))
    share = float((positions[:, 0] == 0).mean())
    assert 0.45 < share < 0.55
    # The club that took second is level with the one above it, in every replicate.
    assert level[:, 1].all() and not level[:, 0].any()


def test_the_level_flag_marks_only_pairs_equal_on_all_three_keys() -> None:
    points, gd, gf = _batch([[10, 10, 10]], [[5, 5, 4]], [[20, 20, 20]])
    _, level = tbl.rank(points, gd, gf, rng=np.random.default_rng(0))
    # Two clubs are level on all three; the third is separated by goal difference.
    assert level.tolist() == [[False, True, False]]


def test_rank_rejects_mismatched_or_flat_arrays() -> None:
    with pytest.raises(SeasonError, match="matching 2-D"):
        tbl.rank(np.zeros(3), np.zeros(3), np.zeros(3), rng=np.random.default_rng(0))


# --- questions --------------------------------------------------------------------------------

def test_a_question_reads_positions_from_the_right_end() -> None:
    positions = np.array([[0, 1, 17, 18, 19]])
    assert Question("top_four", "top", 4).satisfied(positions, 20).tolist() == \
        [[True, True, False, False, False]]
    assert Question("relegation", "bottom", 3).satisfied(positions, 20).tolist() == \
        [[False, False, True, True, True]]


def test_the_cut_is_where_the_playoff_clause_can_fire() -> None:
    assert Question("title", "top", 1).cut(20) == 1
    assert Question("relegation", "bottom", 3).cut(20) == 17


def test_a_question_every_club_satisfies_is_refused() -> None:
    with pytest.raises(SeasonError, match="not a question"):
        Question("all", "top", 20).cut(20)


def test_question_validation() -> None:
    with pytest.raises(SeasonError, match="end must be"):
        Question("x", "middle", 4)
    with pytest.raises(SeasonError, match="at least 1"):
        Question("x", "top", 0)
