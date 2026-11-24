"""Forecasting a fixture nobody has played yet.

Three things are being pinned. That the derived markets are readings of the *same* scoreline
distribution the model already produces, so they cannot disagree with the home/draw/away numbers.
That a name typed by a person reaches the right club or fails loudly, never quietly reaching the
wrong one. And that a club the fit barely knows is flagged — the dangerous case is not the club
with no history, which is visibly pinned at the league average, but the club with a nine-year-old
rating that looks like a measurement.
"""
from __future__ import annotations

import argparse

import numpy as np
import pytest

from plmodel.cli import _parse_fixtures
from plmodel.data.teams import AmbiguousTeamError, resolve_team
from plmodel.model.scoreline import (
    both_teams_to_score,
    collapse_three_class,
    handicap_probability,
    margin_distribution,
    scoreline_matrix,
    scoring_markets,
    top_scorelines,
    total_goals_distribution,
    totals_probability,
)

MAX_GOALS = 12
TEAMS = ["Arsenal", "Man City", "Man United", "Nott'm Forest", "Sheffield United",
         "Sheffield Weds", "Tottenham", "Coventry", "Hull"]


def _grid(lam=(1.7, 0.9), mu=(1.1, 2.0), rho=-0.05):
    return scoreline_matrix(np.array(lam), np.array(mu), np.full(len(lam), rho), MAX_GOALS)


# --- markets ----------------------------------------------------------------------------------

def test_the_markets_are_one_distribution_read_different_ways() -> None:
    """Over plus under is one match, and so is home plus draw plus away."""
    grid = _grid()
    over, under = totals_probability(grid, 2.5)
    assert np.allclose(over + under, 1.0)
    assert np.allclose(collapse_three_class(grid).sum(axis=1), 1.0)


def test_a_half_line_cannot_push_but_a_whole_line_can() -> None:
    """2.5 splits every scoreline; 2.0 leaves the 2-goal games in neither bucket."""
    grid = _grid()
    over, under = totals_probability(grid, 2.0)
    assert np.all(over + under < 0.999)          # exactly-two-goal games are excluded from both
    assert np.allclose(sum(totals_probability(grid, 2.5)), 1.0)


def test_totals_agree_with_a_direct_sum_over_the_grid() -> None:
    grid = _grid()
    x = np.arange(grid.shape[1])[None, :, None]
    y = np.arange(grid.shape[2])[None, None, :]
    direct = grid.sum(axis=(1, 2), where=((x + y) > 2.5))
    assert np.allclose(totals_probability(grid, 2.5)[0], direct)


def test_both_teams_to_score_excludes_every_clean_sheet() -> None:
    grid = _grid()
    clean = grid[:, 0, :].sum(axis=1) + grid[:, :, 0].sum(axis=1) - grid[:, 0, 0]
    assert np.allclose(both_teams_to_score(grid) + clean, 1.0)


def test_top_scorelines_are_sorted_and_are_real_cells() -> None:
    grid = _grid()
    got = top_scorelines(grid, 5)
    assert len(got) == grid.shape[0]
    for match, rows in enumerate(got):
        probs = [p for _, _, p in rows]
        assert probs == sorted(probs, reverse=True)
        for x, y, p in rows:
            assert p == pytest.approx(grid[match, x, y])
        assert max(probs) == pytest.approx(grid[match].max())


def test_a_stronger_home_side_shifts_every_market_the_same_way() -> None:
    weak, strong = _grid(lam=(1.0,), mu=(1.5,)), _grid(lam=(2.5,), mu=(0.8,))
    assert collapse_three_class(strong)[0, 0] > collapse_three_class(weak)[0, 0]
    assert top_scorelines(strong, 1)[0][0][0] > top_scorelines(weak, 1)[0][0][0]


def test_the_derived_markets_agree_with_the_core_ones() -> None:
    """Every derived market is the same grid read another way, so each must agree exactly with a
    market that already existed: margins with home/draw/away, totals with over/under, handicaps
    with margins, clean sheets with scoring, win to nil with both."""
    grid = _grid()
    outcome = collapse_three_class(grid)
    margins = margin_distribution(grid, 3)                 # away 3+ ... draw ... home 3+
    assert np.allclose(margins.sum(axis=1), 1.0)
    assert np.allclose(margins[:, 4:].sum(axis=1), outcome[:, 0])
    assert np.allclose(margins[:, 3], outcome[:, 1])
    assert np.allclose(margins[:, :3].sum(axis=1), outcome[:, 2])

    totals = total_goals_distribution(grid, 6)
    assert np.allclose(totals.sum(axis=1), 1.0)
    assert np.allclose(totals[:, 3:].sum(axis=1), totals_probability(grid, 2.5)[0])

    covers, push, against = handicap_probability(grid, -1.5)
    assert np.allclose(covers, margins[:, 5:].sum(axis=1)) and np.allclose(push, 0.0)
    covers, push, against = handicap_probability(grid, -1.0)
    assert np.allclose(push, margins[:, 4]) and np.allclose(covers + push + against, 1.0)

    scoring = scoring_markets(grid)
    assert np.allclose(scoring["home_clean_sheet"], 1.0 - scoring["away_scores"])
    btts = both_teams_to_score(grid)
    # Home scores = home scores while away also scores, plus home wins to nil.
    assert np.allclose(scoring["home_scores"], btts + scoring["home_win_to_nil"])


def test_market_helpers_reject_a_frame_that_is_not_a_grid() -> None:
    with pytest.raises(ValueError, match="expected an"):
        both_teams_to_score(np.ones((3, 3)))


# --- resolving what a person typed --------------------------------------------------------------

@pytest.mark.parametrize("typed,expected", [
    ("Arsenal", "Arsenal"),
    ("arsenal", "Arsenal"),
    ("Man Utd", "Man United"),
    ("Nottm Forest", "Nott'm Forest"),
    ("nott'm forest", "Nott'm Forest"),
    ("Coventry", "Coventry"),
])
def test_a_typed_name_reaches_the_right_club(typed: str, expected: str) -> None:
    assert resolve_team(typed, TEAMS) == expected


def test_an_alias_is_honoured_before_any_guessing() -> None:
    assert resolve_team("Manchester United", TEAMS,
                        aliases={"Manchester United": "Man United"}) == "Man United"


def test_an_ambiguous_name_raises_with_the_candidates() -> None:
    """Picking the alphabetically-first of two plausible clubs is how a forecast goes silently
    wrong about which team it is describing."""
    with pytest.raises(AmbiguousTeamError, match="Sheffield"):
        resolve_team("Sheffield", TEAMS)


def test_an_unknown_name_raises_rather_than_matching_anything() -> None:
    with pytest.raises(AmbiguousTeamError, match="no club matches"):
        resolve_team("Real Madrid", TEAMS)


def test_resolution_never_reaches_the_ingest_path() -> None:
    """The corpus guard must stay strict; only the command line is forgiving."""
    from plmodel.data import football_data, teams

    source = football_data.__file__
    assert "resolve_team" not in open(source, encoding="utf-8").read()
    assert hasattr(teams, "resolve_team")


# --- parsing fixtures off the command line ------------------------------------------------------

def _args(**kwargs):
    base = {"fixtures": None, "file": None, "home": None, "away": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("text", [
    "Arsenal v Coventry", "Arsenal vs Coventry", "Arsenal - Coventry", "Arsenal V Coventry",
])
def test_every_spelling_of_against_is_accepted(text: str) -> None:
    assert _parse_fixtures(_args(fixtures=text)) == [("Arsenal", "Coventry")]


@pytest.mark.parametrize("text, expected", [
    ("Everton v Hull", ("Everton", "Hull")),
    ("Aston Villa v Hull", ("Aston Villa", "Hull")),
    ("Coventry v Hull", ("Coventry", "Hull")),
    ("Wolves v Hull", ("Wolves", "Hull")),
    ("Aston Villa V Hull", ("Aston Villa", "Hull")),
    ("Everton vs Hull", ("Everton", "Hull")),
    ("Everton - Hull", ("Everton", "Hull")),
])
def test_a_home_side_containing_the_separator_is_not_split_inside(text, expected) -> None:
    """`pl predict --fixtures "Everton v Hull"` used to fail with "no club matches 'E'".

    The guard tests for the SPACED token and the partition then stripped it to a bare "v", so any
    home side carrying one was cut in half. Three of the twenty clubs in the 2026-27 division are
    affected, and every existing spelling test used "Arsenal" as the home side — a name with no
    "v" in it — which is exactly why this survived to a shipped command the README documents.
    """
    assert _parse_fixtures(_args(fixtures=text)) == [expected]


def test_several_fixtures_come_back_in_order() -> None:
    got = _parse_fixtures(_args(fixtures="Arsenal v Coventry, Man Utd v Hull"))
    assert got == [("Arsenal", "Coventry"), ("Man Utd", "Hull")]


def test_a_single_fixture_can_be_given_as_two_flags() -> None:
    assert _parse_fixtures(_args(home="Arsenal", away="Coventry")) == [("Arsenal", "Coventry")]


def test_half_a_fixture_is_refused() -> None:
    with pytest.raises(ValueError, match="must be given together"):
        _parse_fixtures(_args(home="Arsenal"))
    with pytest.raises(ValueError, match="not 'Home v Away'"):
        _parse_fixtures(_args(fixtures="Arsenal"))


def test_a_csv_of_fixtures_is_read(tmp_path) -> None:
    path = tmp_path / "fixtures.csv"
    path.write_text("home_team,away_team\nArsenal,Coventry\nHull,Leeds\n", encoding="utf-8")
    assert _parse_fixtures(_args(file=str(path))) == [("Arsenal", "Coventry"), ("Hull", "Leeds")]


def test_a_csv_missing_its_columns_says_which(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("home,away\nArsenal,Coventry\n", encoding="utf-8")
    with pytest.raises(ValueError, match="away_team"):
        _parse_fixtures(_args(file=str(path)))


def test_the_market_informed_forecast_matches_what_the_live_freeze_records(tmp_path) -> None:
    """`pl predict --arm odds-ratings-ou` and the frozen `odds-ratings-ou` forecast are one model:
    the same ratings at the same date must give the same home/draw/away."""
    import pandas as pd

    from plmodel.cli import _odds_ratings_fit
    from plmodel.config import load_config
    from plmodel.eval.live import freeze_matchday

    cfg = load_config()
    rng = np.random.default_rng(11)
    clubs = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    rows = []
    for d in range(20):
        day = pd.Timestamp("2026-08-01") + pd.Timedelta(days=7 * d)
        for k in range(2):
            h, a = clubs[(d + 2 * k) % 4], clubs[(d + 2 * k + 1) % 4]
            rows.append({"date": day, "season": "2026-27", "division": "E0", "home_team": h,
                         "away_team": a, "result": "H", "home_goals": 1.0, "away_goals": 0.0,
                         "B365H": rng.uniform(1.6, 3.5), "B365D": 3.4, "B365A": rng.uniform(2.0, 5.0),
                         "B365>2.5": rng.uniform(1.6, 2.2), "B365<2.5": rng.uniform(1.6, 2.2)})
    history = pd.DataFrame(rows)
    barrier = history["date"].max() + pd.Timedelta(days=7)
    fixture = pd.DataFrame({"date": [barrier], "season": "2026-27", "division": "E0",
                            "home_team": ["Arsenal"], "away_team": ["Chelsea"], "result": [None]})
    _, block = freeze_matchday(fixture, history, cfg, tmp_path, arm_names=["odds-ratings-ou"])
    frozen = np.array(block["fixtures"][0]["forecasts"]["odds-ratings-ou"])

    ratings = _odds_ratings_fit(cfg, history, barrier, use_over_under=True)
    lam, mu = ratings.rates(fixture["home_team"], fixture["away_team"])
    grid = scoreline_matrix(lam, mu, np.zeros(1), cfg.model.max_goals)
    np.testing.assert_allclose(collapse_three_class(grid)[0], frozen, atol=1e-12)
