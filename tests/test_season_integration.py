"""The season simulator against the real corpus.

The load-bearing test here is the structural one: a forecast made at a barrier must not change when
matches after that barrier are added to or removed from the corpus. Every leak this project could
introduce at season level -- a fit that reaches past its barrier, a table built from the whole
season, a promoted-club roster read from the future -- shows up as a difference between those two
runs, and none of them would show up as a bad score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.eval.backtest import training_frame
from plmodel.model.dixon_coles import fit_dixon_coles
from plmodel.season.simulate import UNCERTAINTY_DRIFT, simulate_season
from plmodel.season.validate import (
    actual_table,
    matchweek_barriers,
    promoted_clubs,
    score_forecast,
)

pytestmark = pytest.mark.integration

SEASON = "2018-19"
REPLICATES = 4000


@pytest.fixture(scope="module")
def season_rows(corpus, cfg):
    rows = corpus[corpus["season"] == SEASON].sort_values("date", kind="stable")
    if rows.empty:
        pytest.skip(f"{SEASON} not in the ingested corpus")
    return rows.reset_index(drop=True)


def _fit(cfg, matches, barrier):
    return fit_dixon_coles(
        training_frame(matches, barrier),
        half_life_days=cfg.model.decay_half_life_days, ref_date=barrier,
        max_goals=cfg.model.max_goals, param_bounds=cfg.model.param_bounds,
        min_effective_share=cfg.model.min_effective_share, max_iter=cfg.model.max_iter,
    )


def _forecast(cfg, matches, rows, barrier, *, uncertainty=None):
    played = rows[(rows["date"] < barrier) & rows["played"]]
    remaining = rows[rows["date"] >= barrier]
    return simulate_season(
        _fit(cfg, matches, barrier), played, remaining,
        spec=cfg.season.spec(uncertainty=uncertainty, n_replicates=REPLICATES),
        seed=cfg.seed, season=SEASON, barrier=barrier,
    )


def test_a_forecast_does_not_move_when_the_future_is_removed(cfg, corpus, season_rows) -> None:
    """The barrier discipline, at season scale. Nothing after it may reach the forecast."""
    barrier = matchweek_barriers(
        season_rows, weeks=(9,), fixtures_per_week=cfg.season.fixtures_per_week
    )[0].date
    truncated = corpus[corpus["date"] < season_rows["date"].max()]
    whole = _forecast(cfg, corpus, season_rows, barrier)
    partial = _forecast(cfg, truncated, season_rows, barrier)
    assert np.array_equal(whole.position_counts, partial.position_counts)
    assert np.array_equal(whole.points_counts, partial.points_counts)


def test_a_promoted_roster_is_read_from_behind_the_barrier(cfg, corpus) -> None:
    """The three clubs that came up are known before a ball is kicked, and only those three."""
    got = promoted_clubs(corpus, SEASON)
    assert len(got) == 3
    previous = corpus[corpus["season"] < SEASON]
    last = previous[previous["season"] == previous["season"].max()]
    assert not (got & (set(last["home_team"]) | set(last["away_team"])))


def test_a_real_preseason_forecast_is_a_probability_distribution(cfg, corpus, season_rows) -> None:
    barrier = season_rows["date"].min()
    forecast = _forecast(cfg, corpus, season_rows, barrier)
    assert len(forecast.teams) == 20
    assert forecast.n_played == 0 and forecast.n_remaining == 380
    assert forecast.probabilities["title"].sum() == pytest.approx(1.0)
    assert forecast.probabilities["relegation"].sum() == pytest.approx(3.0)
    # A 38-match season cannot produce a mean outside these bounds at football rates.
    assert 25 < forecast.probabilities["mean_points"].min() < 45
    assert 60 < forecast.probabilities["mean_points"].max() < 100
    # The league's points must add up: 380 matches give between 760 and 1,140 points.
    total = forecast.probabilities["mean_points"].sum()
    assert 760 <= total <= 1140


def test_the_simulator_beats_a_no_information_forecast(cfg, corpus, season_rows) -> None:
    """A weak but non-vacuous floor: the model must beat "every club is equally likely"."""
    barrier = season_rows["date"].min()
    forecast = _forecast(cfg, corpus, season_rows, barrier)
    teams = tuple(sorted(set(season_rows["home_team"]) | set(season_rows["away_team"])))
    scored = score_forecast(
        forecast, actual_table(season_rows, teams=teams),
        questions=cfg.season.question_specs(), prob_floor=cfg.season.prob_floor, seed=cfg.seed,
        promoted=promoted_clubs(corpus, SEASON),
    )
    uninformed = {"title": 1 / 20, "top_four": 4 / 20, "relegation": 3 / 20}
    floor = float(np.mean([
        (uninformed[row.question] - row.outcome) ** 2 for row in scored.itertuples(index=False)
    ]))
    assert scored["brier"].mean() < floor


def test_drift_widens_a_real_season_without_moving_its_centre(cfg, corpus, season_rows) -> None:
    barrier = season_rows["date"].min()
    point = _forecast(cfg, corpus, season_rows, barrier)
    drift = _forecast(cfg, corpus, season_rows, barrier, uncertainty=UNCERTAINTY_DRIFT)
    band = lambda f: f.points_quantile(0.95) - f.points_quantile(0.05)
    assert band(drift).mean() > band(point).mean()
    # Centres move by less than the widening does: drift is uncertainty, not a different model.
    centre_shift = np.abs(
        drift.probabilities.set_index("team")["mean_points"]
        - point.probabilities.set_index("team")["mean_points"]
    ).max()
    assert centre_shift < (band(drift).mean() - band(point).mean())


def test_the_scored_frame_covers_every_club_and_question(cfg, corpus, season_rows) -> None:
    barrier = season_rows["date"].min()
    forecast = _forecast(cfg, corpus, season_rows, barrier)
    teams = tuple(sorted(set(season_rows["home_team"]) | set(season_rows["away_team"])))
    final = actual_table(season_rows, teams=teams)
    scored = score_forecast(
        forecast, final, questions=cfg.season.question_specs(),
        prob_floor=cfg.season.prob_floor, seed=cfg.seed,
    )
    assert len(scored) == 20 * len(cfg.season.questions)
    assert scored["outcome"].sum() == 1 + 4 + 3          # one champion, four top-four, three down
    assert set(scored["final_position"]) == set(range(1, 21))
    assert not final["level_with_the_club_above"].any()  # no playoff was needed in this season
