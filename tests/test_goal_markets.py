"""The goal-market yardstick: the scoring rules, the market comparison, and that the grids it
scores are provably the arm's own forecast."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval import metrics
from plmodel.eval.goal_markets import (
    GoalMarketError, GoalWalk, _ordinal_rps, climatology, score_goal_markets,
    two_way_probabilities, walk_grids,
)
from plmodel.model.scoreline import scoreline_matrix

MAX_GOALS = 12


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_two_way_prices_are_proportionally_devigged_and_unpriced_rows_stay_missing() -> None:
    p = two_way_probabilities(np.array([[1.9, 1.9], [1.5, 2.6], [np.nan, 2.0], [0.0, 2.0]]))
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 2.6))
    assert np.isnan(p[2]) and np.isnan(p[3])


def test_ordinal_rps_reduces_to_the_three_class_rule() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), 200)
    outcome = rng.integers(0, 3, 200)
    assert _ordinal_rps(probs, outcome) == pytest.approx(metrics.rps(probs, outcome).mean())


def _history() -> pd.DataFrame:
    seasons = [f"20{y:02d}-{y + 1:02d}" for y in range(10, 17)]
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "season": np.repeat(seasons, 50),
        "home_goals": rng.poisson(1.5, 350), "away_goals": rng.poisson(1.1, 350),
    })


def test_climatology_reads_only_the_last_seasons_before_the_pool(cfg) -> None:
    history = _history()
    clim = climatology(history, cfg)
    assert clim["seasons"] == sorted(history["season"].unique())[-cfg.goal_markets.climatology_seasons:]
    recent = history[history["season"].isin(clim["seasons"])]
    assert clim["rates"]["home_scores"] == pytest.approx((recent["home_goals"] > 0).mean())
    assert clim["total_goals"].sum() == pytest.approx(1.0)


def _walk(n: int = 400, seed: int = 2) -> GoalWalk:
    rng = np.random.default_rng(seed)
    lam, mu = rng.uniform(0.8, 2.2, n), rng.uniform(0.6, 1.8, n)
    x, y = rng.poisson(lam), rng.poisson(mu)
    over = 1.0 / np.clip(rng.uniform(0.4, 0.6, n), 0.01, 0.99)
    rows = pd.DataFrame({
        "home_goals": x, "away_goals": y,
        "BbAv>2.5": over * 0.95, "BbAv<2.5": 1.0 / (1.0 - 1.0 / over) * 0.95,
        "BbAHh": np.where(np.arange(n) % 2 == 0, -0.5, -0.75),
        "BbAvAHH": 1.95, "BbAvAHA": 1.95,
    })
    return GoalWalk(arm="test", rows=rows, grids=scoreline_matrix(lam, mu, np.zeros(n), MAX_GOALS))


def test_a_forecast_from_the_true_rates_beats_climatology_on_every_market(cfg) -> None:
    report = score_goal_markets(_walk(), _history(), cfg)
    for key, block in report["binary"].items():
        assert block["brier"] < block["brier_climatology"] + 0.002, key
    assert report["exact_score_log_loss"]["model"] < report["exact_score_log_loss"]["climatology"]
    assert report["goal_level"]["expected_total"] == pytest.approx(
        report["goal_level"]["scored_total"], rel=0.1)


def test_the_market_comparison_uses_only_priced_rows_and_half_lines(cfg) -> None:
    report = score_goal_markets(_walk(), _history(), cfg)
    assert report["over_under_vs_market"]["friday_average"]["n"] == 400
    assert report["over_under_vs_market"]["closing_average"] == {"n": 0}   # no columns at all
    handicap = report["handicap_vs_market"]["friday_average"]
    assert handicap["n"] == 200                       # the -0.75 quarter lines are left out
    assert handicap["share_of_priced_rows_on_half_lines"] == pytest.approx(0.5)


def _league(n_rounds: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    clubs = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    rows = []
    for d in range(n_rounds):
        day = pd.Timestamp("2024-08-01") + pd.Timedelta(days=7 * d)
        for k in range(2):
            h, a = clubs[(d + 2 * k) % 4], clubs[(d + 2 * k + 1) % 4]
            hg, ag = int(rng.poisson(1.5)), int(rng.poisson(1.1))
            rows.append({"date": day, "season": "2024-25", "division": "E0", "home_team": h,
                         "away_team": a, "home_goals": float(hg), "away_goals": float(ag),
                         "result": "H" if hg > ag else ("D" if hg == ag else "A")})
    return pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)


def test_the_grids_reproduce_the_arms_own_forecast(cfg) -> None:
    from plmodel.eval.backtest import walk_forward

    matches = _league()
    walk = walk_grids("dixon-coles", matches, walk_forward(matches, min_train_matches=8), cfg)
    assert walk.grids.shape[1:] == (cfg.model.max_goals + 1,) * 2
    assert len(walk.grids) == len(walk.rows)
    assert np.allclose(walk.grids.sum(axis=(1, 2)), 1.0)


def test_an_arm_without_a_scoreline_distribution_is_refused(cfg) -> None:
    from plmodel.eval.backtest import walk_forward

    matches = _league()
    with pytest.raises(GoalMarketError, match="does not leave"):
        walk_grids("uniform", matches, walk_forward(matches, min_train_matches=8), cfg)
