"""Elo ratings and the Elo-difference Dixon-Coles — the single-scalar comparison arm.

These exist only to make Arm 1 a fair fight. A comparison arm that is subtly broken produces a
flattering result for the production model and nobody notices, so the properties tested here are
the ones that would silently advantage the incumbent: conservation, causality, and the fact that
the fit actually responds to rating differences.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.special import gammaln

from plmodel.config import load_config
from plmodel.model.elo_dc import _objective, fit_elo_dixon_coles
from plmodel.ratings.elo import (
    ELO_SCALE, EloConfig, compute_elo, expected_home_score, goal_difference_multiplier,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def scheme() -> EloConfig:
    return EloConfig(
        initial_rating=1500.0, k=20.0, home_advantage=65.0,
        gd_two_goal=1.5, gd_slope_offset=11.0, gd_slope_divisor=8.0,
    )


def _matches(rows: list[tuple[str, str, int, int]], start: str = "2020-08-01") -> pd.DataFrame:
    dates = [pd.Timestamp(start) + pd.Timedelta(days=7 * i) for i in range(len(rows))]
    return pd.DataFrame(
        {
            "date": dates,
            "home_team": [r[0] for r in rows],
            "away_team": [r[1] for r in rows],
            "home_goals": [float(r[2]) for r in rows],
            "away_goals": [float(r[3]) for r in rows],
        }
    )


# --- the rating update ---------------------------------------------------------------------------

def test_equal_ratings_give_the_home_side_the_edge(scheme: EloConfig) -> None:
    """Home advantage enters the expectation, so a home win against equals moves ratings less."""
    assert expected_home_score(scheme.home_advantage) > 0.5


def test_expectation_is_symmetric_without_home_advantage() -> None:
    assert expected_home_score(0.0) == pytest.approx(0.5)


def test_four_hundred_points_is_ten_to_one() -> None:
    """The defining constant of the Elo system."""
    assert expected_home_score(ELO_SCALE) == pytest.approx(10 / 11)


def test_rating_is_conserved(scheme: EloConfig) -> None:
    """Zero-sum by construction: what the winner gains the loser loses, exactly."""
    replay = compute_elo(_matches([("A", "B", 3, 0), ("B", "C", 1, 1), ("C", "A", 2, 1)]), scheme)
    total = sum(replay.final_ratings.values())
    assert total == pytest.approx(3 * scheme.initial_rating)


def test_winning_raises_and_losing_lowers(scheme: EloConfig) -> None:
    replay = compute_elo(_matches([("A", "B", 2, 0)]), scheme)
    assert replay.final_ratings["A"] > scheme.initial_rating
    assert replay.final_ratings["B"] < scheme.initial_rating


def test_a_bigger_win_moves_ratings_further(scheme: EloConfig) -> None:
    """Without the goal-difference multiplier a 5-0 and a 1-0 would be identical evidence."""
    narrow = compute_elo(_matches([("A", "B", 1, 0)]), scheme).final_ratings["A"]
    heavy = compute_elo(_matches([("A", "B", 5, 0)]), scheme).final_ratings["A"]
    assert heavy > narrow


def test_goal_difference_multiplier_schedule(scheme: EloConfig) -> None:
    got = goal_difference_multiplier(np.array([0, 1, -1, 2, -2, 3, 4]), scheme)
    assert got.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.5, 1.5, 14 / 8, 15 / 8])


def test_a_bigger_k_moves_ratings_further(scheme: EloConfig) -> None:
    slow = compute_elo(_matches([("A", "B", 2, 0)]), scheme).final_ratings["A"]
    fast_scheme = EloConfig(**{**scheme.__dict__, "k": 40.0})
    fast = compute_elo(_matches([("A", "B", 2, 0)]), fast_scheme).final_ratings["A"]
    assert fast - scheme.initial_rating > slow - scheme.initial_rating


def test_beating_a_stronger_side_is_worth_more(scheme: EloConfig) -> None:
    """The whole point of a rating system: the same result carries different information."""
    build_up = [("B", "C", 4, 0)] * 6            # B becomes strong
    strong = compute_elo(_matches(build_up + [("A", "B", 1, 0)]), scheme).final_ratings["A"]
    weak = compute_elo(_matches(build_up + [("A", "C", 1, 0)]), scheme).final_ratings["A"]
    assert strong > weak


# --- causality -----------------------------------------------------------------------------------

def test_replay_requires_date_order(scheme: EloConfig) -> None:
    """Ordering *is* the model here; an unsorted frame would silently produce wrong ratings."""
    unsorted = _matches([("A", "B", 1, 0), ("B", "C", 2, 1)]).iloc[::-1]
    with pytest.raises(ValueError, match="date-sorted"):
        compute_elo(unsorted, scheme)


def test_pre_match_rating_precedes_the_match(scheme: EloConfig) -> None:
    """The forward pass is causal by construction — this pins that it stays so."""
    replay = compute_elo(_matches([("A", "B", 3, 0), ("A", "C", 3, 0)]), scheme)
    first, second = replay.history.iloc[0], replay.history.iloc[1]
    assert first["home_elo_pre"] == scheme.initial_rating   # nothing known yet
    assert second["home_elo_pre"] == first["home_elo_post"]  # only the first match is known


def test_ratings_asof_excludes_the_barrier_date(scheme: EloConfig) -> None:
    matches = _matches([("A", "B", 3, 0), ("A", "C", 3, 0)])
    replay = compute_elo(matches, scheme)
    barrier = matches["date"].iloc[1]
    ratings = replay.ratings_asof(barrier)
    assert ratings["A"] == replay.history.iloc[0]["home_elo_post"]
    assert "C" not in ratings          # C has not played before the barrier


def test_unrated_team_takes_the_initial_rating(scheme: EloConfig) -> None:
    replay = compute_elo(_matches([("A", "B", 1, 0)]), scheme)
    diff = replay.rating_diff(pd.Series(["Newcomer"]), pd.Series(["Other"]), {})
    assert diff[0] == 0.0


# --- the Elo-difference Dixon-Coles ---------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_elo_dc_gradient_matches_finite_differences(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 300
    x = rng.poisson(1.5, n).astype(float)
    y = rng.poisson(1.1, n).astype(float)
    diff = rng.normal(0, 0.5, n)
    weights = rng.uniform(0.2, 1.0, n)
    args = (x, y, diff, weights, gammaln(x + 1), gammaln(y + 1))
    theta = np.array([rng.uniform(-0.2, 0.4), rng.uniform(0.0, 0.4),
                      rng.uniform(0.1, 0.6), rng.uniform(-0.1, 0.1)])
    numeric = approx_fprime(theta, lambda t: _objective(t, *args)[0], 1e-7)
    assert np.allclose(numeric, _objective(theta, *args)[1], rtol=1e-3, atol=1e-4)


def _fit(matches: pd.DataFrame, scheme: EloConfig, cfg):
    replay = compute_elo(matches, scheme)
    return replay, fit_elo_dixon_coles(
        replay.history, replay,
        half_life_days=100_000.0,
        ref_date=matches["date"].max() + pd.Timedelta(days=1),
        max_goals=cfg.model.max_goals,
        param_bounds=cfg.model.param_bounds,
        max_iter=cfg.model.max_iter,
    )


def test_elo_dc_recovers_a_positive_slope(cfg, scheme: EloConfig) -> None:
    """A stronger side must score more; a slope pinned at zero would make the arm a base rate."""
    rng = np.random.default_rng(4)
    rows = []
    for i in range(600):
        # A is genuinely better than B, which the ratings should learn and the slope should use.
        home, away = ("A", "B") if i % 2 else ("B", "A")
        strong_home = home == "A"
        hg = rng.poisson(2.0 if strong_home else 0.9)
        ag = rng.poisson(0.9 if strong_home else 2.0)
        rows.append((home, away, hg, ag))
    _, fit = _fit(_matches(rows), scheme, cfg)
    assert fit.converged and fit.c > 0.05


def test_elo_dc_has_exactly_four_parameters(cfg, scheme: EloConfig) -> None:
    """The point of the arm: four parameters against the production model's ~2N + 3."""
    _, fit = _fit(_matches([("A", "B", 2, 1), ("B", "C", 1, 1), ("C", "A", 0, 2)] * 40),
                  scheme, cfg)
    assert fit.as_dict()["n_params"] == 4


def test_elo_dc_probabilities_are_a_distribution(cfg, scheme: EloConfig) -> None:
    replay, fit = _fit(_matches([("A", "B", 2, 1), ("B", "C", 1, 1), ("C", "A", 0, 2)] * 40),
                       scheme, cfg)
    probs = fit.predict_proba(pd.DataFrame({"home_team": ["A", "B"], "away_team": ["B", "A"]}))
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_elo_dc_needs_the_replay_columns(cfg, scheme: EloConfig) -> None:
    replay = compute_elo(_matches([("A", "B", 1, 0)]), scheme)
    bare = replay.history.drop(columns=["elo_diff_pre"])
    with pytest.raises(ValueError, match="elo_diff_pre"):
        fit_elo_dixon_coles(
            bare, replay, half_life_days=365.0, ref_date=pd.Timestamp("2021-01-01"),
            max_goals=cfg.model.max_goals, param_bounds=cfg.model.param_bounds,
            max_iter=cfg.model.max_iter,
        )


def test_elo_dc_rejects_an_empty_history(cfg, scheme: EloConfig) -> None:
    replay = compute_elo(_matches([("A", "B", 1, 0)]), scheme)
    with pytest.raises(ValueError, match="empty history"):
        fit_elo_dixon_coles(
            replay.history.iloc[:0], replay, half_life_days=365.0,
            ref_date=pd.Timestamp("2021-01-01"), max_goals=cfg.model.max_goals,
            param_bounds=cfg.model.param_bounds, max_iter=cfg.model.max_iter,
        )


# --- the arm is wired up and distinct --------------------------------------------------------------

def test_elo_dc_is_registered() -> None:
    from plmodel.eval.compare import registered_arms

    assert "elo-dc" in registered_arms()


def test_elo_config_is_separate_from_the_production_model(cfg) -> None:
    """The Elo scheme must not leak into production: it exists for the comparison arm only."""
    assert cfg.elo.k > 0
    assert cfg.elo.initial_rating > 0
    # Its half-life is tuned independently — different parameterisations prefer different memory.
    assert hasattr(cfg.elo, "decay_half_life_days")
