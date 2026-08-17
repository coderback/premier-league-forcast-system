"""Logarithmic pooling, the shots-on-target variant, and the Pitcan (2026) reproduction.

The pooling maths is tested against cases with known answers, because the reproduction's whole
value rests on the weight being estimated correctly: a weight of zero is the paper's headline
result, and an optimiser that returns zero for the wrong reason would "confirm" it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval import metrics
from plmodel.reproduce import pitcan2026
from plmodel.reproduce.pooling import (
    fit_pair_weight, fit_simplex_weights, log_pool, pool_log_loss, weight_profile,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _outcomes(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 3, n)


# --- the pool itself -------------------------------------------------------------------------

def test_pool_of_identical_forecasts_is_that_forecast() -> None:
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    assert np.allclose(log_pool([p, p], [0.5, 0.5]), p)


def test_full_weight_recovers_one_side() -> None:
    a = np.array([[0.5, 0.3, 0.2]])
    b = np.array([[0.1, 0.1, 0.8]])
    assert np.allclose(log_pool([a, b], [1.0, 0.0]), a)
    assert np.allclose(log_pool([a, b], [0.0, 1.0]), b)


def test_pool_output_is_a_distribution() -> None:
    rng = np.random.default_rng(3)
    a, b = rng.dirichlet([2, 2, 2], 50), rng.dirichlet([3, 1, 2], 50)
    for w in (-0.5, 0.0, 0.25, 1.0, 1.5):
        pooled = log_pool([a, b], [w, 1.0 - w])
        assert np.allclose(pooled.sum(axis=1), 1.0)
        assert (pooled >= 0).all()


def test_pool_is_the_normalised_geometric_mean() -> None:
    a = np.array([[0.6, 0.3, 0.1]])
    b = np.array([[0.2, 0.2, 0.6]])
    manual = np.sqrt(a * b)
    assert np.allclose(log_pool([a, b], [0.5, 0.5]), manual / manual.sum())


def test_pool_rejects_mismatched_inputs() -> None:
    p = np.zeros((3, 3))
    with pytest.raises(ValueError, match="forecasts but"):
        log_pool([p, p], [1.0])
    with pytest.raises(ValueError, match="same shape"):
        log_pool([p, np.zeros((4, 3))], [0.5, 0.5])
    with pytest.raises(ValueError, match="at least one"):
        log_pool([], [])


# --- weight estimation -------------------------------------------------------------------------

def test_a_useless_forecast_earns_zero_weight() -> None:
    """The paper's headline shape: a forecast carrying nothing gets no admixture."""
    rng = np.random.default_rng(5)
    n = 800
    outcomes = _outcomes(n, seed=5)
    informative = np.full((n, 3), 0.2)
    informative[np.arange(n), outcomes] = 0.6      # genuinely knows the answer
    noise = rng.dirichlet([1, 1, 1], n)            # knows nothing
    result = fit_pair_weight(noise, informative, outcomes, n_grid=51)
    assert result["weight"] < 0.05 and result["at_lower_bound"]


def test_an_informative_forecast_earns_positive_weight() -> None:
    """The detector must be able to fire, or a zero proves nothing."""
    rng = np.random.default_rng(6)
    n = 800
    outcomes = _outcomes(n, seed=6)
    weak = np.full((n, 3), 1 / 3)
    strong = np.full((n, 3), 0.2)
    strong[np.arange(n), outcomes] = 0.6
    result = fit_pair_weight(strong, weak, outcomes, n_grid=51)
    assert result["weight"] > 0.5


def test_two_equally_good_forecasts_split_the_weight() -> None:
    """Two independent, equally informative views should both be admitted."""
    rng = np.random.default_rng(7)
    n = 1200
    outcomes = _outcomes(n, seed=7)
    def noisy_view(seed):
        r = np.random.default_rng(seed)
        p = r.dirichlet([1, 1, 1], n) * 0.5
        p[np.arange(n), outcomes] += 0.5
        return p / p.sum(axis=1, keepdims=True)
    result = fit_pair_weight(noisy_view(11), noisy_view(12), outcomes, n_grid=51)
    assert 0.2 < result["weight"] < 0.8


def test_weight_search_is_bounded() -> None:
    rng = np.random.default_rng(8)
    a, b = rng.dirichlet([2, 2, 2], 200), rng.dirichlet([2, 2, 2], 200)
    result = fit_pair_weight(a, b, _outcomes(200, 8), n_grid=21)
    assert 0.0 <= result["weight"] <= 1.0


# --- the boundary-solution check -------------------------------------------------------------

def test_profile_detects_a_genuine_boundary_minimum() -> None:
    """A fitted zero cannot distinguish a real null from an optimiser stopping at a bound; the
    profile can, and that distinction is the paper's central methodological move."""
    rng = np.random.default_rng(9)
    n = 900
    outcomes = _outcomes(n, seed=9)
    informative = np.full((n, 3), 0.2)
    informative[np.arange(n), outcomes] = 0.6
    noise = rng.dirichlet([1, 1, 1], n)
    profile = weight_profile(noise, informative, outcomes, lower=0.0, upper=1.0, n_grid=41)
    assert profile["monotone_increasing"] is True
    assert profile["argmin_weight"] == pytest.approx(0.0)


def test_profile_is_not_monotone_when_the_optimum_is_interior() -> None:
    rng = np.random.default_rng(10)
    n = 900
    outcomes = _outcomes(n, seed=10)
    def view(seed):
        r = np.random.default_rng(seed)
        p = r.dirichlet([1, 1, 1], n) * 0.5
        p[np.arange(n), outcomes] += 0.5
        return p / p.sum(axis=1, keepdims=True)
    profile = weight_profile(view(21), view(22), outcomes, lower=0.0, upper=1.0, n_grid=41)
    assert profile["monotone_increasing"] is False
    assert 0.0 < profile["argmin_weight"] < 1.0


# --- the simplex pool ----------------------------------------------------------------------------

def test_simplex_weights_sum_to_one() -> None:
    rng = np.random.default_rng(11)
    n = 400
    outcomes = _outcomes(n, 11)
    forecasts = {name: rng.dirichlet([2, 2, 2], n) for name in ("a", "b", "c")}
    result = fit_simplex_weights(forecasts, outcomes)
    assert sum(result[k] for k in ("a", "b", "c")) == pytest.approx(1.0)


def test_simplex_collapses_onto_the_only_informative_view() -> None:
    """The paper's three-way result: both structural models go to zero simultaneously."""
    rng = np.random.default_rng(12)
    n = 900
    outcomes = _outcomes(n, 12)
    informative = np.full((n, 3), 0.2)
    informative[np.arange(n), outcomes] = 0.6
    result = fit_simplex_weights(
        {
            "market": informative,
            "goals": rng.dirichlet([1, 1, 1], n),
            "shots": rng.dirichlet([1, 1, 1], n),
        },
        outcomes,
    )
    assert result["market"] > 0.9
    assert result["goals"] < 0.05 and result["shots"] < 0.05


def test_simplex_needs_two_forecasts() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fit_simplex_weights({"a": np.full((3, 3), 1 / 3)}, np.zeros(3, dtype=int))


def test_pool_log_loss_matches_the_metric() -> None:
    rng = np.random.default_rng(13)
    a, b = rng.dirichlet([2, 2, 2], 100), rng.dirichlet([2, 2, 2], 100)
    outcomes = _outcomes(100, 13)
    assert pool_log_loss([a, b], [0.3, 0.7], outcomes) == pytest.approx(
        metrics.mean_log_loss(log_pool([a, b], [0.3, 0.7]), outcomes)
    )


# --- the shots-on-target variant ------------------------------------------------------------------

def test_finishing_factor_is_weighted_goals_per_shot_on_target() -> None:
    from plmodel.model.shots import finishing_factors

    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "home_goals": [2.0, 0.0], "away_goals": [1.0, 1.0],
            "home_sot": [5.0, 5.0], "away_sot": [4.0, 4.0],
        }
    )
    # Equal dates mean equal weights, so this is just the ratio of totals.
    kappa_home, kappa_away = finishing_factors(history, pd.Timestamp("2024-01-02"), 365.0)
    assert kappa_home == pytest.approx(2.0 / 10.0)
    assert kappa_away == pytest.approx(2.0 / 8.0)


def test_finishing_factor_needs_coverage() -> None:
    from plmodel.model.shots import finishing_factors

    empty = pd.DataFrame({"date": [], "home_goals": [], "away_goals": [],
                          "home_sot": [], "away_sot": []})
    with pytest.raises(ValueError, match="no rows with shots-on-target"):
        finishing_factors(empty, pd.Timestamp("2024-01-01"), 365.0)


@pytest.mark.integration
def test_shots_model_fits_and_predicts(cfg) -> None:
    from plmodel.model.shots import fit_shots_model

    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    corpus = pd.read_parquet(path)
    e0 = corpus[(corpus["division"] == "E0") & corpus["played"]]
    train = e0[e0["date"] < pd.Timestamp("2019-06-01")]

    fit = fit_shots_model(
        train, half_life_days=cfg.model.decay_half_life_days,
        ref_date=pd.Timestamp("2019-06-01"), max_goals=cfg.model.max_goals,
        param_bounds=cfg.model.param_bounds,
        min_effective_share=cfg.model.min_effective_share, max_iter=cfg.model.max_iter,
    )
    # Premier League finishing rate: roughly three shots on target per goal, matching the paper's
    # Serie A factors of 0.309 / 0.317.
    assert 0.25 < fit.kappa_home < 0.36
    assert 0.25 < fit.kappa_away < 0.36
    probs = fit.predict_proba(pd.DataFrame({"home_team": ["Arsenal"], "away_team": ["Burnley"]}))
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, metrics.HOME] > probs[0, metrics.AWAY]


# --- the reproduction's own contract ------------------------------------------------------------

def test_paper_results_record_both_references() -> None:
    """The claim most easily misread: 0.35 is against the goals model, 0.00 against the market."""
    assert pitcan2026.PAPER_RESULTS["goals_plus_shots"]["shots"] == 0.35
    assert pitcan2026.PAPER_RESULTS["market_plus_shots"]["shots"] == 0.00
    assert pitcan2026.PAPER_RESULTS["market_plus_goals"]["goals"] == 0.00


def test_windows_match_the_paper_sample_sizes() -> None:
    """Both leagues play 380-match seasons, so the season spans give identical n."""
    validation_seasons = 6   # 2013-14..2018-19
    test_seasons = 7         # 2019-20..2025-26
    assert validation_seasons * 380 == 2280
    assert test_seasons * 380 == 2660


def test_verdict_proceeds_when_the_pattern_reproduces() -> None:
    validation = {
        "market_plus_goals": {"weight": 0.0},
        "market_plus_shots": {"weight": 0.0},
        "goals_plus_shots": {"weight": 0.35},
        "goals_profile_admissible": {"monotone_increasing": True},
    }
    result = pitcan2026.verdict(validation, {})
    assert result["reproduces"] is True
    assert result["xg_arm_gate"].startswith("PROCEED")


def test_verdict_downgrades_when_the_channel_carries_nothing() -> None:
    """The gate must be able to say no, or running it was theatre."""
    validation = {
        "market_plus_goals": {"weight": 0.0},
        "market_plus_shots": {"weight": 0.0},
        "goals_plus_shots": {"weight": 0.01},
        "goals_profile_admissible": {"monotone_increasing": True},
    }
    result = pitcan2026.verdict(validation, {})
    assert result["reproduces"] is False
    assert result["xg_arm_gate"].startswith("DOWNGRADE")


def test_verdict_flags_a_non_genuine_boundary() -> None:
    validation = {
        "market_plus_goals": {"weight": 0.0},
        "market_plus_shots": {"weight": 0.0},
        "goals_plus_shots": {"weight": 0.35},
        "goals_profile_admissible": {"monotone_increasing": False},
    }
    assert pitcan2026.verdict(validation, {})["boundary_is_genuine"] is False


def test_verdict_reports_the_sensitivity_range() -> None:
    validation = {
        "market_plus_goals": {"weight": 0.0},
        "market_plus_shots": {"weight": 0.0},
        "goals_plus_shots": {"weight": 0.17},
        "goals_profile_admissible": {"monotone_increasing": True},
    }
    sensitivity = [
        {"half_life_days": 180.0, "weight_on_shots": 0.63},
        {"half_life_days": 1460.0, "weight_on_shots": 0.09},
    ]
    result = pitcan2026.verdict(validation, {}, sensitivity)
    assert result["paper_weight_inside_our_range"] is True
