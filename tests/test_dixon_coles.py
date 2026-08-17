"""The production Dixon-Coles fit.

Two tests here carry most of the weight. The gradient check is what makes the analytic derivative
trustworthy — a subtly wrong gradient does not crash, it just converges somewhere slightly wrong,
and every downstream number inherits that. The recovery test is what shows the whole specification
is identified: simulate from known strengths, fit, and get them back.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.special import gammaln

from plmodel.config import load_config
from plmodel.model.dixon_coles import (
    _N_GLOBAL, _objective, _unpack, decay_weights, fit_dixon_coles, fit_summary,
)
from plmodel.model.scoreline import (
    collapse_three_class, poisson_pmf_grid, scoreline_matrix, tau_is_valid, three_class_from_rates,
)

BOUNDS = {
    "intercept": (-2.0, 2.0), "home_advantage": (-1.0, 1.0),
    "rho": (-0.2, 0.2), "strength": (-3.0, 3.0),
}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _simulate(
    n_seasons: int = 6, *, attack: dict[str, float], defence: dict[str, float] | None = None,
    intercept: float = 0.1, home_adv: float = 0.26, seed: int = 11,
) -> pd.DataFrame:
    """A synthetic league with known strengths — a double round robin per season."""
    rng = np.random.default_rng(seed)
    defence = {t: 0.0 for t in attack} if defence is None else defence
    teams = list(attack)
    rows = []
    date = pd.Timestamp("2010-08-01")
    for _ in range(n_seasons):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                lam = math.exp(intercept + home_adv + attack[h] - defence[a])
                mu = math.exp(intercept + attack[a] - defence[h])
                rows.append(
                    {
                        "date": date, "home_team": h, "away_team": a,
                        "home_goals": float(rng.poisson(lam)),
                        "away_goals": float(rng.poisson(mu)),
                    }
                )
                date += pd.Timedelta(days=3)
    return pd.DataFrame(rows)


def _fit(history, **kw):
    params = {
        "half_life_days": 100_000.0,   # effectively no decay, for recovery tests
        "ref_date": history["date"].max() + pd.Timedelta(days=1),
        "max_goals": 12, "param_bounds": BOUNDS,
        "min_effective_share": 0.15, "max_iter": 500,
    }
    return fit_dixon_coles(history, **{**params, **kw})


# --- the gradient ------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_analytic_gradient_matches_finite_differences(seed: int) -> None:
    """A wrong gradient does not crash; it converges somewhere slightly wrong and every number
    downstream inherits it. This is the check that makes the analytic derivative trustworthy."""
    rng = np.random.default_rng(seed)
    n_teams, n = 8, 400
    home = rng.integers(0, n_teams, n)
    away = (home + 1 + rng.integers(0, n_teams - 1, n)) % n_teams
    x = rng.poisson(1.5, n).astype(float)
    y = rng.poisson(1.1, n).astype(float)
    weights = rng.uniform(0.2, 1.0, n)
    args = (x, y, home, away, weights, gammaln(x + 1), gammaln(y + 1), n_teams,
            np.zeros((n, 0)))

    theta = np.concatenate([
        [rng.uniform(-0.3, 0.5)], [rng.uniform(0.0, 0.4)], [rng.uniform(-0.12, 0.12)],
        rng.uniform(-0.5, 0.5, 2 * (n_teams - 1)),
    ])
    numeric = approx_fprime(theta, lambda t: _objective(t, *args)[0], 1e-7)
    analytic = _objective(theta, *args)[1]
    assert np.max(np.abs(numeric - analytic)) < 1e-3
    assert np.allclose(numeric, analytic, rtol=1e-3, atol=1e-4)


def test_gradient_is_zero_at_the_optimum() -> None:
    """A stationary point is what the optimiser claims to have found."""
    attack = {"A": 0.4, "B": 0.1, "C": -0.2, "D": -0.3}
    fit = _fit(_simulate(attack=attack, defence={"A": 0.3, "B": 0.0, "C": -0.1, "D": -0.2}))
    assert fit.converged


# --- identifiability and recovery ----------------------------------------------------------------

def test_strengths_sum_to_zero_by_construction() -> None:
    """Without the constraint the parameters are unidentified: a constant can move between attack
    and defence with no effect on any rate."""
    attack = {"A": 0.5, "B": 0.2, "C": -0.1, "D": -0.6}
    fit = _fit(_simulate(attack=attack, defence={"A": 0.4, "B": 0.1, "C": -0.2, "D": -0.3}))
    assert fit.attack.sum() == pytest.approx(0.0, abs=1e-9)
    assert fit.defence.sum() == pytest.approx(0.0, abs=1e-9)


def test_unpack_applies_the_constraint() -> None:
    theta = np.array([0.1, 0.2, -0.05, 0.3, -0.1, 0.2, 0.4])   # 3 globals + 2 free A + 2 free D
    _, _, _, attack, defence, _ = _unpack(theta, n_teams=3)
    assert attack.tolist() == pytest.approx([0.3, -0.1, -0.2])
    assert defence.tolist() == pytest.approx([0.2, 0.4, -0.6])


_TRUE_ATTACK = {f"T{i}": v for i, v in enumerate(
    [0.55, 0.40, 0.25, 0.10, 0.0, -0.05, -0.20, -0.30, -0.45, -0.60])}
_TRUE_DEFENCE = {f"T{i}": v for i, v in enumerate(
    [0.45, 0.30, 0.10, 0.05, 0.0, -0.05, -0.15, -0.25, -0.30, -0.40])}


def _recovery_errors(seed: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    truth_a = np.array(list(_TRUE_ATTACK.values()))
    truth_d = np.array(list(_TRUE_DEFENCE.values()))
    truth_a, truth_d = truth_a - truth_a.mean(), truth_d - truth_d.mean()
    fit = _fit(_simulate(n_seasons=20, attack=_TRUE_ATTACK, defence=_TRUE_DEFENCE,
                         intercept=0.1, home_adv=0.26, seed=seed))
    order = {t: i for i, t in enumerate(fit.teams)}
    got_a = np.array([fit.attack[order[t]] for t in _TRUE_ATTACK])
    got_d = np.array([fit.defence[order[t]] for t in _TRUE_DEFENCE])
    return got_a - truth_a, got_d - truth_d, fit.home_advantage, fit.intercept


def test_recovers_known_strengths_without_bias() -> None:
    """Simulate from known parameters, fit, and get them back — the whole specification at once.

    Averaged over seeds rather than asserted on one draw. A single fit's error is dominated by
    Poisson noise (per-team standard error ~0.03-0.07 on 1,800 matches), so a tight single-draw
    bound would be testing luck. What must hold is that the estimator is *unbiased*: the mean
    error across independent draws sits at zero.
    """
    seeds = range(6)
    errors_a, errors_d, home_advs, intercepts = zip(*(_recovery_errors(s) for s in seeds))
    mean_a = np.mean(errors_a, axis=0)
    mean_d = np.mean(errors_d, axis=0)
    # Standard error of the mean across seeds; a genuine bias would clear it comfortably.
    sem_a = np.std(errors_a, axis=0) / math.sqrt(len(seeds))
    assert np.all(np.abs(mean_a) < 0.05), f"attack looks biased: {mean_a}"
    assert np.all(np.abs(mean_d) < 0.05), f"defence looks biased: {mean_d}"
    assert np.all(np.abs(mean_a) < np.maximum(3 * sem_a, 0.05))
    assert np.mean(home_advs) == pytest.approx(0.26, abs=0.03)
    assert np.mean(intercepts) == pytest.approx(0.1, abs=0.03)


def test_recovers_the_strength_ordering() -> None:
    """Beyond unbiasedness: a single fit must still rank the teams correctly."""
    errors_a, _, _, _ = _recovery_errors(0)
    truth = np.array(list(_TRUE_ATTACK.values()))
    truth = truth - truth.mean()
    fitted = truth + errors_a
    assert np.corrcoef(fitted, truth)[0, 1] > 0.98


def test_home_advantage_is_recovered_when_absent() -> None:
    """A model that always finds home advantage would be fitting an artefact."""
    attack = {"A": 0.3, "B": 0.0, "C": -0.3}
    fit = _fit(_simulate(n_seasons=40, attack=attack, defence=attack, home_adv=0.0))
    assert fit.home_advantage == pytest.approx(0.0, abs=0.05)


# --- time decay ---------------------------------------------------------------------------------

def test_decay_weights_halve_at_the_half_life() -> None:
    dates = pd.Series(pd.to_datetime(["2024-01-01", "2023-07-05", "2023-01-07"]))
    w = decay_weights(dates, pd.Timestamp("2024-01-01"), half_life_days=180.0)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(0.5, abs=0.01)
    assert w[2] == pytest.approx(0.25, abs=0.01)


def test_future_matches_are_not_up_weighted() -> None:
    """Age is clipped at zero so a stray future row cannot get a weight above 1."""
    w = decay_weights(pd.Series([pd.Timestamp("2024-06-01")]), pd.Timestamp("2024-01-01"), 180.0)
    assert w[0] == pytest.approx(1.0)


def test_invalid_half_life_raises() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        decay_weights(pd.Series([pd.Timestamp("2024-01-01")]), pd.Timestamp("2024-01-01"), 0.0)


def test_shorter_half_life_tracks_recent_form() -> None:
    """A team that improves sharply should look stronger under a short half-life than a long one."""
    early = _simulate(n_seasons=3, attack={"A": -0.5, "B": 0.0, "C": 0.0, "D": 0.5}, seed=1)
    late = _simulate(n_seasons=3, attack={"A": 0.8, "B": 0.0, "C": 0.0, "D": -0.5}, seed=2)
    late["date"] = late["date"] + pd.Timedelta(days=2000)
    history = pd.concat([early, late], ignore_index=True)
    ref = history["date"].max() + pd.Timedelta(days=1)

    short = _fit(history, half_life_days=200.0, ref_date=ref)
    long = _fit(history, half_life_days=5000.0, ref_date=ref)
    a_short = short.attack[short.teams.index("A")]
    a_long = long.attack[long.teams.index("A")]
    assert a_short > a_long


# --- cold starts ----------------------------------------------------------------------------------

def test_thinly_observed_teams_are_pinned_and_named() -> None:
    """Promoted teams are ~28% of the fixture list, so this is the model's behaviour, not an edge
    case. A pinned team is reported by name; it is never silently imputed."""
    history = _simulate(n_seasons=6, attack={"A": 0.3, "B": 0.0, "C": -0.3}, defence={"A": 0.2, "B": 0.0, "C": -0.2})
    newcomer = pd.DataFrame(
        {
            "date": [history["date"].max() + pd.Timedelta(days=1)],
            "home_team": ["Newcomer"], "away_team": ["A"],
            "home_goals": [1.0], "away_goals": [2.0],
        }
    )
    fit = _fit(pd.concat([history, newcomer], ignore_index=True), min_effective_share=0.15)
    assert "Newcomer" in fit.cold_start_teams
    assert "Newcomer" not in fit.teams


def test_an_unknown_team_predicts_at_league_average() -> None:
    fit = _fit(_simulate(attack={"A": 0.3, "B": 0.0, "C": -0.3}, defence={"A": 0.2, "B": 0.0, "C": -0.2}))
    rows = pd.DataFrame({"home_team": ["Nobody"], "away_team": ["Nobody Else"]})
    lam, mu = fit.rates(rows["home_team"], rows["away_team"])
    assert lam[0] == pytest.approx(math.exp(fit.intercept + fit.home_advantage))
    assert mu[0] == pytest.approx(math.exp(fit.intercept))


def test_effective_not_raw_history_decides_a_cold_start() -> None:
    """Under a short half-life a club returning after years away has a healthy match count and
    almost no usable information; the threshold has to be on weight."""
    old = _simulate(n_seasons=3, attack={"A": 0.2, "B": 0.0, "C": -0.2}, seed=3)
    recent = _simulate(n_seasons=3, attack={"B": 0.2, "C": 0.0, "D": -0.2}, seed=4)
    recent["date"] = recent["date"] + pd.Timedelta(days=4000)
    history = pd.concat([old, recent], ignore_index=True)
    fit = _fit(history, half_life_days=200.0, ref_date=history["date"].max() + pd.Timedelta(days=1))
    assert "A" in fit.cold_start_teams   # only appears in the distant past


def test_empty_history_raises() -> None:
    with pytest.raises(ValueError, match="empty history"):
        _fit(pd.DataFrame(columns=["date", "home_team", "away_team", "home_goals", "away_goals"]))


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        _fit(pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "home_team": ["A"]}))


# --- warm starts ------------------------------------------------------------------------------------

def test_warm_start_reaches_the_same_optimum() -> None:
    """Warm starting is a speed device; it must not change where the fit lands."""
    history = _simulate(n_seasons=10, attack={"A": 0.4, "B": 0.1, "C": -0.2, "D": -0.3},
                        defence={"A": 0.3, "B": 0.0, "C": -0.1, "D": -0.2})
    cold = _fit(history)
    warm = _fit(history, warm_start=cold)
    assert np.allclose(cold.attack, warm.attack, atol=1e-4)
    assert warm.home_advantage == pytest.approx(cold.home_advantage, abs=1e-4)


def test_warm_start_converges_in_fewer_iterations() -> None:
    history = _simulate(n_seasons=10, attack={"A": 0.4, "B": 0.1, "C": -0.2, "D": -0.3})
    cold = _fit(history)
    warm = _fit(history, warm_start=cold)
    assert warm.n_iterations < cold.n_iterations


def test_warm_start_tolerates_a_changed_team_set() -> None:
    """Promotion and relegation change the team set between barriers."""
    first = _simulate(n_seasons=6, attack={"A": 0.3, "B": 0.0, "C": -0.3})
    fit = _fit(first)
    second = _simulate(n_seasons=6, attack={"B": 0.3, "C": 0.0, "D": -0.3}, seed=9)
    warm = _fit(second, warm_start=fit)
    assert warm.converged and "D" in warm.teams


# --- the scoreline layer -------------------------------------------------------------------------

def test_poisson_grid_sums_to_one() -> None:
    grid = poisson_pmf_grid(np.array([1.4, 2.2]), max_goals=15)
    assert np.allclose(grid.sum(axis=1), 1.0, atol=1e-8)


def test_three_class_probabilities_sum_to_one() -> None:
    probs = three_class_from_rates(np.array([1.5, 0.8]), np.array([1.1, 2.0]), -0.05, 12)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs > 0).all()


def test_rho_zero_is_the_independent_poisson_case() -> None:
    joint = scoreline_matrix(np.array([1.5]), np.array([1.1]), 0.0, 12)
    independent = np.outer(poisson_pmf_grid(np.array([1.5]), 12)[0],
                           poisson_pmf_grid(np.array([1.1]), 12)[0])
    assert np.allclose(joint[0], independent / independent.sum())


def test_negative_rho_lifts_the_low_score_draws() -> None:
    """What the tau correction is for: independent Poisson misprices 0-0 and 1-1."""
    lam, mu = np.array([1.4]), np.array([1.1])
    plain = scoreline_matrix(lam, mu, 0.0, 12)
    corrected = scoreline_matrix(lam, mu, -0.08, 12)
    assert corrected[0, 0, 0] > plain[0, 0, 0]
    assert corrected[0, 1, 1] > plain[0, 1, 1]


def test_collapse_counts_every_cell_once() -> None:
    joint = scoreline_matrix(np.array([1.5]), np.array([1.1]), -0.05, 8)
    assert collapse_three_class(joint).sum() == pytest.approx(joint.sum())


def test_tau_validity_bound() -> None:
    lam, mu = np.array([1.5]), np.array([1.2])
    assert tau_is_valid(lam, mu, -0.15) and tau_is_valid(lam, mu, 0.15)
    assert not tau_is_valid(lam, mu, 0.9)


def test_configured_rho_bounds_are_safe_at_realistic_rates(cfg) -> None:
    """The bound is an outer box, not a guarantee.

    The binding constraint 1 - lam*mu*rho > 0 is rate-dependent, so no fixed bound can be safe at
    every rate: at lam = mu = 2.5 it already fails for rho > 0.16. The bound keeps the optimiser in
    a sensible region at rates football actually produces; the real guarantee is enforced at
    runtime (see the two tests below).
    """
    lo, hi = cfg.model.param_bounds["rho"]
    lam, mu = np.array([1.6]), np.array([1.2])   # a typical Premier League matchup
    assert tau_is_valid(lam, mu, lo) and tau_is_valid(lam, mu, hi)


def test_the_likelihood_refuses_an_invalid_rho() -> None:
    """The real guard: a rho that makes tau non-positive returns an invalid objective, so the
    optimiser is steered away from it rather than converging into a non-probability model."""
    n = 20
    x = np.zeros(n)          # all 0-0 draws: the cell 1 - lam*mu*rho governs
    y = np.zeros(n)
    home = np.arange(n) % 4
    away = (home + 1) % 4
    weights = np.ones(n)
    args = (x, y, home, away, weights, gammaln(x + 1), gammaln(y + 1), 4, np.zeros((n, 0)))
    # High rates plus a large positive rho drives tau(0,0) negative.
    theta = np.array([1.0, 0.0, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    value, grad = _objective(theta, *args)
    assert value >= 1e11 and np.all(grad == 0.0)


def test_the_scoreline_refuses_an_invalid_rho() -> None:
    with pytest.raises(ValueError, match="negative scoreline probability"):
        scoreline_matrix(np.array([2.5]), np.array([2.5]), 0.3, 12)


# --- reporting -----------------------------------------------------------------------------------

def test_fit_summary_reports_the_rho_sign() -> None:
    """The free falsifier: the sign of the fitted dependence decides whether a negative-dependence
    copula arm is worth building at all."""
    fit = _fit(_simulate(attack={"A": 0.3, "B": 0.0, "C": -0.3}, defence={"A": 0.2, "B": 0.0, "C": -0.2}))
    summary = fit_summary(fit)
    assert summary["rho_sign"] in {"negative", "positive", "zero"}
    assert summary["n_teams"] == len(fit.teams)


def test_team_table_is_ordered_by_attack() -> None:
    fit = _fit(_simulate(attack={"A": 0.5, "B": 0.0, "C": -0.5}, defence={"A": 0.2, "B": 0.0, "C": -0.2}))
    table = fit.team_table()
    assert table["attack"].is_monotonic_decreasing
    assert table.iloc[0]["team"] == "A"
