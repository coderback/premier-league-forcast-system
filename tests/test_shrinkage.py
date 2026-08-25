"""General hierarchical shrinkage (model/shrinkage.py), arm 12.

The load-bearing test here is different from arm 11's. That seam had two sites, so a zero
coefficient was still a different model from the seam being absent. This one has a single site, so
``strength = 0`` must be the baseline **exactly** -- and it has to be, because the tuning grid's zero
point is the incumbent that the resolution rule compares everything against. If zero were merely
close to the baseline, every delta on the grid would be measured from the wrong place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.special import gammaln

from plmodel.model.dixon_coles import _N_GLOBAL, _objective, fit_dixon_coles
from plmodel.model.promotion import PromotionSpec
from plmodel.model.shrinkage import ShrinkageError, ShrinkageSpec, ridge_penalty


def _fit_kwargs(ref):
    from plmodel.config import load_config

    return dict(half_life_days=730.0, ref_date=ref, max_goals=8,
                param_bounds=load_config().model.param_bounds,
                min_effective_share=0.15, max_iter=200)


def _corpus():
    rows = []
    day = pd.Timestamp("1994-08-01")
    seasons = {
        "1994-95": ["A", "B", "C", "D", "E"], "1995-96": ["A", "B", "C", "D", "F"],
        "1996-97": ["A", "B", "C", "D", "G"], "1997-98": ["A", "B", "C", "G", "H"],
    }
    # A is strong and E-H are weak, so there is a real spread for the penalty to compress.
    rate = {"A": (3, 1), "B": (2, 1), "C": (2, 2), "D": (2, 2),
            "E": (1, 3), "F": (1, 3), "G": (1, 3), "H": (1, 3)}
    for season, clubs in seasons.items():
        for i, home in enumerate(clubs):
            for away in clubs[i + 1:]:
                rows.append({
                    "date": day, "season": season, "division": "E0", "played": True,
                    "home_team": home, "away_team": away,
                    "home_goals": rate[home][0], "away_goals": rate[away][0],
                })
                day += pd.Timedelta(days=3)
        day += pd.Timedelta(days=60)
    return pd.DataFrame(rows)


# --- the spec ------------------------------------------------------------------------------------

def test_negative_strength_is_refused() -> None:
    with pytest.raises(ShrinkageError):
        ShrinkageSpec(strength=-0.5)


def test_zero_strength_is_inert() -> None:
    """Honest here in a way the promotion seam's would not have been: one site, nothing else on."""
    assert ShrinkageSpec(strength=0.0).is_inert
    assert not ShrinkageSpec(strength=0.5).is_inert


# --- the penalty ---------------------------------------------------------------------------------

def test_the_penalty_is_the_sum_of_squares() -> None:
    attack = np.array([0.3, -0.2, 0.1])
    defence = np.array([0.0, 0.4, -0.5])
    value, ga, gd = ridge_penalty(attack, defence, np.ones(3, dtype=bool), 0.0, 0.0, 2.0)
    assert value == pytest.approx(2.0 * (attack @ attack + defence @ defence))
    assert np.allclose(ga, 4.0 * attack)
    assert np.allclose(gd, 4.0 * defence)


def test_a_nonzero_centre_is_what_the_promotion_seam_borrows() -> None:
    """One implementation, two centres — this is the line that keeps the derivative single."""
    attack = np.array([0.3, -0.2])
    defence = np.array([0.1, 0.0])
    value, ga, _ = ridge_penalty(attack, defence, np.ones(2, dtype=bool), -0.3, -0.27, 1.0)
    assert value == pytest.approx(
        float(((attack + 0.3) ** 2).sum() + ((defence + 0.27) ** 2).sum())
    )
    assert np.allclose(ga, 2.0 * (attack + 0.3))


def test_zero_strength_produces_nothing() -> None:
    a, d = np.array([0.5, -0.5]), np.array([0.2, 0.1])
    value, ga, gd = ridge_penalty(a, d, np.ones(2, dtype=bool), 0.0, 0.0, 0.0)
    assert value == 0.0 and not ga.any() and not gd.any()


# --- the gradient --------------------------------------------------------------------------------

def _gradient_args(rng, n_teams, n, penalty):
    home = rng.integers(0, n_teams, n)
    away = (home + 1 + rng.integers(0, n_teams - 1, n)) % n_teams
    x = rng.poisson(1.5, n).astype(float)
    y = rng.poisson(1.1, n).astype(float)
    weights = rng.uniform(0.2, 1.0, n)
    return (x, y, home, away, weights, gammaln(x + 1), gammaln(y + 1), n_teams,
            np.zeros((n, 0)), np.zeros((n, 0)), np.zeros((n, 0)), penalty)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_matches_finite_differences(seed: int) -> None:
    """Every club is penalised here, so the sum-to-zero last slot is always in play — unlike the
    promotion seam, where it only mattered if a promoted club happened to land there."""
    rng = np.random.default_rng(seed)
    n_teams, n = 8, 400
    args = _gradient_args(rng, n_teams, n, (np.ones(n_teams, dtype=bool), 0.0, 0.0, 3.0))
    theta = np.concatenate([
        [rng.uniform(-0.3, 0.5)], [rng.uniform(0.0, 0.4)], [rng.uniform(-0.12, 0.12)],
        rng.uniform(-0.5, 0.5, 2 * (n_teams - 1)),
    ])
    numeric = approx_fprime(theta, lambda t: _objective(t, *args)[0], 1e-7)
    analytic = _objective(theta, *args)[1]
    assert np.max(np.abs(numeric - analytic)) < 1e-3
    assert np.allclose(numeric, analytic, rtol=1e-3, atol=1e-4)


def test_strength_zero_is_bit_identical_to_no_penalty_in_the_objective() -> None:
    """The grid's zero point must BE the incumbent, not merely resemble it."""
    rng = np.random.default_rng(9)
    n_teams, n = 7, 300
    theta = np.concatenate([[0.1], [0.25], [-0.05], rng.uniform(-0.4, 0.4, 2 * (n_teams - 1))])

    rng = np.random.default_rng(9)
    zero = _objective(theta, *_gradient_args(
        rng, n_teams, n, (np.ones(n_teams, dtype=bool), 0.0, 0.0, 0.0)))
    rng = np.random.default_rng(9)
    absent = _objective(theta, *_gradient_args(rng, n_teams, n, None))
    assert zero[0] == absent[0]
    assert np.array_equal(zero[1], absent[1])


# --- the fit -------------------------------------------------------------------------------------

def test_the_seam_off_is_the_baseline_fit() -> None:
    df = _corpus()
    ref = pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
    kw = _fit_kwargs(ref)
    base = fit_dixon_coles(df, **kw)
    off = fit_dixon_coles(df, **kw, shrinkage=None)
    assert np.array_equal(base.attack, off.attack)
    assert np.array_equal(base.defence, off.defence)


def test_strength_zero_reproduces_the_baseline_fit() -> None:
    """An inert spec must take the same path as no spec: `shrinkage_spec` returns None at zero, and
    the fit must agree, or the grid's zero point is a different model from the baseline."""
    df = _corpus()
    ref = pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
    kw = _fit_kwargs(ref)
    base = fit_dixon_coles(df, **kw)
    zero = fit_dixon_coles(df, **kw, shrinkage=ShrinkageSpec(strength=0.0))
    assert np.array_equal(base.attack, zero.attack)
    assert np.array_equal(base.defence, zero.defence)


def test_shrinkage_compresses_the_spread_of_team_strengths() -> None:
    """The mechanism check. An arm that did not do this would not be this arm."""
    df = _corpus()
    ref = pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
    kw = _fit_kwargs(ref)
    spreads = []
    for strength in (0.0, 1.0, 5.0, 25.0):
        spec = ShrinkageSpec(strength=strength)
        fit = fit_dixon_coles(df, **kw, shrinkage=spec if strength else None)
        spreads.append(float(np.std(fit.attack)))
    assert spreads == sorted(spreads, reverse=True), f"spread must fall monotonically: {spreads}"
    assert spreads[-1] < spreads[0] * 0.5, "heavy shrinkage must visibly compress the spread"


def test_the_two_penalty_seams_refuse_to_run_together() -> None:
    """Both are ridge terms on the same parameters, so an arm with both moves two axes."""
    df = _corpus()
    ref = pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="cannot run together"):
        fit_dixon_coles(
            df, **_fit_kwargs(ref),
            promotion=PromotionSpec(shrinkage=1.0, min_prior_clubs=3),
            shrinkage=ShrinkageSpec(strength=1.0),
        )
