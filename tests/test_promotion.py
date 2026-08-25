"""The promotion prior (model/promotion.py), arm 11.

The tests that matter here are the ones where the seam could be wrong in a way that still looks
right: a prior that quietly reads the future, a penalty gradient that is subtly inconsistent with
its own value, and a "shrinkage = 0" that is not actually the baseline. Each of those produces a
plausible number rather than a crash, which is exactly why they are pinned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.special import gammaln

from plmodel.model.dixon_coles import _N_GLOBAL, _objective, decay_weights, fit_dixon_coles
from plmodel.model.promotion import (
    PromotionError,
    PromotionPrior,
    PromotionSpec,
    estimate_prior,
    first_season_clubs,
    penalty_and_gradient,
    promoted_at_barrier,
)

SPEC = PromotionSpec(shrinkage=1.0, min_prior_clubs=3)


def season_frame(seasons: dict[str, list[str]], *, goals: dict[str, tuple[int, int]] | None = None):
    """A tiny corpus: each season maps to its member clubs, played as a single round robin.

    ``goals`` optionally overrides a club's (scored, conceded) per match, so a test can make one
    group genuinely weaker instead of asserting against noise.
    """
    rows = []
    day = pd.Timestamp("1994-08-01")
    for season, clubs in seasons.items():
        for i, home in enumerate(clubs):
            for away in clubs[i + 1:]:
                hs, hc = (goals or {}).get(home, (2, 1))
                as_, ac = (goals or {}).get(away, (2, 1))
                rows.append({
                    "date": day, "season": season, "division": "E0", "played": True,
                    "home_team": home, "away_team": away,
                    # Home scores its own rate, concedes the away side's.
                    "home_goals": hs, "away_goals": as_,
                })
                day += pd.Timedelta(days=3)
        day += pd.Timedelta(days=60)
    return pd.DataFrame(rows)


# --- identifying a promoted club -----------------------------------------------------------------

def test_first_season_of_the_corpus_contributes_nobody() -> None:
    """Otherwise every founding club reads as promoted, which is a fact about the corpus boundary."""
    df = season_frame({"1994-95": ["A", "B", "C"], "1995-96": ["A", "B", "D"]})
    pool = first_season_clubs(df)
    assert pool["1994-95"] == set()
    assert pool["1995-96"] == {"D"}


def test_the_penalty_targets_the_latest_seasons_intake() -> None:
    df = season_frame({
        "1994-95": ["A", "B", "C"], "1995-96": ["A", "B", "D"], "1996-97": ["A", "D", "E"],
    })
    assert promoted_at_barrier(df) == ("E",)


def test_season_opening_barrier_carries_the_previous_intake() -> None:
    """The seam's one approximation, pinned so it cannot change silently.

    At a season-opening barrier no match of the new season has been played, so the latest season in
    the training window is the one just finished and its intake -- clubs now starting their SECOND
    season -- is what the penalty targets. Documented in ``promoted_at_barrier`` and recorded in the
    pre-registration; asserted here so the day someone fixes it, this test fails and says so.
    """
    df = season_frame({"1994-95": ["A", "B", "C"], "1995-96": ["A", "B", "D"]})
    assert promoted_at_barrier(df) == ("D",)


def test_an_empty_history_has_no_promoted_clubs() -> None:
    assert promoted_at_barrier(pd.DataFrame(columns=["season", "home_team", "away_team"])) == ()


# --- the prior -----------------------------------------------------------------------------------

def test_the_prior_recovers_a_planted_gap() -> None:
    """A promoted club scoring half as much must come back at log(0.5) against the league."""
    seasons = {"1994-95": ["A", "B", "C", "D"], "1995-96": ["A", "B", "C", "E"]}
    # E is the only promoted club and is deliberately feeble; everyone else scores 2 and concedes 2.
    df = season_frame(seasons, goals={"E": (1, 1)})
    weights = np.ones(len(df))
    prior = estimate_prior(df, weights, min_prior_clubs=1)

    league_gf = float(df["home_goals"].sum() + df["away_goals"].sum()) / (2 * len(df))
    e_rows = df[(df["home_team"] == "E") | (df["away_team"] == "E")]
    e_scored = float(
        np.where(e_rows["home_team"] == "E", e_rows["home_goals"], e_rows["away_goals"]).sum()
    ) / len(e_rows)
    assert prior.attack == pytest.approx(np.log(e_scored / league_gf))
    assert prior.attack < 0.0, "a weaker promoted club must get a negative attack prior"


def test_defence_uses_the_same_league_baseline_as_attack() -> None:
    """Every goal scored is also conceded, so the league's for and against rates are one number.

    A separate `ga_league` would be a second name for the same quantity; this pins that the code
    does not grow one.
    """
    df = season_frame({"1994-95": ["A", "B", "C", "D"], "1995-96": ["A", "B", "C", "E"]},
                      goals={"E": (1, 3)})
    prior = estimate_prior(df, np.ones(len(df)), min_prior_clubs=1)
    league = float(df["home_goals"].sum() + df["away_goals"].sum()) / (2 * len(df))
    e_rows = df[(df["home_team"] == "E") | (df["away_team"] == "E")]
    conceded = float(
        np.where(e_rows["home_team"] == "E", e_rows["away_goals"], e_rows["home_goals"]).sum()
    ) / len(e_rows)
    assert prior.defence == pytest.approx(-np.log(conceded / league))
    assert prior.defence < 0.0, "a leakier promoted club must get a negative defence prior"


def test_too_few_promoted_clubs_yields_no_prior() -> None:
    """Early in a walk there is nothing to learn from, and one club is worse than no prior."""
    df = season_frame({"1994-95": ["A", "B", "C"], "1995-96": ["A", "B", "D"]})
    assert estimate_prior(df, np.ones(len(df)), min_prior_clubs=3) is None


def test_the_prior_does_not_read_the_future() -> None:
    """The load-bearing leakage test: the whole arm rests on an estimate made from the past only.

    A prior computed on history truncated at a barrier must be bit-identical to one computed on the
    same rows when later seasons exist in the corpus but are cut before the call.
    """
    full = season_frame({
        "1994-95": ["A", "B", "C", "D"], "1995-96": ["A", "B", "C", "E"],
        "1996-97": ["A", "B", "E", "F"], "1997-98": ["A", "B", "F", "G"],
    }, goals={"E": (1, 3), "F": (1, 3), "G": (1, 3)})
    barrier = pd.Timestamp(full[full["season"] == "1996-97"]["date"].iloc[-1])

    truncated = full[full["date"] < barrier].reset_index(drop=True)
    early = estimate_prior(truncated, decay_weights(truncated["date"], barrier, 730.0),
                           min_prior_clubs=1)
    # The corpus grows; the answer at the same barrier must not.
    again = estimate_prior(truncated, decay_weights(truncated["date"], barrier, 730.0),
                           min_prior_clubs=1)
    assert early.attack == again.attack and early.defence == again.defence

    later = estimate_prior(full, decay_weights(full["date"], barrier, 730.0), min_prior_clubs=1)
    assert later.attack != early.attack, (
        "a prior that ignored the extra seasons would mean the estimator is not reading its input"
    )


# --- the penalty and its gradient ----------------------------------------------------------------

def _gradient_args(rng, n_teams, n, penalty):
    home = rng.integers(0, n_teams, n)
    away = (home + 1 + rng.integers(0, n_teams - 1, n)) % n_teams
    x = rng.poisson(1.5, n).astype(float)
    y = rng.poisson(1.1, n).astype(float)
    weights = rng.uniform(0.2, 1.0, n)
    return (x, y, home, away, weights, gammaln(x + 1), gammaln(y + 1), n_teams,
            np.zeros((n, 0)), np.zeros((n, 0)), np.zeros((n, 0)), penalty)


@pytest.mark.parametrize("promoted_index", [0, 3, 7])
def test_penalty_gradient_matches_finite_differences(promoted_index: int) -> None:
    """Including the LAST team, which is the case the sum-to-zero reparameterisation makes easy
    to get wrong: that club is not a free parameter, so a penalty on it must be felt by every
    free parameter with a minus sign. A wrong gradient here does not crash, it converges somewhere
    slightly wrong and every number downstream inherits it.
    """
    rng = np.random.default_rng(promoted_index)
    n_teams, n = 8, 400
    mask = np.zeros(n_teams, dtype=bool)
    mask[promoted_index] = True
    prior = PromotionPrior(attack=-0.30, defence=-0.27, n_clubs=9, effective_n=100.0, teams=())
    args = _gradient_args(rng, n_teams, n, (mask, prior, 2.5))

    theta = np.concatenate([
        [rng.uniform(-0.3, 0.5)], [rng.uniform(0.0, 0.4)], [rng.uniform(-0.12, 0.12)],
        rng.uniform(-0.5, 0.5, 2 * (n_teams - 1)),
    ])
    numeric = approx_fprime(theta, lambda t: _objective(t, *args)[0], 1e-7)
    analytic = _objective(theta, *args)[1]
    assert np.max(np.abs(numeric - analytic)) < 1e-3
    assert np.allclose(numeric, analytic, rtol=1e-3, atol=1e-4)


def test_the_last_slot_penalty_reaches_every_free_parameter() -> None:
    """Not just that the gradient is right, but that it is not zero — a penalty on the constrained
    team that produced no gradient at all would pass a tolerance check against its own mistake."""
    rng = np.random.default_rng(11)
    n_teams, n = 6, 200
    mask = np.zeros(n_teams, dtype=bool)
    mask[-1] = True
    prior = PromotionPrior(attack=-0.30, defence=-0.27, n_clubs=9, effective_n=100.0, teams=())
    theta = np.concatenate([[0.1], [0.25], [-0.05], rng.uniform(-0.4, 0.4, 2 * (n_teams - 1))])

    free = n_teams - 1
    off = _objective(theta, *_gradient_args(rng, n_teams, n, None))[1]
    rng = np.random.default_rng(11)  # same draw, so only the penalty differs
    on = _objective(theta, *_gradient_args(rng, n_teams, n, (mask, prior, 2.5)))[1]
    moved = np.abs(on - off)[_N_GLOBAL: _N_GLOBAL + 2 * free]
    assert np.all(moved > 0.0), "every free parameter must feel a penalty on the constrained team"


def test_zero_shrinkage_leaves_the_likelihood_untouched() -> None:
    """So the tuning grid's own zero point IS the baseline rather than something close to it."""
    rng = np.random.default_rng(5)
    n_teams, n = 7, 300
    mask = np.ones(n_teams, dtype=bool)
    prior = PromotionPrior(attack=-0.30, defence=-0.27, n_clubs=9, effective_n=100.0, teams=())
    theta = np.concatenate([[0.1], [0.25], [-0.05], rng.uniform(-0.4, 0.4, 2 * (n_teams - 1))])

    rng = np.random.default_rng(5)
    zero = _objective(theta, *_gradient_args(rng, n_teams, n, (mask, prior, 0.0)))
    rng = np.random.default_rng(5)
    absent = _objective(theta, *_gradient_args(rng, n_teams, n, None))
    assert zero[0] == absent[0]
    assert np.array_equal(zero[1], absent[1])


def test_penalty_is_zero_when_nobody_is_promoted() -> None:
    prior = PromotionPrior(attack=-0.30, defence=-0.27, n_clubs=9, effective_n=100.0, teams=())
    value, ga, gd = penalty_and_gradient(
        np.array([0.1, -0.2]), np.array([0.0, 0.3]), np.zeros(2, dtype=bool), prior, 3.0
    )
    assert value == 0.0 and not ga.any() and not gd.any()


def test_only_promoted_clubs_are_penalised() -> None:
    """What keeps this the promoted-club arm rather than general shrinkage."""
    prior = PromotionPrior(attack=-0.30, defence=-0.27, n_clubs=9, effective_n=100.0, teams=())
    mask = np.array([True, False])
    _, ga, gd = penalty_and_gradient(
        np.array([0.5, 0.5]), np.array([0.5, 0.5]), mask, prior, 1.0
    )
    assert ga[0] != 0.0 and gd[0] != 0.0
    assert ga[1] == 0.0 and gd[1] == 0.0


# --- the spec ------------------------------------------------------------------------------------

def test_negative_shrinkage_is_refused() -> None:
    with pytest.raises(PromotionError):
        PromotionSpec(shrinkage=-0.1, min_prior_clubs=3)


def test_min_prior_clubs_below_one_is_refused() -> None:
    with pytest.raises(PromotionError):
        PromotionSpec(shrinkage=1.0, min_prior_clubs=0)


# --- the two sites, end to end -------------------------------------------------------------------

def _FIT_KWARGS(ref):
    """The production bound names, read from config rather than retyped.

    Typing them out here is how the first version of this test failed: `attack`/`defence` are what
    the parameters are called, `strength` is what the bound is called, and a fixture that invents
    key names tests the fixture.
    """
    from plmodel.config import load_config

    return dict(half_life_days=730.0, ref_date=ref, max_goals=8,
                param_bounds=load_config().model.param_bounds,
                min_effective_share=0.15, max_iter=200)


def _corpus():
    seasons = {
        "1994-95": ["A", "B", "C", "D", "E"], "1995-96": ["A", "B", "C", "D", "F"],
        "1996-97": ["A", "B", "C", "D", "G"], "1997-98": ["A", "B", "C", "G", "H"],
    }
    return season_frame(seasons, goals={"E": (1, 3), "F": (1, 3), "G": (1, 3), "H": (1, 3)})


def test_the_seam_off_is_the_baseline_fit() -> None:
    df = _corpus()
    ref = pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
    common = _FIT_KWARGS(ref)
    base = fit_dixon_coles(df, **common)
    off = fit_dixon_coles(df, **common, promotion=None)
    assert np.array_equal(base.attack, off.attack)
    assert base.promotion_prior is None


def test_an_unknown_club_is_scored_at_the_prior_not_the_league_average() -> None:
    """Site A. The only mechanism that can reach a club with no top-flight rows at all — it never
    appears in the fit's team list, so it is not even a cold start."""
    df = _corpus()
    ref = pd.Timestamp(df["date"].iloc[-1]) + pd.Timedelta(days=1)
    common = _FIT_KWARGS(ref)
    fit = fit_dixon_coles(df, **common, promotion=SPEC)
    assert fit.promotion_prior is not None
    assert fit.promotion_prior.attack < 0.0

    newcomer = pd.DataFrame([{"home_team": "A", "away_team": "ZZZ_NEVER_SEEN",
                              "date": ref, "season": "1998-99"}])
    lam_seam, mu_seam = fit.rates(newcomer["home_team"], newcomer["away_team"], newcomer["date"])
    base = fit_dixon_coles(df, **common)
    lam_base, mu_base = base.rates(newcomer["home_team"], newcomer["away_team"], newcomer["date"])
    # The newcomer is weaker than league average under the prior: it scores less and concedes more.
    assert mu_seam[0] < mu_base[0]
    assert lam_seam[0] > lam_base[0]
