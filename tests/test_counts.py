"""The Weibull-count / Frank-copula scoreline family.

The tests that matter most here are the two *inertness points*. This family has two axes, and each
one has a value at which it must reduce exactly to something already trusted: the Weibull count at
shape 1 is the Poisson, and the Frank copula at kappa 0 is independence. If either identity is even
slightly wrong, every comparison the arm produces is measuring an implementation error rather than a
modelling choice, and it would be measuring it in the direction that flatters the new code.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from plmodel.model import dixon_coles as DC
from plmodel.model.counts import (
    FRANK,
    INDEPENDENT_KAPPA,
    POISSON,
    TAU,
    UNIT_SHAPE,
    WEIBULL,
    CountFamilyError,
    CountSpec,
    count_pmf,
    frank_copula,
    frank_partial_u,
    joint_grid,
    marginals,
    observed_cell_likelihood,
    series_kernel,
)
from plmodel.model.scoreline import scoreline_matrix

MAX_GOALS = 12
N_TERMS = 60
# Rates a Premier League fit actually produces. The production model's widest forecast over the
# test decade is 3.69, so this brackets the real working range with room on both sides.
FOOTBALL_RATES = np.array([0.30, 0.75, 1.10, 1.38, 1.75, 2.40, 3.20, 3.70])


def spec(marginal: str = POISSON, dependence: str = TAU) -> CountSpec:
    return CountSpec(marginal=marginal, dependence=dependence, n_series_terms=N_TERMS)


ALL_FAMILIES = [
    spec(POISSON, TAU),
    spec(POISSON, FRANK),
    spec(WEIBULL, TAU),
    spec(WEIBULL, FRANK),
]


# --- the inertness points ------------------------------------------------------------------------

def test_the_weibull_count_at_shape_one_is_the_poisson() -> None:
    """The first inertness point, and the one everything else rests on."""
    kernel = series_kernel(UNIT_SHAPE, MAX_GOALS, N_TERMS)
    pmf, _, _ = count_pmf(FOOTBALL_RATES, kernel)
    reference = np.vstack([poisson.pmf(np.arange(MAX_GOALS + 1), r) for r in FOOTBALL_RATES])
    assert np.abs(pmf - reference).max() < 1e-12


def test_the_frank_copula_at_zero_is_independence() -> None:
    """The second inertness point. Checked at exactly zero and either side of the limit switch."""
    u = np.array([0.05, 0.3, 0.62, 0.99])
    v = np.array([0.11, 0.5, 0.5, 0.4])
    assert np.allclose(frank_copula(u, v, INDEPENDENT_KAPPA), u * v, atol=0, rtol=0)
    for tiny in (1e-7, -1e-7, 1e-4, -1e-4):
        assert np.abs(frank_copula(u, v, tiny) - u * v).max() < 1e-4


def test_the_production_family_reproduces_the_production_scoreline() -> None:
    """poisson+tau through this module must agree with the Dixon-Coles grid it stands in for.

    Not byte-identical, and deliberately so: this family renormalises over the truncated goal grid
    and the production path does not, because the tau correction is exactly mass-preserving over the
    full support and so needs no constant. The gap IS that truncated tail, and at football rates it
    is a part in ten thousand.
    """
    lam = FOOTBALL_RATES
    mu = FOOTBALL_RATES[::-1]
    rho = -0.045
    ours = joint_grid(spec(), lam, mu, shape=UNIT_SHAPE, rho=rho,
                      kappa=INDEPENDENT_KAPPA, max_goals=MAX_GOALS)
    theirs = scoreline_matrix(lam, mu, rho, MAX_GOALS)
    assert np.abs(ours - theirs).max() < 1e-9


def test_tau_against_poisson_margins_needs_no_normalising_constant() -> None:
    """The identity the module docstring leans on, checked rather than asserted in prose.

    If this were false, the production likelihood would be optimising an unnormalised object and
    every fitted rate in the project would carry a bias.
    """
    lam, mu = FOOTBALL_RATES, FOOTBALL_RATES[::-1]
    for rho in (-0.12, -0.045, 0.0, 0.08):
        joint = scoreline_matrix(lam, mu, rho, 40)
        pmf_h = np.vstack([poisson.pmf(np.arange(41), r) for r in lam])
        pmf_a = np.vstack([poisson.pmf(np.arange(41), r) for r in mu])
        raw = pmf_h[:, :, None] * pmf_a[:, None, :]
        raw[:, 0, 0] *= 1.0 - lam * mu * rho
        raw[:, 0, 1] *= 1.0 + lam * rho
        raw[:, 1, 0] *= 1.0 + mu * rho
        raw[:, 1, 1] *= 1.0 - rho
        assert np.abs(raw.sum(axis=(1, 2)) - 1.0).max() < 1e-12
        assert np.abs(joint.sum(axis=(1, 2)) - 1.0).max() < 1e-12


# --- what each axis actually does -----------------------------------------------------------------

@pytest.mark.parametrize(
    "shape, direction",
    [(0.80, "over"), (0.92, "over"), (1.00, "equal"), (1.10, "under"), (1.30, "under")],
)
def test_the_shape_moves_dispersion_in_the_documented_direction(shape, direction) -> None:
    """Below 1 over-disperses, above 1 under-disperses. The whole point of the extra parameter."""
    kernel = series_kernel(shape, MAX_GOALS, N_TERMS)
    pmf, _, _ = count_pmf(np.array([1.4]), kernel)
    p = pmf[0] / pmf[0].sum()
    n = np.arange(MAX_GOALS + 1)
    mean = float((p * n).sum())
    variance = float((p * (n - mean) ** 2).sum())
    ratio = variance / mean
    if direction == "equal":
        # Not exactly 1: the grid is truncated at max_goals and renormalised, which moves the
        # variance by the mass beyond it. At a rate of 1.4 that mass is around 1e-9.
        assert abs(ratio - 1.0) < 1e-6
    elif direction == "over":
        assert ratio > 1.0
    else:
        assert ratio < 1.0


def test_the_copula_leaves_the_margins_exactly_where_it_found_them() -> None:
    """The structural property tau does not have, and the reason a copula is worth trying at all.

    tau lifts four cells and the grid has to be renormalised afterwards, which perturbs both
    margins. A copula moves mass around the table without touching either one.
    """
    lam, mu = FOOTBALL_RATES, FOOTBALL_RATES[::-1]
    reference = None
    for kappa in (-3.0, -0.5, 0.0, 0.5, 3.0):
        joint = joint_grid(spec(POISSON, FRANK), lam, mu, shape=UNIT_SHAPE, rho=0.0,
                           kappa=kappa, max_goals=MAX_GOALS)
        home_margin = joint.sum(axis=2)
        away_margin = joint.sum(axis=1)
        if reference is None:
            reference = (home_margin, away_margin)
        assert np.abs(home_margin - reference[0]).max() < 1e-12
        assert np.abs(away_margin - reference[1]).max() < 1e-12


def test_the_copula_sign_is_the_direction_of_dependence() -> None:
    """Positive kappa piles mass on the diagonal; negative kappa takes it off."""
    lam, mu = FOOTBALL_RATES, FOOTBALL_RATES[::-1]
    draws = {}
    for kappa in (-2.0, 0.0, 2.0):
        joint = joint_grid(spec(POISSON, FRANK), lam, mu, shape=UNIT_SHAPE, rho=0.0,
                           kappa=kappa, max_goals=MAX_GOALS)
        draws[kappa] = float(sum(joint[:, k, k].sum() for k in range(MAX_GOALS + 1)))
    assert draws[-2.0] < draws[0.0] < draws[2.0]


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.label())
def test_every_family_produces_a_proper_distribution(family) -> None:
    lam, mu = FOOTBALL_RATES, FOOTBALL_RATES[::-1]
    joint = joint_grid(family, lam, mu, shape=1.08, rho=-0.045, kappa=-0.3, max_goals=MAX_GOALS)
    assert joint.min() >= 0.0
    assert np.abs(joint.sum(axis=(1, 2)) - 1.0).max() < 1e-12


# --- the derivatives ------------------------------------------------------------------------------

@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.label())
def test_the_rate_derivatives_match_a_central_difference(family) -> None:
    rng = np.random.default_rng(11)
    n = 300
    lam = rng.uniform(0.4, 3.4, n)
    mu = rng.uniform(0.3, 2.8, n)
    x = rng.integers(0, 6, n).astype(float)
    y = rng.integers(0, 6, n).astype(float)
    shape, rho, kappa = 1.07, -0.045, -0.3
    step = 1e-6

    def probability(rate_home, rate_away):
        marg = marginals(family, rate_home, rate_away, shape=shape, max_goals=MAX_GOALS)
        return observed_cell_likelihood(family, marg, x, y, rho=rho, kappa=kappa)

    prob, d_lam, d_mu = probability(lam, mu)
    numeric_lam = (probability(lam * math.exp(step), mu)[0]
                   - probability(lam * math.exp(-step), mu)[0]) / (2 * step)
    numeric_mu = (probability(lam, mu * math.exp(step))[0]
                  - probability(lam, mu * math.exp(-step))[0]) / (2 * step)
    assert np.abs(d_lam - numeric_lam).max() / np.abs(numeric_lam).max() < 1e-6
    assert np.abs(d_mu - numeric_mu).max() / np.abs(numeric_mu).max() < 1e-6


@pytest.mark.parametrize("kappa", [-3.0, -0.4, 0.4, 3.0])
def test_the_copula_partial_matches_a_central_difference(kappa) -> None:
    u = np.array([0.13, 0.37, 0.55, 0.81, 0.96])
    v = np.array([0.22, 0.44, 0.5, 0.7, 0.9])
    step = 1e-7
    numeric = (frank_copula(u + step, v, kappa) - frank_copula(u - step, v, kappa)) / (2 * step)
    assert np.abs(frank_partial_u(u, v, kappa) - numeric).max() < 1e-7


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.label())
def test_the_fitted_gradient_matches_a_central_difference(family, corpus) -> None:
    """The ~100-long team block is analytic and must be right; the scalars are differenced.

    Compared against a *central* difference at a step chosen for this problem, not against
    ``approx_fprime``'s forward default -- the count series' own cancellation puts a noise floor on
    the objective, and a forward difference at 1e-7 sits below it and reports a disagreement that is
    entirely its own.
    """
    rows = corpus[corpus["season"] == "2015-16"].reset_index(drop=True)
    teams = sorted(set(rows["home_team"]) | set(rows["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    x = rows["home_goals"].to_numpy(dtype=float)
    y = rows["away_goals"].to_numpy(dtype=float)
    home_i = rows["home_team"].map(index).to_numpy()
    away_i = rows["away_team"].map(index).to_numpy()
    weights = np.ones(len(rows))
    n_teams = len(teams)
    ha_design = np.zeros((len(rows), 0))
    n_family = int(family.fits_shape) + int(family.fits_kappa)

    rng = np.random.default_rng(4)
    tail = ([0.05] if family.fits_shape else []) + ([-0.3] if family.fits_kappa else [])
    theta = np.concatenate([
        [0.2], [0.25], [-0.045 if family.fits_rho else 0.0],
        rng.normal(0.0, 0.2, 2 * (n_teams - 1)), tail,
    ])
    args = (x, y, home_i, away_i, weights, n_teams, ha_design, family, n_family, MAX_GOALS)

    analytic = DC._gradient_family(theta, *args)
    step = 1e-4
    numeric = np.zeros_like(theta)
    for k in range(len(theta)):
        up, down = theta.copy(), theta.copy()
        up[k] += step
        down[k] -= step
        numeric[k] = (DC._objective_family(up, *args)
                      - DC._objective_family(down, *args)) / (2 * step)
    # The Weibull shape is the one parameter differenced on both sides of the comparison, at
    # different steps, so it is held to a looser bar than the analytic block.
    rate_block = slice(0, len(theta) - n_family)
    relative = np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1.0)
    assert relative[rate_block].max() < 1e-4
    assert relative.max() < 1e-2


def test_value_and_gradient_agree_with_the_value_only_path(corpus) -> None:
    """The combined callable handed to the optimiser must not have drifted from the plain one."""
    rows = corpus[corpus["season"] == "2015-16"].reset_index(drop=True)
    teams = sorted(set(rows["home_team"]) | set(rows["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    family = spec(WEIBULL, FRANK)
    args = (
        rows["home_goals"].to_numpy(dtype=float),
        rows["away_goals"].to_numpy(dtype=float),
        rows["home_team"].map(index).to_numpy(),
        rows["away_team"].map(index).to_numpy(),
        np.ones(len(rows)),
        len(teams),
        np.zeros((len(rows), 0)),
        family,
        2,
        MAX_GOALS,
    )
    rng = np.random.default_rng(9)
    theta = np.concatenate([[0.2], [0.25], [0.0],
                            rng.normal(0.0, 0.2, 2 * (len(teams) - 1)), [0.04, -0.25]])
    value, _ = DC._value_and_gradient_family(theta, *args)
    assert value == pytest.approx(DC._objective_family(theta, *args), rel=1e-12)


# --- the numerical guards -------------------------------------------------------------------------

def test_the_series_is_trusted_at_football_rates_and_not_beyond() -> None:
    """The condition number is the guard, and it has to actually discriminate.

    A guard that passed everything would let the optimiser fit to noise; one that failed at
    football rates would make the family unusable. Both halves are asserted.
    """
    family = spec()
    football = marginals(family, FOOTBALL_RATES, FOOTBALL_RATES,
                         shape=UNIT_SHAPE, max_goals=MAX_GOALS)
    assert football.is_trustworthy
    assert football.condition < 1e5

    absurd = np.array([14.0, 16.0])
    assert not marginals(family, absurd, absurd, shape=UNIT_SHAPE,
                         max_goals=MAX_GOALS).is_trustworthy


def test_an_untrustworthy_series_raises_rather_than_returning_numbers() -> None:
    with pytest.raises(CountFamilyError, match="cancelled"):
        joint_grid(spec(), np.array([18.0]), np.array([18.0]), shape=UNIT_SHAPE,
                   rho=0.0, kappa=INDEPENDENT_KAPPA, max_goals=MAX_GOALS)


def test_truncating_the_series_too_early_is_visible() -> None:
    """If the length made no difference, the configured value would be decorative.

    Ten terms is not enough at a rate of 3.7 and the pmf comes back wrong; sixty is, and it agrees
    with the Poisson. The test exists so that a future change lowering the default has to argue
    with something.
    """
    rate = np.array([3.7])
    reference = poisson.pmf(np.arange(MAX_GOALS + 1), 3.7)
    short, _, _ = count_pmf(rate, series_kernel(UNIT_SHAPE, MAX_GOALS, 10))
    full, _, _ = count_pmf(rate, series_kernel(UNIT_SHAPE, MAX_GOALS, N_TERMS))
    assert np.abs(short[0] - reference).max() > 1e-3
    assert np.abs(full[0] - reference).max() < 1e-12


def test_negative_probabilities_from_a_broken_series_are_caught() -> None:
    """The invariant that actually matters: these are probabilities."""
    broken = marginals(spec(), np.array([30.0]), np.array([30.0]),
                       shape=UNIT_SHAPE, max_goals=MAX_GOALS)
    assert not broken.valid
    assert not broken.is_trustworthy


def test_the_cached_kernel_cannot_be_corrupted_by_a_caller() -> None:
    kernel = series_kernel(1.05, MAX_GOALS, N_TERMS)
    with pytest.raises(ValueError):
        kernel[0, 0] = 999.0


# --- the specification object ---------------------------------------------------------------------

def test_the_production_pair_is_the_inert_one() -> None:
    assert spec(POISSON, TAU).is_inert
    assert not spec(POISSON, FRANK).is_inert
    assert not spec(WEIBULL, TAU).is_inert
    assert not spec(WEIBULL, FRANK).is_inert


def test_each_axis_declares_which_parameters_it_fits() -> None:
    assert spec(WEIBULL, TAU).fits_shape and not spec(POISSON, TAU).fits_shape
    assert spec(POISSON, FRANK).fits_kappa and not spec(POISSON, TAU).fits_kappa
    assert spec(POISSON, TAU).fits_rho and not spec(POISSON, FRANK).fits_rho


@pytest.mark.parametrize(
    "kwargs",
    [
        {"marginal": "negbin"},
        {"dependence": "gaussian"},
        {"n_series_terms": 0},
    ],
)
def test_an_unrepresentable_family_raises(kwargs) -> None:
    base = {"n_series_terms": N_TERMS}
    with pytest.raises(CountFamilyError):
        CountSpec(**{**base, **kwargs})


def test_the_series_length_has_no_default() -> None:
    """It is an accuracy choice and belongs in config.yaml, not in a dataclass default."""
    with pytest.raises(CountFamilyError, match="no default"):
        CountSpec()


# --- the fit -------------------------------------------------------------------------------------

def _fit(rows: pd.DataFrame, cfg, family: CountSpec | None):
    model = cfg.model
    return DC.fit_dixon_coles(
        rows,
        half_life_days=model.decay_half_life_days,
        ref_date=rows["date"].max() + pd.Timedelta(days=1),
        max_goals=model.max_goals,
        param_bounds=model.param_bounds,
        min_effective_share=model.min_effective_share,
        max_iter=model.max_iter,
        family=family,
    )


@pytest.mark.parametrize("family", ALL_FAMILIES, ids=lambda f: f.label())
def test_every_family_fits_and_reports_its_own_parameters(family, cfg, corpus) -> None:
    rows = corpus[corpus["season"].isin(["2013-14", "2014-15", "2015-16"])].reset_index(drop=True)
    fit = _fit(rows, cfg, family)
    assert fit.converged
    summary = fit.as_dict()
    assert summary["scoreline_family"] == family.label()
    # Whichever axis is off must sit exactly at its inert value rather than drift.
    if not family.fits_shape:
        assert fit.shape == UNIT_SHAPE
    if not family.fits_kappa:
        assert fit.kappa == INDEPENDENT_KAPPA
    if not family.fits_rho:
        assert fit.rho == 0.0
    assert fit.diagnostics["series_condition"] < 1e6
    assert fit.diagnostics["tail_deficit"] < 1e-2


def test_the_family_path_at_its_inert_setting_recovers_the_production_fit(cfg, corpus) -> None:
    """An independent implementation of the same model, agreeing to the truncated tail.

    This is the check that would catch a sign error, a transposed index or a mis-scaled weight
    anywhere in several hundred lines of new numerics, because the production path is known good and
    was not touched.
    """
    rows = corpus[corpus["season"].isin(["2013-14", "2014-15", "2015-16"])].reset_index(drop=True)
    production = _fit(rows, cfg, None)
    through_family = _fit(rows, cfg, spec(POISSON, TAU))
    assert through_family.rho == pytest.approx(production.rho, abs=1e-3)
    assert through_family.home_advantage == pytest.approx(production.home_advantage, abs=1e-3)
    assert through_family.intercept == pytest.approx(production.intercept, abs=1e-3)
    assert np.abs(np.asarray(through_family.attack) - np.asarray(production.attack)).max() < 5e-3


def test_a_family_fit_predicts_probabilities_that_differ_from_the_baseline(cfg, corpus) -> None:
    """The harness contract, at the level of a single fit: an arm must do *something*."""
    rows = corpus[corpus["season"].isin(["2013-14", "2014-15", "2015-16"])].reset_index(drop=True)
    held_out = corpus[corpus["season"] == "2016-17"].head(60).reset_index(drop=True)
    production = _fit(rows, cfg, None).predict_proba(held_out)
    for family in (spec(POISSON, FRANK), spec(WEIBULL, TAU), spec(WEIBULL, FRANK)):
        moved = _fit(rows, cfg, family).predict_proba(held_out)
        assert np.abs(moved - production).max() > 1e-4, f"{family.label()} changed nothing"
        assert np.abs(moved.sum(axis=1) - 1.0).max() < 1e-12


# --- through the harness --------------------------------------------------------------------------

@pytest.mark.integration
def test_the_arms_run_the_walk_and_each_one_moves(cfg, corpus) -> None:
    """The harness contract for this family: every arm differs, and the baseline does not.

    A broken experiment returning "no effect" is otherwise indistinguishable from a correct one
    reporting a genuine null, and this arm was pre-registered as an expected null -- which is
    exactly the situation where that confusion is most expensive.
    """
    import hashlib

    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )

    def digest(probs: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()

    baseline, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    for name in ("dc-copula", "dc-weibull", "dc-weibull-copula"):
        probs, state = run_arm(ArmSpec.parse(name), corpus, splits, cfg)
        assert probs.shape == baseline.shape
        assert np.abs(probs - baseline).max() > 1e-3, f"{name} is indistinguishable from baseline"
        assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12
        fits = state["fits"]
        assert all(f.converged for f in fits)
        assert all(f.diagnostics["series_condition"] < 1e6 for f in fits)

    again, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    assert digest(again) == digest(baseline), "the baseline moved while the family arms ran"


@pytest.mark.integration
def test_the_report_carries_the_family_parameters(cfg, corpus) -> None:
    """A null is only readable if the report says which parameter values produced it."""
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, _fit_summary, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )
    _, state = run_arm(ArmSpec.parse("dc-weibull-copula"), corpus, splits, cfg)
    block = _fit_summary(state)["scoreline_family"]
    assert block["family"] == "weibull+frank"
    assert 0.6 < block["weibull_shape"]["mean"] < 1.7
    assert -5.0 < block["frank_kappa"]["mean"] < 5.0
    assert block["worst_series_condition"] < 1e6
    assert block["worst_tail_deficit"] < 1e-2

    _, plain = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    assert "scoreline_family" not in _fit_summary(plain)
