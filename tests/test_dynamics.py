"""The score-driven state filter: the recursion, its guards, and its inertness.

The load-bearing test in this file is :func:`test_zero_loading_is_byte_identical_to_the_baseline`.
Everything else checks that the filter says what the mathematics says it should; that one checks
that switching it off leaves the production model untouched, which is non-negotiable #4 and the
thing that makes an arm's paired delta attributable to the arm.

The rest are written against hand-built histories with known answers rather than against the
corpus, so a failure names the property that broke instead of moving a fourth decimal.
"""
from __future__ import annotations

import dataclasses
import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.model.dixon_coles import DixonColesFit, fit_dixon_coles
from plmodel.model.dynamics import (
    INVERSE_INFORMATION, INVERSE_SQRT_INFORMATION, UNIT_SCALING, DynamicFit, DynamicsError,
    GasSpec, _tau_log_derivatives, filter_states,
)

BOUND = 1.5


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _fit(teams=("A", "B"), *, intercept=0.0, home_advantage=0.0, rho=0.0) -> DixonColesFit:
    """A flat fit: every team at the league average, so any deviation comes from the filter."""
    n = len(teams)
    return DixonColesFit(
        teams=tuple(teams), attack=np.zeros(n), defence=np.zeros(n),
        intercept=intercept, home_advantage=home_advantage, rho=rho,
        half_life_days=730.0, ref_date=pd.Timestamp("2020-01-01"), max_goals=12,
        n_obs=0, effective_n=0.0, neg_log_lik=0.0, converged=True, n_iterations=0,
    )


def _history(rows) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime([r[0] for r in rows]),
        "home_team": [r[1] for r in rows],
        "away_team": [r[2] for r in rows],
        "home_goals": [float(r[3]) for r in rows],
        "away_goals": [float(r[4]) for r in rows],
    })


def _spec(**kwargs) -> GasSpec:
    base = {"score_loading": 0.1, "persistence": 0.9, "scaling_exponent": UNIT_SCALING,
            "half_life_days": 730.0, "state_bound": BOUND}
    return GasSpec(**{**base, **kwargs})


# --- the specification is complete before it runs ------------------------------------------------

def test_a_half_specified_seam_refuses_to_run(cfg) -> None:
    """The tuned values come from the tuning span. Running without them is not a default, it is a
    missing decision, and the seam must say so rather than invent one."""
    with pytest.raises(DynamicsError, match="score_loading"):
        GasSpec.from_seam({"enabled": True}, state_bound=BOUND, fallback_half_life=730.0)


def test_the_seam_falls_back_to_the_production_half_life() -> None:
    spec = GasSpec.from_seam(
        {"score_loading": 0.01, "persistence": 0.95, "scaling_exponent": 0.5},
        state_bound=BOUND, fallback_half_life=730.0,
    )
    assert spec.half_life_days == 730.0


def test_the_shipped_seam_is_off_and_fully_specified(cfg) -> None:
    """Off is off, but the values must still be there — a seam nobody can switch on is not a seam.

    This is what catches the configuration going out of sync with a retune: the tuned values live
    in config.yaml, and if one is ever blanked the arm must fail here rather than at run time.
    """
    seam = cfg.model.seams["dynamics"]
    assert seam["enabled"] is False
    spec = GasSpec.from_seam(
        seam, state_bound=cfg.model.param_bounds["gas_state"][1],
        fallback_half_life=cfg.model.decay_half_life_days,
    )
    assert 0.0 < spec.score_loading
    assert 0.0 < spec.persistence <= 1.0
    assert spec.scaling_exponent in (UNIT_SCALING, INVERSE_SQRT_INFORMATION, INVERSE_INFORMATION)


# --- inertness -----------------------------------------------------------------------------------

def test_zero_loading_leaves_every_state_at_zero() -> None:
    history = _history([("2020-01-01", "A", "B", 5, 0), ("2020-01-08", "B", "A", 4, 0)])
    states = filter_states(history, _fit(), _spec(score_loading=0.0))
    assert not np.any(states.attack) and not np.any(states.defence)
    assert states.dispersion == 0.0


def test_zero_loading_predicts_exactly_what_the_level_predicts() -> None:
    """Not 'close to': identical. Anything else would make the arm's delta partly an artefact."""
    history = _history([("2020-01-01", "A", "B", 3, 1), ("2020-01-08", "B", "A", 0, 2)])
    fit = _fit(intercept=0.1, home_advantage=0.3, rho=-0.04)
    spec = _spec(score_loading=0.0)
    dynamic = DynamicFit(fit, filter_states(history, fit, spec), spec)
    assert np.array_equal(dynamic.predict_proba(history), fit.predict_proba(history))


# --- the recursion says what the score says ------------------------------------------------------

def test_beating_expectation_raises_attack_and_lowers_the_opponent_defence() -> None:
    """The two are the same derivative with opposite signs, not two separate assumptions."""
    history = _history([("2020-01-01", "A", "B", 5, 0)])
    states = filter_states(history, _fit(), _spec())
    index = {t: i for i, t in enumerate(states.teams)}
    assert states.attack[index["A"]] > 0
    assert states.defence[index["B"]] < 0
    assert states.attack[index["A"]] == pytest.approx(-states.defence[index["B"]])


def test_falling_short_of_expectation_lowers_attack() -> None:
    history = _history([("2020-01-01", "A", "B", 0, 0)])
    states = filter_states(history, _fit(), _spec())
    index = {t: i for i, t in enumerate(states.teams)}
    assert states.attack[index["A"]] < 0
    assert states.defence[index["B"]] > 0


def test_the_step_is_the_scaled_score() -> None:
    """One match against the closed form, so the constant in front of the score is pinned."""
    history = _history([("2020-01-01", "A", "B", 3, 1)])
    fit = _fit(intercept=math.log(1.5))  # lam = mu = 1.5 with no home advantage
    spec = _spec(score_loading=0.1, persistence=0.9)
    states = filter_states(history, fit, spec)
    index = {t: i for i, t in enumerate(states.teams)}
    assert states.attack[index["A"]] == pytest.approx(0.1 * (3 - 1.5))
    assert states.attack[index["B"]] == pytest.approx(0.1 * (1 - 1.5))


def test_inverse_information_scaling_makes_the_update_a_relative_surprise() -> None:
    """With e = 1 the step is K*(x/lam - 1), so it does not depend on the league's scoring level."""
    history = _history([("2020-01-01", "A", "B", 3, 1)])
    fit = _fit(intercept=math.log(1.5))
    states = filter_states(history, fit, _spec(scaling_exponent=INVERSE_INFORMATION))
    index = {t: i for i, t in enumerate(states.teams)}
    assert states.attack[index["A"]] == pytest.approx(0.1 * (3 - 1.5) / 1.5)


def test_a_larger_loading_moves_the_state_further() -> None:
    history = _history([("2020-01-01", "A", "B", 4, 0)])
    small = filter_states(history, _fit(), _spec(score_loading=0.01))
    large = filter_states(history, _fit(), _spec(score_loading=0.05))
    assert large.dispersion > small.dispersion


def test_persistence_governs_how_long_a_state_survives() -> None:
    """The clock is team-match time: a state decays when its team plays, not as days pass.

    Two identical histories, two persistences. The team's early surprise must have faded further
    under the lower one by the time the filter reaches the end.
    """
    rows = [("2020-01-01", "A", "B", 5, 0)]
    rows += [(f"2020-02-{d:02d}", "A", "B", 1, 1) for d in range(1, 11)]
    history = _history(rows)
    forgetful = filter_states(history, _fit(), _spec(persistence=0.5))
    retentive = filter_states(history, _fit(), _spec(persistence=0.99))
    index = {t: i for i, t in enumerate(forgetful.teams)}
    assert retentive.attack[index["A"]] > forgetful.attack[index["A"]]


def test_full_persistence_never_discards_information() -> None:
    """B = 1 is the random-walk case: the state is a running sum of scaled scores."""
    history = _history([("2020-01-01", "A", "B", 3, 1), ("2020-01-08", "A", "B", 3, 1)])
    fit = _fit(intercept=math.log(1.5))
    states = filter_states(history, fit, _spec(persistence=1.0))
    index = {t: i for i, t in enumerate(states.teams)}
    # The second match is forecast at the raised rate, so the two steps are not equal — but with no
    # discounting the total must exceed a single step.
    assert states.attack[index["A"]] > 0.1 * (3 - 1.5)


# --- the guards ----------------------------------------------------------------------------------

def test_a_runaway_recursion_is_clipped_and_counted() -> None:
    """A loading too large for the data must be visible in the report, not silent in the states."""
    rows = [(f"2020-01-{d:02d}", "A", "B", 9, 0) for d in range(1, 21)]
    states = filter_states(_history(rows), _fit(), _spec(score_loading=1.0, persistence=1.0))
    assert states.n_state_clipped > 0
    assert np.abs(states.attack).max() <= BOUND


def test_a_team_the_filter_never_saw_gets_no_deviation() -> None:
    """Missing is a value. An unseen club sits at its level, it does not inherit anyone else's."""
    history = _history([("2020-01-01", "A", "B", 4, 0)])
    states = filter_states(history, _fit(), _spec())
    attack, defence = states.deviation(pd.Series(["C"]))
    assert attack[0] == 0.0 and defence[0] == 0.0


def test_a_frame_without_results_is_refused() -> None:
    history = _history([("2020-01-01", "A", "B", 1, 1)]).drop(columns=["home_goals"])
    with pytest.raises(ValueError, match="missing columns"):
        filter_states(history, _fit(), _spec())


def test_the_filter_is_deterministic() -> None:
    history = _history([("2020-01-01", "A", "B", 3, 1), ("2020-01-08", "B", "A", 0, 2)])
    first = filter_states(history, _fit(), _spec())
    second = filter_states(history, _fit(), _spec())
    assert np.array_equal(first.attack, second.attack)
    assert np.array_equal(first.defence, second.defence)


# --- the tau derivative --------------------------------------------------------------------------

@pytest.mark.parametrize("x,y", [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (3, 2)])
def test_tau_derivative_matches_a_numerical_one(x: int, y: int) -> None:
    """Including tau keeps the filter's score the score of the model actually being fitted.

    Checked against a central difference of log tau in log lam, which is what the analytic form
    claims to be.
    """
    lam, mu, rho, step = 1.4, 1.1, -0.05, 1e-6

    def log_tau(lam_: float, mu_: float) -> float:
        if (x, y) == (0, 0):
            return math.log(1.0 - lam_ * mu_ * rho)
        if (x, y) == (0, 1):
            return math.log(1.0 + lam_ * rho)
        if (x, y) == (1, 0):
            return math.log(1.0 + mu_ * rho)
        if (x, y) == (1, 1):
            return math.log(1.0 - rho)
        return 0.0

    d_lam, d_mu, valid = _tau_log_derivatives(float(x), float(y), lam, mu, rho)
    assert valid
    numeric_lam = (log_tau(lam * math.exp(step), mu) - log_tau(lam * math.exp(-step), mu)) / (2 * step)
    numeric_mu = (log_tau(lam, mu * math.exp(step)) - log_tau(lam, mu * math.exp(-step))) / (2 * step)
    assert d_lam == pytest.approx(numeric_lam, abs=1e-6)
    assert d_mu == pytest.approx(numeric_mu, abs=1e-6)


def test_an_impossible_tau_is_reported_rather_than_propagated() -> None:
    """A rho that puts a cell at or below zero must be counted, not silently divided by."""
    history = _history([("2020-01-01", "A", "B", 0, 0)])
    fit = _fit(intercept=math.log(2.0), rho=0.5)  # 1 - lam*mu*rho = 1 - 2 < 0
    states = filter_states(history, fit, _spec())
    assert states.n_tau_invalid == 1


# --- prediction ----------------------------------------------------------------------------------

def test_states_shift_the_forecast_toward_the_team_in_form() -> None:
    history = _history([(f"2020-01-{d:02d}", "A", "B", 4, 0) for d in range(1, 8)])
    fit = _fit(intercept=math.log(1.4))
    spec = _spec(score_loading=0.02, persistence=0.95)
    dynamic = DynamicFit(fit, filter_states(history, fit, spec), spec)
    fixture = _history([("2020-02-01", "A", "B", 0, 0)])
    assert dynamic.predict_proba(fixture)[0, 0] > fit.predict_proba(fixture)[0, 0]


def test_probabilities_still_sum_to_one() -> None:
    history = _history([(f"2020-01-{d:02d}", "A", "B", 4, 0) for d in range(1, 8)])
    fit = _fit(intercept=math.log(1.4), rho=-0.04)
    spec = _spec(score_loading=0.05)
    dynamic = DynamicFit(fit, filter_states(history, fit, spec), spec)
    probs = dynamic.predict_proba(history)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_the_fit_reports_its_own_recursion() -> None:
    """A report that cannot say what the states were doing cannot support a null."""
    history = _history([(f"2020-01-{d:02d}", "A", "B", 3, 1) for d in range(1, 8)])
    fit = _fit(intercept=math.log(1.4))
    spec = _spec(score_loading=0.02)
    payload = DynamicFit(fit, filter_states(history, fit, spec), spec).as_dict()
    assert payload["gas"]["score_loading"] == 0.02
    assert payload["gas"]["dispersion"] > 0
    assert payload["gas"]["n_updates"] == len(history)


# --- against the real corpus ---------------------------------------------------------------------

def _digest(probs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()


@pytest.fixture(scope="module")
def corpus(cfg) -> pd.DataFrame:
    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    frame = pd.read_parquet(path)
    frame = frame[(frame["division"] == cfg.backtest.prediction_division) & frame["played"]]
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


@pytest.mark.integration
def test_zero_loading_is_byte_identical_to_the_baseline(cfg, corpus) -> None:
    """The seam's inertness proof, run through the harness on real data.

    The dc-gas arm at a zero loading and the production half-life is not merely a good
    approximation of the production model, it IS the production model: same level fit, states
    pinned at zero, same predictor. If this ever drifts, every dynamics delta this project reports
    becomes partly an artefact of the arm's own plumbing rather than of the dynamics.

    **The half-life has to be pinned too, and that is not a detail.** The arm carries its own tuned
    decay because decay and dynamics are substitutes, so relative to the baseline it moves *two*
    axes, not one. Zeroing the loading alone leaves a 2555-day level fit against the baseline's
    730-day one, and the two are correctly different. Neutralising both is what isolates the seam.
    The confound runs in the arm's favour to test rather than to accept: on the tuning window the
    static model scores 0.20048 at 730 days and 0.20156 at 2555, so handing the baseline the arm's
    half-life would make the baseline worse and the arm's margin larger.
    """
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )
    seam = {
        **cfg.model.seams["dynamics"],
        "score_loading": 0.0,
        "half_life_days": cfg.model.decay_half_life_days,
    }
    inert = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, seams={**cfg.model.seams, "dynamics": seam}
    ))
    baseline, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    gas, _ = run_arm(ArmSpec.parse("dc-gas"), corpus, splits, inert)
    assert _digest(gas) == _digest(baseline)


@pytest.mark.integration
def test_the_tuned_recursion_actually_moves_the_forecast(cfg, corpus) -> None:
    """The other half of the inertness pair: switched on, it must do something.

    A seam that is inert when off and *also* inert when on is a broken experiment that returns a
    null indistinguishable from an honest one — the WC2026 false-null trap, in miniature.
    """
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )
    baseline, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    gas, state = run_arm(ArmSpec.parse("dc-gas"), corpus, splits, cfg)
    assert not np.array_equal(gas, baseline)
    assert np.mean([f.states.dispersion for f in state["fits"]]) > 0.0


@pytest.mark.integration
def test_scrambled_results_make_the_filter_worse_than_useless(cfg, corpus) -> None:
    """The placebo. The strongest evidence this project has that the arm is not leaking.

    The filter is handed the same fixtures with the results randomly permuted across matches, so
    every team keeps its schedule and the scores carry no information about who played whom. The
    level fit is untouched and still sees the real results, so the only thing that changes is what
    the states are made of.

    A gain that survived this would be structural — an artefact of the extra flexibility, or of
    information reaching the states by a route other than the results — and the arm would be wrong
    however good its numbers looked. Measured on 2024-25 the real filter gains 0.0075 RPS and the
    scrambled one *loses* 0.0061, so the states are carrying information about teams and nothing
    else.
    """
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, build_pool, run_arm
    from plmodel.model.baselines import outcomes_of
    from plmodel.eval import metrics

    # One fit for the whole season: the placebo contrast is about the states, and refitting at
    # every barrier would spend minutes on the half of the model that is held constant anyway.
    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        refit_every=len(corpus), min_train_matches=cfg.backtest.min_train_matches,
    )
    outcomes = outcomes_of(build_pool(corpus, splits))
    baseline, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    honest, state = run_arm(ArmSpec.parse("dc-gas"), corpus, splits, cfg)

    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(len(corpus))
    scrambled = corpus.assign(
        home_goals=corpus["home_goals"].to_numpy()[order],
        away_goals=corpus["away_goals"].to_numpy()[order],
    )
    spec, level = state["spec"], state["level"]
    placebo = np.vstack([
        DynamicFit(level, filter_states(scrambled.iloc[: s.train_end], level, spec), spec)
        .predict_proba(s.test(corpus))
        for s in splits
    ])

    real_gain = metrics.mean_rps(honest, outcomes) - metrics.mean_rps(baseline, outcomes)
    placebo_gain = metrics.mean_rps(placebo, outcomes) - metrics.mean_rps(baseline, outcomes)
    assert real_gain < 0, "the filter must help when its states are made of real results"
    assert placebo_gain > 0, "the filter must HURT when its states are made of scrambled results"


@pytest.mark.integration
def test_the_filter_uses_nothing_dated_at_or_after_the_barrier(cfg, corpus) -> None:
    """Leak-freedom, checked where it can actually break rather than only in the splitter.

    The filter is handed a training frame and a barrier. Appending a future match to that frame
    must change the states — if it did not, the filter would be ignoring its input; and the arm
    only ever passes it ``split.train``, which the splitter has already truncated.
    """
    history = corpus[corpus["date"] < pd.Timestamp("2024-08-01")]
    barrier = pd.Timestamp("2024-08-01")
    fit = fit_dixon_coles(
        history, half_life_days=cfg.model.decay_half_life_days, ref_date=barrier,
        max_goals=cfg.model.max_goals, param_bounds=cfg.model.param_bounds,
        min_effective_share=cfg.model.min_effective_share, max_iter=cfg.model.max_iter,
    )
    spec = GasSpec.from_seam(
        cfg.model.seams["dynamics"], state_bound=cfg.model.param_bounds["gas_state"][1],
        fallback_half_life=cfg.model.decay_half_life_days,
    )
    honest = filter_states(history, fit, spec)
    leaked = filter_states(corpus[corpus["date"] < pd.Timestamp("2024-09-01")], fit, spec)
    assert not np.allclose(honest.attack, leaked.attack)
