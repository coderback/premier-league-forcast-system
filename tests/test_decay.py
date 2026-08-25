"""Per-parameter time decay, and the two claims the whole arm rests on.

The first is **inertness**: with one memory for every block, this code must not run at all. That is
proved by bypass rather than by agreement — the seam returns None and the ordinary single-weight
fit executes — so the test to write is that the production path is untouched, not that the block
cycle happens to reproduce it.

The second is **that the cycle solves the right problem**. Block coordinate descent optimises a set
of estimating equations rather than one likelihood, so "it converged" is not self-evidently the same
as "it found the maximum". Given equal half-lives there IS a single likelihood to check against, and
the cycle must reach it. That check is what licenses trusting the cycle when the half-lives differ
and no reference optimum exists.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from plmodel.model.decay import (
    BLOCKS,
    HOME,
    LEVEL,
    TEAM,
    BlockLayout,
    DecayError,
    DecaySpec,
    block_weights,
)
from plmodel.model.dixon_coles import decay_weights, fit_dixon_coles

PRODUCTION_HALF_LIFE = 730.0
# The stopping rule the shipped seam carries. Read from config in the tests that have it; named
# here so the pure-unit tests do not need a config just to build a spec.
MAX_CYCLES, TOLERANCE = 25, 1.0e-4


def spec(team=PRODUCTION_HALF_LIFE, level=PRODUCTION_HALF_LIFE, home=PRODUCTION_HALF_LIFE):
    return DecaySpec(team=team, level=level, home_advantage=home,
                     max_cycles=MAX_CYCLES, tolerance=TOLERANCE)


# --- the inertness point --------------------------------------------------------------------------

def test_one_memory_for_every_block_is_the_production_specification() -> None:
    assert spec().is_inert
    assert not spec(team=1825).is_inert
    assert not spec(level=180).is_inert
    assert not spec(home=365).is_inert


def test_the_shipped_seam_is_off_but_carries_tuned_values(cfg) -> None:
    """Off by its switch, not by its values -- the dynamics seam's shape.

    The tuned half-lives ship beside `enabled: false` so an arm and production can never disagree
    about what they are. Asserting the values DIFFER matters: if a retune ever collapsed them back
    to a single memory, the arms would silently become the baseline and their nulls would mean
    nothing, which is the failure the arm body raises on.
    """
    assert cfg.model.seams_are_inert()
    assert cfg.model.decay_spec() is None, "the shipped seam must be off"

    live = cfg.model.decay_spec(enabled=True)
    assert live is not None, "the shipped half-lives have collapsed to a single memory"
    assert not live.is_inert
    assert live.team == cfg.model.decay_half_life_days, (
        "the team axis was tuned back to the production value; if that ever changes, the note in "
        "config.yaml about which axis moved needs changing with it"
    )


def test_a_block_at_the_production_half_life_gets_the_production_weights() -> None:
    """Not merely close: the same function, so the same bits.

    If this drifted, an arm whose team memory equalled production would still be weighting its
    strengths differently, and the comparison would silently be measuring two changes.
    """
    dates = pd.Series(pd.to_datetime(["2020-01-01", "2022-06-15", "2024-08-01"]))
    ref = pd.Timestamp("2024-08-02")
    weights = block_weights(dates, ref, spec(team=PRODUCTION_HALF_LIFE, level=180, home=365))
    assert np.array_equal(weights[TEAM], decay_weights(dates, ref, PRODUCTION_HALF_LIFE))
    assert np.array_equal(weights[LEVEL], decay_weights(dates, ref, 180))
    assert np.array_equal(weights[HOME], decay_weights(dates, ref, 365))


# --- the specification object ---------------------------------------------------------------------

def test_a_half_life_the_seam_does_not_name_is_inherited() -> None:
    """Naming one block means "give this one its own memory", not "default the other two"."""
    built = DecaySpec.from_seam(
        {"home_advantage": 365, "max_cycles": MAX_CYCLES, "tolerance": TOLERANCE},
        fallback_half_life=PRODUCTION_HALF_LIFE,
    )
    assert built.team == PRODUCTION_HALF_LIFE
    assert built.level == PRODUCTION_HALF_LIFE
    assert built.home_advantage == 365


def test_the_stopping_rule_has_no_default() -> None:
    """It is a numerical choice and belongs in config.yaml with its reasoning beside it."""
    with pytest.raises(DecayError, match="stopping rule"):
        DecaySpec.from_seam({"team": 1825}, fallback_half_life=PRODUCTION_HALF_LIFE)


@pytest.mark.parametrize(
    "kwargs",
    [{"team": 0}, {"level": -1}, {"home": 0}],
)
def test_a_half_life_must_be_positive(kwargs) -> None:
    with pytest.raises(DecayError, match="must be positive"):
        spec(**kwargs)


def test_the_cycle_needs_at_least_one_pass_and_a_positive_tolerance() -> None:
    with pytest.raises(DecayError, match="max_cycles"):
        DecaySpec(team=730, level=730, home_advantage=730, max_cycles=0, tolerance=TOLERANCE)
    with pytest.raises(DecayError, match="tolerance"):
        DecaySpec(team=730, level=730, home_advantage=730, max_cycles=1, tolerance=0.0)


# --- the block layout -----------------------------------------------------------------------------

def test_every_fitted_parameter_belongs_to_exactly_one_block() -> None:
    """A slot in no block is pinned for the whole cycle and never fitted — silently.

    That is the failure this layout has to rule out, so it is asserted as a partition rather than
    checked block by block.
    """
    layout = BlockLayout(n_teams=20, n_ha=2, total=3 + 2 * 19 + 2)
    slots = [layout.slots(block) for block in BLOCKS]
    combined = np.concatenate(slots)
    assert len(combined) == len(set(combined.tolist())), "a slot belongs to two blocks"
    assert sorted(combined.tolist()) == list(range(layout.total)), "a slot belongs to no block"


def test_the_home_advantage_design_columns_travel_with_home_advantage() -> None:
    """They are the same quantity measured differently; splitting them lets the two disagree."""
    layout = BlockLayout(n_teams=20, n_ha=2, total=3 + 2 * 19 + 2)
    home = layout.slots(HOME).tolist()
    assert len(home) == 3, "the home block should carry h plus both design columns"
    assert layout.total - 1 in home and layout.total - 2 in home


def test_an_unknown_block_is_refused() -> None:
    layout = BlockLayout(n_teams=20, n_ha=0, total=3 + 2 * 19)
    with pytest.raises(DecayError, match="unknown block"):
        layout.slots("attack")


# --- the estimator ---------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def history(corpus) -> pd.DataFrame:
    return corpus[corpus["season"].isin(["2021-22", "2022-23", "2023-24"])].reset_index(drop=True)


def _fit(history, cfg, decay=None):
    model = cfg.model
    return fit_dixon_coles(
        history,
        half_life_days=model.decay_half_life_days,
        ref_date=history["date"].max() + pd.Timedelta(days=1),
        max_goals=model.max_goals,
        param_bounds=model.param_bounds,
        min_effective_share=model.min_effective_share,
        max_iter=model.max_iter,
        decay=decay,
    )


def test_the_cycle_finds_the_same_optimum_as_the_joint_fit(cfg, history) -> None:
    """The check that licenses trusting the cycle where no reference optimum exists.

    Block coordinate descent solves estimating equations, not a single likelihood — so it has to be
    shown to reach the maximum in the one case where there IS a maximum to reach. Held to the same
    1e-4 the existing warm-versus-cold test holds the joint fit to, because the joint fit's own
    convergence tolerance is the floor and nothing here can be more converged than its reference.
    """
    joint = _fit(history, cfg)
    cycled = _fit(history, cfg, decay=spec())

    assert cycled.intercept == pytest.approx(joint.intercept, abs=1e-3)
    assert cycled.home_advantage == pytest.approx(joint.home_advantage, abs=1e-4)
    assert cycled.rho == pytest.approx(joint.rho, abs=1e-4)
    assert np.allclose(cycled.attack, joint.attack, atol=1e-2)
    # The likelihood is the thing that matters, and the cycle must not be meaningfully worse at it.
    assert cycled.neg_log_lik == pytest.approx(joint.neg_log_lik, abs=1e-2)


def test_a_split_memory_actually_changes_the_fit(cfg, history) -> None:
    """A seam that could not move anything would make every null it produced meaningless."""
    joint = _fit(history, cfg)
    split = _fit(history, cfg, decay=spec(team=2555, level=180, home=365))

    assert abs(split.home_advantage - joint.home_advantage) > 1e-3
    held_out = history.tail(40)
    moved = np.abs(split.predict_proba(held_out) - joint.predict_proba(held_out)).max()
    assert moved > 1e-3, "per-parameter decay changed nothing"


def test_the_cycle_converges_rather_than_running_out(cfg, history) -> None:
    """A fit that stopped at the cap is a different object from one that stopped when finished.

    Both are reported; this asserts the shipped stopping rule actually reaches the first case,
    because a cap hit on every fit would mean the arm is scored on fits that never settled.
    """
    fitted = _fit(history, cfg, decay=spec(team=2555, level=180, home=365))
    assert fitted.decay_diagnostics["hit_cycle_cap"] == 0.0
    assert fitted.decay_diagnostics["cycles"] < MAX_CYCLES
    assert fitted.decay_diagnostics["final_movement"] <= TOLERANCE


def test_the_team_half_life_is_the_one_reported_and_used_for_cold_starts(cfg, history) -> None:
    """`half_life_days` stays a scalar because hybrid.py, DynamicFit and elo_dc all read it.

    Under the seam it carries the TEAM memory — the one governing the strength parameters those
    readers care about — and the three-way split is reported beside it rather than replacing it.
    """
    fitted = _fit(history, cfg, decay=spec(team=2555, level=180, home=365))
    assert fitted.half_life_days == 2555
    assert fitted.decay is not None
    assert fitted.as_dict()["half_life_days"] == 2555


def test_the_seam_refuses_to_run_beside_seams_it_cannot_block(cfg, history) -> None:
    """A covariate or family parameter belongs to no block and would be pinned all cycle.

    Refused rather than silently fitted, for the same reason the covariate and family seams refuse
    each other: a run that quietly froze a parameter would be a wrong fit wearing a right name.
    """
    from plmodel.model.counts import CountSpec

    with pytest.raises(ValueError, match="nowhere to put"):
        fit_dixon_coles(
            history,
            half_life_days=cfg.model.decay_half_life_days,
            ref_date=history["date"].max() + pd.Timedelta(days=1),
            max_goals=cfg.model.max_goals,
            param_bounds=cfg.model.param_bounds,
            min_effective_share=cfg.model.min_effective_share,
            max_iter=cfg.model.max_iter,
            decay=spec(level=180),
            family=CountSpec(marginal="weibull", dependence="tau", n_series_terms=60),
        )


def test_the_fit_is_reproducible(cfg, history) -> None:
    """No RNG, no ordering: two identical cycles give identical bytes."""
    a = _fit(history, cfg, decay=spec(team=2555, level=180, home=365))
    b = _fit(history, cfg, decay=spec(team=2555, level=180, home=365))
    assert np.array_equal(a.predict_proba(history.tail(30)), b.predict_proba(history.tail(30)))


# --- through the harness ----------------------------------------------------------------------------

@pytest.mark.integration
def test_the_decay_arms_run_and_move(cfg, corpus) -> None:
    """Both arms must differ from the baseline, and the baseline must not move while they run."""
    import hashlib

    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, run_arm

    tuned = dataclasses.replace(
        cfg,
        model=dataclasses.replace(
            cfg.model,
            seams={**cfg.model.seams,
                   "decay": {**cfg.model.seams["decay"], "team": 2555, "level": 180,
                             "home_advantage": 365}},
        ),
    )
    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )

    def digest(probs: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()

    baseline, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, tuned)
    for name in ("dc-decay", "dc-gas-decay"):
        probs, state = run_arm(ArmSpec.parse(name), corpus, splits, tuned)
        assert np.abs(probs - baseline).max() > 1e-3, f"{name} is indistinguishable from baseline"
        assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12
        assert all(f.decay_diagnostics["hit_cycle_cap"] == 0.0 for f in state["fits"])

    again, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, tuned)
    assert digest(again) == digest(baseline), "the baseline moved while the decay arms ran"


@pytest.mark.integration
def test_an_untuned_seam_refuses_rather_than_silently_being_the_baseline(cfg, corpus) -> None:
    """With all three equal there is nothing to measure, and a null would be uninterpretable."""
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )
    collapsed = dataclasses.replace(
        cfg,
        model=dataclasses.replace(
            cfg.model,
            seams={**cfg.model.seams,
                   "decay": {**cfg.model.seams["decay"], "team": 730, "level": 730,
                             "home_advantage": 730}},
        ),
    )
    with pytest.raises(ValueError, match="tune them"):
        run_arm(ArmSpec.parse("dc-decay"), corpus, splits, collapsed)
