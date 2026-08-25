"""The two checks added to the tuning protocol on 2026-08-25.

Both exist because the protocol chose against the test span twice in two days and could not tell
either time. The tests that matter are the ones where the check has to disagree with the number a
search would have reported: a winner whose mean genuinely is lower but whose advantage is inside the
window's own noise, and a parameter whose optimum is clean and interior on a window where the
parameter does not move.

A check that only fired on exact ties would have caught neither failure.
"""
from __future__ import annotations

import numpy as np
import pytest

from plmodel.eval import metrics
from plmodel.eval.tuning import parameter_is_tunable, selection_is_resolved, sweep_verdict

N_BOOT, SEED = 2000, 7


def forecasts(rng, n, *, edge=0.0):
    """(n, 3) forecasts, optionally shaded toward the home outcome by ``edge``."""
    base = rng.dirichlet([4.0, 3.0, 3.0], size=n)
    if edge:
        base = base + np.array([edge, -edge / 2, -edge / 2])
        base = np.clip(base, 1e-6, None)
    return base / base.sum(axis=1, keepdims=True)


# --- rule 1: resolution ---------------------------------------------------------------------------

def test_a_winner_inside_the_noise_is_reported_unresolved() -> None:
    """The load-bearing test, and the whole failure being fixed.

    The winner's mean IS lower — a search would report it and move the shipped value. On this much
    data that advantage does not survive its own confidence interval, so the selection has not
    earned the move.
    """
    rng = np.random.default_rng(0)
    n = 1303  # the tuning span's actual pooled size
    outcomes = rng.integers(0, 3, n)
    incumbent = forecasts(rng, n)
    # A vanishingly small perturbation: different forecasts, no real edge.
    winner = np.clip(incumbent + rng.normal(0.0, 0.002, incumbent.shape), 1e-6, None)
    winner = winner / winner.sum(axis=1, keepdims=True)

    out = selection_is_resolved(winner, incumbent, outcomes, n_boot=N_BOOT, seed=SEED)

    assert out["resolved"] is False
    assert out["ci_low"] < 0.0 < out["ci_high"], "the advantage must straddle zero"
    assert "UNRESOLVED" in out["verdict"]


def test_a_genuinely_better_configuration_is_reported_resolved() -> None:
    """The check must still be able to say yes, or it is just a veto."""
    rng = np.random.default_rng(1)
    n = 1303
    outcomes = np.zeros(n, dtype=int)  # home wins throughout
    incumbent = forecasts(rng, n)
    winner = forecasts(rng, n, edge=0.25)  # decisively better on this outcome

    out = selection_is_resolved(winner, incumbent, outcomes, n_boot=N_BOOT, seed=SEED)

    assert out["resolved"] is True
    assert out["delta"] < 0.0, "negative favours the winner"
    assert out["ci_high"] < 0.0


def test_clustering_by_season_widens_the_interval() -> None:
    """A half-life change perturbs a whole season coherently, so matches are not independent.

    Treating them as independent reports an interval narrower than the evidence supports, which is
    the direction that manufactures resolution. The clustered figure is the verdict for that reason,
    and this pins the relationship rather than trusting the docstring.
    """
    rng = np.random.default_rng(2)
    n, per_season = 1300, 130
    outcomes = rng.integers(0, 3, n)
    incumbent = forecasts(rng, n)
    seasons = np.repeat(np.arange(n // per_season), per_season)
    # A season-level shift: every match in a season moves the same way.
    shift = np.repeat(rng.normal(0.0, 0.03, n // per_season), per_season)[:, None]
    winner = np.clip(incumbent + shift * np.array([1.0, -0.5, -0.5]), 1e-6, None)
    winner = winner / winner.sum(axis=1, keepdims=True)

    out = selection_is_resolved(
        winner, incumbent, outcomes, groups=seasons, n_boot=N_BOOT, seed=SEED
    )

    assert out["clustered"] is True
    assert out["n_groups"] == n // per_season
    naive_width = out["unclustered_ci"][1] - out["unclustered_ci"][0]
    clustered_width = out["ci_high"] - out["ci_low"]
    assert clustered_width > naive_width, "clustering must not narrow the interval"


def test_identical_configurations_are_never_resolved() -> None:
    rng = np.random.default_rng(3)
    probs = forecasts(rng, 400)
    outcomes = rng.integers(0, 3, 400)
    out = selection_is_resolved(probs, probs, outcomes, n_boot=N_BOOT, seed=SEED)
    assert out["resolved"] is False
    assert out["delta"] == pytest.approx(0.0, abs=1e-12)


def test_the_check_uses_the_projects_own_bootstrap() -> None:
    """Not a reimplementation: the same instrument the acceptance gate uses, at the same seed."""
    rng = np.random.default_rng(4)
    n = 500
    outcomes = rng.integers(0, 3, n)
    a, b = forecasts(rng, n), forecasts(rng, n)
    out = selection_is_resolved(a, b, outcomes, n_boot=N_BOOT, seed=SEED)
    direct = metrics.paired_delta_losses(
        metrics.rps(a, outcomes), metrics.rps(b, outcomes), n_boot=N_BOOT, seed=SEED
    )
    assert out["delta"] == pytest.approx(direct["delta"])
    assert out["ci_low"] == pytest.approx(direct["ci_low"])


# --- rule 2: stationarity -------------------------------------------------------------------------

def test_a_parameter_that_is_flat_here_is_declared_untunable() -> None:
    """Home advantage's real numbers: sd 0.0117 on the tuning span, 0.0403 on the test span."""
    rng = np.random.default_rng(5)
    out = parameter_is_tunable(
        rng.normal(0.32, 0.0117, 200), rng.normal(0.22, 0.0403, 200), min_ratio=0.5
    )
    assert out["tunable"] is False
    assert out["ratio"] < 0.5
    assert "UNTUNABLE" in out["verdict"]


def test_a_parameter_that_moves_comparably_is_tunable() -> None:
    rng = np.random.default_rng(6)
    out = parameter_is_tunable(
        rng.normal(0.0, 0.04, 200), rng.normal(0.0, 0.045, 200), min_ratio=0.5
    )
    assert out["tunable"] is True
    assert out["ratio"] > 0.5


def test_the_ratio_is_reported_not_just_the_flag() -> None:
    """The number belongs in the ledger next to whatever decision it drove."""
    out = parameter_is_tunable([1.0, 2.0, 3.0], [1.0, 5.0, 9.0], min_ratio=0.5)
    assert set(out) >= {"tunable", "ratio", "tuning_sd", "reference_sd", "min_ratio", "verdict"}
    assert out["ratio"] == pytest.approx(out["tuning_sd"] / out["reference_sd"])


def test_a_motionless_reference_does_not_divide_by_zero() -> None:
    out = parameter_is_tunable([1.0, 2.0], [3.0, 3.0], min_ratio=0.5)
    assert out["tunable"] is True and out["ratio"] == float("inf")


# --- the existing protocol must not move ------------------------------------------------------------

def test_the_grid_edge_rule_is_untouched() -> None:
    """`sweep_verdict` stays a reader of two columns; the new checks live beside it, not inside it.

    Six tests in test_tuning.py feed it a bare half_life_days/rps frame. Folding a resolution
    verdict into it would have broken every one of them, and the separation is deliberate.
    """
    import pandas as pd

    sweep = pd.DataFrame({"half_life_days": [30.0, 730.0, 1825.0], "rps": [0.21, 0.20, 0.205]})
    verdict = sweep_verdict(sweep)
    assert verdict["at_grid_edge"] is False
    assert verdict["best_half_life_days"] == 730.0
    assert "resolved" not in verdict
