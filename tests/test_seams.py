"""Every extension seam proves it is inert.

Non-negotiable #4: a seam switched off must produce **byte-identical** output. This is what makes
future arms affordable — the WC2026 project ran ~25 experiments cheaply because every extension
point had a regression test pinning its off state — and it is what makes "we changed nothing in
production" a checkable claim rather than a hope.

The seams are declared in ``config.yaml`` under ``model.seams`` and are all off in the shipped
configuration. Each will be flipped by exactly one future arm:

    covariates        additive strength terms
    dynamics          score-driven / GAS time-varying states
    observation       xG or shots-on-target as a second channel
    ensemble          blend with an ML goals regressor
    home_advantage    time-varying rather than global
    tiers             joint multi-division fit for promoted-team priors

A seam that is merely *unimplemented* still needs its test: the test is what will fail loudly on
the day someone wires the seam up and accidentally changes the default path while doing it.
"""
from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pytest

from plmodel.eval.backtest import walk_forward
from plmodel.eval.compare import ArmSpec, run_arm

SEAM_NAMES = ("covariates", "dynamics", "observation", "ensemble", "home_advantage",
              "tiers", "scoreline", "decay", "promotion")

# Every seam, explicitly off. Written out here rather than read from config.yaml on purpose.
#
# The detector below asks "does flipping this seam register as on?", and it used to flip from the
# SHIPPED configuration. That works only while everything ships off. The moment one seam ships on —
# which is the whole point of having accepted arms — every flip registers as on regardless of the
# seam under test, all seven cases pass, and the detector stops detecting without ever going red.
# A test that cannot fail is worse than no test, because it reads as coverage.
#
# Anchoring on a known-off mapping keeps the question sharp whatever production happens to be.
ALL_OFF: dict[str, object] = {
    "covariates": [],
    "dynamics": {"enabled": False},
    "observation": {"channels": ["goals"]},
    "ensemble": {"enabled": False},
    "home_advantage": {"mode": "global"},
    "tiers": ["E0"],
    # n_series_terms has no default — CountSpec makes it mandatory so a family can never run on a
    # silently-chosen truncation — so an "off" scoreline still has to carry one.
    "scoreline": {"marginal": "poisson", "dependence": "tau", "n_series_terms": 60},
    # Like dynamics, `enabled` is the switch and the tuned values live beside it, so an off decay
    # seam still has to carry a stopping rule.
    "decay": {"enabled": False, "max_cycles": 25, "tolerance": 1.0e-4},
    # Same shape again: the switch is the off state, and the values that say what "on" would mean
    # sit beside it rather than being invented at the call site.
    "promotion": {"enabled": False, "shrinkage": 1.0, "min_prior_clubs": 3},
}


def _digest(probs: np.ndarray) -> str:
    """A byte-level fingerprint: identical output, not merely close output."""
    return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()


def _run(cfg, corpus, *, seams=None) -> np.ndarray:
    model = cfg.model if seams is None else dataclasses.replace(cfg.model, seams=seams)
    tuned = dataclasses.replace(cfg, model=model)
    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )
    probs, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, tuned)
    return probs


# --- the declared seams --------------------------------------------------------------------------

def test_every_seam_is_declared(cfg) -> None:
    """A seam missing from config cannot be tested for inertness, so its absence must fail here."""
    assert set(cfg.model.seams) == set(SEAM_NAMES)


def test_shipped_configuration_has_every_seam_off(cfg) -> None:
    assert cfg.model.seams_are_inert()


def test_the_all_off_mapping_really_is_off(cfg) -> None:
    """The anchor the detector below stands on. If this drifts, the detector proves nothing."""
    assert dataclasses.replace(cfg.model, seams=ALL_OFF).seams_are_inert()


@pytest.mark.parametrize("seam", SEAM_NAMES)
def test_seam_is_recognised_as_on_when_flipped(cfg, seam: str) -> None:
    """The inertness check must be able to *tell* — a detector that never fires proves nothing.

    Flipped from ALL_OFF rather than from the shipped configuration, so this keeps working — and
    keeps failing when it should — after a seam is ever promoted to production.
    """
    on = {
        "covariates": ["rest"],
        "dynamics": {"enabled": True},
        "observation": {"channels": ["goals", "xg"]},
        "ensemble": {"enabled": True},
        "home_advantage": {**ALL_OFF["home_advantage"], "mode": "trend"},
        "tiers": ["E0", "E1"],
        "scoreline": {**ALL_OFF["scoreline"], "marginal": "weibull", "dependence": "frank"},
        "decay": {**ALL_OFF["decay"], "enabled": True},
        "promotion": {**ALL_OFF["promotion"], "enabled": True},
    }[seam]
    flipped = dataclasses.replace(cfg.model, seams={**ALL_OFF, seam: on})
    assert not flipped.seams_are_inert(), f"flipping {seam} was not detected"


# --- inertness on the real corpus ----------------------------------------------------------------

@pytest.mark.integration
def test_seams_off_is_byte_identical_to_no_seams_at_all(cfg, corpus) -> None:
    """The load-bearing regression: the production path must not depend on the seams existing."""
    with_seams = _run(cfg, corpus)
    without = _run(cfg, corpus, seams={})
    assert _digest(with_seams) == _digest(without)


@pytest.mark.integration
@pytest.mark.parametrize("seam", SEAM_NAMES)
def test_each_seam_off_explicitly_is_byte_identical(cfg, corpus, seam: str) -> None:
    """Setting a seam to its off value must change nothing at all."""
    off = {
        "covariates": [],
        "dynamics": {"enabled": False},
        "observation": {"channels": ["goals"]},
        "ensemble": {"enabled": False},
        "home_advantage": {**cfg.model.seams["home_advantage"], "mode": "global"},
        "tiers": ["E0"],
        "scoreline": {**cfg.model.seams["scoreline"], "marginal": "poisson",
                      "dependence": "tau"},
        "decay": {**cfg.model.seams["decay"], "enabled": False},
    }[seam]
    baseline = _run(cfg, corpus)
    explicit = _run(cfg, corpus, seams={**cfg.model.seams, seam: off})
    assert _digest(baseline) == _digest(explicit)


@pytest.mark.integration
def test_the_production_path_is_reproducible(cfg, corpus) -> None:
    """Two identical runs give identical bytes — no RNG, no ordering, no parallel nondeterminism.

    The WC2026 project was bitten here: scikit-learn's random forest fits seed-stably but
    *predicts* in a nondeterministic chunk order, so two runs of the same model disagreed in the
    last decimal. Nothing in this path is parallel, and this test is what keeps it that way.
    """
    assert _digest(_run(cfg, corpus)) == _digest(_run(cfg, corpus))


# Seams this project has adopted into production. Empty today. When one is adopted its name goes
# here in the same commit that flips `enabled`, which makes adoption a one-line reviewable change
# rather than someone deleting a red test at the end of a long day.
ADOPTED_SEAMS: tuple[str, ...] = ()


def test_every_unadopted_seam_ships_off(cfg) -> None:
    """Phrased against ADOPTED_SEAMS rather than against "all of them", so the day a seam is
    promoted this asserts the new truth instead of being deleted for asserting the old one."""
    for seam in SEAM_NAMES:
        if seam in ADOPTED_SEAMS:
            continue
        flipped = dataclasses.replace(cfg.model, seams={**ALL_OFF, seam: cfg.model.seams[seam]})
        assert flipped.seams_are_inert(), f"{seam} ships on but is not in ADOPTED_SEAMS"


@pytest.mark.integration
def test_the_baseline_arm_stays_plain_dixon_coles_whatever_the_seams_say(cfg, corpus) -> None:
    """The acceptance instrument's baseline is a fixed point, not a pointer to production.

    If the `dixon-coles` arm ever followed `model.seams`, then the day a seam is adopted every
    paired delta this project has recorded would silently become a comparison against a different
    baseline — and not one of them would be reproducible. Nothing would go red; the numbers in
    NOTES.md would simply stop meaning what they say.

    So the arm is pinned here rather than trusted: flipping every seam on must leave its forecasts
    byte-identical.
    """
    from plmodel.model.dixon_coles import DixonColesFit

    all_on = {
        "covariates": ["rest"],
        "dynamics": {**cfg.model.seams["dynamics"], "enabled": True},
        "observation": {**cfg.model.seams["observation"], "channels": ["goals"]},
        "ensemble": {**cfg.model.seams["ensemble"], "enabled": True},
        "home_advantage": {**cfg.model.seams["home_advantage"], "mode": "trend"},
        "tiers": ["E0", "E1"],
        "scoreline": {**cfg.model.seams["scoreline"], "marginal": "weibull"},
        "decay": {**cfg.model.seams["decay"], "enabled": True},
    }
    assert not dataclasses.replace(cfg.model, seams=all_on).seams_are_inert()
    assert _digest(_run(cfg, corpus)) == _digest(_run(cfg, corpus, seams=all_on))

    splits = walk_forward(corpus, first_season="2024-25", last_season="2024-25",
                          min_train_matches=cfg.backtest.min_train_matches)
    _, state = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits,
                       dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, seams=all_on)))
    assert all(isinstance(f, DixonColesFit) for f in state["fits"])
