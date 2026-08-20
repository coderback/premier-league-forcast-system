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
              "tiers", "scoreline")


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


@pytest.mark.parametrize("seam", SEAM_NAMES)
def test_seam_is_recognised_as_on_when_flipped(cfg, seam: str) -> None:
    """The inertness check must be able to *tell* — a detector that never fires proves nothing."""
    on = {
        "covariates": ["rest"],
        "dynamics": {"enabled": True},
        "observation": {"channels": ["goals", "xg"]},
        "ensemble": {"enabled": True},
        "home_advantage": {**cfg.model.seams["home_advantage"], "mode": "trend"},
        "tiers": ["E0", "E1"],
        "scoreline": {**cfg.model.seams["scoreline"], "marginal": "weibull",
                      "dependence": "frank"},
    }[seam]
    flipped = dataclasses.replace(cfg.model, seams={**cfg.model.seams, seam: on})
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
