"""The Murphy decomposition, and the discretisation error it carries.

`BS = reliability - resolution + uncertainty` is exact only when forecasts are grouped by
*identical value*. Bin distinct values together and the identity acquires a residual equal to the
within-bin spread — which is not a rounding nuisance but the quantity that decides whether a
resolution difference between two arms means anything.

NOTES.md 2026-09-22 is what these tests exist for: a +0.00054 draw-resolution gap between two arms
was read as a lead and turned out to be their binning residuals differing by 0.00037. Nothing in
the codebase was wrong — `brier_decomposition` computed and returned both numbers all along — but
the readout printed one of them, so the pair that makes the error visible never reached a reader.
"""
from __future__ import annotations

import numpy as np

from plmodel.cli import _calibration_lines
from plmodel.eval.calibration import brier_decomposition

# Three distinct forecast values, so a fine grid can give each its own bin and a coarse one cannot.
PROBS = np.array([0.05, 0.05, 0.25, 0.25, 0.25, 0.85, 0.85])
OUTCOMES = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])


def test_the_decomposition_is_exact_when_each_value_gets_its_own_bin() -> None:
    """Grouping by identical forecast value is the case Murphy's identity is stated for."""
    d = brier_decomposition(PROBS, OUTCOMES, n_bins=20)
    assert d["brier"] == d["brier_from_decomposition"]


def test_collapsing_distinct_values_into_one_bin_opens_a_residual() -> None:
    """The residual IS the within-bin spread, and it is not small at a coarse grid."""
    d = brier_decomposition(PROBS, OUTCOMES, n_bins=2)
    residual = d["brier"] - d["brier_from_decomposition"]
    assert abs(residual) > 1e-3
    # Coarser binning cannot see between-bin variation, so it understates resolution.
    fine = brier_decomposition(PROBS, OUTCOMES, n_bins=20)
    assert d["resolution"] < fine["resolution"]


def test_the_audit_readout_prints_the_residual_beside_the_brier() -> None:
    """The visibility that `calibration.py`'s docstring promises has to reach a reader.

    Printing the Brier alone is what let a binning artifact read as a draw-resolution finding for
    a day; the pair is what makes it checkable at a glance.
    """
    calibration = {
        "draw": {"decomposition": brier_decomposition(PROBS, OUTCOMES, n_bins=2)},
    }
    lines = _calibration_lines(calibration)
    assert "resid" in lines[0] and "from bins" in lines[0]

    d = calibration["draw"]["decomposition"]
    row = next(line for line in lines if line.strip().startswith("draw"))
    assert f"{d['brier']:.4f}" in row
    assert f"{d['brier_from_decomposition']:.4f}" in row
    assert f"{d['brier'] - d['brier_from_decomposition']:+.5f}" in row


def test_the_readout_tells_the_reader_what_the_residual_is_for() -> None:
    """A column nobody knows how to read is not visibility."""
    lines = _calibration_lines(
        {"draw": {"decomposition": brier_decomposition(PROBS, OUTCOMES, n_bins=2)}}
    )
    note = " ".join(lines)
    assert "resid" in note and "bin" in note
    assert "2026-09-22" in note, "the reader needs somewhere to go for the case that earned this"
