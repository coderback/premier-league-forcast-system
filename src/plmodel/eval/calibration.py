"""Calibration diagnostics: reliability curves and the Murphy decomposition.

Discrimination alone is not enough. Two forecasters with the same RPS can differ entirely in
*why* — one honest but blunt, the other sharp but biased — and the acceptance rule's two gates
cannot tell them apart on their own.

The Murphy (1973) decomposition of the Brier score splits it into three interpretable parts:

    BS = reliability - resolution + uncertainty

* **reliability** (lower better) — how far the forecast probabilities sit from the outcome
  frequencies they imply. Zero means perfectly calibrated.
* **resolution** (higher better) — how far the binned outcome frequencies sit from the base rate.
  Zero means the forecast never distinguishes one match from another.
* **uncertainty** — the base rate's own variance. A property of the matches, not the forecaster,
  so it is identical across arms and useful only as a sanity check that two arms saw the same
  pool.

**Draw resolution is the metric to watch and expect nothing from.** Nothing in the literature
moves it, and every WC2026 arm either left it alone or traded it for home/away sharpness. It is
reported as its own curve so a flat result reads as the confirmation it is, rather than as a
missing result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Reliability bins span [0, 1]; the last bin is closed so p = 1.0 lands inside it.
_BIN_MIN, _BIN_MAX = 0.0, 1.0


def _binned(probs: np.ndarray, outcomes: np.ndarray, n_bins: int):
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=float)
    if p.ndim != 1 or o.ndim != 1:
        raise ValueError("reliability works on one outcome at a time; pass 1-D arrays")
    if p.shape != o.shape:
        raise ValueError(f"probs and outcomes must align; got {p.shape} and {o.shape}")
    if len(p) == 0:
        raise ValueError("cannot bin an empty forecast")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive; got {n_bins}")
    edges = np.linspace(_BIN_MIN, _BIN_MAX, n_bins + 1)
    # searchsorted puts p exactly on an edge into the upper bin; clip keeps p = 1.0 in the last.
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    return p, o, edges, index


def reliability_curve(
    probs: np.ndarray, outcomes: np.ndarray, *, n_bins: int
) -> pd.DataFrame:
    """Per-bin forecast probability vs observed frequency — the reliability diagram's data.

    ``outcomes`` is a 0/1 indicator of the event ``probs`` forecasts. Empty bins are returned with
    ``n = 0`` and NaN statistics rather than dropped, so the curve's shape is honest about where
    the forecaster never ventures.
    """
    p, o, edges, index = _binned(probs, outcomes, n_bins)
    rows = []
    for b in range(n_bins):
        mask = index == b
        n = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "n": n,
                "mean_forecast": float(p[mask].mean()) if n else np.nan,
                "observed_rate": float(o[mask].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def brier_decomposition(
    probs: np.ndarray, outcomes: np.ndarray, *, n_bins: int
) -> dict[str, float]:
    """Murphy decomposition of the single-event Brier score.

    Returns reliability, resolution, uncertainty, the base rate, and both the direct Brier score
    and the one implied by the decomposition. The two agree only up to binning error, so reporting
    both makes the discretisation visible instead of hiding it.
    """
    p, o, _, index = _binned(probs, outcomes, n_bins)
    n = len(p)
    base_rate = float(o.mean())

    reliability = 0.0
    resolution = 0.0
    for b in np.unique(index):
        mask = index == b
        weight = float(mask.sum()) / n
        mean_p = float(p[mask].mean())
        mean_o = float(o[mask].mean())
        reliability += weight * (mean_p - mean_o) ** 2
        resolution += weight * (mean_o - base_rate) ** 2
    uncertainty = base_rate * (1.0 - base_rate)

    return {
        "n": int(n),
        "base_rate": base_rate,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "brier": float(np.mean((p - o) ** 2)),
        "brier_from_decomposition": reliability - resolution + uncertainty,
        "n_bins": int(n_bins),
    }


def calibration_report(
    probs: np.ndarray, outcomes: np.ndarray, *, n_bins: int
) -> dict[str, object]:
    """Per-outcome calibration for a three-class forecast.

    The draw block is the one to read first — and the one to expect not to move.
    """
    from plmodel.eval.metrics import AWAY, DRAW, HOME

    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=int)
    if p.ndim != 2 or p.shape[1] != len({HOME, DRAW, AWAY}):
        raise ValueError(f"probs must be (N, 3); got {p.shape}")

    out: dict[str, object] = {}
    for name, code in (("home", HOME), ("draw", DRAW), ("away", AWAY)):
        indicator = (o == code).astype(float)
        out[name] = {
            "decomposition": brier_decomposition(p[:, code], indicator, n_bins=n_bins),
            "curve": reliability_curve(p[:, code], indicator, n_bins=n_bins).to_dict("records"),
        }
    return out
