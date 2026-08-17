"""Proper scoring rules for ordered three-class (home/draw/away) forecasts.

Probabilities are arrays of shape (N, 3) ordered [home, draw, away]; outcomes are integer codes
0=home, 1=draw, 2=away. Report RPS *and* log loss; the uniform (1/3,1/3,1/3)
forecast scores RPS = 0.2222 in expectation — the sanity floor.

RPS (Constantinou & Fenton 2012) is primary because it respects the ordinality of home/draw/away
and is the scale every published number in this field is quoted on. Log loss is reported alongside
as the more sensitive discriminator: Wheatcroft (2019, arXiv 1908.08980) argues the Ignorance
(log) score selects the better forecaster more reliably than RPS, and that sensitivity matters
most in the small sub-samples the calibration slices produce.

**RPS IS NOT COMPARABLE ACROSS COMPETITIONS.** A Premier League RPS must never be compared to a
World Cup RPS in any report, chart or ledger entry. International tournament fields are far more
lopsided than a 20-team top division, so outcomes are more predictable and RPS is mechanically
lower — the WC2026 project's production model scored 0.1835 where a good PL model is expected to
land near 0.196-0.206. Reading that difference as a regression would be a straightforward error of
units. Compare within a competition, against a baseline and against that competition's market.

Everything from :data:`HOME` down to :func:`paired_delta` is ported **verbatim** from the WC2026
project (``wc2026/eval/metrics.py``) and is byte-identity tested against it on a fixed fixture in
``tests/fixtures/wc2026_metrics_golden.json``. Do not "improve" those functions: their value is
that numbers computed here are directly comparable with numbers computed there. New tests below
that line.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

HOME, DRAW, AWAY = 0, 1, 2
EPS = 1e-15


def outcome_from_scores(home_score: np.ndarray, away_score: np.ndarray) -> np.ndarray:
    """Map full-time scores to outcome codes (0 home, 1 draw, 2 away)."""
    hs = np.asarray(home_score)
    as_ = np.asarray(away_score)
    out = np.full(hs.shape, DRAW, dtype=int)
    out[hs > as_] = HOME
    out[hs < as_] = AWAY
    return out


def _check(probs: np.ndarray, outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(probs, dtype=float)
    o = np.asarray(outcomes, dtype=int)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"probs must be (N, 3); got {p.shape}")
    if o.shape[0] != p.shape[0]:
        raise ValueError("probs and outcomes length mismatch")
    return p, o


def rps(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match ranked probability score (ordered 3-class). Lower is better.

    RPS = 1/2 * [ (C1 - O1)^2 + (C2 - O2)^2 ], C = cumulative forecast, O = cumulative outcome.
    """
    p, o = _check(probs, outcomes)
    c1 = p[:, HOME]
    c2 = p[:, HOME] + p[:, DRAW]
    o1 = (o == HOME).astype(float)
    o2 = (o <= DRAW).astype(float)  # home or draw
    return 0.5 * ((c1 - o1) ** 2 + (c2 - o2) ** 2)


def mean_rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(rps(probs, outcomes)))


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match negative log-likelihood of the realised outcome."""
    p, o = _check(probs, outcomes)
    picked = p[np.arange(p.shape[0]), o]
    return -np.log(np.clip(picked, EPS, 1.0))


def mean_log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(log_loss(probs, outcomes)))


def brier(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    """Per-match multiclass Brier score: sum_k (p_k - 1[outcome=k])^2."""
    p, o = _check(probs, outcomes)
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), o] = 1.0
    return np.sum((p - onehot) ** 2, axis=1)


def mean_brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean(brier(probs, outcomes)))


def uniform_baseline(n: int) -> np.ndarray:
    """(n, 3) array of (1/3, 1/3, 1/3) — the reference floor (mean RPS ~ 0.2222)."""
    return np.full((n, 3), 1.0 / 3.0)


def summary(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(outcomes)),
        "rps": mean_rps(probs, outcomes),
        "log_loss": mean_log_loss(probs, outcomes),
        "brier": mean_brier(probs, outcomes),
    }


def skill(rps_value: float, rps_uniform: float) -> float:
    """Fractional RPS improvement over the uniform forecast: (uniform - model) / uniform.

    The headline "% better than randomness" number. 0 = no skill; bookmaker odds on World Cups
    sit around 0.17-0.20 on this scale (see NOTES.md).
    """
    if rps_uniform <= 0:
        raise ValueError("rps_uniform must be positive")
    return (rps_uniform - rps_value) / rps_uniform


def paired_delta(
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    outcomes: np.ndarray,
    *,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired-bootstrap comparison of two forecasts on the *same* matches (RPS; lower better).

    Pairing is what buys sensitivity: per-match RPS differences of two similar models are highly
    correlated, so the delta's CI is far tighter than either model's own CI — this resolves
    ~0.002-level RPS differences on ~1,000 matches where unpaired comparison cannot. Returns the
    mean delta (A - B; negative favours A), a percentile bootstrap CI, and the fraction of
    resamples in which A beats B (``p_a_better``, a one-sided posterior-style probability).
    """
    pa, o = _check(probs_a, outcomes)
    pb, _ = _check(probs_b, outcomes)
    if pa.shape != pb.shape:
        raise ValueError("probs_a and probs_b must have the same shape")
    diffs = rps(pa, o) - rps(pb, o)
    n = diffs.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n": int(n),
        "delta_rps": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_a_better": float(np.mean(boot_means < 0.0)),
        "n_boot": int(n_boot),
    }


# ==============================================================================================
# Added for this project: a corroborating significance test and multiplicity control.
# Nothing below is part of the verbatim port.
# ==============================================================================================

# Newey & West (1994) automatic Bartlett bandwidth: lag = floor(SCALE * (n / BASE) ** EXP).
# The standard plug-in rule; used when no explicit lag is supplied.
_NW_LAG_SCALE = 4.0
_NW_LAG_BASE = 100.0
_NW_LAG_EXP = 2.0 / 9.0


def newey_west_lag(n: int) -> int:
    """Automatic Bartlett truncation lag for a sample of ``n`` loss differentials."""
    if n <= 0:
        raise ValueError("n must be positive")
    return int(np.floor(_NW_LAG_SCALE * (n / _NW_LAG_BASE) ** _NW_LAG_EXP))


def _long_run_variance(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC estimate of the long-run variance of ``d``.

    gamma_0 + 2 * sum_k w_k * gamma_k with Bartlett weights w_k = 1 - k / (lag + 1), which keeps
    the estimate non-negative. Per-match RPS differences are serially correlated — a team's form
    persists across its fixtures — so the naive i.i.d. variance understates the standard error.
    """
    n = d.shape[0]
    centred = d - d.mean()
    variance = float(centred @ centred) / n
    for k in range(1, lag + 1):
        gamma_k = float(centred[k:] @ centred[:-k]) / n
        weight = 1.0 - k / (lag + 1.0)
        variance += 2.0 * weight * gamma_k
    return variance


def diebold_mariano(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    lag: int | None = None,
    horizon: int = 1,
) -> dict[str, float]:
    """HAC-corrected Diebold-Mariano test with the Harvey-Leybourne-Newbold small-sample fix.

    ``loss_a`` and ``loss_b`` are per-match losses (typically :func:`rps` output) for the same
    matches. The differential is ``d = loss_a - loss_b``, so a **negative** statistic favours A —
    the same sign convention as :func:`paired_delta`.

    Reported alongside the paired bootstrap rather than instead of it: agreement between two tests
    that make different assumptions is a robustness signal. Note Diebold's own caveat (2015, NBER
    w18391) that DM compares *forecasts*, not *models*, and can favour the simpler benchmark. That
    bias is conservative under a two-gate acceptance rule, which is why it is acceptable here.

    Returns the raw and HLN-corrected statistics, the two-sided p-value from t(n-1), and the
    truncation lag actually used.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"loss arrays must have the same shape; got {a.shape} and {b.shape}")
    if a.ndim != 1:
        raise ValueError(f"loss arrays must be 1-D per-match losses; got {a.shape}")
    n = a.shape[0]
    if n < 2:
        raise ValueError("need at least two observations for a DM test")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    d = a - b
    used_lag = newey_west_lag(n) if lag is None else int(lag)
    if used_lag < 0 or used_lag >= n:
        raise ValueError(f"lag {used_lag} out of range for n={n}")

    variance = _long_run_variance(d, used_lag)
    mean_d = float(d.mean())
    if variance <= 0.0:
        # Degenerate: identical forecasts, or a variance estimate driven negative by noise. No
        # evidence of a difference either way, reported as such rather than as a divide-by-zero.
        return {
            "n": int(n), "lag": int(used_lag), "mean_diff": mean_d,
            "dm_stat": 0.0, "dm_hln": 0.0, "p_value": 1.0, "horizon": int(horizon),
        }

    dm_stat = mean_d / np.sqrt(variance / n)
    # Harvey, Leybourne & Newbold (1997): the DM statistic is oversized in small samples.
    hln_scale = np.sqrt(
        (n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n
    )
    dm_hln = float(dm_stat * hln_scale)
    p_value = float(2.0 * stats.t.sf(abs(dm_hln), df=n - 1))
    return {
        "n": int(n),
        "lag": int(used_lag),
        "mean_diff": mean_d,
        "dm_stat": float(dm_stat),
        "dm_hln": dm_hln,
        "p_value": p_value,
        "horizon": int(horizon),
    }


def benjamini_hochberg(p_values: dict[str, float] | np.ndarray, *, alpha: float) -> dict:
    """Benjamini-Hochberg FDR control across a family of arms.

    ``alpha`` is required, not defaulted: the false-discovery rate the project is willing to
    tolerate is a decision that belongs in config.yaml, not in a function signature.

    Testing many candidate features against one backtest without correction is the norm in this
    literature and is how a two-gate rule eventually passes a false positive by chance. With ~25
    arms in the WC2026 ledger and a comparable plan here, the correction is not optional.

    Accepts a name -> p-value mapping (order preserved in the output) or a bare array. Returns the
    BH-adjusted p-values, the rejection decisions, and the step-up threshold.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    names: list[str] | None = None
    if isinstance(p_values, dict):
        names = list(p_values)
        raw = np.asarray([p_values[k] for k in names], dtype=float)
    else:
        raw = np.asarray(p_values, dtype=float)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("p_values must be a non-empty 1-D collection")
    if np.any(raw < 0.0) or np.any(raw > 1.0):
        raise ValueError("p-values must lie in [0, 1]")

    m = raw.size
    order = np.argsort(raw, kind="stable")
    ranked = raw[order]
    ranks = np.arange(1, m + 1, dtype=float)

    # Step-up: the largest k with p_(k) <= k/m * alpha; everything up to it is rejected.
    below = ranked <= ranks / m * alpha
    n_reject = int(np.max(np.nonzero(below)[0]) + 1) if below.any() else 0
    threshold = float(ranked[n_reject - 1]) if n_reject else 0.0

    # Adjusted p-values, enforced monotone non-decreasing from the largest rank downwards.
    adjusted_sorted = np.minimum.accumulate((ranked * m / ranks)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted

    rejected = np.zeros(m, dtype=bool)
    rejected[order[:n_reject]] = True

    out: dict[str, object] = {
        "alpha": float(alpha),
        "n_tests": int(m),
        "n_rejected": n_reject,
        "threshold": threshold,
    }
    if names is None:
        out["p_adjusted"] = adjusted.tolist()
        out["rejected"] = rejected.tolist()
    else:
        out["p_adjusted"] = {k: float(v) for k, v in zip(names, adjusted)}
        out["rejected"] = {k: bool(v) for k, v in zip(names, rejected)}
    return out
