"""De-vigged bookmaker odds: the hard benchmark the model is judged against.

The odds arrive as raw columns on the match corpus (the ingest passes them through unparsed).
This module resolves which columns constitute a usable market benchmark in a given era, strips
the bookmaker's margin, and reports coverage.

Two de-vig methods, both kept in every report:

- ``proportional`` — ``p_i = (1/o_i) / sum_j(1/o_j)``. Spreads the margin evenly, which overstates
  longshot probabilities because bookmakers pad longshot margins hardest (the favourite-longshot
  bias).
- ``shin`` — Shin (1993): prices arise from a bookmaker defending against a fraction ``z`` of
  insider money; solving for ``z`` recovers probabilities that undo the asymmetric margin.

**Shin is primary.** Strumbelj (2016, IJF 32(2):462-472) finds Shin probabilities lowest-RPS
across 412 bookmaker/competition pairs, and Koning & Zijm (2023) find them unbiased for the EPL
specifically (though not for La Liga). The theoretical case is contested — Whelan calls it
"relatively weak" — which is exactly why proportional is retained as a sensitivity rather than
discarded. Both appear in every report.

Choosing the benchmark family
-----------------------------
There is no single market-average closing column covering the whole corpus, so the choice is a
real one, made from measured coverage rather than preference (verified 2026-08-16, see NOTES.md):

* ``AvgCH/AvgCD/AvgCA`` — market-average **closing**, 2019/20 onward, 100% covered, still being
  published. **This is the acceptance rule's gate-2 benchmark**: 2,660 E0 matches.
* ``PSCH/PSCD/PSCA`` — Pinnacle **closing**, 2012/13 onward and the sharper single book, but the
  feed **stops dead after 2026-01-08**. Reaches further back (3,630 of the 3,800 test-decade
  matches) and is reported as a historical diagnostic, never as the gate — a benchmark that dies
  mid-season cannot judge a model that has to score next season.
* ``BbAvH/...`` — Betbrain average, 2005/06-2018/19, but **pre-close, not closing**. Mixing it
  with a closing benchmark inside one pool would silently change what the gate means, so
  settlement timing is recorded on every family and comparisons across it are refused.

Missing and invalid prices
--------------------------
The source encodes "no price" as ``0.0``, not as a blank: six Bet365 rows in the corpus carry
``B365H = 0.0`` with plausible draw and away prices. Those are counted as uncovered, never
imputed and never allowed to reach the de-vig maths. The ported de-vig functions keep their strict
contract (finite, > 1.0) precisely so a sentinel value cannot pass for a price; sanitising happens
before they are called.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ODDS_OUTCOMES: tuple[str, str, str] = ("home", "draw", "away")

# Bisection tolerance and iteration cap for the Shin solve (ported).
_SHIN_TOL = 1e-12
_SHIN_MAX_ITER = 200
# Upper bracket for the insider fraction z, kept strictly below 1 where the formula divides by
# (1 - z).
_SHIN_Z_MAX = 1.0 - 1e-9


class OddsError(ValueError):
    """Raised when odds are malformed or a requested market family cannot be resolved."""


@dataclass(frozen=True)
class MarketFamily:
    """One set of 1X2 columns, with the metadata that makes it comparable — or not."""

    name: str
    columns: tuple[str, str, str]   # home, draw, away
    settlement: str                 # "closing" | "pre-close"
    kind: str                       # "market average" | "single book"
    note: str = ""

    @property
    def is_closing(self) -> bool:
        return self.settlement == "closing"


# The families the source publishes, with the era each covers. Column names are a property of the
# source, so they live here rather than in config; WHICH family is the gate benchmark is a
# decision, so that lives in config.yaml.
FAMILIES: dict[str, MarketFamily] = {
    "avg_closing": MarketFamily(
        "avg_closing", ("AvgCH", "AvgCD", "AvgCA"), "closing", "market average",
        "2019/20 onward; 100% covered and still published — the gate-2 benchmark",
    ),
    "pinnacle_closing": MarketFamily(
        "pinnacle_closing", ("PSCH", "PSCD", "PSCA"), "closing", "single book",
        "2012/13 onward, sharpest single book, but the feed stops after 2026-01-08",
    ),
    "avg_preclose": MarketFamily(
        "avg_preclose", ("AvgH", "AvgD", "AvgA"), "pre-close", "market average",
        "2019/20 onward",
    ),
    "betbrain_avg": MarketFamily(
        "betbrain_avg", ("BbAvH", "BbAvD", "BbAvA"), "pre-close", "market average",
        "2005/06-2018/19",
    ),
    "bet365": MarketFamily(
        "bet365", ("B365H", "B365D", "B365A"), "pre-close", "single book", "2002/03 onward",
    ),
    "william_hill": MarketFamily(
        "william_hill", ("WHH", "WHD", "WHA"), "pre-close", "single book", "2000/01-2024/25",
    ),
}


# --- de-vig (ported verbatim from the WC2026 project) -----------------------------------------

def devig(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proportionally de-vig an (N, 3) decimal-odds array -> ((N, 3) probabilities, overround).

    Implied probabilities 1/o_i sum to 1 + vig; dividing by their sum recovers a proper
    distribution. Overround = sum(1/o_i) - 1 (typically 0.02-0.08 for closing 1X2 odds).
    """
    o = np.asarray(odds, dtype=float)
    if o.ndim != 2 or o.shape[1] != 3:
        raise OddsError(f"odds must be (N, 3); got {o.shape}")
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        raise OddsError("all decimal odds must be finite and > 1.0")
    implied = 1.0 / o
    total = implied.sum(axis=1)
    return implied / total[:, None], total - 1.0


def _shin_probs(implied: np.ndarray, z: float) -> np.ndarray:
    """Shin probabilities for one book at insider fraction ``z`` (implied = 1/odds row)."""
    beta = implied.sum()
    return (np.sqrt(z * z + 4.0 * (1.0 - z) * implied**2 / beta) - z) / (2.0 * (1.0 - z))


def devig_shin(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shin-de-vig an (N, 3) decimal-odds array -> ((N, 3) probabilities, overround).

    Solves per book for the insider fraction ``z`` such that the Shin probabilities sum to 1,
    by bisection on z in [0, 1): the objective f(z) = sum p_i(z) - 1 has f(0) = sqrt(beta) - 1 > 0
    for any overround beta > 1 and f(1-) = sum pi_i^2/beta - 1 < 0 (since each pi_i < 1), so a root
    is bracketed. A fair or sub-fair book (beta <= 1) yields z = 0 and returns the implied
    probabilities normalised.

    Sub-fair books are not hypothetical here: averaging decimal odds across bookmakers can push
    the implied sum below 1, and the lower divisions contain such rows. E0 contains none.
    """
    o = np.asarray(odds, dtype=float)
    if o.ndim != 2 or o.shape[1] != 3:
        raise OddsError(f"odds must be (N, 3); got {o.shape}")
    if np.any(~np.isfinite(o)) or np.any(o <= 1.0):
        raise OddsError("all decimal odds must be finite and > 1.0")
    implied = 1.0 / o
    total = implied.sum(axis=1)

    probs = np.empty_like(implied)
    for i, row in enumerate(implied):
        if total[i] <= 1.0 + _SHIN_TOL:  # fair or sub-fair book: nothing to remove
            probs[i] = row / total[i]
            continue
        lo, hi = 0.0, _SHIN_Z_MAX
        for _ in range(_SHIN_MAX_ITER):
            mid = 0.5 * (lo + hi)
            if _shin_probs(row, mid).sum() > 1.0:
                lo = mid
            else:
                hi = mid
            if hi - lo < _SHIN_TOL:
                break
        p = _shin_probs(row, 0.5 * (lo + hi))
        probs[i] = p / p.sum()  # remove residual bisection error
    return probs, total - 1.0


DEVIG_METHODS = {"proportional": devig, "shin": devig_shin}


# --- resolving a family on the corpus ----------------------------------------------------------

def get_family(name: str) -> MarketFamily:
    if name not in FAMILIES:
        raise OddsError(f"unknown market family {name!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[name]


def resolve_family(df: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract a family's odds from the corpus.

    Returns ``(odds, covered, n_invalid)`` where ``odds`` is (N, 3) with NaN on uncovered rows,
    ``covered`` is the boolean mask of rows carrying a usable price, and ``n_invalid`` counts rows
    that had values but not usable ones (the ``0.0`` sentinel, or any non-finite / <= 1.0 price).
    Invalid rows are counted and excluded — never imputed, never silently dropped from the count.
    """
    family = get_family(name)
    missing = [c for c in family.columns if c not in df.columns]
    if missing:
        raise OddsError(f"family {name!r} needs columns {missing}, absent from the frame")

    odds = df.loc[:, list(family.columns)].to_numpy(dtype=float)
    present = np.isfinite(odds).all(axis=1)
    usable = present & (odds > 1.0).all(axis=1)
    n_invalid = int((present & ~usable).sum())
    odds = np.where(usable[:, None], odds, np.nan)
    return odds, usable, n_invalid


def market_probabilities(
    df: pd.DataFrame, name: str, method: str, *, sum_tolerance: float
) -> pd.DataFrame:
    """De-vigged market probabilities for one family, as a frame aligned to ``df``.

    Uncovered rows are NaN throughout — the market simply did not price them, which is a value,
    not a hole to fill.
    """
    if method not in DEVIG_METHODS:
        raise OddsError(f"unknown de-vig method {method!r}; expected {sorted(DEVIG_METHODS)}")
    odds, covered, _ = resolve_family(df, name)

    out = pd.DataFrame(
        {f"p_{k}": np.full(len(df), np.nan) for k in ODDS_OUTCOMES}, index=df.index
    )
    out["overround"] = np.nan
    if covered.any():
        probs, overround = DEVIG_METHODS[method](odds[covered])
        _check_normalised(probs, name, method, sum_tolerance)
        for k, col in enumerate(ODDS_OUTCOMES):
            out.loc[covered, f"p_{col}"] = probs[:, k]
        out.loc[covered, "overround"] = overround
    return out


def _check_normalised(probs: np.ndarray, name: str, method: str, tolerance: float) -> None:
    """De-vigged probabilities must be a distribution. A silent drift here corrupts every score."""
    sums = probs.sum(axis=1)
    worst = float(np.max(np.abs(sums - 1.0)))
    if worst > tolerance:
        raise OddsError(
            f"{name}/{method}: de-vigged probabilities deviate from 1.0 by {worst:.3e} "
            f"(tolerance {tolerance:.1e})"
        )
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise OddsError(f"{name}/{method}: de-vigged probabilities outside [0, 1]")


def attach_market(
    df: pd.DataFrame, name: str, method: str, *, prefix: str, sum_tolerance: float
) -> pd.DataFrame:
    """Return ``df`` with ``{prefix}_home/draw/away`` and ``{prefix}_overround`` columns added."""
    probs = market_probabilities(df, name, method, sum_tolerance=sum_tolerance)
    out = df.copy()
    for col in ODDS_OUTCOMES:
        out[f"{prefix}_{col}"] = probs[f"p_{col}"].to_numpy()
    out[f"{prefix}_overround"] = probs["overround"].to_numpy()
    return out


def assert_comparable(*names: str) -> None:
    """Refuse to mix settlement timings in one benchmark.

    A pool that is closing odds for one era and pre-close for another silently changes what the
    market gate measures partway through, which would look like a model effect.
    """
    settlements = {get_family(n).settlement for n in names}
    if len(settlements) > 1:
        detail = ", ".join(f"{n}={get_family(n).settlement}" for n in names)
        raise OddsError(
            f"cannot combine market families with different settlement timings ({detail}); "
            "a benchmark that changes definition mid-pool is not a benchmark"
        )


# --- coverage ----------------------------------------------------------------------------------

def family_coverage(
    df: pd.DataFrame, *, names: tuple[str, ...] | None = None, by: tuple[str, ...] = ("season",)
) -> pd.DataFrame:
    """Per family (and per grouping): rows priced, rows invalid, and the season span covered."""
    rows: list[dict[str, object]] = []
    for name in names or tuple(FAMILIES):
        family = FAMILIES[name]
        if not all(c in df.columns for c in family.columns):
            continue
        _, covered, n_invalid = resolve_family(df, name)
        frame = df.assign(_covered=covered)
        for keys, group in frame.groupby(list(by), sort=True):
            key_tuple = keys if isinstance(keys, tuple) else (keys,)
            rows.append(
                {
                    "family": name,
                    "settlement": family.settlement,
                    **dict(zip(by, key_tuple)),
                    "n_rows": int(len(group)),
                    "n_priced": int(group["_covered"].sum()),
                    "coverage": float(group["_covered"].mean()),
                }
            )
        rows.append(
            {
                "family": name, "settlement": family.settlement,
                **{k: "ALL" for k in by},
                "n_rows": int(len(df)), "n_priced": int(covered.sum()),
                "coverage": float(covered.mean()), "n_invalid": n_invalid,
            }
        )
    return pd.DataFrame(rows)
