"""Model-free forecasters — the reference points every model is judged against.

Three of them, and none needs a fitted parameter:

* **uniform** — (1/3, 1/3, 1/3). The skill floor. A model that cannot beat this has none.
* **home-always** — all mass on the home win. Degenerate by design: it is the "is the harness
  wired up?" arm, because its RPS is trivially predictable from the home-win rate alone.
* **market** — the de-vigged closing line. Not a floor but a ceiling: the PL 1X2 closing line is
  among the most efficient markets in the world, and the acceptance rule's second gate is measured
  against it.

home-always assigns zero probability to two of three outcomes, so its log loss is the clipped
value (~34.5 per wrong match) rather than infinity. That is honest rather than convenient — RPS is
the headline metric precisely because it stays finite and ordinal under degenerate forecasts, and
the clipping is visible in the report rather than hidden.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from plmodel.data.odds import market_probabilities
from plmodel.eval.metrics import AWAY, DRAW, HOME

N_OUTCOMES = 3


def uniform(rows: pd.DataFrame) -> np.ndarray:
    """(N, 3) of (1/3, 1/3, 1/3) — the skill floor."""
    return np.full((len(rows), N_OUTCOMES), 1.0 / N_OUTCOMES)


def home_always(rows: pd.DataFrame) -> np.ndarray:
    """(N, 3) of (1, 0, 0) — all mass on the home win."""
    probs = np.zeros((len(rows), N_OUTCOMES))
    probs[:, HOME] = 1.0
    return probs


def home_rate(rows: pd.DataFrame, *, rate: tuple[float, float, float]) -> np.ndarray:
    """(N, 3) of a fixed base rate — the non-degenerate cousin of home-always.

    Useful as a sanity contrast: it shows how much of a model's apparent skill is simply knowing
    the league's long-run home/draw/away split rather than anything about the two teams.
    """
    if len(rate) != N_OUTCOMES or not np.isclose(sum(rate), 1.0):
        raise ValueError(f"rate must be three probabilities summing to 1; got {rate}")
    return np.tile(np.asarray(rate, dtype=float), (len(rows), 1))


def market(
    rows: pd.DataFrame, *, family: str, method: str, sum_tolerance: float
) -> np.ndarray:
    """(N, 3) de-vigged market probabilities; NaN on rows the market did not price.

    NaN rather than a fallback: an unpriced match is a coverage fact, and filling it with a base
    rate would quietly turn the benchmark into a blend of the market and a guess.
    """
    probs = market_probabilities(rows, family, method, sum_tolerance=sum_tolerance)
    return probs[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)


def empirical_base_rate(history: pd.DataFrame) -> tuple[float, float, float]:
    """The home/draw/away split of a results frame, for :func:`home_rate`."""
    if len(history) == 0:
        raise ValueError("cannot take a base rate from an empty history")
    result = history["result"]
    n = float(len(result))
    return (
        float((result == "H").sum()) / n,
        float((result == "D").sum()) / n,
        float((result == "A").sum()) / n,
    )


def outcomes_of(rows: pd.DataFrame) -> np.ndarray:
    """Outcome codes for a match frame, from its own result column."""
    codes = {"H": HOME, "D": DRAW, "A": AWAY}
    missing = set(rows["result"].dropna().unique()) - set(codes)
    if missing:
        raise ValueError(f"unexpected result codes {sorted(missing)}")
    if rows["result"].isna().any():
        raise ValueError("cannot score rows without a result; filter to played matches first")
    return rows["result"].map(codes).to_numpy(dtype=int)
