"""Home-advantage design terms — the seam for a home advantage that is not a single constant.

The production model fits one global ``h`` per barrier on decay-weighted data, so it already
*tracks* drift at the speed the half-life allows. This module lets ``h`` carry explicit structure
instead:

* **trend** — ``h(t) = h + h_trend * years_before_barrier``. Older matches are allowed their own,
  larger home advantage, so they can inform the *shape* of the decline without dragging the current
  estimate up with them. Prediction happens at zero years before the barrier, so the forecast uses
  ``h`` alone: the current value, not the average over the window.
* **empty** — ``h(t) = h + h_empty * 1[played without a crowd]``. Lets the model attribute the
  2020-21 collapse to the regime that caused it rather than smearing it across neighbouring
  seasons through the decay.

Both are single extra parameters entering only the home rate, fitted jointly by the same weighted
likelihood, and both contribute exactly zero at prediction time — a future match is neither in the
past nor behind closed doors.

The empty-stadium window is defined causally, not empirically
--------------------------------------------------------------
The window covers every Premier League match played without a crowd: the 2019-20 restart from
2020-06-17, and all of 2020-21 through 2021-05-23.

Measured on the corpus, those two periods behave completely differently:

    pre-COVID (n=1808)   home 45.7%  away 30.3%  home:away goals 1.279
    restart   (n=92)     home 46.7%  away 31.5%  home:away goals 1.315
    2020-21   (n=380)    home 37.9%  away 40.3%  home:away goals 1.008

**The restart shows no loss of home advantage at all**, despite being played in empty stadiums.
Only the full 2020-21 season collapses, and there away wins outnumber home wins — the first time in
English top-flight history.

The window still includes the restart. Excluding it because it fails to show the expected effect
would be fitting the definition to the outcome, and the resulting dummy would look stronger than
the evidence deserves. Defining it causally means the arm tests the crowd hypothesis honestly: if
the dummy underperforms because the restart dilutes it, that is a finding about crowds, not a
reason to move the boundary. Ninety-two matches is also small enough that the restart's apparent
normality is not conclusive on its own.

Two known impurities inside the window, both left in for the same reason: a limited crowd return in
December 2020 (2,000 spectators at some grounds) and again in May 2021 (up to 10,000). The data
hints at both — the December-to-May portion is slightly less depressed than the September-to-
December portion (38.3% against 36.7% home wins) — but attendance figures are not in the corpus, so
a graded term would be invention rather than measurement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Days per year, for expressing the trend in interpretable units.
_DAYS_PER_YEAR = 365.25

MODE_GLOBAL = "global"
MODE_TREND = "trend"
MODE_EMPTY = "empty"
MODE_BOTH = "trend+empty"
MODES: tuple[str, ...] = (MODE_GLOBAL, MODE_TREND, MODE_EMPTY, MODE_BOTH)


def parse_mode(mode: str) -> tuple[bool, bool]:
    """``mode`` -> (wants_trend, wants_empty)."""
    if mode not in MODES:
        raise ValueError(f"unknown home_advantage mode {mode!r}; expected one of {MODES}")
    return mode in (MODE_TREND, MODE_BOTH), mode in (MODE_EMPTY, MODE_BOTH)


def empty_stadium_flag(
    dates: pd.Series, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> np.ndarray:
    """1.0 for matches played without a crowd, 0.0 otherwise."""
    stamps = pd.to_datetime(dates)
    return (
        (stamps >= pd.Timestamp(start)) & (stamps <= pd.Timestamp(end))
    ).to_numpy(dtype=float)


def design(
    dates: pd.Series,
    ref_date: pd.Timestamp,
    *,
    mode: str,
    empty_start: str | pd.Timestamp | None = None,
    empty_end: str | pd.Timestamp | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """(N, K) home-advantage design matrix and its parameter names.

    K is zero under ``global``, which makes the seam exactly inert: the contribution is an empty
    matrix product, the parameter vector gains nothing, and the fit is byte-identical.
    """
    wants_trend, wants_empty = parse_mode(mode)
    columns: list[np.ndarray] = []
    names: list[str] = []

    if wants_trend:
        seconds = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.total_seconds()
        age_days = seconds / 86400.0  # MATH: seconds per day
        # Years BEFORE the barrier, so the term vanishes at prediction and h is the current value.
        columns.append(np.clip(age_days.to_numpy() / _DAYS_PER_YEAR, 0.0, None))
        names.append("ha_trend")

    if wants_empty:
        if empty_start is None or empty_end is None:
            raise ValueError("empty-stadium mode needs a window; set model.seams.home_advantage")
        columns.append(empty_stadium_flag(dates, empty_start, empty_end))
        names.append("ha_empty")

    n = len(dates)
    if not columns:
        return np.zeros((n, 0)), ()
    return np.column_stack(columns), tuple(names)


def prediction_design(n_rows: int, names: tuple[str, ...]) -> np.ndarray:
    """The design matrix for matches being forecast: all zeros.

    A match being predicted is zero years before its own barrier and is not behind closed doors, so
    every structural term contributes nothing and the forecast uses the current ``h``. Making that
    explicit is the point of the seam — a model that averaged home advantage across a decade would
    forecast today's matches with a home advantage that has not existed for years.
    """
    return np.zeros((n_rows, len(names)))
