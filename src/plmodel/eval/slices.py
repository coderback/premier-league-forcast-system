"""Calibration slices: does the model systematically misprice an identifiable group?

The WC2026 project sliced by confederation. There are no confederations here, so the partitions
that matter in a domestic league are:

* **promoted teams** — three arrive each season with no top-flight history. Roughly 15% of
  fixtures involve one, and it is exactly where the market's information advantage is largest.
* **big six vs the rest** — the persistent-strength partition, and the one where a model that has
  simply learned "the rich clubs win" would look good overall while being useless elsewhere.
* **home/away favourite** — asks whether errors are symmetric in the direction of the mismatch.
* **by season** — catches a result that is really an artefact of one era.
* **staleness regime** — where a fitted-then-frozen strength is most out of date: the opening
  weeks of a season, and the weeks after the January transfer window shuts. The research report
  notes that the theoretical case for dynamic models predicts their edge is concentrated exactly
  there, and that **no paper it found isolates the effect size by regime**. This partition is what
  makes that measurable here.

A slice is a diagnostic, not a gate. A model that is well calibrated overall and badly calibrated
on promoted teams is not failing the acceptance rule; it is telling you where its next improvement
lives.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The favourite slice is defined by the MARKET, not by the arm being scored: an arm-defined
# favourite would make the partition differ between arms, so the slices would no longer be
# comparable. It therefore exists only where the market priced the match.
FAVOURITE_UNPRICED = "unpriced"

# The staleness partition, defined causally from the calendar and from how many matches each side
# has played — both known before kickoff, neither read off the result.
#
# A team is "early season" for its first six matches: that is the window in which a model fitted
# on history is carrying last summer's squad into this season's fixtures, and it is about a sixth
# of a 38-match season. February is the month after the January window shuts, so it is where
# mid-season transfers first show up on the pitch. Neither number is tuned and neither gates
# anything; they partition a diagnostic.
EARLY_SEASON_MATCHES = 6
POST_WINDOW_MONTH = 2  # February, the month after the January window closes
STALENESS_EARLY = "early_season"
STALENESS_POST_WINDOW = "post_january_window"
STALENESS_SETTLED = "settled"


def promoted_teams(history: pd.DataFrame, division: str) -> dict[str, set[str]]:
    """Season -> the teams newly present in ``division`` that season.

    Derived from the corpus rather than a hardcoded list, so it stays correct as seasons are
    added. The first season in the corpus has no predecessor and therefore no promoted teams —
    reported as an empty set rather than as "every team", which would be an artefact of the
    corpus boundary rather than a fact about football.
    """
    div = history[history["division"] == division]
    by_season = {
        season: set(g["home_team"]) | set(g["away_team"])
        for season, g in div.groupby("season", sort=True)
    }
    seasons = sorted(by_season)
    out: dict[str, set[str]] = {}
    for i, season in enumerate(seasons):
        out[season] = set() if i == 0 else by_season[season] - by_season[seasons[i - 1]]
    return out


def add_slice_columns(
    rows: pd.DataFrame,
    *,
    history: pd.DataFrame,
    division: str,
    big_six: tuple[str, ...],
    market_probs: np.ndarray | None = None,
) -> pd.DataFrame:
    """Attach the slice keys to a scored match frame."""
    promoted = promoted_teams(history, division)
    out = rows.copy()

    is_promoted = [
        (r.home_team in promoted.get(r.season, set())) or
        (r.away_team in promoted.get(r.season, set()))
        for r in rows.itertuples(index=False)
    ]
    out["slice_promoted"] = np.where(is_promoted, "involves_promoted", "established_only")

    six = set(big_six)
    n_big = rows["home_team"].isin(six).astype(int) + rows["away_team"].isin(six).astype(int)
    out["slice_big_six"] = np.select(
        [n_big == 2, n_big == 1], ["big_six_derby", "big_six_vs_rest"], default="rest_vs_rest"
    )

    # Priority matters: a February fixture in a team's first six matches does not exist, but an
    # explicit order means the partition can never depend on evaluation order.
    played = np.maximum(
        rows["home_match_index"].to_numpy(), rows["away_match_index"].to_numpy()
    ) if {"home_match_index", "away_match_index"} <= set(rows.columns) else None
    if played is not None:
        out["slice_staleness"] = np.select(
            [played <= EARLY_SEASON_MATCHES,
             rows["date"].dt.month.to_numpy() == POST_WINDOW_MONTH],
            [STALENESS_EARLY, STALENESS_POST_WINDOW],
            default=STALENESS_SETTLED,
        )

    if market_probs is None:
        out["slice_favourite"] = FAVOURITE_UNPRICED
    else:
        from plmodel.eval.metrics import AWAY, HOME

        p = np.asarray(market_probs, dtype=float)
        priced = ~np.isnan(p).any(axis=1)
        favourite = np.where(p[:, HOME] >= p[:, AWAY], "home_favourite", "away_favourite")
        out["slice_favourite"] = np.where(priced, favourite, FAVOURITE_UNPRICED)

    return out


SLICE_COLUMNS: tuple[str, ...] = (
    "season", "slice_promoted", "slice_big_six", "slice_favourite", "slice_staleness",
)


def slice_metrics(
    rows: pd.DataFrame, probs: np.ndarray, outcomes: np.ndarray, *, by: str, n_bins: int
) -> pd.DataFrame:
    """Per-group RPS, log loss and draw calibration for one slice key."""
    from plmodel.eval import metrics
    from plmodel.eval.calibration import brier_decomposition

    if by not in rows.columns:
        raise ValueError(f"slice column {by!r} not present; call add_slice_columns first")

    per_match_rps = metrics.rps(probs, outcomes)
    frame = rows[[by]].copy()
    frame["_rps"] = per_match_rps

    out = []
    for group, idx in frame.groupby(by, sort=True).groups.items():
        positions = frame.index.get_indexer(idx)
        group_probs, group_outcomes = probs[positions], outcomes[positions]
        draw = brier_decomposition(
            group_probs[:, metrics.DRAW],
            (group_outcomes == metrics.DRAW).astype(float),
            n_bins=n_bins,
        )
        uniform = metrics.mean_rps(metrics.uniform_baseline(len(positions)), group_outcomes)
        rps_value = metrics.mean_rps(group_probs, group_outcomes)
        out.append(
            {
                "slice": by,
                "group": str(group),
                "n": int(len(positions)),
                "rps": rps_value,
                "log_loss": metrics.mean_log_loss(group_probs, group_outcomes),
                "rps_uniform": uniform,
                "skill": metrics.skill(rps_value, uniform),
                "draw_reliability": draw["reliability"],
                "draw_resolution": draw["resolution"],
            }
        )
    return pd.DataFrame(out)


def all_slices(
    rows: pd.DataFrame, probs: np.ndarray, outcomes: np.ndarray, *, n_bins: int
) -> pd.DataFrame:
    """Every slice key present on ``rows``, stacked into one frame."""
    frames = [
        slice_metrics(rows, probs, outcomes, by=key, n_bins=n_bins)
        for key in SLICE_COLUMNS
        if key in rows.columns
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
