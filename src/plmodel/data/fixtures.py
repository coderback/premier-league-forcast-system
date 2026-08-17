"""Calendar structure: the walk-forward barrier unit, and a reporting round index.

Two different things that are easy to conflate:

* **matchday** — the index of a distinct *match date* within a (division, season). This is the
  unit the walk-forward harness rolls over. Train strictly before a matchday's date, predict every
  match on it, roll forward.
* **team match index** — each side's own match number within the season, 1-based.
  **Reporting only.** Never a barrier, for the reason below.

Why a date and not a round is the barrier
-----------------------------------------
A round index is not monotonic in time: a fixture postponed in December and replayed in March
still belongs to round 17, so training "strictly before round 17" would mean training on data from
after some of round 17 was played. The barrier has to be a *date*, and a date is what a matchday
is. football-data.co.uk carries no round column at all, so a round would have to be reconstructed
— and every reconstruction is a guess (see :func:`derive_team_match_index`).

Why not a date-gap rule
-----------------------
An earlier design grouped dates into blocks whenever the gap between consecutive match dates
exceeded a threshold. Measured against the real calendar it collapses: English league football has
matches on most days, so with a 3-day threshold the blocks chain transitively and whole months
merge into one. On E0 it produced 19-31 blocks per season instead of the expected ~38, silently
withholding weeks of information from each prediction. Measured on the corpus, E0 averages ~106
distinct match dates per season (min 95, max 135), so per-date barriers are both exact and cheap:
1,153 barriers across the ten-season test span.

The bonus is one fewer arbitrary knob — the gap threshold was the only WEAK-EVIDENCE tunable in
the ingest, and per-date barriers do not need it.
"""
from __future__ import annotations

import pandas as pd


def derive_matchdays(dates: pd.Series) -> pd.Series:
    """1-based index of each row's distinct match date. Input need not be sorted.

    This is the walk-forward barrier unit: every match sharing a date shares a barrier, and no
    match is ever predicted from information dated on or after its own date.
    """
    stamps = pd.to_datetime(dates)
    ordered = pd.Index(sorted(stamps.unique()))
    index_of_date = {d: i for i, d in enumerate(ordered, start=1)}
    return stamps.map(index_of_date).astype(int)


def derive_team_match_index(
    dates: pd.Series, home: pd.Series, away: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Each side's own match number within the season — ``(home_index, away_index)``, 1-based.

    Two exact numbers rather than one approximate one. There is no way to recover the league's
    official round from results alone: a single "matchweek" per match has to reconcile two teams
    that may have played different numbers of games, and every reconciliation is a guess. An
    earlier attempt assigned ``max(games played) + 1`` and inflated E0 seasons to 40-53 rounds
    instead of 38, because rearranged fixtures ratchet both sides forward.

    A team's own match count, by contrast, is exactly defined and runs 1..38 in a 20-team season.
    Reporting can use either side's; nothing is claimed that the data does not support.

    **Reporting only.** Use :func:`derive_matchdays` for anything that splits train from test.
    """
    stamps = pd.to_datetime(dates)
    order = stamps.sort_values(kind="stable").index
    played: dict[str, int] = {}
    home_idx = pd.Series(0, index=stamps.index, dtype=int)
    away_idx = pd.Series(0, index=stamps.index, dtype=int)
    for idx in order:
        h, a = home[idx], away[idx]
        played[h] = home_idx[idx] = played.get(h, 0) + 1
        played[a] = away_idx[idx] = played.get(a, 0) + 1
    return home_idx, away_idx


def calendar_summary(
    df: pd.DataFrame, *, group_cols: tuple[str, ...] = ("division", "season")
) -> pd.DataFrame:
    """Per (division, season): match, matchday and per-team counts — the derivation's audit.

    ``max_team_matches`` must equal (teams - 1) * 2 for a completed season: 38 in a 20-team
    league, 46 in a 24-team one. A season that misses it is the signal that the calendar was
    mis-tracked, which is the failure mode worth watching.
    """
    return (
        df.groupby(list(group_cols), sort=True)
        .agg(
            n_matches=("date", "size"),
            n_matchdays=("matchday", "nunique"),
            max_team_matches=("home_match_index", "max"),
            max_per_matchday=("matchday", lambda s: int(s.value_counts().max())),
        )
        .reset_index()
    )
