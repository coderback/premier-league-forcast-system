"""The football-data.co.uk column contract, and the canonical frame it maps to.

The source's schema is **not** stable across its 33 seasons — it grew from seven columns in
1993/94 to well over a hundred today. Validation is therefore era-aware: a small core is required
in every file, and every other column group is validated *when present* and recorded as coverage.
Assuming a column exists because it exists in recent files is how an ingest silently drops a
decade.

Verified against the live source on 2026-08-16 (see NOTES.md):

* core columns are present in every season of every division;
* half-time scores appear from 1995/96;
* match statistics (shots, corners, cards, referee) appear from 2000/01;
* a ``Time`` column appears only from 2019/20 — which is why the walk-forward barrier is
  date-granular rather than kickoff-granular.

Odds columns are deliberately **passed through unparsed**. Resolving the era-varying odds ladder
and de-vigging it is the odds loader's job; doing it here would bury the market benchmark inside
the ingest.
"""
from __future__ import annotations

import pandas as pd

# --- source column groups ---------------------------------------------------------------------

# Required in every season file of every division. A file missing any of these is malformed.
CORE_COLUMNS: tuple[str, ...] = ("Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")

# Optional groups: validated when present, counted as coverage when absent. The mapping is
# source column -> canonical column.
HALFTIME_COLUMNS: dict[str, str] = {
    "HTHG": "ht_home_goals",
    "HTAG": "ht_away_goals",
    "HTR": "ht_result",
}

MATCH_STAT_COLUMNS: dict[str, str] = {
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_sot",
    "AST": "away_sot",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HC": "home_corners",
    "AC": "away_corners",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}

TEXT_COLUMNS: dict[str, str] = {"Referee": "referee", "Time": "kickoff_time"}

# Any column starting with one of these prefixes is an odds column, for the odds loader.
# Kept as a prefix rule rather than a fixed list because the bookmaker panel changes every season.
ODDS_PREFIXES: tuple[str, ...] = (
    "B365", "BW", "BF", "BS", "BV", "BMGM", "CL", "GB", "IW", "LB", "PS", "P>", "P<", "PC",
    "SB", "SJ", "VC", "WH", "1XB", "Max", "Avg", "Bb", "AH",
)

# Integer count columns whose values must be non-negative when present.
_COUNT_COLUMNS: tuple[str, ...] = ("home_goals", "away_goals", *MATCH_STAT_COLUMNS.values(),
                                   "ht_home_goals", "ht_away_goals")

# Valid full-time / half-time result codes.
RESULT_CODES: frozenset[str] = frozenset({"H", "D", "A"})

# Canonical identity columns, in order. Every downstream join keys on these.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "date", "division", "season", "matchday", "home_team", "away_team",
)


class SchemaError(ValueError):
    """Raised when a season file does not match the expected contract."""


def is_odds_column(name: str) -> bool:
    return name.startswith(ODDS_PREFIXES)


def check_core_columns(columns: list[str], source: str) -> None:
    """Fail loudly on a file missing any core column."""
    missing = [c for c in CORE_COLUMNS if c not in columns]
    if missing:
        raise SchemaError(f"{source}: missing core columns {missing}")


def present_groups(columns: list[str]) -> dict[str, bool]:
    """Which optional column groups this file carries — the input to the coverage report."""
    cols = set(columns)
    return {
        "halftime": set(HALFTIME_COLUMNS) <= cols,
        "match_stats": set(MATCH_STAT_COLUMNS) <= cols,
        "referee": "Referee" in cols,
        "kickoff_time": "Time" in cols,
        "odds": any(is_odds_column(c) for c in cols),
    }


def validate_frame(df: pd.DataFrame, source: str) -> None:
    """Post-canonicalisation invariants.

    These are the checks that catch a plausible-looking but wrong frame: a mis-parsed date, a
    result code inconsistent with its own goals, a negative count from a shifted column.

    Score-dependent checks apply only to **played** rows. The current season's file carries its
    unplayed fixtures with blank scores; those are the fixture list the season simulator and
    ``pl live`` need, not malformed rows.
    """
    if len(df) == 0:
        raise SchemaError(f"{source}: no match rows after filtering")
    played = df[df["played"]] if "played" in df.columns else df

    if df["date"].isna().any():
        bad = int(df["date"].isna().sum())
        raise SchemaError(f"{source}: {bad} row(s) with unparseable dates")

    for col in ("home_team", "away_team"):
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if blank.any():
            raise SchemaError(f"{source}: {int(blank.sum())} row(s) with a blank {col}")

    if (df["home_team"] == df["away_team"]).any():
        raise SchemaError(f"{source}: row(s) where a team plays itself")

    for col in _COUNT_COLUMNS:
        if col not in df.columns:
            continue
        values = df[col].dropna()
        if len(values) and (values < 0).any():
            raise SchemaError(f"{source}: negative values in {col}")

    if len(played) == 0:
        return  # a fixture-list-only file (a season that has not kicked off) is valid

    bad_codes = set(played["result"].dropna().unique()) - RESULT_CODES
    if bad_codes:
        raise SchemaError(f"{source}: unexpected result codes {sorted(bad_codes)}")

    # The source's own FTR must agree with its own goals. The brief requires verifying the
    # full-time column conventions per season file rather than trusting them: the WC2026 project
    # found football-data.co.uk's World Cup workbook had FT columns inconsistent across sheets.
    derived = derive_result(played["home_goals"], played["away_goals"])
    stated = played["result"]
    mismatch = stated.notna() & (derived != stated)
    if mismatch.any():
        rows = played.loc[mismatch, ["date", "home_team", "away_team", "home_goals", "away_goals"]]
        raise SchemaError(
            f"{source}: {int(mismatch.sum())} row(s) where FTR disagrees with FTHG/FTAG:\n{rows}"
        )


def derive_result(home_goals: pd.Series, away_goals: pd.Series) -> pd.Series:
    """H/D/A from goals — the source of truth when FTR and the goals disagree."""
    out = pd.Series("D", index=home_goals.index, dtype=object)
    out[home_goals > away_goals] = "H"
    out[home_goals < away_goals] = "A"
    return out
