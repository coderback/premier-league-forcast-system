"""The coverage report — what the corpus actually contains, emitted on every run.

Non-negotiable #6: missing data is a value, not a hole. Nothing is imputed, so every consumer
needs to know what is present and what is absent, per feature and per era. This module turns the
per-season ingest metadata into a JSON-serialisable report.

It also carries the **known discontinuities**. Both fall in January 2026 and both would otherwise
look like a gradual decline in coverage rather than a hard stop:

* **Pinnacle closing odds** vanish from the football-data.co.uk feed after 2026-01-08. The 2025/26
  season shows 55% coverage as a result. This is why the market gate uses the market-average
  closing columns instead.
* **FBref** lost its Opta advanced-stats licence in January 2026. Its archive through 2025 is
  intact and usable for backtesting, but it does not update — leaving Understat as the single live
  xG source, with no cross-provider redundancy.

The brief requires these to be flagged explicitly rather than smoothed over, so they are reported
as first-class entries even when the affected columns are not loaded.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # avoids a cycle; only needed for annotations
    from plmodel.data.football_data import SeasonMeta

# Documented breaks in the sources, surfaced in every report.
KNOWN_DISCONTINUITIES: tuple[dict[str, str], ...] = (
    {
        "source": "football-data.co.uk",
        "field": "Pinnacle closing odds (PSCH/PSCD/PSCA)",
        "date": "2026-01-08",
        "effect": "present for every match through 2026-01-08, absent thereafter; "
                  "2025/26 season coverage 55%",
        "consequence": "not used as the market gate; market-average closing (AvgC*) is, and "
                       "Pinnacle is reported as a historical diagnostic only",
    },
    {
        "source": "FBref",
        "field": "Opta advanced stats (incl. xG)",
        "date": "2026-01",
        "effect": "licence lost; archive through 2025 intact but no longer updating",
        "consequence": "Understat is the single live xG source; no code path may assume "
                       "cross-provider xG redundancy",
    },
)


def build_report(
    corpus: pd.DataFrame, metas: list["SeasonMeta"], *, skipped: list[str] | None = None
) -> dict[str, Any]:
    """Assemble the coverage report for a loaded corpus."""
    played = corpus[corpus["played"]]
    per_season = [
        {
            "division": m.division,
            "season": m.season,
            "n_matches": m.n_matches,
            "n_played": m.n_played,
            "encoding": m.encoding,
            "date_min": str(m.date_min.date()),
            "date_max": str(m.date_max.date()),
            "n_matchdays": m.n_matchdays,
            "max_team_matches": m.max_team_matches,
            "column_groups": dict(m.groups),
            "market_columns": dict(m.market_coverage),
        }
        for m in metas
    ]

    return {
        "totals": {
            "n_rows": int(len(corpus)),
            "n_played": int(len(played)),
            "n_unplayed_fixtures": int(len(corpus) - len(played)),
            "n_teams": int(len(set(corpus["home_team"]) | set(corpus["away_team"]))),
            "date_min": str(corpus["date"].min().date()),
            "date_max": str(corpus["date"].max().date()),
            "divisions": sorted(corpus["division"].unique()),
            "n_seasons": int(corpus["season"].nunique()),
        },
        "by_division": _by_division(corpus),
        "encodings": _value_counts(per_season, "encoding"),
        "column_group_first_seen": _first_seen_groups(per_season),
        "market_column_coverage": _market_coverage(per_season),
        "per_season": per_season,
        "skipped_seasons": list(skipped or []),
        "known_discontinuities": [dict(d) for d in KNOWN_DISCONTINUITIES],
    }


def _by_division(corpus: pd.DataFrame) -> list[dict[str, Any]]:
    grouped = corpus.groupby("division", sort=True)
    return [
        {
            "division": str(div),
            "n_matches": int(len(g)),
            "n_seasons": int(g["season"].nunique()),
            "date_min": str(g["date"].min().date()),
            "date_max": str(g["date"].max().date()),
        }
        for div, g in grouped
    ]


def _value_counts(per_season: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_season:
        counts[str(row[key])] = counts.get(str(row[key]), 0) + 1
    return dict(sorted(counts.items()))


def _first_seen_groups(per_season: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Per division, the earliest season carrying each optional column group."""
    out: dict[str, dict[str, str]] = {}
    for row in per_season:
        div = out.setdefault(row["division"], {})
        for group, present in row["column_groups"].items():
            if present and (group not in div or row["season"] < div[group]):
                div[group] = row["season"]
    return out


def _market_coverage(per_season: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per market family: the season span it covers and how many matches it prices."""
    out: dict[str, dict[str, Any]] = {}
    for row in per_season:
        for family, n in row["market_columns"].items():
            if n <= 0:
                continue
            entry = out.setdefault(family, {"n_priced": 0, "first_season": row["season"],
                                            "last_season": row["season"]})
            entry["n_priced"] += int(n)
            entry["first_season"] = min(entry["first_season"], row["season"])
            entry["last_season"] = max(entry["last_season"], row["season"])
    return dict(sorted(out.items()))
