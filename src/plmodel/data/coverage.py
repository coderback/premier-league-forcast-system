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
        "source": "football-data.co.uk",
        "field": "shots on target (HST/AST), every division",
        "date": "2013-08",
        "effect": "definition changes at the start of 2013-14: E0 mean per match 14.24 -> 8.92 "
                  "(x0.63) while total shots stay flat; E1 x0.77, E2 x0.84",
        "consequence": "the dc+sot-form trailing differential reads raw counts, so windows "
                       "spanning the break mix two scales and the term was under-weighted until "
                       "about 2020; E0/E1 comparisons pool two regimes. Below the gate's floor "
                       "(-0.00026 sensitivity, -0.00012 test when harmonised) and out of live "
                       "forecasts. See NOTES.md, 2026-11-18",
    },
    {
        "source": "football-data.co.uk",
        "field": "market-average closing odds (AvgCH/AvgCD/AvgCA)",
        "date": "2024-08",
        "effect": "the bookmakers behind the average change: VC, IW and V end in 2024-25; "
                  "BFD, BMGM, BV, CL and LB appear in 2025-26. Margin 0.039-0.042 through "
                  "2024-25, 0.057 in 2025-26, 0.061 in 2026-27 to date",
        "consequence": "gate 2's benchmark is not quite the same benchmark either side of the "
                       "change. After Shin de-vigging it agrees with bet365 within 0.0002 RPS, so "
                       "no recorded gap moves; if the two separate by more than 0.0005 in a "
                       "season, read gate 2 against bet365 too. See NOTES.md, 2026-11-18",
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


def market_benchmark_block(corpus: pd.DataFrame, cfg) -> dict[str, Any]:
    """What each market family prices, and which one the acceptance rule's gate 2 uses.

    Emitted on every run so the benchmark behind a reported market gap is never implicit — and so
    the Pinnacle termination shows up as a coverage number, not just as prose.
    """
    from plmodel.data.odds import FAMILIES, resolve_family

    families: dict[str, Any] = {}
    for name, family in FAMILIES.items():
        if not all(c in corpus.columns for c in family.columns):
            continue
        _, covered, n_invalid = resolve_family(corpus, name)
        priced = corpus.loc[covered]
        families[name] = {
            "columns": list(family.columns),
            "settlement": family.settlement,
            "kind": family.kind,
            "note": family.note,
            "n_priced": int(covered.sum()),
            "n_invalid": n_invalid,
            "coverage": float(covered.mean()),
            "first_priced": str(priced["date"].min().date()) if len(priced) else None,
            "last_priced": str(priced["date"].max().date()) if len(priced) else None,
            "by_division": {
                str(div): int(g.sum())
                for div, g in corpus.assign(_c=covered).groupby("division")["_c"]
            },
        }
    return {
        "gate_benchmark": cfg.odds.gate_benchmark,
        "diagnostic_benchmarks": list(cfg.odds.diagnostic_benchmarks),
        "devig_primary": cfg.odds.devig_primary,
        "devig_sensitivity": cfg.odds.devig_sensitivity,
        "families": families,
    }


def build_report(
    corpus: pd.DataFrame,
    metas: list["SeasonMeta"],
    *,
    skipped: list[str] | None = None,
    cfg=None,
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

    report_market = market_benchmark_block(corpus, cfg) if cfg is not None else None

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
        "market_benchmark": report_market,
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
