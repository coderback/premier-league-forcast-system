"""Expected goals: per-match team xG, loaded from a Kaggle mirror of Understat.

Provenance, stated plainly
--------------------------
The underlying figures are Understat's. Understat's own site now disallows all crawling
(``robots.txt``: ``Disallow: /``) and its public match pages return 404 as of 2026-08-18, so this
project does not scrape it. The data is obtained instead from a Kaggle dataset whose uploader
published it under an explicit licence, which is a different artefact obtained on different terms.
That distinction was put to the project owner and the route chosen deliberately; it is recorded
here so nobody has to reconstruct the reasoning later.

    dataset : yarknyorulmaz/understat-match-team-metrics-dataset-epl-v16-v24
    licence : Open Database License (ODbL)
    coverage: 3,420 Premier League matches, 2015/16 through 2023/24

Why this source is trusted
--------------------------
Not because it looks plausible, but because it agrees with data we already hold from an entirely
independent source. Joined to the football-data.co.uk corpus on date and teams:

* **goals match on 3,394 of 3,394 rows, both sides** - exact agreement;
* shots on target match on ~98% of rows, the residual being genuine provider disagreement about
  what counts as on target;
* xG is essentially unbiased against outcomes: mean home xG 1.567 against 1.555 home goals
  actually scored, away 1.260 against 1.266.

A mirror that had been mangled, misaligned or fabricated would fail the first of those instantly,
so the check runs on every join rather than once at review time.

The coverage cliff is the real constraint
-----------------------------------------
xG stops after 2023/24. The test span runs to 2025/26, so the xG channel exists for **8 of 10 test
seasons** and **not at all for the 2026/27 season the model must score live**. Any arm built on it
is therefore evaluated on a subset, and could not be wired into production even if it won, until a
live feed is found. Reported by :func:`coverage_summary` rather than left to be discovered.
"""
from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from plmodel.config import Config
from plmodel.data.teams import canonicalise, load_aliases, load_roster

DATASET_REF = "yarknyorulmaz/understat-match-team-metrics-dataset-epl-v16-v24"
DATASET_LICENCE = "Open Database License (ODbL)"
MEMBER = "understat_match_1524.csv"

# Understat timestamps a match by kickoff, football-data.co.uk by calendar date, so a late kickoff
# can land either side of midnight relative to our corpus. Verified: 26 of 3,420 rows need this,
# and all 26 resolve at exactly one day.
JOIN_TOLERANCE_DAYS = 1

# Columns the mirror must carry for this loader to mean anything.
REQUIRED_COLUMNS = ("date", "team_h", "team_a", "h_goals", "a_goals", "h_xg", "a_xg")

_DOWNLOAD_TIMEOUT = 120


class ExpectedGoalsError(ValueError):
    """Raised when the xG source is missing, malformed, or disagrees with the match corpus."""


def cache_path(cfg: Config) -> Path:
    return cfg.cache_dir / "understat" / "understat_match.zip"


def _token() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if token:
        return token.strip()
    token_file = Path.home() / ".kaggle" / "access_token"
    if not token_file.exists():
        raise ExpectedGoalsError(
            "no Kaggle credential: set KAGGLE_API_TOKEN or write ~/.kaggle/access_token"
        )
    return token_file.read_text(encoding="utf-8").strip()


def download(cfg: Config, *, refresh: bool = False) -> Path:
    """Fetch the dataset archive into the cache using the Kaggle API token.

    Kept as an explicit opt-in step rather than folded into the default ingest, so that pulling
    third-party data is always a deliberate act rather than a side effect of refreshing results.
    """
    dest = cache_path(cfg)
    if dest.exists() and not refresh:
        return dest

    url = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_REF}"
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {_token()}"})
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:
        payload = response.read()
    if not payload.startswith(b"PK"):
        raise ExpectedGoalsError(f"{DATASET_REF} did not return a zip archive")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(payload)
    return dest


def load_raw(cfg: Config) -> pd.DataFrame:
    """The mirror's rows, canonicalised to our team names and column conventions."""
    path = cache_path(cfg)
    if not path.exists():
        raise ExpectedGoalsError(f"no cached xG archive at {path}; run `pl ingest --with-xg` first")

    with zipfile.ZipFile(path) as archive:
        if MEMBER not in archive.namelist():
            raise ExpectedGoalsError(f"{path} does not contain {MEMBER}")
        raw = pd.read_csv(io.BytesIO(archive.read(MEMBER)))

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ExpectedGoalsError(f"xG source missing columns: {missing}")

    aliases, roster = load_aliases(cfg.static_dir), load_roster(cfg.static_dir)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]).dt.normalize(),
        "home_team": canonicalise(raw["team_h"], aliases, roster, source="understat xG"),
        "away_team": canonicalise(raw["team_a"], aliases, roster, source="understat xG"),
        "home_xg": pd.to_numeric(raw["h_xg"], errors="coerce"),
        "away_xg": pd.to_numeric(raw["a_xg"], errors="coerce"),
        "xg_home_goals": pd.to_numeric(raw["h_goals"], errors="coerce"),
        "xg_away_goals": pd.to_numeric(raw["a_goals"], errors="coerce"),
    })
    if out[["home_xg", "away_xg"]].isna().any().any():
        raise ExpectedGoalsError("xG source contains unparseable expected-goals values")
    if (out[["home_xg", "away_xg"]] < 0).to_numpy().any():
        raise ExpectedGoalsError("xG source contains negative expected goals")
    return out


def attach(matches: pd.DataFrame, xg: pd.DataFrame) -> pd.DataFrame:
    """Add ``home_xg``/``away_xg`` to a match frame; NaN where the source does not cover it.

    Joined on teams plus a date within :data:`JOIN_TOLERANCE_DAYS`, because the two sources
    timestamp a match differently. Exact dates are tried first and only then the neighbouring
    days, so a fixture can never steal a different meeting's xG.

    Every joined row is checked against the score we already hold. Two independent providers must
    agree on what actually happened before a quantity derived from one of them is believed; a
    mismatch means the join is wrong or a source is corrupt, and either way it raises.
    """
    lookup: dict[tuple, tuple[float, float, float, float]] = {}
    for row in xg.itertuples(index=False):
        lookup[(row.date, row.home_team, row.away_team)] = (
            row.home_xg, row.away_xg, row.xg_home_goals, row.xg_away_goals
        )

    home_xg: list[float] = []
    away_xg: list[float] = []
    disagreements: list[str] = []
    for row in matches.itertuples(index=False):
        base = pd.Timestamp(row.date)
        found = None
        for offset in range(JOIN_TOLERANCE_DAYS + 1):
            days = ({base} if offset == 0 else
                    {base + pd.Timedelta(days=offset), base - pd.Timedelta(days=offset)})
            for day in days:
                found = lookup.get((day, row.home_team, row.away_team))
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            home_xg.append(np.nan)
            away_xg.append(np.nan)
            continue
        h_xg, a_xg, h_goals, a_goals = found
        if not (h_goals == row.home_goals and a_goals == row.away_goals):
            disagreements.append(
                f"{base.date()} {row.home_team} v {row.away_team}: corpus "
                f"{row.home_goals:.0f}-{row.away_goals:.0f} vs xG source {h_goals:.0f}-{a_goals:.0f}"
            )
        home_xg.append(h_xg)
        away_xg.append(a_xg)

    if disagreements:
        raise ExpectedGoalsError(
            f"{len(disagreements)} match(es) where the xG source disagrees with the corpus on the "
            "score, so the join is wrong or a source is corrupt:\n  "
            + "\n  ".join(disagreements[:10])
        )

    out = matches.copy()
    out["home_xg"] = home_xg
    out["away_xg"] = away_xg
    return out


def coverage_summary(matches: pd.DataFrame) -> dict[str, object]:
    """What the xG channel covers - and, more importantly, where it stops."""
    covered = (matches["home_xg"].notna() & matches["away_xg"].notna()).to_numpy()
    priced = matches.loc[covered]
    seasons_covered: list[str] = []
    seasons_uncovered: list[str] = []
    if "season" in matches.columns:
        grouped = matches.assign(_covered=covered).groupby("season")["_covered"].sum()
        seasons_covered = sorted(str(s) for s, n in grouped.items() if n > 0)
        seasons_uncovered = sorted(str(s) for s, n in grouped.items() if n == 0)
    return {
        "dataset": DATASET_REF,
        "licence": DATASET_LICENCE,
        "source": "understat.com via a Kaggle mirror; understat.com itself disallows crawling",
        "n_covered": int(covered.sum()),
        "n_total": int(len(matches)),
        "first_covered": str(priced["date"].min().date()) if len(priced) else None,
        "last_covered": str(priced["date"].max().date()) if len(priced) else None,
        "seasons_covered": seasons_covered,
        "seasons_uncovered": seasons_uncovered,
        "discontinuity": (
            "xG stops after 2023/24: no coverage for 2024-25, 2025-26, or the live 2026/27 season"
        ),
    }
