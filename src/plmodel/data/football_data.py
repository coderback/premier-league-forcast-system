"""Ingest the football-data.co.uk match corpus: download, decode, validate, cache.

Source: https://www.football-data.co.uk/ — free for personal and research use with attribution.
URL pattern ``{base_url}/{season_code}/{division}.csv``, e.g. ``.../mmz4281/2425/E0.csv``.

Every quirk handled below was verified against the live source on 2026-08-16 rather than assumed
(see NOTES.md for the full survey). They are the difference between a working ingest and one that
silently produces a plausible but wrong frame:

* **Encoding is not uniform.** Most files are utf-8, some with a BOM — but 2004/05 is cp1252 and
  raises ``UnicodeDecodeError`` on byte 0xa0. Decoding falls back rather than failing, and records
  which codec was used per file.
* **Dates mix two-digit and four-digit years**, always dayfirst. Both formats are tried explicitly;
  inference is never used, because a mis-inferred dayfirst turns 03/04/1995 into the wrong month
  for two thirds of the corpus without erroring.
* **Files carry trailing blank rows and trailing empty columns**, which must be dropped before
  validation or every count check is wrong.
* **Some files have rows wider than their own header** — stray trailing commas. Verified across
  the whole corpus (17 files, 1993/94-2004/05): every extra field is empty, so they are absorbed
  into padding columns and dropped. A non-empty extra field raises rather than being discarded.
* **Column sets grow over time**, so validation is era-aware (see :mod:`plmodel.data.schema`).
* **The server substitutes a different file when the requested one does not exist.** Requesting a
  season that has not been published yet returns HTTP 300 "Multiple Choices" — and for some
  divisions it silently serves *another division's data*: on 2026-08-17, `2627/E0.csv` returned the
  National League file (``Div`` = ``EC``). ``raise_for_status()`` does not catch a 300, and the body
  is a valid CSV, so nothing downstream would have noticed Premier League rows being National
  League matches. Three guards below close this: the status must be exactly 200, the body must
  start with the expected header, and **every row's ``Div`` must equal the requested division**.

Odds columns are passed through unparsed; the odds loader resolves them.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from plmodel.config import Config
from plmodel.data import schema
from plmodel.data.fixtures import derive_matchdays, derive_team_match_index
from plmodel.data.teams import canonicalise, load_aliases, load_roster

# Codecs tried in order. utf-8-sig strips a BOM when present and is a superset of plain utf-8;
# cp1252 is the fallback that 2004/05 needs and never itself fails on this source's bytes.
_CODECS: tuple[str, ...] = ("utf-8-sig", "cp1252")

# A season file's dates must fall inside its own season. The window opens in July of the start
# year and closes at the end of August the following year — wide enough for 2019/20, which COVID
# pushed to a 2020-07-26 finish, and tight enough to catch a century mis-parse.
_SEASON_OPENS_MONTH = 7
_SEASON_CLOSES_MONTH = 8

# Two-digit season codes at or above this belong to the 1900s ("9394" -> 1993, "0001" -> 2000).
_CENTURY_PIVOT = 90

# Probe columns for the coverage report: one representative per market family, so the report shows
# at a glance which benchmark is available in which era. The odds loader owns the full ladder.
_MARKET_PROBES: dict[str, str] = {
    "avg_closing": "AvgCH",     # market-average closing — the gate-2 benchmark, 2019/20 ->
    "pinnacle_closing": "PSCH",  # Pinnacle closing — historical diagnostic, dies 2026-01-08
    "betbrain_avg": "BbAvH",    # Betbrain average, PRE-close — 2005/06 to 2018/19
    "bet365": "B365H",
    "william_hill": "WHH",
}

# Timeout for a single season-file download, in seconds. The files are small (30-300 KB).
_DOWNLOAD_TIMEOUT = 60

# Every genuine season file starts with the division column. Used to reject the server's HTML
# "300 Multiple Choices" page, which is otherwise a non-empty 200-ish body.
_EXPECTED_FIRST_FIELD = "Div"


class IngestError(RuntimeError):
    """Raised when a season file cannot be fetched or read."""


class MissingSeasonError(IngestError):
    """The season has not been published for this division yet — an expected absence.

    Distinguished from IngestError because it is normal: on 2026-08-17 the 2026/27 files for E1
    and E2 did not exist. A season that is genuinely absent is skipped and recorded; a season that
    is present but *wrong* still raises.
    """


@dataclass
class SeasonMeta:
    """What the ingest learned about one season file — the coverage report's raw material."""

    division: str
    season: str
    season_code: str
    n_matches: int
    n_played: int
    encoding: str
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    n_matchdays: int
    max_team_matches: int
    groups: dict[str, bool] = field(default_factory=dict)
    market_coverage: dict[str, int] = field(default_factory=dict)
    from_cache: bool = True


# --- season codes -----------------------------------------------------------------------------

def season_start_year(code: str) -> int:
    """``"9394" -> 1993``, ``"0001" -> 2000``, ``"2526" -> 2025``."""
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"malformed season code {code!r}; expected four digits like '2425'")
    yy = int(code[:2])
    return 1900 + yy if yy >= _CENTURY_PIVOT else 2000 + yy


def season_code(start_year: int) -> str:
    """``1993 -> "9394"``, ``2025 -> "2526"``."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(code: str) -> str:
    """``"2425" -> "2024-25"`` — the human-readable form used in reports."""
    start = season_start_year(code)
    return f"{start}-{(start + 1) % 100:02d}"


def season_codes(first_code: str, through_year: int) -> list[str]:
    """Every season code from ``first_code`` up to the season starting in ``through_year``."""
    first = season_start_year(first_code)
    if through_year < first:
        raise ValueError(f"through_year {through_year} precedes first season {first}")
    return [season_code(y) for y in range(first, through_year + 1)]


def is_in_progress(code: str, current_season_start: int) -> bool:
    """Whether this season's file can still gain rows.

    A completed season's CSV never changes again, so a cached copy of it is permanently valid. The
    in-progress season's grows every matchday, which makes a cached copy stale by construction the
    moment another match is played. The distinction is what lets ``pl ingest`` stay current with
    four requests instead of the 136 that ``--refresh`` costs.
    """
    return season_start_year(code) >= current_season_start


def latest_started_season(today: pd.Timestamp) -> int:
    """The start year of the most recent season to have kicked off.

    A season starting in August means that before July the current season is the previous year's.
    """
    return today.year if today.month >= _SEASON_OPENS_MONTH else today.year - 1


# --- fetch + decode ---------------------------------------------------------------------------

def cache_path(cfg: Config, division: str, code: str) -> Path:
    return cfg.cache_dir / "football-data" / f"{code}_{division}.csv"


def fetch_season(cfg: Config, division: str, code: str, *, refresh: bool = False) -> tuple[Path, bool]:
    """Download a season file into the cache if absent. Returns (path, came_from_cache).

    Nothing reaches the cache until it looks like the requested file. ``raise_for_status`` is not
    sufficient here: the source answers an unpublished season with HTTP 300 and a body that is
    either an HTML page or, worse, a different division's perfectly valid CSV.
    """
    dest = cache_path(cfg, division, code)
    if dest.exists() and not refresh:
        return dest, True
    url = f"{cfg.data.base_url}/{code}/{division}.csv"
    try:
        resp = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
    except requests.RequestException as exc:
        raise IngestError(f"could not fetch {url}: {exc}") from exc
    if resp.status_code != requests.codes.ok:
        raise MissingSeasonError(f"{url} returned HTTP {resp.status_code} (not published yet?)")
    body = resp.content.strip()
    if not body:
        raise IngestError(f"{url} returned an empty body")
    text, _ = decode(body, url)
    if not text.lstrip("﻿").lstrip().startswith(_EXPECTED_FIRST_FIELD):
        raise MissingSeasonError(
            f"{url} did not return a season CSV (body starts {text.lstrip()[:40]!r}); "
            "the source serves an HTML page for seasons it has not published"
        )
    # Check the division before writing, so a substituted file never reaches the cache. Verified
    # necessary on 2026-08-17: `2627/E0.csv` answered 200 with a valid National League CSV.
    served = _first_division(text)
    if served is not None and served != division:
        raise MissingSeasonError(
            f"{url} served division {served!r}, not {division!r} — the source substitutes another "
            "competition's file for a season it has not published"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    return dest, False


def _first_division(text: str) -> str | None:
    """The ``Div`` value of the first data row, or None if the file has no data rows."""
    reader = csv.reader(io.StringIO(text))
    try:
        header = [c.strip() for c in next(reader)]
        col = header.index(_EXPECTED_FIRST_FIELD)
    except (StopIteration, ValueError):
        return None
    for row in reader:
        if len(row) > col and row[col].strip():
            return row[col].strip()
    return None


def _rows_from_text(text: str, source: str) -> tuple[list[str], list[list[str]]]:
    """Split a season file into (header, body rows), absorbing this source's ragged rows.

    Rows wider than the header occur in 17 files across 1993/94-2004/05. Every extra field was
    verified empty across the whole corpus, so they are padded out and dropped — but a non-empty
    one raises, because that would be data the header does not describe.
    """
    rows = [r for r in csv.reader(io.StringIO(text))]
    if not rows:
        raise schema.SchemaError(f"{source}: file is empty")
    header = [c.strip() for c in rows[0]]
    body = [r for r in rows[1:] if any(c.strip() for c in r)]

    widest = max((len(r) for r in body), default=len(header))
    if widest > len(header):
        for row in body:
            extras = [c for c in row[len(header):] if c.strip()]
            if extras:
                raise schema.SchemaError(
                    f"{source}: row wider than header carries data in its extra fields: {extras}"
                )
        header = header + [f"_pad{i}" for i in range(widest - len(header))]

    # Blank header cells become padding too, so a duplicate-name check below stays meaningful.
    header = [c if c else f"_pad_blank{i}" for i, c in enumerate(header)]
    duplicates = sorted({c for c in header if header.count(c) > 1})
    if duplicates:
        raise schema.SchemaError(f"{source}: duplicate column names {duplicates}")

    padded = [row + [""] * (len(header) - len(row)) for row in body]
    return header, padded


def decode(raw: bytes, source: str) -> tuple[str, str]:
    """Decode a season file, returning (text, codec used). Tries utf-8 then cp1252."""
    for codec in _CODECS:
        try:
            return raw.decode(codec), codec
        except UnicodeDecodeError:
            continue
    raise IngestError(f"{source}: undecodable with any of {_CODECS}")


def parse_dates(raw: pd.Series, source: str) -> pd.Series:
    """Parse the source's dayfirst dates, which mix two- and four-digit years.

    Both formats are tried explicitly. Inference is never used: pandas guessing dayfirst wrongly
    would silently transpose day and month for every fixture before the 13th of a month.
    """
    text = raw.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(text[missing], format="%d/%m/%y", errors="coerce")
    still_missing = parsed.isna()
    if still_missing.any():
        offenders = sorted(set(text[still_missing]))
        raise schema.SchemaError(f"{source}: unparseable dates {offenders[:10]}")
    return parsed.dt.normalize()


def _check_season_window(dates: pd.Series, code: str, source: str) -> None:
    """Every date must fall inside its own season — the guard against a century mis-parse."""
    start = season_start_year(code)
    opens = pd.Timestamp(year=start, month=_SEASON_OPENS_MONTH, day=1)
    closes = pd.Timestamp(year=start + 1, month=_SEASON_CLOSES_MONTH, day=31)
    outside = (dates < opens) | (dates > closes)
    if outside.any():
        bad = sorted(set(dates[outside].dt.strftime("%Y-%m-%d")))
        raise schema.SchemaError(
            f"{source}: {int(outside.sum())} date(s) outside the "
            f"{opens.date()}..{closes.date()} season window: {bad[:10]}"
        )


# --- read one season --------------------------------------------------------------------------

def read_season(
    path: Path, division: str, code: str, *,
    aliases: dict[str, str], roster: set[str],
) -> tuple[pd.DataFrame, SeasonMeta]:
    """Read one cached season file into the canonical frame, validating as it goes."""
    source = f"{division} {season_label(code)}"
    text, codec = decode(path.read_bytes(), source)
    header, body = _rows_from_text(text, source)
    raw = pd.DataFrame(body, columns=header, dtype=str)
    raw = raw.loc[:, [c for c in raw.columns if not c.startswith("_pad")]]
    raw = raw.replace({"": None})

    schema.check_core_columns(list(raw.columns), source)
    raw = raw[raw["Div"].notna() & raw["Date"].notna()].copy()

    # The requested division must be the division we got. The source substitutes another file
    # when the requested one does not exist, and that substitute is a valid CSV of the wrong
    # competition — the one failure here that nothing downstream could detect.
    divisions_seen = sorted(raw["Div"].astype(str).str.strip().unique())
    if divisions_seen != [division]:
        raise schema.SchemaError(
            f"{source}: requested division {division!r} but the file contains {divisions_seen}. "
            "The source served a different competition's file; do not trust the cached copy."
        )

    groups = schema.present_groups(list(raw.columns))

    # Assembled as a dict and built in one pass: a recent season carries ~130 odds columns, and
    # inserting them one at a time fragments the frame badly enough for pandas to warn.
    date = parse_dates(raw["Date"], source)
    _check_season_window(date, code, source)
    home_goals = pd.to_numeric(raw["FTHG"], errors="coerce")
    away_goals = pd.to_numeric(raw["FTAG"], errors="coerce")

    cols: dict[str, object] = {
        "date": date,
        "division": division,
        "season": season_label(code),
        "home_team": canonicalise(raw["HomeTeam"], aliases, roster, source=source),
        "away_team": canonicalise(raw["AwayTeam"], aliases, roster, source=source),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result": raw["FTR"].astype(str).str.strip().replace({"": None, "nan": None, "None": None}),
        # The in-progress season's file carries its unplayed fixtures with blank scores. They are
        # kept — they are the fixture list `pl live` and the season simulator need — but flagged,
        # so no fitting or scoring path can pick them up by accident.
        "played": home_goals.notna() & away_goals.notna(),
    }
    for src_col, dest in {**schema.HALFTIME_COLUMNS, **schema.MATCH_STAT_COLUMNS}.items():
        if src_col in raw.columns:
            cols[dest] = pd.to_numeric(raw[src_col], errors="coerce")
    for src_col, dest in schema.TEXT_COLUMNS.items():
        if src_col in raw.columns:
            cols[dest] = raw[src_col].astype(str).str.strip().replace({"": None, "nan": None})
    # Odds pass through unparsed; the odds loader resolves the era-varying ladder and de-vigs.
    for col in raw.columns:
        if schema.is_odds_column(col):
            cols[col] = pd.to_numeric(raw[col], errors="coerce")

    out = pd.DataFrame(cols, index=raw.index)
    out = out.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    out["matchday"] = derive_matchdays(out["date"])
    out["home_match_index"], out["away_match_index"] = derive_team_match_index(
        out["date"], out["home_team"], out["away_team"]
    )
    schema.validate_frame(out, source)

    meta = SeasonMeta(
        division=division,
        season=season_label(code),
        season_code=code,
        n_matches=len(out),
        n_played=int(out["played"].sum()),
        encoding=codec,
        date_min=out["date"].min(),
        date_max=out["date"].max(),
        n_matchdays=int(out["matchday"].nunique()),
        max_team_matches=int(out["home_match_index"].max()),
        groups=groups,
        market_coverage={
            name: int(out[col].notna().sum()) if col in out.columns else 0
            for name, col in _MARKET_PROBES.items()
        },
    )
    return out, meta


# --- the corpus -------------------------------------------------------------------------------

def load_matches(
    cfg: Config,
    *,
    divisions: tuple[str, ...] | None = None,
    through_year: int | None = None,
    refresh: bool = False,
    today: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[SeasonMeta]]:
    """Fetch, validate and concatenate every season file into one canonical frame."""
    divisions = divisions or cfg.data.divisions
    now = today if today is not None else pd.Timestamp.today().normalize()
    last = through_year if through_year is not None else latest_started_season(now)
    codes = season_codes(cfg.data.first_season, last)

    aliases = load_aliases(cfg.static_dir)
    roster = load_roster(cfg.static_dir)

    frames: list[pd.DataFrame] = []
    metas: list[SeasonMeta] = []
    skipped: list[str] = []
    for division in divisions:
        for code in codes:
            try:
                # Always re-fetch the in-progress season: see is_in_progress.
                path, cached = fetch_season(
                    cfg, division, code, refresh=refresh or is_in_progress(code, last),
                )
            except MissingSeasonError as exc:
                # Normal near a season boundary: the source publishes divisions at different
                # times. Recorded so an unexpected gap is still visible in the coverage report.
                skipped.append(f"{division} {season_label(code)}: {exc}")
                continue
            frame, meta = read_season(
                path, division, code,
                aliases=aliases, roster=roster,
            )
            meta.from_cache = cached
            frames.append(frame)
            metas.append(meta)

    if not frames:
        raise IngestError(f"no season files loaded for divisions {divisions}")
    _check_row_floors(metas, cfg, current_season_start=latest_started_season(now))
    corpus = pd.concat(frames, ignore_index=True)
    corpus.attrs["skipped_seasons"] = skipped
    corpus = corpus.sort_values(
        ["date", "division", "home_team"], kind="stable"
    ).reset_index(drop=True)
    _check_no_duplicate_fixtures(corpus)
    return corpus, metas


def _check_row_floors(metas: list[SeasonMeta], cfg: Config, *, current_season_start: int) -> None:
    """Every completed season file must carry at least its division's expected match count.

    The smoke test against a truncated or partial download — the failure that otherwise looks like
    a real dip in a division's fixture count. Seasons still in progress are exempt: they are the
    ones legitimately short, identified by carrying unplayed fixtures or by being the current one.
    """
    floors = cfg.data.min_expected_rows
    if not floors:
        return
    short: list[str] = []
    for m in metas:
        floor = floors.get(m.division)
        if floor is None:
            continue
        in_progress = (
            m.n_played < m.n_matches or season_start_year(m.season_code) >= current_season_start
        )
        if not in_progress and m.n_matches < floor:
            short.append(f"{m.division} {m.season}: {m.n_matches} rows < floor {floor}")
    if short:
        raise schema.SchemaError(
            "season file(s) below the expected row count — suspect a truncated download:\n  "
            + "\n  ".join(short)
        )


def _check_no_duplicate_fixtures(df: pd.DataFrame) -> None:
    """One row per (date, division, home, away). A duplicate means a file was concatenated twice."""
    key = ["date", "division", "home_team", "away_team"]
    dupes = df.duplicated(subset=key, keep=False)
    if dupes.any():
        raise schema.SchemaError(
            f"{int(dupes.sum())} duplicate fixture row(s):\n{df.loc[dupes, key].head(20)}"
        )


def observed_team_names(
    cfg: Config, *, divisions: tuple[str, ...] | None = None, through_year: int | None = None
) -> set[str]:
    """Every raw source team spelling in the cached corpus — the roster-curation input.

    Deliberately bypasses canonicalisation so it can be run *before* the roster exists.
    """
    divisions = divisions or cfg.data.divisions
    last = through_year if through_year is not None else latest_started_season(
        pd.Timestamp.today().normalize()
    )
    names: set[str] = set()
    for division in divisions:
        for code in season_codes(cfg.data.first_season, last):
            path = cache_path(cfg, division, code)
            if not path.exists():
                continue
            source = f"{division} {season_label(code)}"
            text, _ = decode(path.read_bytes(), source)
            header, body = _rows_from_text(text, source)
            raw = pd.DataFrame(body, columns=header, dtype=str)
            if "Div" in raw.columns and sorted(raw["Div"].str.strip().unique()) != [division]:
                raise schema.SchemaError(f"{source}: cached file is not division {division}")
            for col in ("HomeTeam", "AwayTeam"):
                if col in raw.columns:
                    names.update(raw[col].dropna().astype(str).str.strip())
    return {n for n in names if n}


# --- the pre-match fixture feed -----------------------------------------------------------------
#
# A season file appears only once it has results in it. That is fine for everything historical and
# useless for the one job that cannot be redone later: freezing a forecast before the first ball of
# a season is kicked. The source publishes a separate rolling feed of upcoming fixtures, and this
# reads it into the same canonical frame the season files produce, so `pl live` takes one path
# whichever supplied the row.

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"


def fetch_fixtures(
    cfg: Config, *, divisions: tuple[str, ...] | None = None, today: pd.Timestamp | None = None
) -> pd.DataFrame:
    """Upcoming fixtures from the rolling feed, canonicalised and flagged unplayed.

    Returns an empty frame with the right columns when the feed carries nothing for the requested
    divisions, which is the normal state for most of the week rather than an error.

    The team names go through the same alias map and the same closed roster as the ingest path. A
    fixture naming a club the roster does not know is a name change or a promotion the static files
    have not caught up with, and it must be fixed deliberately rather than silently admitted.
    """
    wanted = tuple(divisions or (cfg.backtest.prediction_division,))
    now = today if today is not None else pd.Timestamp.today().normalize()
    try:
        resp = requests.get(FIXTURES_URL, timeout=_DOWNLOAD_TIMEOUT)
    except requests.RequestException as exc:
        raise IngestError(f"could not fetch {FIXTURES_URL}: {exc}") from exc
    if resp.status_code != requests.codes.ok:
        raise IngestError(f"{FIXTURES_URL} returned HTTP {resp.status_code}")
    text, _ = decode(resp.content, FIXTURES_URL)
    if not text.lstrip("\ufeff").lstrip().startswith(_EXPECTED_FIRST_FIELD):
        raise IngestError(
            f"{FIXTURES_URL} did not return a fixtures CSV (body starts {text.lstrip()[:40]!r})"
        )

    header, body = _rows_from_text(text, FIXTURES_URL)
    raw = pd.DataFrame(body, columns=header, dtype=str)
    raw = raw.loc[:, [c for c in raw.columns if not c.startswith("_pad")]].replace({"": None})
    raw = raw[raw["Div"].notna() & raw["Date"].notna()]
    raw = raw[raw["Div"].astype(str).str.strip().isin(wanted)].copy()

    columns = ["date", "division", "season", "home_team", "away_team", "home_goals",
               "away_goals", "result", "played", "matchday", "home_match_index",
               "away_match_index"]
    if raw.empty:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in columns})

    aliases = load_aliases(cfg.static_dir)
    roster = load_roster(cfg.static_dir)
    date = parse_dates(raw["Date"], FIXTURES_URL)
    cols: dict[str, object] = {
        "date": date,
        "division": raw["Div"].astype(str).str.strip(),
        "season": [season_label(season_code(latest_started_season(d))) for d in date],
        "home_team": canonicalise(raw["HomeTeam"], aliases, roster, source=FIXTURES_URL),
        "away_team": canonicalise(raw["AwayTeam"], aliases, roster, source=FIXTURES_URL),
        # No scores exist yet, and that is the point: these rows are the fixture list, never
        # training data. `played` is False for every one of them by construction.
        "home_goals": pd.Series([pd.NA] * len(raw), index=raw.index, dtype="Float64"),
        "away_goals": pd.Series([pd.NA] * len(raw), index=raw.index, dtype="Float64"),
        "result": None,
        "played": False,
    }
    for src_col, dest in schema.TEXT_COLUMNS.items():
        if src_col in raw.columns:
            cols[dest] = raw[src_col].astype(str).str.strip().replace({"": None, "nan": None})
    for col in raw.columns:
        if schema.is_odds_column(col):
            cols[col] = pd.to_numeric(raw[col], errors="coerce")

    out = pd.DataFrame(cols, index=raw.index)
    out = out[out["date"] >= now]
    out = out.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
    out["matchday"] = derive_matchdays(out["date"]) if len(out) else pd.Series(dtype=int)
    if len(out):
        out["home_match_index"], out["away_match_index"] = derive_team_match_index(
            out["date"], out["home_team"], out["away_team"]
        )
    else:
        out["home_match_index"] = out["away_match_index"] = pd.Series(dtype=int)
    return out


def upcoming_fixtures(
    cfg: Config, corpus: pd.DataFrame, *, division: str | None = None,
    use_feed: bool = True, today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Every unplayed fixture known for a division, from the corpus and the rolling feed.

    The corpus wins on any fixture both describe: once the season file carries a row, that row is
    the one every other command sees, and a live ledger built off a second description of the same
    match would be scored against the first.
    """
    div = division or cfg.backtest.prediction_division
    known = corpus[(corpus["division"] == div) & ~corpus["played"]]
    if not use_feed:
        return known.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)

    feed = fetch_fixtures(cfg, divisions=(div,), today=today)
    if feed.empty:
        return known.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)

    seen = {
        (pd.Timestamp(r.date).normalize(), r.home_team, r.away_team)
        for r in corpus[corpus["division"] == div].itertuples(index=False)
    }
    fresh = feed[[
        (pd.Timestamp(r.date).normalize(), r.home_team, r.away_team) not in seen
        for r in feed.itertuples(index=False)
    ]]
    combined = pd.concat([known, fresh], ignore_index=True)
    return combined.sort_values(["date", "home_team"], kind="stable").reset_index(drop=True)
