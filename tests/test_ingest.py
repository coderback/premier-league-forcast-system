"""The data spine.

The tests that matter here are the ones guarding against a frame that looks right and is wrong —
a mis-parsed date, a silently substituted division, a team name that quietly became a second club.
Each of these was found in the real source (see NOTES.md), not imagined.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.data import schema
from plmodel.data.football_data import (
    MissingSeasonError, _first_division, _rows_from_text, _check_row_floors, SeasonMeta,
    latest_started_season, parse_dates, read_season, season_code, season_codes, season_label,
    season_start_year,
)
from plmodel.data.fixtures import derive_matchdays, derive_team_match_index
from plmodel.data.teams import (
    TeamNameError, canonicalise, find_near_duplicates, load_aliases, load_roster,
)

STATIC_DIR = Path(__file__).resolve().parents[1] / "data" / "static"

_HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR"


def _write_season(tmp_path: Path, rows: str, *, header: str = _HEADER, name: str = "s.csv",
                  encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(f"{header}\n{rows}".encode(encoding))
    return path


def _read(path: Path, division: str = "E0", code: str = "2425", **kw):
    defaults = {"aliases": {}, "roster": {"Arsenal", "Chelsea", "Everton", "Fulham"}}
    return read_season(path, division, code, **{**defaults, **kw})


# --- team names -------------------------------------------------------------------------------

def test_unmapped_team_name_raises() -> None:
    """An unrecognised name must raise, never be dropped or fuzzy-matched."""
    names = pd.Series(["Arsenal", "Arsnal FC"])
    with pytest.raises(TeamNameError, match="not on the roster"):
        canonicalise(names, {}, {"Arsenal"})


def test_alias_resolves_before_the_roster_check() -> None:
    out = canonicalise(pd.Series(["Man United"]), {"Man United": "Manchester United"},
                       {"Manchester United"})
    assert list(out) == ["Manchester United"]


def test_team_names_are_stripped() -> None:
    assert list(canonicalise(pd.Series(["  Arsenal "]), {}, {"Arsenal"})) == ["Arsenal"]


def test_committed_roster_loads_and_is_closed() -> None:
    roster = load_roster(STATIC_DIR)
    assert len(roster) > 100, "the roster should cover every club in E0-E3 since 1993/94"
    assert "Arsenal" in roster and "Man United" in roster


def test_committed_roster_has_no_near_duplicates() -> None:
    """Two spellings of one club would silently split its history in two."""
    assert find_near_duplicates(load_roster(STATIC_DIR)) == []


def test_wimbledon_entities_stay_distinct() -> None:
    """Three separate clubs that fuzzy matching would wrongly merge."""
    roster = load_roster(STATIC_DIR)
    assert {"Wimbledon", "AFC Wimbledon", "Milton Keynes Dons"} <= roster


def test_near_duplicate_detector_has_teeth() -> None:
    assert find_near_duplicates({"Nott'm Forest", "Nottm Forest"})


def test_alias_file_loads() -> None:
    assert isinstance(load_aliases(STATIC_DIR), dict)


def test_missing_team_file_raises(tmp_path: Path) -> None:
    with pytest.raises(TeamNameError, match="not found"):
        load_roster(tmp_path)


# --- season codes -----------------------------------------------------------------------------

@pytest.mark.parametrize(("code", "year"), [("9394", 1993), ("9900", 1999), ("0001", 2000),
                                            ("2425", 2024), ("2526", 2025)])
def test_season_start_year(code: str, year: int) -> None:
    assert season_start_year(code) == year
    assert season_code(year) == code


def test_season_label() -> None:
    assert season_label("2425") == "2024-25"
    assert season_label("9900") == "1999-00"


def test_season_codes_span() -> None:
    codes = season_codes("9394", 2025)
    assert codes[0] == "9394" and codes[-1] == "2526" and len(codes) == 33


def test_malformed_season_code_raises() -> None:
    with pytest.raises(ValueError, match="malformed season code"):
        season_start_year("94")


def test_latest_started_season_respects_the_august_boundary() -> None:
    assert latest_started_season(pd.Timestamp("2026-08-17")) == 2026
    assert latest_started_season(pd.Timestamp("2026-05-17")) == 2025


# --- date parsing -----------------------------------------------------------------------------

def test_parses_two_and_four_digit_years() -> None:
    out = parse_dates(pd.Series(["14/08/93", "17/08/2002"]), "test")
    assert list(out) == [pd.Timestamp("1993-08-14"), pd.Timestamp("2002-08-17")]


def test_dates_are_dayfirst() -> None:
    """03/04 is 3 April, not 4 March. Inference would get two thirds of the corpus wrong."""
    assert parse_dates(pd.Series(["03/04/1995"]), "test")[0] == pd.Timestamp("1995-04-03")


def test_unparseable_date_raises() -> None:
    with pytest.raises(schema.SchemaError, match="unparseable dates"):
        parse_dates(pd.Series(["not-a-date"]), "test")


def test_date_outside_the_season_window_raises(tmp_path: Path) -> None:
    path = _write_season(tmp_path, "E0,14/08/2019,Arsenal,Chelsea,1,0,H\n")
    with pytest.raises(schema.SchemaError, match="season window"):
        _read(path, code="2425")


# --- the source's structural quirks -----------------------------------------------------------

def test_cp1252_season_file_decodes(tmp_path: Path) -> None:
    """2004/05 is cp1252 across all four divisions; utf-8 raises on byte 0xa0."""
    path = tmp_path / "s.csv"
    path.write_bytes(_HEADER.encode() + b"\nE0,14/08/2004,Arsenal,Chelsea,1,0,H\xa0\n")
    frame, meta = _read(path, code="0405")
    assert meta.encoding == "cp1252" and len(frame) == 1


def test_bom_is_stripped(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_bytes("﻿".encode() + f"{_HEADER}\nE0,17/08/2024,Arsenal,Chelsea,1,0,H\n".encode())
    frame, _ = _read(path)
    assert len(frame) == 1


def test_ragged_rows_with_empty_extras_are_absorbed() -> None:
    """17 files across 1993/94-2004/05 have rows wider than their header — all extras empty."""
    header, body = _rows_from_text(f"{_HEADER}\nE0,17/08/2024,Arsenal,Chelsea,1,0,H,,,\n", "t")
    assert body[0][:7] == ["E0", "17/08/2024", "Arsenal", "Chelsea", "1", "0", "H"]
    assert all(c.startswith("_pad") for c in header[7:])


def test_ragged_row_carrying_data_raises() -> None:
    """An extra field with content is data the header does not describe — never dropped."""
    with pytest.raises(schema.SchemaError, match="extra fields"):
        _rows_from_text(f"{_HEADER}\nE0,17/08/2024,Arsenal,Chelsea,1,0,H,SURPRISE\n", "t")


def test_blank_rows_are_dropped() -> None:
    _, body = _rows_from_text(f"{_HEADER}\nE0,17/08/2024,Arsenal,Chelsea,1,0,H\n,,,,,,\n", "t")
    assert len(body) == 1


def test_duplicate_column_names_raise() -> None:
    with pytest.raises(schema.SchemaError, match="duplicate column"):
        _rows_from_text("Div,Date,Div\nE0,17/08/2024,E0\n", "t")


def test_missing_core_column_raises(tmp_path: Path) -> None:
    path = _write_season(tmp_path, "E0,17/08/2024,Arsenal,Chelsea\n",
                         header="Div,Date,HomeTeam,AwayTeam")
    with pytest.raises(schema.SchemaError, match="missing core columns"):
        _read(path)


# --- the substituted-division guard -----------------------------------------------------------

def test_wrong_division_raises(tmp_path: Path) -> None:
    """The source answers an unpublished season with another competition's valid CSV.

    Verified 2026-08-17: `2627/E0.csv` returned the National League file. Nothing downstream
    could detect Premier League rows that are really National League matches, so the check that
    the served division matches the requested one is load-bearing.
    """
    path = _write_season(tmp_path, "EC,17/08/2024,Arsenal,Chelsea,1,0,H\n")
    with pytest.raises(schema.SchemaError, match="requested division"):
        _read(path, division="E0")


def test_first_division_probe() -> None:
    assert _first_division(f"{_HEADER}\nEC,17/08/2024,Arsenal,Chelsea,1,0,H\n") == "EC"
    assert _first_division("not,a,csv\n") is None


# --- validation invariants --------------------------------------------------------------------

def test_ftr_disagreeing_with_goals_raises(tmp_path: Path) -> None:
    """The brief requires verifying the source's own full-time conventions, not trusting them."""
    path = _write_season(tmp_path, "E0,17/08/2024,Arsenal,Chelsea,1,0,A\n")
    with pytest.raises(schema.SchemaError, match="FTR disagrees"):
        _read(path)


def test_team_playing_itself_raises(tmp_path: Path) -> None:
    path = _write_season(tmp_path, "E0,17/08/2024,Arsenal,Arsenal,1,0,H\n")
    with pytest.raises(schema.SchemaError, match="plays itself"):
        _read(path)


def test_unplayed_fixtures_are_kept_and_flagged(tmp_path: Path) -> None:
    """The in-progress season's file carries its fixture list with blank scores."""
    path = _write_season(
        tmp_path,
        "E0,17/08/2024,Arsenal,Chelsea,1,0,H\nE0,24/08/2024,Everton,Fulham,,,\n",
    )
    frame, meta = _read(path)
    assert len(frame) == 2 and meta.n_played == 1
    assert list(frame["played"]) == [True, False]


def test_negative_counts_raise(tmp_path: Path) -> None:
    path = _write_season(tmp_path, "E0,17/08/2024,Arsenal,Chelsea,-1,0,A\n")
    with pytest.raises(schema.SchemaError, match="negative values"):
        _read(path)


# --- calendar derivation ----------------------------------------------------------------------

def test_matchdays_index_distinct_dates() -> None:
    dates = pd.Series(pd.to_datetime(["2024-08-17", "2024-08-17", "2024-08-24", "2024-08-20"]))
    assert list(derive_matchdays(dates)) == [1, 1, 3, 2]


def test_matchdays_are_order_independent() -> None:
    """The blocks are the splits, so a non-deterministic derivation makes backtests irreproducible."""
    dates = pd.to_datetime(["2024-08-24", "2024-08-17", "2024-08-20"])
    forward = derive_matchdays(pd.Series(dates))
    reversed_ = derive_matchdays(pd.Series(dates[::-1]))
    assert list(forward) == [3, 1, 2] and list(reversed_) == [2, 1, 3]


def test_team_match_index_counts_each_side_separately() -> None:
    dates = pd.Series(pd.to_datetime(["2024-08-17", "2024-08-24", "2024-08-31"]))
    home = pd.Series(["Arsenal", "Chelsea", "Arsenal"])
    away = pd.Series(["Chelsea", "Everton", "Everton"])
    h, a = derive_team_match_index(dates, home, away)
    assert list(h) == [1, 2, 2]   # Arsenal's 1st, Chelsea's 2nd, Arsenal's 2nd
    assert list(a) == [1, 1, 2]   # Chelsea's 1st, Everton's 1st, Everton's 2nd


# --- row-count floors -------------------------------------------------------------------------

def _meta(division: str, season: str, n: int, played: int | None = None) -> SeasonMeta:
    return SeasonMeta(
        division=division, season=season, season_code="2223", n_matches=n,
        n_played=n if played is None else played, encoding="utf-8-sig",
        date_min=pd.Timestamp("2022-08-01"), date_max=pd.Timestamp("2023-05-01"),
        n_matchdays=100, max_team_matches=38,
    )


def test_row_floor_catches_a_truncated_season() -> None:
    cfg = load_config()
    with pytest.raises(schema.SchemaError, match="below the expected row count"):
        _check_row_floors([_meta("E0", "2022-23", 200)], cfg, current_season_start=2025)


def test_row_floor_exempts_a_season_in_progress() -> None:
    cfg = load_config()
    _check_row_floors([_meta("E0", "2022-23", 200, played=100)], cfg, current_season_start=2025)
    _check_row_floors([_meta("E0", "2022-23", 200)], cfg, current_season_start=2022)


def test_completed_seasons_meet_their_floor() -> None:
    cfg = load_config()
    _check_row_floors([_meta("E0", "2022-23", 380)], cfg, current_season_start=2025)


# --- the real corpus --------------------------------------------------------------------------

@pytest.mark.integration
def test_real_corpus_loads() -> None:
    from plmodel.data.football_data import load_matches

    cfg = load_config()
    corpus, metas = load_matches(cfg)
    assert len(corpus) > 60_000
    e0 = corpus[corpus["division"] == "E0"]
    assert len(e0) == 12_704, "E0 1993/94-2025/26 is 462 + 462 + 31 * 380"
    assert corpus["date"].min() == pd.Timestamp("1993-08-14")
    # 2004/05 is the only cp1252 era, and it is cp1252 in all four divisions.
    assert {m.season for m in metas if m.encoding == "cp1252"} == {"2004-05"}


def test_canonical_frame_carries_the_identity_columns(tmp_path: Path) -> None:
    """Every downstream join keys on these; a rename must break here, not silently downstream."""
    path = _write_season(tmp_path, "E0,17/08/2024,Arsenal,Chelsea,1,0,H\n")
    frame, _ = _read(path)
    assert set(schema.IDENTITY_COLUMNS) <= set(frame.columns)


@pytest.mark.integration
def test_real_corpus_team_match_counts_are_exact() -> None:
    """Every deviation from the standard season length must be real history, not a parse bug."""
    from plmodel.data.football_data import load_matches

    cfg = load_config()
    corpus, _ = load_matches(cfg, divisions=("E0",))
    per_season = corpus.groupby("season")["home_match_index"].max()
    # 42 for the 22-team seasons, 38 thereafter. Nothing else is acceptable for a completed E0.
    assert set(per_season.unique()) == {38, 42}
    assert set(per_season[per_season == 42].index) == {"1993-94", "1994-95"}
