"""The rolling pre-match fixture feed.

A season file appears only once it carries results, so a ledger built from the corpus alone cannot
freeze the first matchday of a season — the one forecast that provably cannot be reconstructed
afterwards. The feed closes that hole, and everything below is about it closing it *safely*: the
same roster guard as the ingest path, no scores admitted, and the corpus winning any disagreement.
"""
from __future__ import annotations

import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.data import football_data as fd
from plmodel.data.teams import TeamNameError

HEADER = "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A"


def _feed(rows: str) -> bytes:
    return (HEADER + "\n" + rows).encode("utf-8")


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def serve(monkeypatch):
    """Point the feed reader at a body of our choosing, without touching the network."""
    def _serve(body: bytes, status: int = 200):
        class Response:
            status_code = status
            content = body

        monkeypatch.setattr(fd.requests, "get", lambda *a, **k: Response())
    return _serve


def test_a_fixture_is_read_into_the_canonical_frame(cfg, serve) -> None:
    serve(_feed("E0,22/08/2026,15:00,Arsenal,Chelsea,2.10,3.40,3.50\n"))
    got = fd.fetch_fixtures(cfg, divisions=("E0",), today=pd.Timestamp("2026-08-21"))
    assert len(got) == 1
    row = got.iloc[0]
    assert row["date"] == pd.Timestamp("2026-08-22")
    assert (row["home_team"], row["away_team"]) == ("Arsenal", "Chelsea")
    assert row["division"] == "E0" and row["season"] == "2026-27"
    assert row["B365H"] == pytest.approx(2.10)


def test_a_feed_row_can_never_carry_a_result(cfg, serve) -> None:
    """These rows are the fixture list. Nothing may fit or score on them."""
    serve(_feed("E0,22/08/2026,15:00,Arsenal,Chelsea,2.10,3.40,3.50\n"))
    got = fd.fetch_fixtures(cfg, divisions=("E0",), today=pd.Timestamp("2026-08-21"))
    assert not got["played"].any()
    assert got["home_goals"].isna().all() and got["away_goals"].isna().all()
    assert got["result"].isna().all()


def test_only_the_requested_division_is_returned(cfg, serve) -> None:
    """The feed carries several competitions, and a Spanish club is not a roster failure."""
    serve(_feed(
        "E0,22/08/2026,15:00,Arsenal,Chelsea,2.10,3.40,3.50\n"
        "SP1,22/08/2026,20:00,Rayo Vallecano,Alaves,2.00,3.30,3.90\n"
    ))
    got = fd.fetch_fixtures(cfg, divisions=("E0",), today=pd.Timestamp("2026-08-21"))
    assert list(got["division"].unique()) == ["E0"]
    assert "Rayo Vallecano" not in set(got["home_team"])


def test_stale_rows_are_dropped(cfg, serve) -> None:
    """The feed lags: it was observed carrying two-day-old fixtures. Those are not upcoming."""
    serve(_feed(
        "E0,19/08/2026,20:00,Arsenal,Chelsea,2.10,3.40,3.50\n"
        "E0,22/08/2026,15:00,Everton,Fulham,2.10,3.40,3.50\n"
    ))
    got = fd.fetch_fixtures(cfg, divisions=("E0",), today=pd.Timestamp("2026-08-21"))
    assert list(got["home_team"]) == ["Everton"]


def test_an_unknown_club_stops_the_feed_rather_than_entering_the_corpus(cfg, serve) -> None:
    """The ingest-path guard applies here unchanged: a near-miss is how one club becomes two."""
    serve(_feed("E0,22/08/2026,15:00,Arsenal FC,Chelsea,2.10,3.40,3.50\n"))
    with pytest.raises(TeamNameError, match="not on the roster"):
        fd.fetch_fixtures(cfg, divisions=("E0",), today=pd.Timestamp("2026-08-21"))


def test_the_observed_feed_spellings_resolve(cfg, serve) -> None:
    """'Sheffield Wed' and 'Bradford City' are how the feed spells them; both were added by hand."""
    serve(_feed("E2,22/08/2026,15:00,Sheffield Wed,Bradford City,2.10,3.40,3.50\n"))
    got = fd.fetch_fixtures(cfg, divisions=("E2",), today=pd.Timestamp("2026-08-21"))
    assert (got.iloc[0]["home_team"], got.iloc[0]["away_team"]) == ("Sheffield Weds", "Bradford")


def test_an_empty_feed_is_normal_rather_than_an_error(cfg, serve) -> None:
    serve(_feed("SP1,22/08/2026,20:00,Rayo Vallecano,Alaves,2.00,3.30,3.90\n"))
    got = fd.fetch_fixtures(cfg, divisions=("E0",), today=pd.Timestamp("2026-08-21"))
    assert got.empty and "home_team" in got.columns


def test_a_non_csv_body_is_refused(cfg, serve) -> None:
    serve(b"<html>we are down for maintenance</html>")
    with pytest.raises(fd.IngestError, match="did not return a fixtures CSV"):
        fd.fetch_fixtures(cfg, divisions=("E0",))


def test_an_http_error_is_refused(cfg, serve) -> None:
    serve(_feed(""), status=500)
    with pytest.raises(fd.IngestError, match="HTTP 500"):
        fd.fetch_fixtures(cfg, divisions=("E0",))


# --- merging the two sources --------------------------------------------------------------------

_CORPUS_COLUMNS = ["date", "division", "season", "home_team", "away_team", "played"]


def _corpus(rows: list[tuple[str, str, str, bool]]) -> pd.DataFrame:
    """A corpus slice. Empty is the interesting case: it is a season file with no rows yet."""
    frame = pd.DataFrame([
        {"date": pd.Timestamp(d), "division": "E0", "season": "2026-27",
         "home_team": h, "away_team": a, "played": p,
         "home_goals": 1.0 if p else None, "away_goals": 0.0 if p else None}
        for d, h, a, p in rows
    ], columns=_CORPUS_COLUMNS + ["home_goals", "away_goals"])
    return frame.astype({"played": bool})


def test_the_corpus_wins_a_fixture_both_sources_describe(cfg, serve) -> None:
    """Two descriptions of one match would be frozen against the wrong one when scored."""
    serve(_feed("E0,22/08/2026,15:00,Arsenal,Chelsea,9.99,9.99,9.99\n"))
    corpus = _corpus([("2026-08-22", "Arsenal", "Chelsea", False)])
    got = fd.upcoming_fixtures(cfg, corpus, today=pd.Timestamp("2026-08-21"))
    assert len(got) == 1
    assert "B365H" not in got.columns or pd.isna(got.iloc[0].get("B365H"))


def test_the_feed_supplies_what_the_corpus_has_never_seen(cfg, serve) -> None:
    """The case this exists for: a season file with no rows in it yet."""
    serve(_feed("E0,22/08/2026,15:00,Arsenal,Chelsea,2.10,3.40,3.50\n"))
    got = fd.upcoming_fixtures(cfg, _corpus([]), today=pd.Timestamp("2026-08-21"))
    assert len(got) == 1 and got.iloc[0]["home_team"] == "Arsenal"


def test_a_played_fixture_is_never_resurrected_by_the_feed(cfg, serve) -> None:
    serve(_feed("E0,22/08/2026,15:00,Arsenal,Chelsea,2.10,3.40,3.50\n"))
    corpus = _corpus([("2026-08-22", "Arsenal", "Chelsea", True)])
    got = fd.upcoming_fixtures(cfg, corpus, today=pd.Timestamp("2026-08-21"))
    assert got.empty


def test_the_feed_can_be_switched_off(cfg, serve) -> None:
    serve(_feed("E0,22/08/2026,15:00,Arsenal,Chelsea,2.10,3.40,3.50\n"))
    assert fd.upcoming_fixtures(cfg, _corpus([]), use_feed=False).empty
