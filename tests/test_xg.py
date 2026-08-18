"""The expected-goals source: loading, joining, and the check that makes it trustworthy.

This data comes from a third-party mirror of a site we cannot verify against directly, so the
tests here are mostly about refusing to believe it. The score-agreement guard is the important
one: two independent providers must agree on what actually happened before a quantity derived from
one of them is used.
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.data.xg import (
    DATASET_LICENCE, DATASET_REF, JOIN_TOLERANCE_DAYS, MEMBER, REQUIRED_COLUMNS,
    ExpectedGoalsError, attach, cache_path, coverage_summary, load_raw,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _matches(rows: list[tuple[str, str, str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime([r[0] for r in rows]),
        "home_team": [r[1] for r in rows],
        "away_team": [r[2] for r in rows],
        "home_goals": [r[3] for r in rows],
        "away_goals": [r[4] for r in rows],
    })


def _xg(rows: list[tuple[str, str, str, float, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime([r[0] for r in rows]),
        "home_team": [r[1] for r in rows],
        "away_team": [r[2] for r in rows],
        "home_xg": [r[3] for r in rows],
        "away_xg": [r[4] for r in rows],
        "xg_home_goals": [r[5] for r in rows],
        "xg_away_goals": [r[6] for r in rows],
    })


# --- provenance is recorded in the code, not just in prose --------------------------------------

def test_dataset_identity_is_pinned() -> None:
    """Which mirror and which licence must be greppable from the source, not only the ledger."""
    assert DATASET_REF == "yarknyorulmaz/understat-match-team-metrics-dataset-epl-v16-v24"
    assert "ODbL" in DATASET_LICENCE or "Open Database" in DATASET_LICENCE
    assert set(REQUIRED_COLUMNS) >= {"date", "team_h", "team_a", "h_xg", "a_xg"}


# --- the join ---------------------------------------------------------------------------------

def test_exact_join_attaches_xg() -> None:
    matches = _matches([("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0)])
    out = attach(matches, _xg([("2020-01-01", "Arsenal", "Chelsea", 1.8, 0.9, 2.0, 1.0)]))
    assert out.loc[0, "home_xg"] == pytest.approx(1.8)
    assert out.loc[0, "away_xg"] == pytest.approx(0.9)


def test_join_tolerates_a_one_day_kickoff_offset() -> None:
    """Understat timestamps by kickoff, football-data by calendar date, so a late kickoff can land
    either side of midnight. 26 of 3,420 real rows need this, all at exactly one day."""
    matches = _matches([("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0)])
    for offset in (1, -1):
        source_date = (pd.Timestamp("2020-01-01") + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
        out = attach(matches, _xg([(source_date, "Arsenal", "Chelsea", 1.8, 0.9, 2.0, 1.0)]))
        assert out.loc[0, "home_xg"] == pytest.approx(1.8)


def test_join_does_not_reach_beyond_the_tolerance() -> None:
    """A fixture must never inherit a different meeting's xG."""
    matches = _matches([("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0)])
    far = (pd.Timestamp("2020-01-01") + pd.Timedelta(days=JOIN_TOLERANCE_DAYS + 1)).strftime("%Y-%m-%d")
    out = attach(matches, _xg([(far, "Arsenal", "Chelsea", 1.8, 0.9, 2.0, 1.0)]))
    assert np.isnan(out.loc[0, "home_xg"])


def test_exact_date_wins_over_a_neighbouring_one() -> None:
    """Both meetings exist; the same-day one must be chosen."""
    matches = _matches([("2020-01-02", "Arsenal", "Chelsea", 2.0, 1.0)])
    source = _xg([
        ("2020-01-01", "Arsenal", "Chelsea", 9.9, 9.9, 2.0, 1.0),
        ("2020-01-02", "Arsenal", "Chelsea", 1.8, 0.9, 2.0, 1.0),
    ])
    assert attach(matches, source).loc[0, "home_xg"] == pytest.approx(1.8)


def test_uncovered_matches_are_nan_not_imputed() -> None:
    """Coverage is a value. A match the source does not carry must not acquire an xG."""
    matches = _matches([
        ("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0),
        ("2013-01-01", "Everton", "Fulham", 1.0, 1.0),      # before the source begins
    ])
    out = attach(matches, _xg([("2020-01-01", "Arsenal", "Chelsea", 1.8, 0.9, 2.0, 1.0)]))
    assert out["home_xg"].notna().tolist() == [True, False]


# --- the guard that makes the source trustworthy -------------------------------------------------

def test_a_score_disagreement_raises() -> None:
    """The load-bearing check.

    Two independent providers must agree on what actually happened before a quantity derived from
    one of them is believed. A mismatch means the join is wrong or a source is corrupt, and either
    way the run must stop rather than quietly model fabricated numbers. On the real corpus this
    passes on all 3,394 joined rows.
    """
    matches = _matches([("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0)])
    wrong = _xg([("2020-01-01", "Arsenal", "Chelsea", 1.8, 0.9, 3.0, 1.0)])
    with pytest.raises(ExpectedGoalsError, match="disagrees with the corpus"):
        attach(matches, wrong)


def test_the_disagreement_message_names_the_match() -> None:
    """A guard that fires without saying which row is far less useful."""
    matches = _matches([("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0)])
    wrong = _xg([("2020-01-01", "Arsenal", "Chelsea", 1.8, 0.9, 0.0, 4.0)])
    with pytest.raises(ExpectedGoalsError, match=r"Arsenal v Chelsea"):
        attach(matches, wrong)


# --- coverage reporting --------------------------------------------------------------------------

def test_coverage_summary_reports_the_cliff() -> None:
    matches = _matches([
        ("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0),
        ("2025-01-01", "Everton", "Fulham", 1.0, 1.0),
    ]).assign(season=["2019-20", "2024-25"])
    out = attach(matches, _xg([("2020-01-01", "Arsenal", "Chelsea", 1.8, 0.9, 2.0, 1.0)]))
    summary = coverage_summary(out)
    assert summary["n_covered"] == 1 and summary["n_total"] == 2
    assert summary["seasons_covered"] == ["2019-20"]
    assert summary["seasons_uncovered"] == ["2024-25"]
    assert "2026/27" in summary["discontinuity"]
    assert summary["licence"] == DATASET_LICENCE


def test_coverage_summary_names_the_provenance() -> None:
    """Anyone reading a report must be able to see where the numbers came from."""
    matches = _matches([("2020-01-01", "Arsenal", "Chelsea", 2.0, 1.0)])
    joined = attach(matches, _xg([("2020-01-01", "Arsenal", "Chelsea", 1.8, 0.9, 2.0, 1.0)]))
    summary = coverage_summary(joined)
    assert "understat" in summary["source"].lower()
    assert "disallows crawling" in summary["source"]


# --- loading and validation ------------------------------------------------------------------------

def _write_archive(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(MEMBER, buffer.getvalue())


def _source_frame(**overrides) -> pd.DataFrame:
    base = {
        "date": ["2020-01-01 15:00:00"], "team_h": ["Manchester United"], "team_a": ["Chelsea"],
        "h_goals": [2], "a_goals": [1], "h_xg": [1.8], "a_xg": [0.9],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_load_raw_applies_aliases(cfg, tmp_path, monkeypatch) -> None:
    """Understat's full club names must resolve to the corpus's short ones."""
    import dataclasses

    local = dataclasses.replace(cfg, cache_dir=tmp_path)
    _write_archive(cache_path(local), _source_frame())
    out = load_raw(local)
    assert out.loc[0, "home_team"] == "Man United"


def test_load_raw_rejects_a_missing_column(cfg, tmp_path) -> None:
    import dataclasses

    local = dataclasses.replace(cfg, cache_dir=tmp_path)
    _write_archive(cache_path(local), _source_frame().drop(columns=["h_xg"]))
    with pytest.raises(ExpectedGoalsError, match="missing columns"):
        load_raw(local)


def test_load_raw_rejects_negative_expected_goals(cfg, tmp_path) -> None:
    import dataclasses

    local = dataclasses.replace(cfg, cache_dir=tmp_path)
    _write_archive(cache_path(local), _source_frame(h_xg=[-0.5]))
    with pytest.raises(ExpectedGoalsError, match="negative expected goals"):
        load_raw(local)


def test_load_raw_rejects_an_unknown_team(cfg, tmp_path) -> None:
    """Same fail-loudly discipline as the match ingest: never fuzzy-match a club."""
    import dataclasses

    local = dataclasses.replace(cfg, cache_dir=tmp_path)
    _write_archive(cache_path(local), _source_frame(team_h=["Real Madrid"]))
    with pytest.raises(Exception, match="not on the roster"):
        load_raw(local)


def test_missing_archive_gives_an_actionable_message(cfg, tmp_path) -> None:
    import dataclasses

    with pytest.raises(ExpectedGoalsError, match="pl ingest --with-xg"):
        load_raw(dataclasses.replace(cfg, cache_dir=tmp_path))


# --- the real corpus --------------------------------------------------------------------------------

@pytest.mark.integration
def test_real_xg_source_agrees_with_the_corpus(cfg) -> None:
    """The whole validation, run end to end: if the mirror were wrong, attach() would raise."""
    if not cache_path(cfg).exists():
        pytest.skip("run `pl ingest --with-xg` first")
    corpus = pd.read_parquet(cfg.cache_dir / "matches.parquet")
    e0 = corpus[(corpus["division"] == "E0") & corpus["played"]]
    joined = attach(e0.drop(columns=["home_xg", "away_xg"], errors="ignore"), load_raw(cfg))
    summary = coverage_summary(joined)
    assert summary["n_covered"] == 3420
    assert summary["first_covered"] == "2015-08-08"
    assert summary["last_covered"] == "2024-05-19"


@pytest.mark.integration
def test_real_xg_is_calibrated_against_realised_goals(cfg) -> None:
    """An independent sanity check on the mirror: xG should be roughly unbiased against outcomes.

    A feed that had been rescaled, truncated or fabricated would fail this even if it happened to
    join correctly.
    """
    if not cache_path(cfg).exists():
        pytest.skip("run `pl ingest --with-xg` first")
    corpus = pd.read_parquet(cfg.cache_dir / "matches.parquet")
    covered = corpus[corpus["home_xg"].notna()]
    assert abs(covered["home_xg"].mean() - covered["home_goals"].mean()) < 0.05
    assert abs(covered["away_xg"].mean() - covered["away_goals"].mean()) < 0.05
    # Home xG must exceed away xG, for the same reason home teams score more.
    assert covered["home_xg"].mean() > covered["away_xg"].mean()
