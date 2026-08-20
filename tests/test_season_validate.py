"""Scoring a long-horizon forecast, and the two things that make it easy to get wrong.

**The sample size is seasons, not team-seasons.** Twenty rows come out of every season but exactly
one of them wins the title, so an interval built by resampling rows would be several times too
narrow. The cluster bootstrap is tested in tests/test_metrics.py; what is tested here is that this
module actually routes through it and refuses to compare two specs that are not aligned row for
row.

**The PIT has to be randomised.** Final points are a discrete quantity, so the ordinary transform
cannot be uniform however good the forecast is. The test below simulates from a known distribution
and scores draws from that same distribution: the transform is uniform only if the randomisation
is right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.season import validate
from plmodel.season.simulate import SeasonForecast
from plmodel.season.table import Question, SeasonError

_COLUMNS = ["date", "season", "home_team", "away_team", "home_goals", "away_goals"]


def _season(n_teams: int = 6, *, seed: int = 0) -> pd.DataFrame:
    teams = [chr(ord("A") + i) for i in range(n_teams)]
    rng = np.random.default_rng(seed)
    rows = []
    day = pd.Timestamp("2020-08-08")
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            rows.append({"date": day, "season": "2020-21", "home_team": home, "away_team": away,
                         "home_goals": int(rng.integers(0, 4)),
                         "away_goals": int(rng.integers(0, 4))})
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows, columns=_COLUMNS)


# --- barriers ---------------------------------------------------------------------------------

def test_barriers_partition_the_season_at_the_right_place() -> None:
    rows = _season(6)          # 30 fixtures, one per day
    got = validate.matchweek_barriers(rows, weeks=(0, 1, 2), fixtures_per_week=5)
    assert [b.n_played for b in got] == [0, 5, 10]
    assert [b.n_remaining for b in got] == [30, 25, 20]
    assert got[0].horizon == 1.0 and got[2].horizon == pytest.approx(20 / 30)


def test_a_barrier_never_falls_inside_a_matchday() -> None:
    """Two halves of one afternoon must not land on opposite sides of the split."""
    rows = _season(6)
    rows["date"] = pd.Timestamp("2020-08-08")          # every fixture on one day
    rows.loc[15:, "date"] = pd.Timestamp("2020-08-15")
    got = validate.matchweek_barriers(rows, weeks=(1,), fixtures_per_week=5)
    assert got[0].n_played == 0                        # snapped back to the start of the day


def test_a_barrier_past_the_end_of_a_season_is_refused() -> None:
    with pytest.raises(SeasonError, match="past the end"):
        validate.matchweek_barriers(_season(6), weeks=(10,), fixtures_per_week=5)


# --- the actual table -------------------------------------------------------------------------

def test_the_actual_table_carries_positions_and_a_level_flag() -> None:
    rows = _season(6, seed=3)
    teams = tuple(sorted(set(rows["home_team"])))
    got = validate.actual_table(rows, teams=teams)
    assert got["position"].tolist() == list(range(1, 7))
    assert not got["level_with_the_club_above"].iloc[0]


def test_a_deduction_reaches_the_actual_table_only() -> None:
    rows = _season(6, seed=1)
    teams = tuple(sorted(set(rows["home_team"])))
    plain = validate.actual_table(rows, teams=teams).set_index("team")
    docked = validate.actual_table(rows, teams=teams, deductions={"A": 10}).set_index("team")
    assert docked.loc["A", "points"] == plain.loc["A", "points"] - 10
    assert docked.loc["B", "points"] == plain.loc["B", "points"]


# --- the probability integral transform -------------------------------------------------------

def _forecast_from_counts(counts: np.ndarray, teams: tuple[str, ...], floor: int = 0):
    n = int(counts.sum(axis=1)[0])
    return SeasonForecast(
        season="2020-21", barrier=pd.Timestamp("2020-08-08"), teams=teams,
        table=pd.DataFrame({"team": list(teams)}), n_played=0, n_remaining=1, n_replicates=n,
        uncertainty="point", probabilities=pd.DataFrame({"team": list(teams)}),
        position_counts=np.zeros((len(teams), len(teams)), dtype=np.int64),
        points_counts=counts, points_floor=floor,
    )


def test_the_randomised_transform_is_uniform_when_the_forecast_is_right() -> None:
    """Scored against draws from the very distribution it simulated, the PIT must be flat.

    Without the randomisation it cannot be: points are discrete, so the ordinary transform piles
    mass on a handful of values and a perfect forecast would look badly calibrated.
    """
    rng = np.random.default_rng(0)
    n_teams, replicates = 400, 4000
    true_mean = rng.integers(30, 80, size=n_teams)
    span = 120
    counts = np.zeros((n_teams, span), dtype=np.int64)
    for i, mean in enumerate(true_mean):
        draws = np.clip(rng.poisson(mean, replicates), 0, span - 1)
        counts[i] = np.bincount(draws, minlength=span)
    actual = np.clip(rng.poisson(true_mean), 0, span - 1)

    forecast = _forecast_from_counts(counts, tuple(str(i) for i in range(n_teams)))
    pit = validate._points_pit(forecast, actual, seed=1)
    summary = validate.pit_summary(pit)
    # Kolmogorov-Smirnov 5% critical value at this sample size is about 1.36/sqrt(n).
    assert summary["ks"] < 1.36 / np.sqrt(n_teams)
    assert summary["tail_mass"] == pytest.approx(0.2, abs=0.05)


def test_a_forecast_that_is_too_narrow_pushes_the_transform_into_the_tails() -> None:
    """The failure a point-estimate season simulation is accused of, in its diagnostic form."""
    rng = np.random.default_rng(2)
    n_teams, replicates, span = 400, 4000, 120
    true_mean = rng.integers(30, 80, size=n_teams)
    counts = np.zeros((n_teams, span), dtype=np.int64)
    for i, mean in enumerate(true_mean):
        narrow = np.clip(np.round(rng.normal(mean, 1.0, replicates)).astype(int), 0, span - 1)
        counts[i] = np.bincount(narrow, minlength=span)
    actual = np.clip(rng.poisson(true_mean), 0, span - 1)
    summary = validate.pit_summary(
        validate._points_pit(_forecast_from_counts(counts, tuple(map(str, range(n_teams)))),
                             actual, seed=1)
    )
    assert summary["tail_mass"] > 0.5
    assert summary["ks"] > 0.2


def test_the_points_log_score_reads_the_replicate_histogram() -> None:
    """Strictly proper, one number per club-season, so a bootstrap can put an interval on it."""
    counts = np.zeros((2, 10), dtype=np.int64)
    counts[0, 4] = 250; counts[0, 5] = 750       # club 0: a quarter of replicates on 4 points
    counts[1, 9] = 1000                          # club 1: every replicate on 9
    forecast = _forecast_from_counts(counts, ("A", "B"))
    got = validate.points_log_score(forecast, np.array([4, 9]), floor=1e-4)
    assert got[0] == pytest.approx(-np.log(0.25))
    assert got[1] == pytest.approx(0.0)


def test_the_points_log_score_floors_an_outcome_the_simulation_never_reached() -> None:
    counts = np.zeros((1, 10), dtype=np.int64)
    counts[0, 5] = 1000
    forecast = _forecast_from_counts(counts, ("A",))
    got = validate.points_log_score(forecast, np.array([0]), floor=1e-4)
    assert got[0] == pytest.approx(-np.log(1e-4))


def test_pit_summary_rejects_an_empty_sample() -> None:
    with pytest.raises(SeasonError, match="empty PIT"):
        validate.pit_summary(np.array([]))


# --- scoring and aggregation ------------------------------------------------------------------

def _scored_frame(spec: str, brier: list[float], seasons: list[str]) -> pd.DataFrame:
    n = len(brier)
    return pd.DataFrame({
        "season": seasons, "barrier": pd.Timestamp("2020-08-08"), "horizon": 1.0,
        "uncertainty": spec, "team": [f"T{i}" for i in range(n)],
        "question": "title", "probability": 0.5, "outcome": 0.0,
        "brier": brier, "log_loss": brier, "floored": False, "cold_start": False,
        "promoted": False,
        "final_position": 1, "points_pit": np.linspace(0.05, 0.95, n),
        "points_log_score": 3.0,
        "spec": spec, "week": 0, "n_cold_start": 0,
    })


def test_summarise_reports_the_season_count_beside_the_row_count() -> None:
    """The honest sample size for a long-horizon forecast is the smaller of the two."""
    seasons = [f"20{y:02d}-{y + 1:02d}" for y in range(10) for _ in range(20)]
    base = _scored_frame("point", [0.25] * 200, seasons)
    better = _scored_frame("drift", [0.20] * 200, seasons)
    got = validate.summarise(pd.concat([base, better], ignore_index=True),
                             n_boot=500, seed=0, baseline="point")
    assert got["specs"]["point"]["n_rows"] == 200
    assert got["specs"]["point"]["n_seasons"] == 10
    delta = got["specs"]["drift"]["vs_baseline"]
    assert delta["n"] == 200 and delta["n_groups"] == 10
    assert delta["delta"] == pytest.approx(-0.05)


def test_summarise_refuses_specs_that_are_not_aligned_row_for_row() -> None:
    seasons = [f"20{y:02d}-{y + 1:02d}" for y in range(4) for _ in range(5)]
    base = _scored_frame("point", [0.25] * 20, seasons)
    shuffled = _scored_frame("drift", [0.25] * 20, seasons[::-1])
    with pytest.raises(SeasonError, match="not aligned"):
        validate.summarise(pd.concat([base, shuffled], ignore_index=True),
                           n_boot=100, seed=0, baseline="point")


def test_summarise_refuses_an_unknown_baseline() -> None:
    frame = _scored_frame("point", [0.25] * 5, ["2020-21"] * 5)
    with pytest.raises(SeasonError, match="not among the specs"):
        validate.summarise(frame, n_boot=100, seed=0, baseline="drift")


def test_run_span_refuses_specs_asking_different_questions(cfg) -> None:
    specs = {
        "a": cfg.season.spec(n_replicates=10),
        "b": cfg.season.spec(n_replicates=10).__class__(
            n_replicates=10, chunk_size=10, questions=(Question("title", "top", 1),)),
    }
    with pytest.raises(SeasonError, match="same questions"):
        validate.run_span(_season(6), cfg, seasons=("2020-21",), specs=specs, weeks=(0,),
                          fixtures_per_week=5, prob_floor=1e-4)
