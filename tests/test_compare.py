"""The acceptance harness: arms, gates, and the guards that keep it honest.

The most important test in this file is :func:`test_an_arm_that_does_nothing_is_refused`. A broken
experiment returning "no effect" looks exactly like a correct experiment returning "no effect", and
no amount of care distinguishes them — only an assertion does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval.backtest import walk_forward
from plmodel.eval.compare import (
    ArmResult, ArmSpec, assert_aligned, assert_arms_differ, gate_verdicts, register,
    registered_arms, report_json, run_compare,
)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _corpus(n_days: int = 12, *, season: str = "2024-25") -> pd.DataFrame:
    """A small synthetic league: two matches per matchday, alternating results."""
    dates, home, away, results = [], [], [], []
    teams = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    for d in range(n_days):
        day = pd.Timestamp("2024-08-01") + pd.Timedelta(days=d * 7)
        for k in range(2):
            dates.append(day)
            home.append(teams[(d + k) % len(teams)])
            away.append(teams[(d + k + 1) % len(teams)])
            results.append(["H", "D", "A"][(d + k) % 3])
    return pd.DataFrame(
        {
            "date": dates, "season": season, "division": "E0",
            "home_team": home, "away_team": away, "result": results,
            "AvgCH": 2.10, "AvgCD": 3.40, "AvgCA": 3.60,
        }
    ).sort_values("date", kind="stable").reset_index(drop=True)


def _run(cfg, arms, corpus=None, **kw):
    corpus = _corpus() if corpus is None else corpus
    splits = walk_forward(corpus, min_train_matches=2)
    return run_compare(
        corpus, splits, cfg, arms,
        n_bins=cfg.audit.calibration_bins, big_six=cfg.audit.big_six, **kw
    )


# --- the arm registry ---------------------------------------------------------------------------

def test_baseline_arms_are_registered() -> None:
    assert {"uniform", "home-always", "home-rate"} <= set(registered_arms())


def test_unknown_arm_raises() -> None:
    with pytest.raises(ValueError, match="unknown arm"):
        ArmSpec.parse("no-such-arm")


def test_duplicate_arms_are_refused(cfg) -> None:
    with pytest.raises(ValueError, match="duplicate arms"):
        _run(cfg, ["uniform", "uniform"])


def test_registering_a_name_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register("uniform")(lambda ctx: None)


# --- the guards ---------------------------------------------------------------------------------

def test_an_arm_that_does_nothing_is_refused() -> None:
    """The false-null trap: an arm silently reproducing the baseline returns an uninformative
    null that is indistinguishable from a real one. This assertion is the only defence."""
    probs = np.full((4, 3), 1.0 / 3.0)
    results = [ArmResult(name="baseline", probs=probs),
               ArmResult(name="broken", probs=probs.copy())]
    with pytest.raises(ValueError, match="not doing anything"):
        assert_arms_differ(results)


def test_arms_that_differ_are_accepted() -> None:
    a = ArmResult(name="baseline", probs=np.full((4, 3), 1.0 / 3.0))
    b = ArmResult(name="variant", probs=np.tile([0.5, 0.25, 0.25], (4, 1)))
    assert_arms_differ([a, b])


def test_misaligned_arms_are_refused() -> None:
    """A paired comparison on mismatched rows is meaningless, so it fails rather than reindexes."""
    left = _corpus(4)
    right = left.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="do not align"):
        assert_aligned([left, right])


def test_baseline_is_reproducible_bit_for_bit(cfg) -> None:
    """Half of the false-null guard: the baseline must not drift between runs."""
    first = _run(cfg, ["uniform", "home-rate"])
    second = _run(cfg, ["uniform", "home-rate"])
    assert np.array_equal(first.arms[0].probs, second.arms[0].probs)
    assert first.arms[1].pooled == second.arms[1].pooled


def test_an_arm_with_partial_coverage_is_refused(cfg) -> None:
    """A forecaster that cannot cover the pool is a benchmark, not an arm."""
    @register("_holey")
    def _holey(ctx):
        probs = np.full((len(ctx.test), 3), 1.0 / 3.0)
        probs[0] = np.nan
        return probs

    with pytest.raises(ValueError, match="NaN probabilities"):
        _run(cfg, ["_holey"])


# --- the report ----------------------------------------------------------------------------------

def test_report_embeds_the_acceptance_rule_verbatim(cfg) -> None:
    """The brief's requirement: a report can never claim a rule the config does not state."""
    payload = report_json(_run(cfg, ["uniform", "home-rate"]))
    assert payload["acceptance_rule"] == cfg.acceptance_rule
    assert "Shin de-vigged market" in payload["acceptance_rule"]


def test_report_carries_the_required_blocks(cfg) -> None:
    payload = report_json(_run(cfg, ["uniform", "home-rate"]))
    assert set(payload) >= {"acceptance_rule", "splits", "market", "verdicts", "arms", "fdr"}
    variant = payload["arms"]["home-rate"]
    assert variant["vs_baseline"]["n"] > 0            # paired delta with CI
    assert "p_value" in variant["vs_baseline_dm"]     # DM statistic
    assert variant["slices"]                          # calibration slices
    assert set(variant["calibration"]) == {"home", "draw", "away"}


def test_baseline_has_no_delta_against_itself(cfg) -> None:
    report = _run(cfg, ["uniform", "home-rate"])
    assert report.arms[0].vs_baseline is None
    assert report.arms[1].vs_baseline is not None


def test_pooled_metrics_are_consistent(cfg) -> None:
    from plmodel.eval import metrics

    report = _run(cfg, ["uniform"])
    arm = report.arms[0]
    assert arm.pooled["rps"] == metrics.mean_rps(arm.probs, report.outcomes)
    assert arm.pooled["n"] == len(report.outcomes)


# --- the two gates ---------------------------------------------------------------------------------

def test_gate1_passes_on_a_confident_improvement(cfg) -> None:
    report = _run(cfg, ["uniform", "home-rate"])
    # Force a decisive delta so the gate logic, not the toy data, is what is under test.
    report.arms[1].vs_baseline = {
        "delta_rps": -0.01, "ci_low": -0.02, "ci_high": -0.005, "p_a_better": 0.999, "n": 100
    }
    assert gate_verdicts(report)["home-rate"]["gate1_vs_baseline"] is True


def test_gate1_fails_when_the_ci_straddles_zero(cfg) -> None:
    report = _run(cfg, ["uniform", "home-rate"])
    report.arms[1].vs_baseline = {
        "delta_rps": -0.001, "ci_low": -0.004, "ci_high": +0.002, "p_a_better": 0.80, "n": 100
    }
    assert gate_verdicts(report)["home-rate"]["gate1_vs_baseline"] is False


def test_gate1_passes_on_p_better_alone(cfg) -> None:
    """The rule is explicitly an OR: CI excluding zero, or P(better) >= 0.95."""
    report = _run(cfg, ["uniform", "home-rate"])
    report.arms[1].vs_baseline = {
        "delta_rps": -0.002, "ci_low": -0.005, "ci_high": +0.0001, "p_a_better": 0.96, "n": 100
    }
    verdict = gate_verdicts(report)["home-rate"]
    assert verdict["gate1_vs_baseline"] is True and "P(better)" in verdict["gate1_reason"]


def test_gate2_fails_when_the_market_gap_widens(cfg) -> None:
    """The WC2026 `rsfit` shape: better on the pool, worse against the market. Gate 2 exists
    solely to catch a change that optimises the instrument rather than the target."""
    report = _run(cfg, ["uniform", "home-rate"])
    report.arms[1].vs_baseline = {
        "delta_rps": -0.01, "ci_low": -0.02, "ci_high": -0.005, "p_a_better": 0.999, "n": 100
    }
    report.arms[0].vs_market = {"delta_rps": +0.0060}
    report.arms[1].vs_market = {"delta_rps": +0.0077}
    verdict = gate_verdicts(report)["home-rate"]
    assert verdict["gate2_vs_market"] is False
    assert verdict["accepted"] is False, "gate 1 alone must not accept an arm"


def test_acceptance_needs_both_gates(cfg) -> None:
    report = _run(cfg, ["uniform", "home-rate"])
    report.arms[1].vs_baseline = {
        "delta_rps": -0.01, "ci_low": -0.02, "ci_high": -0.005, "p_a_better": 0.999, "n": 100
    }
    report.arms[0].vs_market = {"delta_rps": +0.0060}
    report.arms[1].vs_market = {"delta_rps": +0.0041}
    assert gate_verdicts(report)["home-rate"]["accepted"] is True


# --- the real corpus ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real(cfg):
    """The real corpus, projected to the columns this file's tests actually read.

    The cached corpus carries 193 columns and materialising all of them costs ~400 MB — enough,
    on a machine whose commit limit is already stretched, to leave the paired bootstrap unable to
    allocate its two contiguous ~300 MB arrays. Projecting in the read rather than afterwards is
    what fixes that; trimming a loaded frame gives back almost nothing, because pandas has already
    built every block. Every market family is kept, so no test loses a benchmark it could ask for.
    """
    import pyarrow.parquet as pq

    from plmodel.data.odds import FAMILIES

    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    keep = [
        "date", "division", "season", "played", "result",
        "home_team", "away_team", "home_goals", "away_goals",
        "home_match_index", "away_match_index", "home_sot", "away_sot",
        *(column for family in FAMILIES.values() for column in family.columns),
    ]
    available = set(pq.ParquetFile(path).schema.names)
    corpus = pd.read_parquet(path, columns=[c for c in keep if c in available])
    div = corpus[corpus["division"] == cfg.backtest.prediction_division]
    played = div[div["played"]].sort_values("date", kind="stable").reset_index(drop=True)
    return div, played


@pytest.mark.integration
def test_the_first_milestone(cfg, real) -> None:
    """`pl compare --arms uniform,home-always` end to end — the harness before any model."""
    history, matches = real
    splits = walk_forward(
        matches,
        first_season=cfg.backtest.test_span.first_season,
        last_season=cfg.backtest.test_span.last_season,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    report = run_compare(
        matches, splits, cfg, ["uniform", "home-always"], history=history,
        n_bins=cfg.audit.calibration_bins, big_six=cfg.audit.big_six,
    )
    payload = report_json(report)

    assert payload["acceptance_rule"] == cfg.acceptance_rule
    assert payload["splits"]["n_splits"] == 1153
    assert report.arms[0].pooled["n"] == 3800
    assert payload["market"]["n_covered"] == 2660
    assert payload["arms"]["home-always"]["vs_baseline"]["delta_rps"] > 0  # far worse, as it must be
    assert payload["verdicts"]["home-always"]["accepted"] is False


@pytest.mark.integration
def test_market_lands_in_the_published_band(cfg, real) -> None:
    """An external check on the whole odds path: the de-vigged closing line should score
    RPS ~0.19-0.20, the range the literature reports for a sharp market."""
    history, matches = real
    splits = walk_forward(
        matches,
        first_season=cfg.backtest.test_span.first_season,
        last_season=cfg.backtest.test_span.last_season,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    report = run_compare(
        matches, splits, cfg, ["uniform"], history=history,
        n_bins=cfg.audit.calibration_bins, big_six=cfg.audit.big_six,
    )
    assert 0.19 <= report.market["rps_market"] <= 0.20


def test_the_staleness_slice_is_defined_before_kickoff(cfg) -> None:
    """The regime partition the dynamic-model literature never isolates.

    Both keys are causal: how many matches each side has already played, and what month it is.
    Neither reads the result, so a match can be assigned to its regime the moment it is scheduled.
    """
    from plmodel.eval.slices import add_slice_columns

    rows = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-17", "2024-02-10", "2024-11-09", "2024-02-03"]),
        "season": ["2024-25"] * 4,
        "home_team": ["Arsenal", "Arsenal", "Arsenal", "Everton"],
        "away_team": ["Chelsea", "Chelsea", "Chelsea", "Fulham"],
        "home_match_index": [1, 24, 12, 3],
        "away_match_index": [1, 25, 13, 4],
    })
    out = add_slice_columns(
        rows, history=rows.assign(division="E0"), division="E0", big_six=cfg.audit.big_six,
    )
    assert out["slice_staleness"].tolist() == [
        "early_season", "post_january_window", "settled", "early_season"
    ]


@pytest.mark.integration
def test_slices_partition_the_pool(cfg, real) -> None:
    """Every slice must account for every match exactly once."""
    history, matches = real
    splits = walk_forward(
        matches,
        first_season=cfg.backtest.test_span.first_season,
        last_season=cfg.backtest.test_span.last_season,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    report = run_compare(
        matches, splits, cfg, ["home-rate"], history=history,
        n_bins=cfg.audit.calibration_bins, big_six=cfg.audit.big_six,
    )
    slices = report.arms[0].slices
    for key, group in slices.groupby("slice"):
        assert group["n"].sum() == 3800, f"slice {key} does not partition the pool"


@pytest.mark.integration
def test_promoted_teams_are_a_meaningful_share(cfg, real) -> None:
    """Three teams come up each season and play 38 matches apiece, so roughly 108 of a season's
    380 fixtures involve one — about 28%, not the ~15% the build brief estimated."""
    history, matches = real
    splits = walk_forward(
        matches,
        first_season=cfg.backtest.test_span.first_season,
        last_season=cfg.backtest.test_span.last_season,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    report = run_compare(
        matches, splits, cfg, ["uniform"], history=history,
        n_bins=cfg.audit.calibration_bins, big_six=cfg.audit.big_six,
    )
    counts = report.matches["slice_promoted"].value_counts()
    share = counts["involves_promoted"] / counts.sum()
    assert 0.25 <= share <= 0.32
