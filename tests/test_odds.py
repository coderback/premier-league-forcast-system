"""De-vigging and market-family resolution.

The de-vig is the benchmark half of the acceptance rule, so the tests here are about the two ways
it can be quietly wrong: probabilities that are not a distribution, and a benchmark that changes
definition partway through a pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.data.odds import (
    FAMILIES, OddsError, assert_comparable, attach_market, devig, devig_shin, family_coverage,
    get_family, market_probabilities, resolve_family,
)

TOL = 1e-9

# A pronounced favourite-longshot fixture: a heavy favourite and a 40/1 away side.
FAV_LONGSHOT = np.array([[1.04, 19.42, 40.86]])


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --- de-vig arithmetic -------------------------------------------------------------------------

@pytest.mark.parametrize("method", [devig, devig_shin])
def test_devigged_probabilities_sum_to_one(method) -> None:
    odds = np.array([[2.10, 3.40, 3.60], [1.30, 5.50, 11.0], [4.00, 3.50, 1.95]])
    probs, _ = method(odds)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=TOL)
    assert (probs > 0).all() and (probs < 1).all()


@pytest.mark.parametrize("method", [devig, devig_shin])
def test_overround_is_reported(method) -> None:
    odds = np.array([[2.10, 3.40, 3.60]])
    _, overround = method(odds)
    expected = float((1 / odds).sum() - 1)
    assert overround[0] == pytest.approx(expected)


def test_shin_differs_from_proportional_on_a_favourite_longshot() -> None:
    """The de-vig choice is a real decision, not a formality — it moves the benchmark."""
    shin, _ = devig_shin(FAV_LONGSHOT)
    prop, _ = devig(FAV_LONGSHOT)
    assert not np.allclose(shin, prop, atol=1e-4)


def test_shin_moves_probability_from_longshots_to_favourites() -> None:
    """Shin's whole purpose: bookmakers pad longshot margins hardest, so proportional
    de-vigging overstates longshot probabilities. Verified on the real corpus as a clean
    monotone effect (-0.008 below p=0.05, +0.013 above p=0.70)."""
    shin, _ = devig_shin(FAV_LONGSHOT)
    prop, _ = devig(FAV_LONGSHOT)
    assert shin[0, 0] > prop[0, 0]   # favourite gains
    assert shin[0, 2] < prop[0, 2]   # longshot loses


def test_fair_book_is_unchanged_by_either_method() -> None:
    """With no margin to remove, Shin reduces to the implied probabilities."""
    odds = np.array([[3.0, 3.0, 3.0]])
    shin, over = devig_shin(odds)
    prop, _ = devig(odds)
    assert np.allclose(shin, prop, atol=TOL)
    assert over[0] == pytest.approx(0.0, abs=TOL)


def test_sub_fair_book_is_handled() -> None:
    """Averaging odds across bookmakers can push the implied sum below 1; the lower divisions
    contain such rows, so this must normalise rather than diverge."""
    odds = np.array([[3.2, 3.4, 3.5]])
    probs, overround = devig_shin(odds)
    assert overround[0] < 0
    assert probs.sum() == pytest.approx(1.0, abs=TOL)


@pytest.mark.parametrize("method", [devig, devig_shin])
def test_invalid_odds_raise(method) -> None:
    """The ported functions keep a strict contract so a sentinel value cannot pass for a price."""
    with pytest.raises(OddsError, match="finite and > 1.0"):
        method(np.array([[0.0, 3.4, 3.6]]))
    with pytest.raises(OddsError, match="finite and > 1.0"):
        method(np.array([[np.nan, 3.4, 3.6]]))
    with pytest.raises(OddsError, match=r"odds must be \(N, 3\)"):
        method(np.array([[2.0, 3.0]]))


# --- family resolution -------------------------------------------------------------------------

def test_zero_is_treated_as_missing_not_as_a_price() -> None:
    """The source encodes 'no price' as 0.0 — six Bet365 rows in the corpus do exactly this."""
    df = _frame([
        {"B365H": 2.0, "B365D": 3.4, "B365A": 3.6},
        {"B365H": 0.0, "B365D": 3.6, "B365A": 3.25},
    ])
    odds, covered, n_invalid = resolve_family(df, "bet365")
    assert list(covered) == [True, False]
    assert n_invalid == 1
    assert np.isnan(odds[1]).all()


def test_absent_prices_are_uncovered_not_invalid() -> None:
    """A match the market never priced is uncovered; only a malformed price is invalid."""
    df = _frame([
        {"AvgCH": 2.0, "AvgCD": 3.4, "AvgCA": 3.6},
        {"AvgCH": np.nan, "AvgCD": np.nan, "AvgCA": np.nan},
    ])
    _, covered, n_invalid = resolve_family(df, "avg_closing")
    assert list(covered) == [True, False] and n_invalid == 0


def test_unknown_family_raises() -> None:
    with pytest.raises(OddsError, match="unknown market family"):
        get_family("nope")


def test_missing_columns_raise() -> None:
    with pytest.raises(OddsError, match="absent from the frame"):
        resolve_family(_frame([{"AvgCH": 2.0}]), "avg_closing")


def test_market_probabilities_leave_uncovered_rows_nan() -> None:
    """Coverage is a value, not a hole: an unpriced match must not acquire a probability."""
    df = _frame([
        {"AvgCH": 2.0, "AvgCD": 3.4, "AvgCA": 3.6},
        {"AvgCH": np.nan, "AvgCD": np.nan, "AvgCA": np.nan},
    ])
    out = market_probabilities(df, "avg_closing", "shin", sum_tolerance=TOL)
    assert out.loc[0, ["p_home", "p_draw", "p_away"]].sum() == pytest.approx(1.0, abs=TOL)
    assert out.loc[1, ["p_home", "p_draw", "p_away"]].isna().all()


def test_no_covered_rows_is_not_an_error() -> None:
    df = _frame([{"AvgCH": np.nan, "AvgCD": np.nan, "AvgCA": np.nan}])
    out = market_probabilities(df, "avg_closing", "shin", sum_tolerance=TOL)
    assert out["p_home"].isna().all()


def test_unknown_method_raises() -> None:
    with pytest.raises(OddsError, match="unknown de-vig method"):
        market_probabilities(_frame([{"AvgCH": 2.0, "AvgCD": 3.4, "AvgCA": 3.6}]),
                             "avg_closing", "magic", sum_tolerance=TOL)


def test_attach_market_adds_prefixed_columns() -> None:
    df = _frame([{"AvgCH": 2.0, "AvgCD": 3.4, "AvgCA": 3.6}])
    out = attach_market(df, "avg_closing", "shin", prefix="mkt", sum_tolerance=TOL)
    assert {"mkt_home", "mkt_draw", "mkt_away", "mkt_overround"} <= set(out.columns)
    assert out[["mkt_home", "mkt_draw", "mkt_away"]].sum(axis=1)[0] == pytest.approx(1.0, abs=TOL)


# --- the guard against a benchmark that changes meaning ----------------------------------------

def test_mixing_settlement_timings_is_refused() -> None:
    """A pool that is closing odds for one era and pre-close for another silently changes what
    the market gate measures, which would read as a model effect."""
    with pytest.raises(OddsError, match="different settlement timings"):
        assert_comparable("avg_closing", "betbrain_avg")


def test_same_settlement_timing_is_allowed() -> None:
    assert_comparable("avg_closing", "pinnacle_closing")


def test_every_family_declares_its_settlement() -> None:
    for name, family in FAMILIES.items():
        assert family.settlement in {"closing", "pre-close"}, name
        assert family.kind in {"market average", "single book"}, name
        assert len(family.columns) == 3, name


# --- the real corpus ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    cfg = load_config()
    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    return pd.read_parquet(path)


@pytest.mark.integration
def test_gate_benchmark_coverage_on_the_real_corpus(corpus: pd.DataFrame) -> None:
    """The gate-2 pool is 2,660 E0 matches from 2019/20 — the figure the plan committed to."""
    cfg = load_config()
    e0 = corpus[corpus["division"] == "E0"]
    _, covered, n_invalid = resolve_family(e0, cfg.odds.gate_benchmark)
    assert int(covered.sum()) == 2660
    assert n_invalid == 0
    assert e0.loc[covered, "season"].min() == "2019-20"


@pytest.mark.integration
def test_pinnacle_diagnostic_stops_in_january_2026(corpus: pd.DataFrame) -> None:
    """The discontinuity that disqualified Pinnacle as the gate benchmark."""
    e0 = corpus[corpus["division"] == "E0"]
    _, covered, _ = resolve_family(e0, "pinnacle_closing")
    priced = e0.loc[covered, "date"]
    assert priced.max() == pd.Timestamp("2026-01-08")
    # Still the wider historical reach, which is why it is kept as a diagnostic.
    assert int(covered.sum()) > 5000


@pytest.mark.integration
def test_real_odds_devig_to_valid_distributions(corpus: pd.DataFrame) -> None:
    cfg = load_config()
    e0 = corpus[corpus["division"] == "E0"]
    for method in (cfg.odds.devig_primary, cfg.odds.devig_sensitivity):
        out = market_probabilities(e0, cfg.odds.gate_benchmark, method,
                                   sum_tolerance=cfg.odds.sum_tolerance)
        covered = out["p_home"].notna()
        sums = out.loc[covered, ["p_home", "p_draw", "p_away"]].sum(axis=1)
        assert np.allclose(sums, 1.0, atol=cfg.odds.sum_tolerance)


@pytest.mark.integration
def test_shin_longshot_correction_is_monotone_on_real_data(corpus: pd.DataFrame) -> None:
    """Measured on 2,660 real matches: Shin moves probability from longshots to favourites, and
    the size of the shift increases monotonically with the favourite's price."""
    e0 = corpus[corpus["division"] == "E0"]
    odds, covered, _ = resolve_family(e0, "avg_closing")
    shin, _ = devig_shin(odds[covered])
    prop, _ = devig(odds[covered])

    flat_prop, diff = prop.ravel(), (shin - prop).ravel()
    edges = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0]
    means = []
    for lo, hi in zip(edges, edges[1:]):
        bucket = (flat_prop >= lo) & (flat_prop < hi)
        means.append(float(diff[bucket].mean()))
    assert means[0] < 0 and means[-1] > 0
    assert all(a < b for a, b in zip(means, means[1:])), f"not monotone: {means}"


@pytest.mark.integration
def test_bet365_zero_sentinels_are_counted(corpus: pd.DataFrame) -> None:
    """Six rows corpus-wide carry B365H = 0.0; they must be counted, not crash and not impute."""
    _, _, n_invalid = resolve_family(corpus, "bet365")
    assert n_invalid == 6


@pytest.mark.integration
def test_family_coverage_report(corpus: pd.DataFrame) -> None:
    cov = family_coverage(corpus[corpus["division"] == "E0"], names=("avg_closing",))
    assert set(cov["family"]) == {"avg_closing"}
    overall = cov[cov["season"] == "ALL"].iloc[0]
    assert overall["n_priced"] == 2660


@pytest.mark.integration
def test_ingest_report_names_the_benchmark(corpus: pd.DataFrame) -> None:
    """Every run must state which market a reported gap was measured against."""
    from plmodel.data.coverage import market_benchmark_block

    cfg = load_config()
    block = market_benchmark_block(corpus, cfg)
    assert block["gate_benchmark"] == cfg.odds.gate_benchmark
    assert block["devig_primary"] == cfg.odds.devig_primary
    assert block["devig_sensitivity"] == cfg.odds.devig_sensitivity
    gate = block["families"][cfg.odds.gate_benchmark]
    assert gate["settlement"] == "closing" and gate["n_priced"] > 0
    # The discontinuity that disqualified Pinnacle shows up as a date, not just as prose.
    assert block["families"]["pinnacle_closing"]["last_priced"] == "2026-01-08"
