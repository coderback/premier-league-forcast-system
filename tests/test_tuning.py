"""Hyperparameter search, and the discipline around it.

The tests here are mostly about *where* a search may run and *how* its result may be read, because
those are the parts that go wrong. A grid winner is a lead to confirm on data the search never
saw, never a value to adopt on the spot.
"""
from __future__ import annotations

import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval.tuning import sweep_verdict


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _sweep(pairs: list[tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"half_life_days": hl, "rps": rps} for hl, rps in pairs]
    )


# --- the search window is kept clean --------------------------------------------------------------

def test_tuning_span_is_disjoint_from_both_evaluation_spans(cfg) -> None:
    """The standing rule: never tune against your acceptance instrument.

    The sensitivity span must be clean too — a sensitivity check that has itself been tuned on is
    not a check on anything.
    """
    bt = cfg.backtest
    assert bt.tuning_span.last_season < bt.sensitivity_span.first_season
    assert bt.sensitivity_span.last_season < bt.test_span.first_season


def test_the_grid_brackets_the_published_optimum(cfg) -> None:
    """Published club-football optima sit near 210-385 days. The grid must extend well past that
    on both sides, so the data can choose rather than be steered into the expected answer."""
    grid = cfg.backtest.half_life_grid_days
    assert min(grid) < 210, "grid does not reach below the published band"
    assert max(grid) > 385, "grid does not reach above the published band"
    assert any(210 <= g <= 385 for g in grid), "grid skips the published band entirely"


def test_the_grid_spans_the_inherited_value(cfg) -> None:
    """1825 days is the WC2026 international value. It is in the grid so the data can reject it
    explicitly rather than it being excluded by construction."""
    assert 1825.0 in cfg.backtest.half_life_grid_days


# --- reading a sweep -------------------------------------------------------------------------------

def test_verdict_picks_the_minimum() -> None:
    verdict = sweep_verdict(_sweep([(180, 0.2050), (365, 0.2010), (730, 0.2030)]))
    assert verdict["best_half_life_days"] == 365.0
    assert verdict["best_rps"] == pytest.approx(0.2010)


def test_verdict_reports_the_margin_over_the_runner_up() -> None:
    """A winner is only meaningful next to its margin: a 0.0001 win on a flat curve is a coin
    toss dressed as a decision."""
    verdict = sweep_verdict(_sweep([(365, 0.20064), (730, 0.20048), (1095, 0.20072)]))
    assert verdict["margin_over_runner_up"] == pytest.approx(0.00016, abs=1e-5)


def test_a_winner_at_the_grid_edge_is_flagged() -> None:
    """A grid edge means the search never bracketed the optimum — the WC2026 red flag."""
    verdict = sweep_verdict(_sweep([(180, 0.2010), (365, 0.2030), (730, 0.2050)]))
    assert verdict["at_grid_edge"] is True
    assert "never bracketed" in verdict["warning"]


def test_an_interior_winner_is_not_flagged() -> None:
    verdict = sweep_verdict(_sweep([(180, 0.2050), (365, 0.2010), (730, 0.2030)]))
    assert verdict["at_grid_edge"] is False and verdict["warning"] is None


def test_verdict_reports_the_spread() -> None:
    """The spread separates 'the curve has a shape' from 'every grid point is the same'."""
    verdict = sweep_verdict(_sweep([(30, 0.2237), (365, 0.2006), (1825, 0.2014)]))
    assert verdict["spread_across_grid"] == pytest.approx(0.0231, abs=1e-4)


def test_empty_sweep_raises() -> None:
    with pytest.raises(ValueError, match="empty sweep"):
        sweep_verdict(pd.DataFrame())


# --- the sweep on real data --------------------------------------------------------------------------

@pytest.mark.integration
def test_sweep_runs_and_ranks(cfg) -> None:
    """A short real sweep: two grid points, a few seasons, asserting only that it runs and ranks.

    The full twelve-point sweep takes minutes and its numbers live in NOTES.md; this is the smoke
    test that the plumbing still works.
    """
    from plmodel.config import SeasonSpan
    from plmodel.eval.tuning import sweep_half_life

    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    corpus = pd.read_parquet(path)
    e0 = corpus[(corpus["division"] == "E0") & corpus["played"]]
    e0 = e0.sort_values("date", kind="stable").reset_index(drop=True)

    sweep = sweep_half_life(
        e0, cfg, span=SeasonSpan("2024-25", "2024-25"), grid=(180.0, 730.0)
    )
    assert len(sweep) == 2
    assert sweep["rps"].between(0.15, 0.30).all()
    assert (sweep["n_fits"] > 0).all()
    # Under a sane half-life the fit never needs its dependence parameter pulled into range.
    assert (sweep["rho_clamped"] == 0).all()
    verdict = sweep_verdict(sweep)
    assert verdict["best_half_life_days"] in (180.0, 730.0)
