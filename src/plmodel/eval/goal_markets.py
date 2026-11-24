"""The goal-market yardstick: how good are the model's goal-based forecasts, and against what?

Every selection in this project has been made on home/draw/away RPS, which is nearly blind to the
level of scoring: the three-way outcome depends on the difference between the two goal rates, not
on how many goals there are. Over/under, both-teams-to-score, clean sheets, handicaps and exact
scores are not blind to it, so they have never been measured, let alone optimised. This module
measures them, for any arm whose forecast is a full scoreline distribution.

**What it scores**, on a walk-forward pool, each against a climatology (the base rates of the
seasons before the pool, so nothing is read from the future) and, where the source publishes one,
against the bookmakers:

* the goal level: mean expected home, away and total goals against what was scored;
* eight binary markets (over the configured line, both teams to score, each side to score, each
  side's clean sheet, each side to win to nil): Brier score, log loss and skill over climatology;
* the total-goals distribution, by RPS over 0..cap-or-more;
* the exact score, by log loss over the grid;
* over/under against three bookmaker families (the Friday market average, bet365's Friday price
  and the closing average), as a Brier gap and as the share of the market's skill the model has;
* Asian handicaps against the Friday market average, on half-goal lines only. A whole or quarter
  line can refund all or half the stake, so it has no single yes/no outcome to score.

**Two-way de-vig.** The project's Shin de-vig is written for three outcomes. Over/under and
handicap prices are two-way and are normalised proportionally here; on 1X2 the two methods agree
to the fifth decimal (0.19676 both, NOTES.md), so the choice does not move a verdict.

Nothing here is a gate. It is the instrument a goal-market arm would later be judged with.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plmodel.config import Config
from plmodel.data.odds import HANDICAP_FAMILIES, TOTALS_FAMILIES
from plmodel.eval import metrics
from plmodel.model.scoreline import (
    both_teams_to_score,
    collapse_three_class,
    scoreline_matrix,
    scoring_markets,
    total_goals_distribution,
    totals_probability,
)


class GoalMarketError(ValueError):
    """Raised when an arm cannot supply a scoreline distribution, or the grids disagree with it."""


# --- the scoreline grids an arm forecast -----------------------------------------------------

@dataclass
class GoalWalk:
    """Every pool row's full scoreline distribution, aligned with the pool."""

    arm: str
    rows: pd.DataFrame
    grids: np.ndarray            # (N, G+1, G+1): home goals by away goals


def walk_grids(arm: str, matches: pd.DataFrame, splits, cfg: Config) -> GoalWalk:
    """Walk one arm and keep its scoreline grid for every test row, not just the 1X2 collapse.

    The arm's own forecaster runs exactly as in a compare walk; the grid is then rebuilt from the
    fit it left in its state, and checked against the probabilities the arm returned, so the goal
    markets scored here are provably the same forecast the gates judged.
    """
    from plmodel.eval.compare import _REGISTRY, ArmContext, ArmSpec, build_pool

    spec = ArmSpec.parse(arm)
    state: dict = {"matches": matches}
    blocks = []
    for split in splits:
        test, train = split.test(matches), split.train(matches)
        probs = np.asarray(_REGISTRY[spec.forecaster](ArmContext(split, test, train, cfg, state)),
                           dtype=float)
        grid = _grid_from_state(arm, state, test, train, cfg)
        drift = float(np.abs(collapse_three_class(grid) - probs).max()) if len(test) else 0.0
        if drift > cfg.goal_markets.consistency_tolerance:
            raise GoalMarketError(
                f"{arm}: the rebuilt grid's home/draw/away differs from the arm's forecast by "
                f"{drift:.2e} at {split.barrier.date()}; the goal markets would not be its forecast"
            )
        blocks.append(grid)
    return GoalWalk(arm=arm, rows=build_pool(matches, splits), grids=np.concatenate(blocks))


def _grid_from_state(arm: str, state: dict, test: pd.DataFrame, train: pd.DataFrame,
                     cfg: Config) -> np.ndarray:
    """Rates from whatever fit the arm left behind, then the grid the arm's own model implies."""
    from plmodel.model.dixon_coles import _RHO_CLAMP_MARGIN, DixonColesFit
    from plmodel.model.scoreline import clamp_rho_for_rates

    max_goals = cfg.model.max_goals
    if state.get("odds_fits") and arm in ("odds-ratings", "odds-ratings-ou"):
        ratings = state["odds_fits"][-1]
        lam, mu = ratings.rates(test["home_team"], test["away_team"])
        return scoreline_matrix(lam, mu, np.zeros(len(test)), max_goals)
    fit = state.get("fit")
    if isinstance(fit, DixonColesFit) and fit.family is None:
        lam, mu = fit.match_rates(test, train)
        rho, _ = clamp_rho_for_rates(lam, mu, fit.rho, margin=_RHO_CLAMP_MARGIN)
        return scoreline_matrix(lam, mu, rho, max_goals)
    raise GoalMarketError(
        f"arm {arm!r} does not leave a Poisson-and-tau fit in its state, so its scoreline "
        "distribution cannot be read; supported: the Dixon-Coles family and odds-ratings"
    )


# --- scoring ---------------------------------------------------------------------------------

def two_way_probabilities(odds: np.ndarray) -> np.ndarray:
    """(N,) probability of the first of two outcomes, proportionally de-vigged; NaN if unpriced."""
    o = np.asarray(odds, dtype=float)
    ok = np.isfinite(o).all(axis=1) & (o > 1.0).all(axis=1)
    out = np.full(len(o), np.nan)
    implied = 1.0 / o[ok]
    out[ok] = implied[:, 0] / implied.sum(axis=1)
    return out


def _binary_block(p: np.ndarray, y: np.ndarray, base_rate: float) -> dict[str, float]:
    p = np.clip(p, metrics.EPS, 1.0 - metrics.EPS)
    brier = float(((p - y) ** 2).mean())
    clim = float(((base_rate - y) ** 2).mean())
    return {
        "n": int(len(y)),
        "observed_rate": float(y.mean()),
        "forecast_rate": float(p.mean()),
        "brier": brier,
        "brier_climatology": clim,
        "skill": 1.0 - brier / clim if clim > 0 else float("nan"),
        "log_loss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
    }


def _events(grid: np.ndarray, line: float) -> dict[str, np.ndarray]:
    """The eight binary markets as forecast probabilities, (N,) each."""
    scoring = scoring_markets(grid)
    return {
        f"over_{line:g}": totals_probability(grid, line)[0],
        "both_teams_to_score": both_teams_to_score(grid),
        **scoring,
    }


def _outcomes(x: np.ndarray, y: np.ndarray, line: float) -> dict[str, np.ndarray]:
    """What happened, in the same keys as :func:`_events`."""
    return {
        f"over_{line:g}": (x + y > line).astype(float),
        "both_teams_to_score": ((x > 0) & (y > 0)).astype(float),
        "home_scores": (x > 0).astype(float),
        "away_scores": (y > 0).astype(float),
        "home_clean_sheet": (y == 0).astype(float),
        "away_clean_sheet": (x == 0).astype(float),
        "home_win_to_nil": ((x > 0) & (y == 0)).astype(float),
        "away_win_to_nil": ((y > 0) & (x == 0)).astype(float),
    }


def climatology(history: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Base rates from the seasons immediately before the pool: the forecast to beat."""
    gm = cfg.goal_markets
    seasons = sorted(history["season"].unique())[-gm.climatology_seasons:]
    past = history[history["season"].isin(seasons)]
    x = past["home_goals"].to_numpy(dtype=int)
    y = past["away_goals"].to_numpy(dtype=int)
    rates = {k: float(v.mean()) for k, v in _outcomes(x, y, gm.over_under_line).items()}
    totals = np.bincount(np.minimum(x + y, gm.total_goals_cap), minlength=gm.total_goals_cap + 1)
    size = cfg.model.max_goals + 1
    # Exact scores: observed frequencies with one pseudo-count spread over the grid, so a score the
    # window never saw still has a probability and the log loss stays finite.
    cells = np.full((size, size), 1.0 / (size * size))
    np.add.at(cells, (np.minimum(x, size - 1), np.minimum(y, size - 1)), 1.0)
    return {
        "seasons": seasons,
        "rates": rates,
        "total_goals": totals / totals.sum(),
        "scores": cells / cells.sum(),
    }


def score_goal_markets(walk: GoalWalk, history: pd.DataFrame, cfg: Config) -> dict[str, object]:
    """Every goal market the grid implies, against results, climatology and the bookmakers."""
    gm = cfg.goal_markets
    rows, grid = walk.rows, walk.grids
    x = rows["home_goals"].to_numpy(dtype=int)
    y = rows["away_goals"].to_numpy(dtype=int)
    clim = climatology(history, cfg)
    goals = np.arange(grid.shape[1])

    expected_home = (grid.sum(axis=2) * goals).sum(axis=1)
    expected_away = (grid.sum(axis=1) * goals).sum(axis=1)
    report: dict[str, object] = {
        "arm": walk.arm,
        "n": int(len(rows)),
        "climatology_seasons": clim["seasons"],
        "goal_level": {
            "expected_home": float(expected_home.mean()), "scored_home": float(x.mean()),
            "expected_away": float(expected_away.mean()), "scored_away": float(y.mean()),
            "expected_total": float((expected_home + expected_away).mean()),
            "scored_total": float((x + y).mean()),
        },
    }

    forecasts = _events(grid, gm.over_under_line)
    observed = _outcomes(x, y, gm.over_under_line)
    report["binary"] = {
        key: _binary_block(forecasts[key], observed[key], clim["rates"][key]) for key in forecasts
    }

    totals = total_goals_distribution(grid, gm.total_goals_cap)
    actual_total = np.minimum(x + y, gm.total_goals_cap)
    report["total_goals_rps"] = {
        "model": _ordinal_rps(totals, actual_total),
        "climatology": _ordinal_rps(np.tile(clim["total_goals"], (len(rows), 1)), actual_total),
    }

    size = grid.shape[1]
    hit = grid[np.arange(len(rows)), np.minimum(x, size - 1), np.minimum(y, size - 1)]
    clim_hit = clim["scores"][np.minimum(x, size - 1), np.minimum(y, size - 1)]
    report["exact_score_log_loss"] = {
        "model": float(-np.log(np.clip(hit, metrics.EPS, 1.0)).mean()),
        "climatology": float(-np.log(clim_hit).mean()),
    }

    over_key = f"over_{gm.over_under_line:g}"
    report["over_under_vs_market"] = {
        name: _versus_market(
            forecasts[over_key], observed[over_key], _family_prices(rows, columns),
            clim["rates"][over_key],
        )
        for name, columns in TOTALS_FAMILIES.items()
    }
    report["handicap_vs_market"] = {
        name: _handicap_versus_market(rows, grid, x, y, columns)
        for name, columns in HANDICAP_FAMILIES.items()
    }
    return report


def _ordinal_rps(probs: np.ndarray, outcome: np.ndarray) -> float:
    """RPS over ordered categories 0..K, the same rule the 1X2 gate uses, widened to K+1 classes."""
    cum = np.cumsum(probs, axis=1)
    observed = (np.arange(probs.shape[1])[None, :] >= outcome[:, None]).astype(float)
    return float(((cum - observed) ** 2)[:, :-1].sum(axis=1).mean() / (probs.shape[1] - 1))


def _family_prices(rows: pd.DataFrame, eras: tuple[tuple[str, ...], ...]) -> np.ndarray:
    """(N,) de-vigged probability of the first outcome, taking each row from the first era that
    priced it. Every era in a family has the same settlement timing (data/odds.py)."""
    out = np.full(len(rows), np.nan)
    for columns in eras:
        if not all(c in rows.columns for c in columns):
            continue
        p = two_way_probabilities(rows[list(columns)].to_numpy(dtype=float))
        take = np.isnan(out) & ~np.isnan(p)
        out[take] = p[take]
    return out


def _versus_market(p_model: np.ndarray, y: np.ndarray, p_market: np.ndarray,
                   base_rate: float) -> dict[str, float]:
    ok = ~np.isnan(p_market)
    if not ok.any():
        return {"n": 0}
    model = _binary_block(p_model[ok], y[ok], base_rate)
    market = _binary_block(p_market[ok], y[ok], base_rate)
    return {
        "n": int(ok.sum()),
        "brier_model": model["brier"],
        "brier_market": market["brier"],
        "gap": model["brier"] - market["brier"],
        "skill_model": model["skill"],
        "skill_market": market["skill"],
        "share_of_market_skill": (model["skill"] / market["skill"]
                                  if market["skill"] > 0 else float("nan")),
    }


def _handicap_versus_market(rows: pd.DataFrame, grid: np.ndarray, x: np.ndarray, y: np.ndarray,
                            eras: tuple[tuple[str, ...], ...]) -> dict[str, float]:
    """Home side covers its Asian handicap, on half-goal lines only."""
    n = len(rows)
    line = np.full(n, np.nan)
    p_market = np.full(n, np.nan)
    for columns in eras:
        line_col, home_col, away_col = columns
        if not all(c in rows.columns for c in columns):
            continue
        p = two_way_probabilities(rows[[home_col, away_col]].to_numpy(dtype=float))
        h = rows[line_col].to_numpy(dtype=float)
        take = np.isnan(p_market) & ~np.isnan(p) & np.isfinite(h)
        line[take], p_market[take] = h[take], p[take]
    half = np.isfinite(line) & np.isclose(np.abs(line) % 1.0, 0.5)  # MATH: a half-goal line
    if not half.any():
        return {"n": 0}
    goals = np.arange(grid.shape[1])
    margin = goals[:, None] - goals[None, :]
    idx = np.flatnonzero(half)
    p_model = np.array([grid[i][margin + line[i] > 0].sum() for i in idx])
    covered = ((x[idx] + line[idx] - y[idx]) > 0).astype(float)
    base = float(covered.mean())
    return {**_versus_market(p_model, covered, p_market[idx], base),
            "share_of_priced_rows_on_half_lines": float(half.sum() / np.isfinite(line).sum())}
