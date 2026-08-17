"""Hyperparameter search, run where it cannot contaminate the acceptance instrument.

The standing rule from the WC2026 project: **never tune against your acceptance instrument, and
treat a grid winner as a lead to confirm rather than a value to adopt.** Every tuning pass there
came with an honesty note — the half-life grid was seven looks at the test set, the K-weight
coordinate descent was twenty-four, and two tiers ended up at grid edges, which is the signature of
having optimised the measurement rather than the model.

So the search runs on ``backtest.tuning_span``, a third window that is neither the test span nor
the sensitivity span. The winner is then *confirmed* on the test span through the ordinary
acceptance harness, and the sensitivity span shows whether the optimum is era-stable — a question
worth asking, because a half-life tuned on 1996-2006 is only useful for 2016-2026 if the answer
does not move.

A winner at an edge of the grid means the search never bracketed the optimum. That is reported as
a red flag rather than adopted.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from plmodel.config import Config, SeasonSpan
from plmodel.eval import metrics
from plmodel.eval.backtest import walk_forward
from plmodel.eval.compare import ArmSpec, run_arm
from plmodel.model.baselines import outcomes_of


def sweep_half_life(
    matches: pd.DataFrame,
    cfg: Config,
    *,
    span: SeasonSpan,
    arm: str = "dixon-coles",
    grid: tuple[float, ...] | None = None,
    refit_every: int | None = None,
) -> pd.DataFrame:
    """Walk-forward RPS at each half-life on one span. Point estimates only.

    Point estimates by design: this is a search, and attaching bootstrap CIs to every grid point
    would invite reading significance into what is a selection step. The chosen value earns a full
    paired comparison afterwards, on data this search never saw.
    """
    values = tuple(grid) if grid is not None else cfg.backtest.half_life_grid_days
    splits = walk_forward(
        matches,
        first_season=span.first_season,
        last_season=span.last_season,
        refit_every=refit_every if refit_every is not None else cfg.backtest.refit_every,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    rows_pool = pd.concat([s.test(matches) for s in splits], ignore_index=True)
    outcomes = outcomes_of(rows_pool)
    uniform_rps = metrics.mean_rps(metrics.uniform_baseline(len(outcomes)), outcomes)

    # "At its bound" is measured against the configured bound, not a hardcoded number, so the two
    # cannot drift apart when the bound is edited.
    rho_lo, rho_hi = cfg.model.param_bounds["rho"]

    spec = ArmSpec.parse(arm)
    records: list[dict[str, object]] = []
    for half_life in values:
        tuned = dataclasses.replace(cfg, model=dataclasses.replace(
            cfg.model, decay_half_life_days=float(half_life)
        ))
        probs, state = run_arm(spec, matches, splits, tuned)
        fits = state.get("fits") or []
        rps = metrics.mean_rps(probs, outcomes)
        records.append(
            {
                "half_life_days": float(half_life),
                "rps": rps,
                "log_loss": metrics.mean_log_loss(probs, outcomes),
                "skill": metrics.skill(rps, uniform_rps),
                "n": int(len(outcomes)),
                "mean_rho": float(np.mean([f.rho for f in fits])) if fits else np.nan,
                "mean_home_adv": (
                    float(np.mean([f.home_advantage for f in fits])) if fits else np.nan
                ),
                "mean_cold_start": (
                    float(np.mean([len(f.cold_start_teams) for f in fits])) if fits else np.nan
                ),
                "converged": (
                    int(sum(f.converged for f in fits)) if fits else 0
                ),
                # A grid point needing rho clamped is unstable; reported, not hidden.
                "rho_clamped": int(sum(f.diagnostics.get("rho_clamped", 0) for f in fits)),
                "rho_at_bound": int(
                    sum(np.isclose(f.rho, rho_lo) or np.isclose(f.rho, rho_hi) for f in fits)
                ),
                "n_fits": len(fits),
            }
        )
    return pd.DataFrame(records)


def sweep_verdict(sweep: pd.DataFrame) -> dict[str, object]:
    """Read a sweep: the winner, its margin, and whether it sits at a grid edge."""
    if sweep.empty:
        raise ValueError("empty sweep")
    ordered = sweep.sort_values("rps", kind="stable").reset_index(drop=True)
    best = ordered.iloc[0]
    grid = sorted(sweep["half_life_days"])
    at_edge = best["half_life_days"] in (grid[0], grid[-1])
    runner_up = ordered.iloc[1] if len(ordered) > 1 else None
    return {
        "best_half_life_days": float(best["half_life_days"]),
        "best_rps": float(best["rps"]),
        "margin_over_runner_up": (
            float(runner_up["rps"] - best["rps"]) if runner_up is not None else None
        ),
        "spread_across_grid": float(sweep["rps"].max() - sweep["rps"].min()),
        "at_grid_edge": bool(at_edge),
        "warning": (
            "winner sits at a grid edge: the search never bracketed the optimum, so this is a "
            "lead to widen the grid on, not a value to adopt"
            if at_edge else None
        ),
    }
