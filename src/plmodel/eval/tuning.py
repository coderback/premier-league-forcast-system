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

AMENDED 2026-08-25, after the protocol chose against the test span twice in two days
------------------------------------------------------------------------------------
Both failures looked like clean convergence at the time, and neither was detectable by anything
the protocol measured.

**Selection noise.** The mandated `dc-gas` retune won this window by 0.19806 -> 0.19780 — 0.00027
on 1,303 matches — with every axis interior after coordinate cycles run to convergence. On the test
span it turned -0.0019 at DM p=0.011, which clears all four gates, into -0.0013 at p=0.0513, which
clears three. Ten of this project's shipped values were selected on margins *smaller* than that
0.00027, four of them on 0.00001 or less.

**Structural blindness.** The per-parameter decay search chose a home-advantage memory of 1460
against production's 730 — the wrong direction — because home advantage is flat across this window
(+0.3215 -> +0.3217 in seven years) and only starts falling afterwards. A longer memory is free
variance reduction for a parameter that does not move.

So two checks, and neither of them contradicts the point-estimate policy below:

**1. Resolution.** A selection that MOVES a shipped value must beat the value it replaces by more
than this window's own sampling noise, or the incumbent stands. See :func:`selection_is_resolved`.
The existing policy's answer to selection noise is "the chosen value earns a full paired comparison
afterwards" — but a *gate* can only reject an arm, never correct a selection. By the time that
comparison runs the value is baked into the arm being judged, the arm is spent, and its scorings
have enlarged the family every future re-scoring must be corrected against. This catches the same
failure while it is still free.

**2. Stationarity.** A decay half-life may only be selected for a parameter that actually varies on
this window. See :func:`parameter_is_tunable`. This is not a new idea — the paragraph above already
says a half-life tuned on 1996-2006 "is only useful for 2016-2026 if the answer does not move". It
was simply never asked out loud.

Both report rather than raise. A protocol that refuses to run is a protocol people work around.
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


def selection_is_resolved(
    winner: np.ndarray,
    incumbent: np.ndarray,
    outcomes: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Did the search actually distinguish these two configurations, or is the gap its own noise?

    One comparison, not a confidence interval on every grid point: the winner against the value it
    would replace. The question is not "is this grid point significantly better than that one" but
    "did this search resolve anything at all", and the honest answer on a flat surface is no.

    ``groups`` should be the season of each match. A half-life change perturbs a whole season
    coherently -- every fit in it shifts the same way -- so treating matches as independent reports
    an interval several times too narrow. The clustered figure is the verdict; the unclustered one
    is reported beside it so the difference between the two is visible rather than asserted.

    Sign convention follows the rest of the project: negative favours the winner.
    """
    from plmodel.eval import metrics

    loss_winner = metrics.rps(winner, outcomes)
    loss_incumbent = metrics.rps(incumbent, outcomes)
    naive = metrics.paired_delta_losses(
        loss_winner, loss_incumbent, n_boot=n_boot, seed=seed
    )
    clustered = None
    if groups is not None:
        clustered = metrics.paired_delta_clustered(
            loss_winner, loss_incumbent, np.asarray(groups), n_boot=n_boot, seed=seed
        )
    verdict = clustered if clustered is not None else naive
    # Resolved when the winner's advantage does not straddle zero. Straddling means the search
    # cannot tell the two apart, and moving a shipped value on that is churn dressed as progress.
    resolved = bool(verdict["ci_high"] < 0.0)
    return {
        "resolved": resolved,
        "delta": float(verdict["delta"]),
        "ci_low": float(verdict["ci_low"]),
        "ci_high": float(verdict["ci_high"]),
        "p_winner_better": float(verdict["p_a_better"]),
        "clustered": clustered is not None,
        "n_groups": int(verdict.get("n_groups", 0)) or None,
        "unclustered_ci": (float(naive["ci_low"]), float(naive["ci_high"])),
        "verdict": (
            "resolved: the winner beats the incumbent by more than this window's noise"
            if resolved else
            "UNRESOLVED: the difference is inside this window's own noise, so the incumbent "
            "stands and the search has not earned a move"
        ),
    }


def parameter_is_tunable(
    tuning_values: np.ndarray, reference_values: np.ndarray, *, min_ratio: float
) -> dict[str, object]:
    """Does the parameter vary enough on the tuning window for a memory to be selectable there?

    A decay half-life is a statement about how fast something moves. Choose one on a window where
    the parameter is flat and the search will ask for a longer memory, correctly for that window
    and uselessly for any other -- which is exactly what happened to home advantage.

    ``tuning_values`` and ``reference_values`` are the parameter's fitted trajectory across the
    tuning window and across a reference window. Variation is compared as a standard-deviation
    ratio, reported rather than reduced to a flag, because the number belongs in the ledger next to
    whatever decision it drives.
    """
    tuning_sd = float(np.std(np.asarray(tuning_values, dtype=float)))
    reference_sd = float(np.std(np.asarray(reference_values, dtype=float)))
    ratio = tuning_sd / reference_sd if reference_sd > 0 else float("inf")
    tunable = bool(ratio >= min_ratio)
    return {
        "tunable": tunable,
        "ratio": ratio,
        "tuning_sd": tuning_sd,
        "reference_sd": reference_sd,
        "min_ratio": float(min_ratio),
        "verdict": (
            "tunable: the parameter varies on this window comparably to the reference"
            if tunable else
            "UNTUNABLE ON THIS WINDOW: the parameter is materially flatter here than where it "
            "will be judged, so a memory selected here is selected for a different problem"
        ),
    }


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
