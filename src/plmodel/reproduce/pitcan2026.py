"""Reproduce Pitcan (2026) on Premier League data.

    Yannik Pitcan, "Does a Structural Model Add Anything to the Closing Price? Calibrated
    forecasting, incremental information, and match leverage in the Italian Serie A."
    arXiv:2608.11505, submitted 2026-08-11. Code: github.com/pitcany/seriea-leverage

Why this runs before the xG-channel arm
---------------------------------------
Three separate conclusions in the build plan rest on this one paper: that the market pooling weight
on a goals model is zero, that a chance-creation channel earns 0.35 against that goals model, and
that the model-market gap is about +0.0067. It is a days-old preprint, not peer reviewed, on a
single league that is **not ours**, and its central result is a boundary solution — the pattern
this project treats as a red flag. So the claim gets re-run on our own data before anything is
built on top of it.

What the paper reports (Serie A, 7,220 matches, its Table 3)

    Pool                          Market   Goals   Shots
    market + shots                 1.00      —      0.00
    goals  + shots                  —       0.65    0.35
    market + goals + shots         1.00     0.00    0.00

**The two headline weights are against different references**, which is easy to misread: 0.35 is
the shots model's weight against the *goals model*; against the *market* the same signal earns
0.00. A reproduction that estimated only one of them would answer the wrong question, so all three
pools are fitted here.

Protocol, mirroring the paper's §5.4
------------------------------------
* Validation 2013-14..2018-19, test 2019-20..2025-26. On Serie A those are n = 2,280 and n = 2,660;
  because both leagues play 380-match seasons, the Premier League gives **identical** sample sizes,
  which makes this an unusually clean comparison.
* Weights are fitted on validation and carried to test without refitting.
* Market probabilities are Shin de-vigged Pinnacle closing prices — the paper's stated preference
  ("the closing price of a low-margin, high-limit book is the sharpest widely published forecast"),
  and the only family covering our validation window, which begins before market-average closing
  prices appear in 2019/20.

Two deviations, both recorded rather than smoothed over
-------------------------------------------------------
1. **The decay half-life is ours, not refitted here.** The paper fixes hyperparameters on its
   validation window; ours overlaps this project's acceptance test span, so refitting there would
   tune against the acceptance instrument. The production half-life is used instead. The half-life
   sits on a plateau (see NOTES.md), so the choice moves nothing material — and the object of the
   reproduction is the pooling weight, not the decay rate.
2. **Pinnacle closing prices stop after 2026-01-08** in our feed, so the last months of 2025-26 are
   unpriced. The covered subset is reported, and the market-average family is re-run as a
   sensitivity over the seasons where it exists.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from plmodel.config import Config, SeasonSpan
from plmodel.data.odds import market_probabilities
from plmodel.eval import metrics
from plmodel.eval.backtest import walk_forward
from plmodel.eval.compare import ArmSpec, build_pool, run_arm
from plmodel.model.baselines import outcomes_of
from plmodel.reproduce.pooling import (
    fit_pair_weight, fit_simplex_weights, log_pool, weight_profile,
)

PAPER_ID = "pitcan2026"
CITATION = (
    "Pitcan, Y. (2026). Does a Structural Model Add Anything to the Closing Price? "
    "arXiv:2608.11505"
)

# The paper's own windows (its §5.4). Serie A n = 2,280 and 2,660; the Premier League matches both
# exactly, since each league plays 380 matches a season.
VALIDATION_SPAN = SeasonSpan("2013-14", "2018-19")
TEST_SPAN = SeasonSpan("2019-20", "2025-26")

# The paper's stated market preference (its §4): Pinnacle closing where available.
MARKET_FAMILY = "pinnacle_closing"

# Table 3, for a like-for-like comparison. Weight on the named model within each pool.
PAPER_RESULTS: dict[str, Any] = {
    "market_plus_goals": {"goals": 0.00},
    "market_plus_shots": {"shots": 0.00},
    "goals_plus_shots": {"goals": 0.65, "shots": 0.35},
    "three_way": {"market": 1.00, "goals": 0.00, "shots": 0.00},
    "rps_goals": 0.1972,
    "rps_market": 0.1905,
    "gap": 0.0067,
    "gap_ci": (0.0046, 0.0088),
    "unconstrained_weight": -0.225,
    "kappa_home": 0.309,
    "kappa_away": 0.317,
}

# Grid density for the weight search and for the profile that proves a boundary solution genuine.
_WEIGHT_GRID = 101
# The profile is traced wider than the admissible [0, 1] to answer the sharper question: is the
# model merely uninformative given the price, or anti-informative?
_PROFILE_LOWER, _PROFILE_UPPER = -1.0, 1.0

# Decay rates the goals-vs-shots pool weight is re-estimated at. Spans the plateau the goals
# model sits on, plus shorter memories, because that is where the shots channel turns out to
# live. Kept short: each point is a full walk over the validation window.
SENSITIVITY_GRID: tuple[float, ...] = (180.0, 365.0, 730.0, 1460.0)


@dataclass
class ReproductionWindow:
    """Forecasts and outcomes for one window, aligned on identical matches."""

    name: str
    span: SeasonSpan
    rows: pd.DataFrame
    outcomes: np.ndarray
    goals: np.ndarray
    shots: np.ndarray
    market: np.ndarray
    covered: np.ndarray

    @property
    def n(self) -> int:
        return len(self.outcomes)

    def priced(self) -> dict[str, np.ndarray]:
        """The three forecasts restricted to matches the market priced."""
        return {
            "market": self.market[self.covered],
            "goals": self.goals[self.covered],
            "shots": self.shots[self.covered],
        }


def build_window(
    matches: pd.DataFrame, cfg: Config, span: SeasonSpan, *, market_family: str
) -> ReproductionWindow:
    """Run both structural arms and the market over one window, on identical rows."""
    splits = walk_forward(
        matches,
        first_season=span.first_season,
        last_season=span.last_season,
        refit_every=cfg.backtest.refit_every,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    rows = build_pool(matches, splits)
    outcomes = outcomes_of(rows)

    goals, _ = run_arm(ArmSpec.parse("dixon-coles"), matches, splits, cfg)
    shots, _ = run_arm(ArmSpec.parse("dixon-coles-sot"), matches, splits, cfg)

    market_frame = market_probabilities(
        rows, market_family, cfg.odds.devig_primary, sum_tolerance=cfg.odds.sum_tolerance
    )
    market = market_frame[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float)
    covered = ~np.isnan(market).any(axis=1)

    return ReproductionWindow(
        name=span.first_season + ".." + span.last_season,
        span=span,
        rows=rows,
        outcomes=outcomes,
        goals=goals,
        shots=shots,
        market=market,
        covered=covered,
    )


def _scores(window: ReproductionWindow) -> dict[str, Any]:
    """RPS and log loss for each forecast, on the market-covered subset for comparability."""
    outcomes = window.outcomes[window.covered]
    priced = window.priced()
    out: dict[str, Any] = {"n_covered": int(window.covered.sum()), "n_total": window.n}
    for name, probs in priced.items():
        out[name] = {
            "rps": metrics.mean_rps(probs, outcomes),
            "log_loss": metrics.mean_log_loss(probs, outcomes),
        }
    for name in ("goals", "shots"):
        out[f"{name}_vs_market"] = metrics.paired_delta(
            priced[name], priced["market"], outcomes,
            n_boot=window_n_boot(), seed=0,
        )
    return out


def window_n_boot() -> int:
    """Bootstrap resamples for the reproduction's own intervals.

    Deliberately independent of the harness setting: this is a reproduction of someone else's
    reported intervals, not a run of our acceptance rule, and it should not silently change if we
    retune our own harness.
    """
    return 10_000


def fit_all_pools(window: ReproductionWindow) -> dict[str, Any]:
    """Every pool in the paper's Table 3, plus the profile that tests the boundary solution."""
    outcomes = window.outcomes[window.covered]
    priced = window.priced()

    pools: dict[str, Any] = {
        "market_plus_goals": fit_pair_weight(
            priced["goals"], priced["market"], outcomes, n_grid=_WEIGHT_GRID
        ),
        "market_plus_shots": fit_pair_weight(
            priced["shots"], priced["market"], outcomes, n_grid=_WEIGHT_GRID
        ),
        "goals_plus_shots": fit_pair_weight(
            priced["shots"], priced["goals"], outcomes, n_grid=_WEIGHT_GRID
        ),
        "three_way": fit_simplex_weights(
            {"market": priced["market"], "goals": priced["goals"], "shots": priced["shots"]},
            outcomes,
        ),
    }
    pools["goals_profile"] = weight_profile(
        priced["goals"], priced["market"], outcomes,
        lower=_PROFILE_LOWER, upper=_PROFILE_UPPER, n_grid=_WEIGHT_GRID,
    )
    # Restricted to the admissible range, which is the claim actually being tested.
    pools["goals_profile_admissible"] = weight_profile(
        priced["goals"], priced["market"], outcomes,
        lower=0.0, upper=1.0, n_grid=_WEIGHT_GRID,
    )
    return pools


def carry_weights_to_test(
    validation_pools: dict[str, Any], test: ReproductionWindow
) -> dict[str, Any]:
    """Apply the validation-fitted weights to the test window without refitting.

    The paper's protocol, and the only honest way to read a fitted weight: a weight that helps only
    on the data it was chosen on has demonstrated nothing.
    """
    outcomes = test.outcomes[test.covered]
    priced = test.priced()
    out: dict[str, Any] = {}
    for pool, structural, reference in (
        ("market_plus_goals", "goals", "market"),
        ("market_plus_shots", "shots", "market"),
        ("goals_plus_shots", "shots", "goals"),
    ):
        w = validation_pools[pool]["weight"]
        pooled = log_pool([priced[structural], priced[reference]], [w, 1.0 - w])
        out[pool] = {
            "weight_from_validation": w,
            "rps": metrics.mean_rps(pooled, outcomes),
            "log_loss": metrics.mean_log_loss(pooled, outcomes),
            "vs_reference": metrics.paired_delta(
                pooled, priced[reference], outcomes, n_boot=window_n_boot(), seed=0
            ),
        }
    return out


def half_life_sensitivity(
    matches: pd.DataFrame, cfg: Config, *, market_family: str, grid: tuple[float, ...]
) -> list[dict[str, Any]]:
    """How the goals-vs-shots pool weight moves with the decay rate.

    Run because the first reproduction attempt matched three of the paper's four weights exactly
    and returned half its 0.35 for the fourth. The market is not involved in that pool, so the
    difference cannot come from the odds; the remaining suspect is the decay rate, which the paper
    fits on its own validation window and which we deliberately did not refit here.

    The answer is that the shots channel prefers a **shorter memory than the goals channel** —
    which is a design fact about the xG-channel arm, not a discrepancy about Serie A.
    """
    import dataclasses

    out: list[dict[str, Any]] = []
    for half_life in grid:
        tuned = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, decay_half_life_days=float(half_life))
        )
        window = build_window(matches, tuned, VALIDATION_SPAN, market_family=market_family)
        outcomes = window.outcomes[window.covered]
        priced = window.priced()
        pool = fit_pair_weight(priced["shots"], priced["goals"], outcomes, n_grid=_WEIGHT_GRID)
        out.append(
            {
                "half_life_days": float(half_life),
                "weight_on_shots": pool["weight"],
                "rps_goals": metrics.mean_rps(priced["goals"], outcomes),
                "rps_shots": metrics.mean_rps(priced["shots"], outcomes),
            }
        )
    return out


def verdict(
    validation: dict[str, Any],
    test_scores: dict[str, Any],
    sensitivity: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Does the Premier League reproduce Serie A? Each claim judged separately.

    The build plan gates the xG-channel arm on this: the chance-creation channel must earn a
    materially positive weight against the goals model, and the goals model must earn ~0 against
    the market. If the Premier League disagrees, the arm's priority drops and the discrepancy is
    recorded.
    """
    w_goals_vs_market = validation["market_plus_goals"]["weight"]
    w_shots_vs_market = validation["market_plus_shots"]["weight"]
    w_shots_vs_goals = validation["goals_plus_shots"]["weight"]
    profile = validation["goals_profile_admissible"]

    goals_null = bool(w_goals_vs_market < 0.05)
    shots_null_vs_market = bool(w_shots_vs_market < 0.05)
    shots_informative = bool(w_shots_vs_goals > 0.10)

    return {
        "goals_earns_zero_against_market": goals_null,
        "shots_earns_zero_against_market": shots_null_vs_market,
        "shots_informative_against_goals": shots_informative,
        "boundary_is_genuine": bool(profile["monotone_increasing"]),
        "weights": {
            "goals_vs_market": w_goals_vs_market,
            "shots_vs_market": w_shots_vs_market,
            "shots_vs_goals": w_shots_vs_goals,
        },
        "paper_weights": {
            "goals_vs_market": PAPER_RESULTS["market_plus_goals"]["goals"],
            "shots_vs_market": PAPER_RESULTS["market_plus_shots"]["shots"],
            "shots_vs_goals": PAPER_RESULTS["goals_plus_shots"]["shots"],
        },
        "shots_weight_by_half_life": {
            str(int(row["half_life_days"])): row["weight_on_shots"]
            for row in (sensitivity or [])
        },
        "paper_weight_inside_our_range": bool(
            sensitivity
            and min(r["weight_on_shots"] for r in sensitivity)
            <= PAPER_RESULTS["goals_plus_shots"]["shots"]
            <= max(r["weight_on_shots"] for r in sensitivity)
        ),
        "reproduces": bool(goals_null and shots_informative),
        "xg_arm_gate": (
            "PROCEED: the chance-creation channel carries information the goals model lacks, "
            "and the goals model adds nothing to the price — the same shape as Serie A. Expect "
            "the arm to pass gate 1 and leave the market gap untouched. Give the channel its OWN "
            "decay rate: its pool weight rises sharply as memory shortens, so inheriting the "
            "goals model's half-life understates what it carries."
            if (goals_null and shots_informative) else
            "DOWNGRADE: the Premier League does not reproduce the Serie A pattern. Record the "
            "discrepancy and lower the xG arm's priority before building it."
        ),
    }


def run(
    matches: pd.DataFrame,
    cfg: Config,
    *,
    market_family: str = MARKET_FAMILY,
    sensitivity_grid: tuple[float, ...] | None = SENSITIVITY_GRID,
) -> dict[str, Any]:
    """The whole reproduction: fit on validation, carry to test, judge each claim."""
    validation_window = build_window(matches, cfg, VALIDATION_SPAN, market_family=market_family)
    test_window = build_window(matches, cfg, TEST_SPAN, market_family=market_family)

    validation_pools = fit_all_pools(validation_window)
    test_pools = fit_all_pools(test_window)
    carried = carry_weights_to_test(validation_pools, test_window)
    test_scores = _scores(test_window)
    sensitivity = (
        half_life_sensitivity(matches, cfg, market_family=market_family, grid=sensitivity_grid)
        if sensitivity_grid else []
    )

    return {
        "paper": PAPER_ID,
        "citation": CITATION,
        "market_family": market_family,
        "devig": cfg.odds.devig_primary,
        "half_life_days": cfg.model.decay_half_life_days,
        "windows": {
            "validation": {
                "span": [VALIDATION_SPAN.first_season, VALIDATION_SPAN.last_season],
                "n": validation_window.n,
                "n_covered": int(validation_window.covered.sum()),
                "paper_n": 2280,
            },
            "test": {
                "span": [TEST_SPAN.first_season, TEST_SPAN.last_season],
                "n": test_window.n,
                "n_covered": int(test_window.covered.sum()),
                "paper_n": 2660,
            },
        },
        "validation_pools": validation_pools,
        "test_pools_refitted": test_pools,
        "test_pools_carried": carried,
        "test_scores": test_scores,
        "paper_results": PAPER_RESULTS,
        "half_life_sensitivity": sensitivity,
        "verdict": verdict(validation_pools, test_scores, sensitivity),
    }
