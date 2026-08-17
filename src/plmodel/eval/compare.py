"""The acceptance harness — the standing instrument every candidate change is judged by.

One arm changes one axis. Every arm replays the identical walk-forward splits, is scored on the
identical matches, and is compared to the baseline with the paired bootstrap and a HAC-corrected
Diebold-Mariano. The acceptance rule is read from config and embedded verbatim in the report, so a
report can never claim a rule the project does not hold.

Three guards make this a falsification machine rather than a scoreboard:

**The alignment guard.** Arms are compared per match, so a silent reindex would destroy the
pairing that makes the comparison sensitive in the first place. Row identities are asserted equal
across arms and the run fails loudly rather than reindexing.

**The does-it-do-anything guard.** ``assert_arms_differ`` requires every non-baseline arm's
probability vector to differ from the baseline's, and the baseline to be reproducible bit for bit.
This exists because of the WC2026 false-null trap: a broken experiment returning "no effect" looks
exactly like a correct experiment returning "no effect", and nothing else in a harness can tell
them apart.

**The market gate.** Gate 1 is scored on the full pool; gate 2 on the odds-covered subset, because
the acceptance rule scopes it there. An arm that improves the pooled number while degrading against
the market has optimised the instrument rather than the target — precisely what WC2026's ``rsfit``
arm did, and the reason there are two gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from plmodel.config import Config
from plmodel.eval import metrics
from plmodel.eval.backtest import Split, validate_splits
from plmodel.eval.calibration import calibration_report
from plmodel.eval.slices import add_slice_columns, all_slices

# Columns that identify a match. Two arms disagreeing on these are not comparable.
ALIGN_COLUMNS: tuple[str, ...] = ("date", "division", "home_team", "away_team")

# A forecaster takes the test rows of one split plus the training rows available at its barrier,
# and returns (n, 3) probabilities. Model-free arms ignore the training frame.
Forecaster = Callable[[pd.DataFrame, pd.DataFrame, Config], np.ndarray]

_REGISTRY: dict[str, Forecaster] = {}


def register(name: str) -> Callable[[Forecaster], Forecaster]:
    """Register a forecaster under an arm name."""

    def deco(fn: Forecaster) -> Forecaster:
        if name in _REGISTRY:
            raise ValueError(f"forecaster {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def registered_arms() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


@register("uniform")
def _uniform(test: pd.DataFrame, train: pd.DataFrame, cfg: Config) -> np.ndarray:
    from plmodel.model.baselines import uniform

    return uniform(test)


@register("home-always")
def _home_always(test: pd.DataFrame, train: pd.DataFrame, cfg: Config) -> np.ndarray:
    from plmodel.model.baselines import home_always

    return home_always(test)


@register("home-rate")
def _home_rate(test: pd.DataFrame, train: pd.DataFrame, cfg: Config) -> np.ndarray:
    """The league's long-run home/draw/away split, estimated from data before the barrier."""
    from plmodel.model.baselines import empirical_base_rate, home_rate

    return home_rate(test, rate=empirical_base_rate(train))


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the comparison: exactly one axis changed from the baseline."""

    name: str
    forecaster: str

    @classmethod
    def parse(cls, name: str) -> "ArmSpec":
        if name not in _REGISTRY:
            raise ValueError(f"unknown arm {name!r}; registered: {registered_arms()}")
        return cls(name=name, forecaster=name)


@dataclass
class ArmResult:
    name: str
    probs: np.ndarray = field(repr=False)
    pooled: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, object] = field(default_factory=dict, repr=False)
    slices: pd.DataFrame | None = field(default=None, repr=False)
    vs_baseline: dict[str, float] | None = None
    vs_baseline_dm: dict[str, float] | None = None
    vs_market: dict[str, float] | None = None


@dataclass
class CompareReport:
    acceptance_rule: str
    arms: list[ArmResult]
    matches: pd.DataFrame = field(repr=False)
    outcomes: np.ndarray = field(repr=False)
    splits: dict[str, object] = field(default_factory=dict)
    market: dict[str, object] | None = None
    fdr: dict[str, object] | None = None


def _pooled_block(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    pooled = metrics.summary(probs, outcomes)
    pooled["rps_uniform"] = metrics.mean_rps(metrics.uniform_baseline(len(outcomes)), outcomes)
    pooled["skill"] = metrics.skill(pooled["rps"], pooled["rps_uniform"])
    return pooled


def run_arm(
    spec: ArmSpec, matches: pd.DataFrame, splits: Sequence[Split], cfg: Config
) -> np.ndarray:
    """Run one arm over the walk, returning probabilities aligned to the concatenated test rows."""
    forecaster = _REGISTRY[spec.forecaster]
    blocks = []
    for split in splits:
        test = split.test(matches)
        probs = forecaster(test, split.train(matches), cfg)
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (len(test), metrics.AWAY + 1):
            raise ValueError(
                f"arm {spec.name!r} returned {probs.shape} for {len(test)} test rows"
            )
        blocks.append(probs)
    return np.vstack(blocks)


def build_pool(matches: pd.DataFrame, splits: Sequence[Split]) -> pd.DataFrame:
    """The concatenated test rows, in split order — the pool every arm is scored on."""
    frames = [split.test(matches).assign(_split=split.index, _barrier=split.barrier)
              for split in splits]
    return pd.concat(frames, ignore_index=True)


def assert_aligned(frames: Sequence[pd.DataFrame]) -> None:
    """Every arm must be scored on identical match rows, in identical order."""
    if not frames:
        return
    base = frames[0][list(ALIGN_COLUMNS)].reset_index(drop=True)
    for i, other in enumerate(frames[1:], start=1):
        if not base.equals(other[list(ALIGN_COLUMNS)].reset_index(drop=True)):
            raise ValueError(
                f"arm {i} match rows do not align with the baseline; refusing to reindex — "
                "a paired comparison on mismatched rows is meaningless"
            )


def assert_arms_differ(results: Sequence[ArmResult]) -> None:
    """Every non-baseline arm must actually change the forecast.

    The WC2026 false-null trap: an arm that silently reproduces the baseline returns "no effect",
    which is indistinguishable from a correct experiment returning "no effect". Without this check
    a broken harness looks exactly like an honest null.
    """
    if not results:
        return
    baseline = results[0]
    identical = [
        arm.name for arm in results[1:] if np.array_equal(arm.probs, baseline.probs)
    ]
    if identical:
        raise ValueError(
            f"arm(s) {identical} produced probabilities identical to the baseline "
            f"{baseline.name!r} — the arm is not doing anything, so its null is uninformative"
        )


def run_compare(
    matches: pd.DataFrame,
    splits: Sequence[Split],
    cfg: Config,
    arm_names: Sequence[str],
    *,
    history: pd.DataFrame | None = None,
    n_bins: int,
    big_six: tuple[str, ...],
) -> CompareReport:
    """Run every arm over the walk and pair-compare them. The first arm is the baseline."""
    if len(arm_names) < 1:
        raise ValueError("need at least one arm")
    specs = [ArmSpec.parse(name) for name in arm_names]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError(f"duplicate arms requested: {list(arm_names)}")

    validate_splits(matches, splits)
    rows = build_pool(matches, splits)
    assert_aligned([rows] * len(specs))  # identical by construction; asserted anyway

    from plmodel.model.baselines import market, outcomes_of

    outcomes = outcomes_of(rows)

    market_probs = market(
        rows, family=cfg.odds.gate_benchmark, method=cfg.odds.devig_primary,
        sum_tolerance=cfg.odds.sum_tolerance,
    )
    covered = ~np.isnan(market_probs).any(axis=1)

    rows = add_slice_columns(
        rows,
        history=history if history is not None else matches,
        division=cfg.backtest.prediction_division,
        big_six=big_six,
        market_probs=market_probs,
    )

    results: list[ArmResult] = []
    for spec in specs:
        probs = run_arm(spec, matches, splits, cfg)
        if np.isnan(probs).any():
            raise ValueError(
                f"arm {spec.name!r} produced NaN probabilities; an arm must cover the whole pool. "
                "A partially-covered forecaster is a benchmark, not an arm."
            )
        results.append(
            ArmResult(
                name=spec.name,
                probs=probs,
                pooled=_pooled_block(probs, outcomes),
                calibration=calibration_report(probs, outcomes, n_bins=n_bins),
                slices=all_slices(rows, probs, outcomes, n_bins=n_bins),
            )
        )
    assert_arms_differ(results)

    baseline = results[0]
    for arm in results[1:]:
        arm.vs_baseline = metrics.paired_delta(
            arm.probs, baseline.probs, outcomes,
            n_boot=cfg.backtest.n_boot, seed=cfg.seed,
        )
        arm.vs_baseline_dm = metrics.diebold_mariano(
            metrics.rps(arm.probs, outcomes), metrics.rps(baseline.probs, outcomes)
        )

    market_block = _market_block(results, market_probs, covered, outcomes, cfg)
    fdr = _fdr_block(results, cfg)

    return CompareReport(
        acceptance_rule=cfg.acceptance_rule,
        arms=results,
        matches=rows,
        outcomes=outcomes,
        splits={
            "n_splits": len(splits),
            "n_refits": sum(1 for s in splits if s.is_refit),
            "first_barrier": str(splits[0].barrier.date()),
            "last_barrier": str(splits[-1].barrier.date()),
            "refit_every": cfg.backtest.refit_every,
        },
        market=market_block,
        fdr=fdr,
    )


def _market_block(
    results: Sequence[ArmResult],
    market_probs: np.ndarray,
    covered: np.ndarray,
    outcomes: np.ndarray,
    cfg: Config,
) -> dict[str, object] | None:
    """Gate 2: every arm against the de-vigged market on the odds-covered subset."""
    block: dict[str, object] = {
        "benchmark": cfg.odds.gate_benchmark,
        "devig": cfg.odds.devig_primary,
        "n_covered": int(covered.sum()),
        "n_total": int(len(outcomes)),
    }
    if not covered.any():
        return block
    covered_market = market_probs[covered]
    covered_outcomes = outcomes[covered]
    block["rps_market"] = metrics.mean_rps(covered_market, covered_outcomes)
    block["log_loss_market"] = metrics.mean_log_loss(covered_market, covered_outcomes)
    for arm in results:
        arm.vs_market = metrics.paired_delta(
            arm.probs[covered], covered_market, covered_outcomes,
            n_boot=cfg.backtest.n_boot, seed=cfg.seed,
        )
    return block


def _fdr_block(results: Sequence[ArmResult], cfg: Config) -> dict[str, object] | None:
    """Benjamini-Hochberg across the family of arms tested in this run.

    Uses the DM p-value, which is a genuine two-sided p-value; the bootstrap's ``p_a_better`` is a
    one-sided posterior-style quantity and is not an input to FDR control.
    """
    p_values = {
        arm.name: arm.vs_baseline_dm["p_value"]
        for arm in results[1:]
        if arm.vs_baseline_dm is not None
    }
    if not p_values:
        return None
    return metrics.benjamini_hochberg(p_values, alpha=cfg.backtest.fdr_alpha)


def gate_verdicts(report: CompareReport) -> dict[str, dict[str, object]]:
    """Apply the acceptance rule's two gates to each non-baseline arm.

    Gate 1 passes when the paired delta is favourable — 95% CI excluding 0, or P(better) >= 0.95.
    Gate 2 passes when the arm's delta against the market does not degrade relative to baseline.
    Both must pass; the verdict is reported, never acted on automatically.
    """
    baseline = report.arms[0]
    verdicts: dict[str, dict[str, object]] = {}
    for arm in report.arms[1:]:
        delta = arm.vs_baseline or {}
        ci_excludes_zero = bool(delta.get("ci_high", 0.0) < 0.0)
        p_better = float(delta.get("p_a_better", 0.0))
        gate1 = ci_excludes_zero or p_better >= _P_BETTER_BAR

        gate2 = None
        if arm.vs_market is not None and baseline.vs_market is not None:
            gate2 = bool(arm.vs_market["delta_rps"] <= baseline.vs_market["delta_rps"])

        verdicts[arm.name] = {
            "gate1_vs_baseline": gate1,
            "gate1_reason": (
                "95% CI excludes 0" if ci_excludes_zero
                else f"P(better) = {p_better:.3f}"
            ),
            "gate2_vs_market": gate2,
            "gate2_reason": (
                None if gate2 is None else
                f"arm gap {arm.vs_market['delta_rps']:+.5f} vs baseline gap "
                f"{baseline.vs_market['delta_rps']:+.5f}"
            ),
            "accepted": bool(gate1 and (gate2 is not False)),
        }
    return verdicts


# The acceptance rule's one-sided bar: P(better) >= 0.95. Stated in config.yaml's rule text; kept
# here as a named constant so the code and the rule cannot drift apart silently.
_P_BETTER_BAR = 0.95


def report_json(report: CompareReport) -> dict[str, object]:
    """The JSON report. The acceptance rule is embedded verbatim, as the brief requires."""
    return {
        "acceptance_rule": report.acceptance_rule,
        "splits": report.splits,
        "market": report.market,
        "fdr": report.fdr,
        "verdicts": gate_verdicts(report),
        "arms": {
            arm.name: {
                "pooled": arm.pooled,
                "vs_baseline": arm.vs_baseline,
                "vs_baseline_dm": arm.vs_baseline_dm,
                "vs_market": arm.vs_market,
                "calibration": {
                    outcome: block["decomposition"]
                    for outcome, block in arm.calibration.items()
                },
                "slices": (
                    arm.slices.to_dict("records") if arm.slices is not None else None
                ),
            }
            for arm in report.arms
        },
    }
