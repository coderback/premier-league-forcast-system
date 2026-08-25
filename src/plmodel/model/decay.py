"""Per-parameter time decay: a separate memory for team strength, the league level and home advantage.

The production fit weights every observation by ``exp(-ln2 * age / half_life)`` and uses **one**
half-life for all of it. That single number is doing two incompatible jobs, and three measurements
say so:

* Home advantage falls **+0.2711 -> +0.1774** across the test decade. A 730-day memory tracks a
  decade-long slide badly.
* The same drift shows up as a rate error. The model under-predicts total goals by 2.46%, and
  **three times worse on away goals (+3.9%) than home (+1.3%)** -- it is carrying too much of the
  league's scoring on the home side.
* 2023-24 is a level shift no single memory could follow: **3.279 goals per match actual against
  2.809 predicted**, a season spent a sixth of a goal short on every match.

Meanwhile the score-driven arm wants the *opposite*: a long memory, because its states are what
make old data usable. Today it can only buy that by also buying a stale home advantage.

Why this is not a config change
-------------------------------
The obvious implementation -- give each parameter block its own weight vector inside the existing
likelihood -- is wrong, and quietly so. ``_objective`` returns a value and a gradient that
``L-BFGS-B`` trusts as a matched pair::

    total = sum(weights * ll_per)                    # the value
    g_lam = weights * (x - lam + dt_dloglam)         # every block's gradient flows from here

Use a different weight vector per block and ``-total`` stops being the antiderivative of ``-grad``.
The optimiser is then handed an inconsistent pair and will converge to something that is not the
optimum of anything -- with no error, no warning, and a plausible-looking answer.

So the estimator is **block coordinate descent**. Each block is a genuine weighted maximum
likelihood problem under its own half-life, holding the other blocks fixed, and the blocks are
cycled until nothing moves:

===========  =========================================  ====================
block        parameters                                 half-life
===========  =========================================  ====================
``team``     attack, defence                            long
``level``    intercept, rho                             short
``home``     home advantage (and its design columns)    short
===========  =========================================  ====================

Within any single block solve the value and the gradient come from one weight vector, so the
``jac=True`` contract holds exactly. What the cycle solves is a set of estimating equations rather
than one likelihood; that is the honest description of the object, and it is the same shape as any
other backfitting or profile estimator.

**The likelihood is not reimplemented.** Each block solve calls the production ``_objective`` with
out-of-block parameters pinned to their current values by equal bounds. One implementation, one
gradient, nothing to drift -- at the cost of doing full-vector work to move a single scalar, which
is a price worth paying for not having a second copy of the model.

Inertness
---------
When all three half-lives are equal this module must not run at all. :meth:`DecaySpec.is_inert`
says so, ``Config.decay_spec()`` returns ``None``, and ``fit_dixon_coles`` takes its ordinary
single-weight path. The seam is absent from the call graph rather than reproducing the same answer
by a longer route -- the same discipline the scoreline family follows.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# The three blocks, in the order they are cycled. Team strength first because it carries by far the
# most parameters and the other two are close to conditional means given it, so leading with it
# makes the first cycle do most of the work.
TEAM, LEVEL, HOME = "team", "level", "home_advantage"
BLOCKS: tuple[str, ...] = (TEAM, LEVEL, HOME)

# Sentinel for a half-life the seam did not name: it inherits the production value rather than
# silently taking some default nobody chose. Mirrors GasSpec.from_seam's `fallback_half_life`.
_INHERIT = None


class DecayError(ValueError):
    """Raised when a per-parameter decay configuration cannot be honoured."""


@dataclass(frozen=True)
class DecaySpec:
    """One half-life per parameter block, in days.

    ``is_inert`` is equality of all three rather than a flag, for the same reason the scoreline
    family's inertness is a property of its marginal and dependence: the configuration that means
    "off" should be a *value* somebody can read, not a boolean that can disagree with the values
    beside it.
    """

    team: float
    level: float
    home_advantage: float
    max_cycles: int
    tolerance: float

    def __post_init__(self) -> None:
        for name in BLOCKS:
            value = float(getattr(self, name))
            if value <= 0:
                raise DecayError(f"{name} half-life must be positive; got {value}")
        if self.max_cycles < 1:
            raise DecayError(f"max_cycles must be at least 1; got {self.max_cycles}")
        if self.tolerance <= 0:
            raise DecayError(f"tolerance must be positive; got {self.tolerance}")

    @property
    def is_inert(self) -> bool:
        """True when every block shares one memory -- which IS the production specification."""
        return self.team == self.level == self.home_advantage

    def half_life(self, block: str) -> float:
        if block not in BLOCKS:
            raise DecayError(f"unknown block {block!r}; expected one of {BLOCKS}")
        return float(getattr(self, block))

    @classmethod
    def from_seam(cls, seam: dict, *, fallback_half_life: float) -> "DecaySpec":
        """Build from ``model.seams.decay``, inheriting any half-life the seam does not name.

        Inheriting rather than defaulting is what makes a partially-specified seam mean something
        precise: naming only ``home_advantage`` says "give home advantage its own memory and leave
        everything else where production has it", which is a configuration somebody might well want
        and should not have to spell out three times to get.
        """
        settings = seam or {}
        chosen = {
            name: float(settings.get(name, _INHERIT) or fallback_half_life) for name in BLOCKS
        }
        missing = [k for k in ("max_cycles", "tolerance") if settings.get(k) is None]
        if missing:
            raise DecayError(
                f"model.seams.decay is missing {missing}: the cycle's stopping rule has no "
                "defensible default and belongs in config.yaml with its reasoning"
            )
        return cls(
            max_cycles=int(settings["max_cycles"]),
            tolerance=float(settings["tolerance"]),
            **chosen,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "team_half_life_days": self.team,
            "level_half_life_days": self.level,
            "home_advantage_half_life_days": self.home_advantage,
            "max_cycles": self.max_cycles,
            "tolerance": self.tolerance,
        }

    def label(self) -> str:
        return f"team={self.team:g} level={self.level:g} ha={self.home_advantage:g}"


def block_weights(
    dates: pd.Series, ref_date: pd.Timestamp, spec: DecaySpec
) -> dict[str, np.ndarray]:
    """One decay-weight vector per block, from the production ``decay_weights``.

    Deliberately the same function the single-half-life path uses, so a block whose half-life
    equals the production value gets a vector that is identical bit for bit.
    """
    from plmodel.model.dixon_coles import decay_weights

    return {block: decay_weights(dates, ref_date, spec.half_life(block)) for block in BLOCKS}


@dataclass(frozen=True)
class BlockLayout:
    """Which slots of the optimiser vector belong to which block.

    Built from the same offsets ``_unpack`` uses. Home-advantage *design* columns (the trend and
    empty-stadium terms) join the home block rather than the level block, because they are the same
    quantity measured differently and splitting them would let the two disagree.

    Covariate and family parameters are deliberately absent: those seams cannot run alongside this
    one, and a slot that belongs to no block is pinned throughout, which is the safe direction.
    """

    n_teams: int
    n_ha: int
    total: int

    def slots(self, block: str) -> np.ndarray:
        from plmodel.model.dixon_coles import _HOME_ADV, _INTERCEPT, _N_GLOBAL, _RHO

        free = self.n_teams - 1
        if block == TEAM:
            return np.arange(_N_GLOBAL, _N_GLOBAL + 2 * free)
        if block == LEVEL:
            return np.array([_INTERCEPT, _RHO])
        if block == HOME:
            ha_start = _N_GLOBAL + 2 * free
            return np.concatenate(
                [np.array([_HOME_ADV]), np.arange(ha_start, ha_start + self.n_ha)]
            ).astype(int)
        raise DecayError(f"unknown block {block!r}; expected one of {BLOCKS}")


def fit_blockwise(
    theta0: np.ndarray,
    bounds: list[tuple[float, float]],
    spec: DecaySpec,
    weights: dict[str, np.ndarray],
    layout: BlockLayout,
    *,
    objective,
    args_without_weights: tuple,
    weights_position: int,
    max_iter: int,
) -> dict[str, object]:
    """Cycle the blocks to convergence and return the solution plus how it got there.

    ``args_without_weights`` is the production objective's argument tuple with the weight vector
    removed, and ``weights_position`` says where to splice each block's own vector back in. Passing
    the objective and its arguments in rather than rebuilding them here is what keeps this module
    free of any second copy of the likelihood.

    Returns the solution, whether the cycle converged, how many cycles it took, and the final
    movement -- all reported, because a fit that stopped because it ran out of cycles is a
    diagnostic and must not look like one that stopped because it was finished.
    """
    theta = np.asarray(theta0, dtype=float).copy()
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    movement = float("inf")
    cycles = 0
    n_iterations = 0
    converged_blocks = True
    last_value = float("nan")

    for cycles in range(1, spec.max_cycles + 1):
        previous = theta.copy()
        for block in BLOCKS:
            slots = layout.slots(block)
            if slots.size == 0:
                continue
            # Everything outside the block is pinned by equal bounds, so this solve is a genuine
            # weighted MLE over the block alone, using the one production likelihood.
            block_lo, block_hi = theta.copy(), theta.copy()
            block_lo[slots] = lo[slots]
            block_hi[slots] = hi[slots]
            args = (
                *args_without_weights[:weights_position],
                weights[block],
                *args_without_weights[weights_position:],
            )
            result = minimize(
                objective,
                theta,
                args=args,
                method="L-BFGS-B",
                jac=True,
                bounds=list(zip(block_lo, block_hi)),
                options={"maxiter": max_iter},
            )
            theta = np.clip(result.x, lo, hi)
            n_iterations += int(result.nit)
            converged_blocks = converged_blocks and bool(result.success)
            last_value = float(result.fun)

        movement = float(np.max(np.abs(theta - previous)))
        if movement <= spec.tolerance:
            break

    return {
        "theta": theta,
        "converged": bool(converged_blocks and movement <= spec.tolerance),
        "cycles": int(cycles),
        "hit_cycle_cap": bool(movement > spec.tolerance),
        "movement": movement,
        "n_iterations": int(n_iterations),
        "value": last_value,
    }
