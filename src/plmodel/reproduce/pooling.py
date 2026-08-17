"""Logarithmic opinion pooling, and the weight that answers "does this model add anything?".

Genest & Zidek (1986). For forecasts p^(1)..p^(K) over the same outcomes and weights w on the
simplex,

    p_i ∝ prod_k (p_i^(k)) ** w_k,     renormalised to sum to one.

The logarithmic form is preferred to the linear one because it is externally Bayesian and operates
naturally on log-odds — a linear pool of two sharp, disagreeing forecasts produces a bimodal
mixture, whereas the logarithmic pool produces a compromise.

Why the weight is the interesting quantity
------------------------------------------
Reporting a model's accuracy next to the market's answers the wrong question: two forecasts can
score almost identically while one is entirely redundant given the other. The fitted pool weight
asks the question directly — *given the market price, does admitting any amount of this model
improve the forecast?* A weight indistinguishable from zero says the market has already priced
everything the model knows.

A fitted weight of exactly zero is a boundary solution, and a boundary solution cannot by itself
distinguish a genuine null from an optimiser that stopped at a bound. That is why
:func:`weight_profile` exists: tracing the whole loss curve shows whether zero is the argument
minimum or merely where the search halted.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from plmodel.eval.metrics import EPS

# The admissible range for a pool weight: outside [0, 1] the result is no longer a mixture of two
# opinions. The profile is deliberately traced wider than this (see `weight_profile`) to answer the
# sharper question of whether a model is *anti*-informative given the price.
POOL_MIN, POOL_MAX = 0.0, 1.0


def log_pool(forecasts: list[np.ndarray], weights: np.ndarray | list[float]) -> np.ndarray:
    """Logarithmic pool of K forecasts with the given weights. Weights need not sum to 1."""
    if len(forecasts) != len(weights):
        raise ValueError(f"{len(forecasts)} forecasts but {len(weights)} weights")
    if not forecasts:
        raise ValueError("need at least one forecast")
    shape = forecasts[0].shape
    if any(f.shape != shape for f in forecasts):
        raise ValueError("all forecasts must have the same shape")

    log_p = np.zeros(shape, dtype=float)
    for forecast, weight in zip(forecasts, weights):
        log_p += float(weight) * np.log(np.clip(forecast, EPS, 1.0))
    # Subtract the row max before exponentiating: the pooled log-scores are unnormalised and a
    # negative weight can drive them far from zero.
    log_p -= log_p.max(axis=1, keepdims=True)
    pooled = np.exp(log_p)
    return pooled / pooled.sum(axis=1, keepdims=True)


def pool_log_loss(
    forecasts: list[np.ndarray], weights: np.ndarray | list[float], outcomes: np.ndarray
) -> float:
    """Mean log loss of the pooled forecast — the objective the weight is fitted on."""
    from plmodel.eval import metrics

    return metrics.mean_log_loss(log_pool(forecasts, weights), outcomes)


def fit_pair_weight(
    structural: np.ndarray,
    reference: np.ndarray,
    outcomes: np.ndarray,
    *,
    lower: float = POOL_MIN,
    upper: float = POOL_MAX,
    n_grid: int,
) -> dict[str, float]:
    """Weight on ``structural`` in a two-way pool against ``reference``.

    Follows the paper's convention: ``p ∝ (p_reference)^(1-w) (p_structural)^w``, so ``w`` is the
    weight on the *structural* model and the reference carries the remainder. Minimises
    out-of-sample log loss.

    Solved by a dense grid followed by a local polish. A pure local optimiser started in the
    interior can stop at a bound and report it as a solution, which is exactly the ambiguity this
    whole exercise is about; the grid makes the reported minimum a global one over the range.
    """
    grid = np.linspace(lower, upper, n_grid)
    losses = np.array([
        pool_log_loss([structural, reference], [w, 1.0 - w], outcomes) for w in grid
    ])
    best = int(np.argmin(losses))
    result = minimize(
        lambda w: pool_log_loss([structural, reference], [w[0], 1.0 - w[0]], outcomes),
        x0=np.array([grid[best]]),
        method="L-BFGS-B",
        bounds=[(lower, upper)],
    )
    weight = float(result.x[0])
    return {
        "weight": weight,
        "log_loss": float(result.fun),
        "log_loss_at_zero": float(losses[0]) if lower == POOL_MIN else
        pool_log_loss([structural, reference], [0.0, 1.0], outcomes),
        "at_lower_bound": bool(np.isclose(weight, lower, atol=1e-6)),
        "at_upper_bound": bool(np.isclose(weight, upper, atol=1e-6)),
        "lower": float(lower),
        "upper": float(upper),
    }


def weight_profile(
    structural: np.ndarray,
    reference: np.ndarray,
    outcomes: np.ndarray,
    *,
    lower: float,
    upper: float,
    n_grid: int,
) -> dict[str, object]:
    """Trace log loss across the whole weight range.

    The check a fitted boundary value cannot provide on its own: if the loss increases
    monotonically from the lower bound, zero is the genuine argument minimum rather than the point
    where an optimiser gave up.
    """
    grid = np.linspace(lower, upper, n_grid)
    losses = np.array([
        pool_log_loss([structural, reference], [w, 1.0 - w], outcomes) for w in grid
    ])
    increasing = bool(np.all(np.diff(losses) > 0))
    return {
        "weights": grid.tolist(),
        "log_loss": losses.tolist(),
        "argmin_weight": float(grid[int(np.argmin(losses))]),
        "monotone_increasing": increasing,
        "loss_at_lower": float(losses[0]),
        "loss_at_upper": float(losses[-1]),
    }


def fit_simplex_weights(
    forecasts: dict[str, np.ndarray], outcomes: np.ndarray
) -> dict[str, float]:
    """Weights on the simplex for a K-way pool, minimising out-of-sample log loss.

    The multi-model extension: each forecast is raised to its own weight, the weights are
    non-negative and sum to one. Used for the three-way market/goals/shots pool, where the question
    is whether two structural models built on different targets survive *together* against the
    price.
    """
    names = list(forecasts)
    stack = [forecasts[n] for n in names]
    k = len(names)
    if k < 2:
        raise ValueError("need at least two forecasts to pool")

    def objective(w: np.ndarray) -> float:
        return pool_log_loss(stack, w, outcomes)

    start = np.full(k, 1.0 / k)
    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
    )
    weights = np.clip(result.x, 0.0, 1.0)
    weights = weights / weights.sum()
    return {
        **{name: float(w) for name, w in zip(names, weights)},
        "log_loss": float(objective(weights)),
        "converged": bool(result.success),
    }
