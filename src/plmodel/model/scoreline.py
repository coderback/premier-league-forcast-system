"""Scoreline distributions and their collapse to home/draw/away.

Two independent Poisson counts with the Dixon-Coles low-score correction:

    P(X = x, Y = y) = tau(x, y; lam, mu, rho) * Poisson(x; lam) * Poisson(y; mu)

The tau correction touches only the four lowest-scoring cells, where independent Poisson is known
to misprice English league football:

    tau(0,0) = 1 - lam*mu*rho     tau(0,1) = 1 + lam*rho
    tau(1,0) = 1 + mu*rho         tau(1,1) = 1 - rho

and 1 everywhere else. A negative rho lifts the 0-0 and 1-1 cells; a positive one suppresses them.
Which sign the Premier League actually shows is measured, not assumed — see
:func:`plmodel.model.dixon_coles.fit_summary`, because it decides whether a negative-dependence
model is worth building at all.

Working in the scoreline rather than directly in three classes is what makes over/under, both-teams-
to-score and correct-score fall out of the same fit later, and it is why the goals-based approach
is preferred to an ordered-probit on the outcome (Koopman & Lit 2019).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from plmodel.eval.metrics import AWAY, DRAW, HOME

N_OUTCOMES = 3
# The four cells the Dixon-Coles correction touches.
_TAU_CELLS = ((0, 0), (0, 1), (1, 0), (1, 1))


def poisson_pmf_grid(rates: np.ndarray, max_goals: int) -> np.ndarray:
    """(N, max_goals+1) Poisson probabilities for each rate, computed in log space.

    Log space rather than a direct factorial: at the rates football produces the difference is
    immaterial, but it keeps the routine safe if it is ever pointed at a higher-scoring sport.
    """
    lam = np.asarray(rates, dtype=float).reshape(-1, 1)
    if np.any(lam <= 0):
        raise ValueError("Poisson rates must be positive")
    goals = np.arange(max_goals + 1, dtype=float).reshape(1, -1)
    return np.exp(goals * np.log(lam) - lam - gammaln(goals + 1.0))


def tau_matrix(lam: np.ndarray, mu: np.ndarray, rho: np.ndarray | float, max_goals: int) -> np.ndarray:
    """(N, G+1, G+1) Dixon-Coles correction factors — 1 outside the four low-score cells."""
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    rho = np.broadcast_to(np.asarray(rho, dtype=float), lam.shape)
    tau = np.ones((lam.shape[0], max_goals + 1, max_goals + 1))
    tau[:, 0, 0] = 1.0 - lam * mu * rho
    tau[:, 0, 1] = 1.0 + lam * rho
    tau[:, 1, 0] = 1.0 + mu * rho
    tau[:, 1, 1] = 1.0 - rho
    return tau


def scoreline_matrix(
    lam: np.ndarray, mu: np.ndarray, rho: np.ndarray | float, max_goals: int
) -> np.ndarray:
    """(N, G+1, G+1) joint scoreline probabilities, renormalised over the truncated grid.

    Truncating at ``max_goals`` discards a vanishing tail (P(>12 goals for one side) is ~1e-9 at
    football rates), and the tau correction is not a probability-preserving transform, so the grid
    is renormalised. Without that the three-class probabilities would not sum to 1.
    """
    home = poisson_pmf_grid(lam, max_goals)
    away = poisson_pmf_grid(mu, max_goals)
    joint = home[:, :, None] * away[:, None, :]
    joint = joint * tau_matrix(lam, mu, rho, max_goals)
    if np.any(joint < 0):
        raise ValueError(
            "negative scoreline probability: rho is outside the region where tau stays positive"
        )
    total = joint.sum(axis=(1, 2), keepdims=True)
    return joint / total


def collapse_three_class(joint: np.ndarray) -> np.ndarray:
    """(N, 3) home/draw/away probabilities from a scoreline grid."""
    if joint.ndim != N_OUTCOMES:
        raise ValueError(f"expected an (N, G+1, G+1) grid; got {joint.shape}")
    n, rows, cols = joint.shape
    x = np.arange(rows).reshape(1, -1, 1)
    y = np.arange(cols).reshape(1, 1, -1)
    probs = np.empty((n, N_OUTCOMES))
    probs[:, HOME] = joint.sum(axis=(1, 2), where=(x > y))
    probs[:, DRAW] = joint.sum(axis=(1, 2), where=(x == y))
    probs[:, AWAY] = joint.sum(axis=(1, 2), where=(x < y))
    return probs


def three_class_from_rates(
    lam: np.ndarray, mu: np.ndarray, rho: np.ndarray | float, max_goals: int
) -> np.ndarray:
    """(N, 3) home/draw/away probabilities straight from the expected goals."""
    return collapse_three_class(scoreline_matrix(lam, mu, rho, max_goals))


def valid_rho_range(lam: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-match ``(lo, hi)`` bounds keeping every tau cell strictly positive.

    From the four cells:  rho < 1/(lam*mu),  rho < 1,  rho > -1/lam,  rho > -1/mu.

    The range is **rate-dependent**, which is why no fixed configuration bound can guarantee
    validity: a fit whose rho is fine on its training rates can still meet an extreme matchup at
    prediction time whose rates put the same rho out of range.
    """
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    hi = np.minimum(1.0 / (lam * mu), 1.0)
    lo = np.maximum(-1.0 / lam, -1.0 / mu)
    return lo, hi


def clamp_rho_for_rates(
    lam: np.ndarray, mu: np.ndarray, rho: float, *, margin: float
) -> tuple[np.ndarray, int]:
    """Clamp rho into each match's valid range. Returns ``(rho_per_match, n_clamped)``.

    Applied at prediction, never during fitting — the likelihood already steers the optimiser away
    from an invalid rho on its own sample. This exists for the extreme matchup the fit never saw,
    where the alternative is a negative probability.

    The clamp is **counted and surfaced**, not silent. On a well-behaved fit it never binds: it was
    measured to trigger only at half-lives of 30-60 days, where the fit is degenerate anyway (rho
    pinned at its configuration bound and implied rates reaching lam*mu = 13). A non-zero count is
    therefore a signal that the configuration, not the match, is the problem.
    """
    lo, hi = valid_rho_range(lam, mu)
    # Strict inequality: back off from the boundary so tau is positive, not merely non-negative.
    lo, hi = lo * (1.0 - margin), hi * (1.0 - margin)
    clamped = np.clip(rho, lo, hi)
    return clamped, int(np.sum(clamped != rho))


def tau_is_valid(lam: np.ndarray, mu: np.ndarray, rho: float) -> bool:
    """Whether rho keeps every tau cell positive at these rates.

    tau(0,0) = 1 - lam*mu*rho is the binding constraint: it goes negative for rho above
    1/(lam*mu), which at typical rates is around 0.4. The optimiser is bounded well inside this,
    but the check exists so a bound loosened later cannot silently produce negative probabilities.
    """
    lam = np.asarray(lam, dtype=float)
    mu = np.asarray(mu, dtype=float)
    return bool(
        np.all(1.0 - lam * mu * rho > 0)
        and np.all(1.0 + lam * rho > 0)
        and np.all(1.0 + mu * rho > 0)
        and (1.0 - rho > 0)
    )


# --- markets derived from the scoreline grid ---------------------------------------------------
#
# All of these are the same object read differently: the joint distribution over (home, away)
# goals that the model already produces. Nothing here is a new model, which is the point --
# over/under and both-teams-to-score fall out of a goals model for free, and a separate fit for
# each would be several models that could disagree about the same match.

def totals_probability(joint: np.ndarray, line: float) -> tuple[np.ndarray, np.ndarray]:
    """``(over, under)`` for a total-goals line. A whole-number line would push; a .5 line cannot."""
    x, y = _goal_axes(joint)
    total = x + y
    return joint.sum(axis=(1, 2), where=(total > line)), joint.sum(axis=(1, 2), where=(total < line))


def both_teams_to_score(joint: np.ndarray) -> np.ndarray:
    """Probability each side scores at least once."""
    x, y = _goal_axes(joint)
    return joint.sum(axis=(1, 2), where=((x > 0) & (y > 0)))


def top_scorelines(joint: np.ndarray, n: int) -> list[list[tuple[int, int, float]]]:
    """The ``n`` most likely exact scores per match, most likely first."""
    out: list[list[tuple[int, int, float]]] = []
    for grid in joint:
        flat = np.argsort(grid, axis=None)[::-1][:n]
        rows, cols = np.unravel_index(flat, grid.shape)
        out.append([(int(i), int(j), float(grid[i, j])) for i, j in zip(rows, cols)])
    return out


def _goal_axes(joint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if joint.ndim != N_OUTCOMES:
        raise ValueError(f"expected an (N, G+1, G+1) grid; got {joint.shape}")
    x = np.arange(joint.shape[1]).reshape(1, -1, 1)
    y = np.arange(joint.shape[2]).reshape(1, 1, -1)
    return x, y
