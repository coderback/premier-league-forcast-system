"""Dixon-Coles parameterised by a single Elo rating difference — the comparison arm.

    d = (R_home - R_away) / elo_scale
    log lam = a + h + c * d
    log mu  = a     - c * d

Four free parameters — base log rate ``a``, home advantage ``h``, Elo slope ``c``, and the tau
correction ``rho`` — against the production model's ~2N + 3. This is the WC2026 project's design,
reproduced here to answer one question: **was moving to per-team attack and defence the right call
for a domestic league?**

Everything except the strength parameterisation is deliberately identical to
:mod:`plmodel.model.dixon_coles`: the same tau correction, the same exponential decay, the same
weighted likelihood, the same scoreline collapse, the same analytic-gradient optimiser. One arm,
one axis.

Why the single scalar might lose
--------------------------------
A rating difference cannot express "great attack, leaky defence". Two clubs on identical ratings
that arrive there by opposite routes get identical forecasts, and in a 20-team league that
distinction is both large and persistent. Against that, four parameters estimated from thousands of
matches are far better determined than forty from the same data, which is precisely why the scalar
was the right choice for international football.

Why the fit is trivially fast
-----------------------------
The Elo replay does the work of estimating strength; this model only has to calibrate how a rating
difference maps to goal rates. Four parameters means the optimiser converges in a handful of
iterations, so no warm start is needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from plmodel.model.dixon_coles import (
    _LOG_RATE_MAX, _LOG_RATE_MIN, _RHO_CLAMP_MARGIN, _INVALID_NLL, _tau_terms, decay_weights,
)
from plmodel.model.scoreline import clamp_rho_for_rates, three_class_from_rates
from plmodel.ratings.elo import ELO_SCALE, EloReplay

# Optimiser vector layout: base log rate, home advantage, Elo slope, tau correction.
_A, _H, _C, _RHO = 0, 1, 2, 3
_N_PARAMS = 4


@dataclass(frozen=True)
class EloDixonColesFit:
    """A fitted Elo-difference Dixon-Coles, plus the ratings it reads strength from."""

    a: float
    h: float
    c: float
    rho: float
    ratings: dict[str, float]
    initial_rating: float
    half_life_days: float
    ref_date: pd.Timestamp
    max_goals: int
    n_obs: int
    effective_n: float
    neg_log_lik: float
    converged: bool
    n_iterations: int

    def rates(self, home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        base = self.initial_rating
        diff = np.array(
            [self.ratings.get(x, base) - self.ratings.get(y, base) for x, y in zip(home, away)],
            dtype=float,
        ) / ELO_SCALE
        log_lam = np.clip(self.a + self.h + self.c * diff, _LOG_RATE_MIN, _LOG_RATE_MAX)
        log_mu = np.clip(self.a - self.c * diff, _LOG_RATE_MIN, _LOG_RATE_MAX)
        return np.exp(log_lam), np.exp(log_mu)

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        lam, mu = self.rates(rows["home_team"], rows["away_team"])
        rho, _ = clamp_rho_for_rates(lam, mu, self.rho, margin=_RHO_CLAMP_MARGIN)
        return three_class_from_rates(lam, mu, rho, self.max_goals)

    def as_dict(self) -> dict[str, object]:
        return {
            "a": self.a, "h": self.h, "c": self.c, "rho": self.rho,
            "half_life_days": self.half_life_days,
            "ref_date": str(pd.Timestamp(self.ref_date).date()),
            "n_params": _N_PARAMS,
            "n_obs": self.n_obs,
            "effective_n": self.effective_n,
            "neg_log_lik": self.neg_log_lik,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
        }


def _objective(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    diff: np.ndarray,
    weights: np.ndarray,
    lgamma_x: np.ndarray,
    lgamma_y: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Weighted negative log-likelihood and its analytic gradient."""
    a, h, c, rho = theta[_A], theta[_H], theta[_C], theta[_RHO]
    log_lam = np.clip(a + h + c * diff, _LOG_RATE_MIN, _LOG_RATE_MAX)
    log_mu = np.clip(a - c * diff, _LOG_RATE_MIN, _LOG_RATE_MAX)
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    tau, dt_dloglam, dt_dlogmu, dt_drho = _tau_terms(x, y, lam, mu, rho)
    if np.any(tau <= 0):
        return _INVALID_NLL, np.zeros_like(theta)

    ll = x * log_lam - lam - lgamma_x + y * log_mu - mu - lgamma_y + np.log(tau)
    total = float(np.sum(weights * ll))
    if not np.isfinite(total):
        return _INVALID_NLL, np.zeros_like(theta)

    g_lam = weights * (x - lam + dt_dloglam)
    g_mu = weights * (y - mu + dt_dlogmu)

    grad = np.empty_like(theta)
    grad[_A] = g_lam.sum() + g_mu.sum()
    grad[_H] = g_lam.sum()
    # The slope enters lam with +d and mu with -d.
    grad[_C] = float(np.sum(g_lam * diff) - np.sum(g_mu * diff))
    grad[_RHO] = float(np.sum(weights * dt_drho))
    return -total, -grad


def fit_elo_dixon_coles(
    history: pd.DataFrame,
    replay: EloReplay,
    *,
    half_life_days: float,
    ref_date: pd.Timestamp,
    max_goals: int,
    param_bounds: dict[str, tuple[float, float]],
    max_iter: int,
) -> EloDixonColesFit:
    """Calibrate the rating-difference-to-goal-rate map on pre-barrier matches.

    ``history`` must already be restricted to matches before the barrier, and ``replay`` must be
    the single global Elo pass — its ``elo_diff_pre`` column is the pre-match difference, which is
    causal by construction.
    """
    if len(history) == 0:
        raise ValueError("cannot fit on an empty history")
    if "elo_diff_pre" not in history.columns:
        raise ValueError("history needs elo_diff_pre; pass rows from the Elo replay")

    weights = decay_weights(history["date"], ref_date, half_life_days)
    x = history["home_goals"].to_numpy(dtype=float)
    y = history["away_goals"].to_numpy(dtype=float)
    diff = history["elo_diff_pre"].to_numpy(dtype=float) / ELO_SCALE
    lgamma_x, lgamma_y = gammaln(x + 1.0), gammaln(y + 1.0)

    total_w = weights.sum()
    mean_home = float((weights * x).sum() / total_w)
    mean_away = float((weights * y).sum() / total_w)
    theta0 = np.array([
        math.log(max(mean_away, np.finfo(float).tiny)),
        math.log(max(mean_home, np.finfo(float).tiny)) - math.log(max(mean_away, np.finfo(float).tiny)),
        0.0,
        0.0,
    ])
    bounds = [
        tuple(param_bounds["intercept"]),
        tuple(param_bounds["home_advantage"]),
        tuple(param_bounds["elo_slope"]),
        tuple(param_bounds["rho"]),
    ]
    theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])

    result = minimize(
        _objective,
        theta0,
        args=(x, y, diff, weights, lgamma_x, lgamma_y),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    return EloDixonColesFit(
        a=float(result.x[_A]),
        h=float(result.x[_H]),
        c=float(result.x[_C]),
        rho=float(result.x[_RHO]),
        ratings=replay.ratings_asof(ref_date),
        initial_rating=replay.config.initial_rating,
        half_life_days=float(half_life_days),
        ref_date=pd.Timestamp(ref_date),
        max_goals=max_goals,
        n_obs=int(len(history)),
        effective_n=float(total_w),
        neg_log_lik=float(result.fun),
        converged=bool(result.success),
        n_iterations=int(result.nit),
    )
