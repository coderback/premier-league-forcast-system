"""Per-team Dixon-Coles: the production baseline.

    log lam_home = c + h + A[home] - D[away]
    log mu_away  = c     + A[away] - D[home]

with attack ``A``, defensive solidity ``D`` (higher = concedes less), a global intercept ``c``, a
fitted global home advantage ``h``, and the tau low-score correction ``rho``. Fitted by weighted
maximum likelihood with exponential time decay ``w = exp(-ln2 * age_days / half_life)``.

This is the specification the literature says is hard to beat, and it is Koopman & Lit's
*semi-dynamic* row (ARPS 0.2014 on 2,660 EPL matches) rather than their static one — worth
remembering when reading the result, because the static row's 0.2062 is a different model.

Three implementation decisions that are cheap now and expensive later
--------------------------------------------------------------------

**Identifiability by construction.** 2N attack/defence parameters are unidentified: adding a
constant to every attack and subtracting it from every defence leaves every rate unchanged. The
constraint is imposed *structurally* — the last team's value is minus the sum of the others, so
only N-1 free parameters per side ever exist. Fitting all 2N and projecting afterwards would leave
the optimiser working on a singular Hessian the whole way, which is slow at best and
non-reproducible at worst.

**An analytic gradient, not a numerical one.** The parameter vector is ~100 long, so finite
differences would need ~100 likelihood evaluations per gradient step and turn a single walk-forward
arm into hours. Every derivative here is closed-form and checked against ``approx_fprime``.

**Warm starts.** Consecutive barriers differ by one matchday, so the previous solution is an
excellent starting point. Combined with the analytic gradient this is what makes 1,153 refits per
arm practical.

Cold starts are counted, never imputed
--------------------------------------
A team with too little weighted history to identify its own parameters is pinned at the league
average and **reported by name**. Under a short half-life a club returning after years away has
almost no effective history even though its raw match count looks fine, so the threshold is on the
sum of decay weights rather than on a count. Promoted teams are ~28% of the fixture list, so this
is a first-class part of the model rather than an edge case.

The threshold is a **share of the median team's effective history**, not an absolute number of
matches. An absolute threshold does not survive a half-life sweep: at a 30-day half-life a team
playing every nine days accumulates only ~4.6 effective matches in total, so a fixed bar of 5 pins
the entire league, and the grid point scores badly for a reason that has nothing to do with how
well a 30-day memory forecasts football. Scaling to the median makes the criterion say what it
means — *this team has far less usable history than a normal member of this league* — at any decay
rate.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from plmodel.model.counts import (
    CountFamilyError,
    CountSpec,
    INDEPENDENT_KAPPA,
    UNIT_SHAPE,
    joint_grid,
    log_probabilities,
    marginals,
    observed_cell_likelihood,
)
from plmodel.model.covariates import CovariateSpec
from plmodel.model.covariates import design as cov_design_matrix
from plmodel.model.decay import BlockLayout, DecaySpec, block_weights, fit_blockwise
from plmodel.model.promotion import PromotionPrior, PromotionSpec, estimate_prior
from plmodel.model.shrinkage import ShrinkageSpec, ridge_penalty
from plmodel.model.home_advantage import MODE_GLOBAL as HA_GLOBAL
from plmodel.model.home_advantage import design as ha_design_matrix
from plmodel.model.home_advantage import prediction_design
from plmodel.model.scoreline import (
    clamp_rho_for_rates,
    collapse_three_class,
    three_class_from_rates,
)

# Fixed layout of the non-team parameters at the head of the optimiser vector.
_INTERCEPT, _HOME_ADV, _RHO = 0, 1, 2
_N_GLOBAL = 3

# Clip the linear predictor so the optimiser cannot overflow exp() while exploring its bounds.
# Wide enough to be inert at football rates: exp(-6) is a thousandth of a goal, exp(4) is 55.
_LOG_RATE_MIN, _LOG_RATE_MAX = -6.0, 4.0

# Returned in place of the likelihood when the parameters put a tau cell at or below zero.
_INVALID_NLL = 1e12

# Central-difference step for the handful of parameters that do not enter through a rate. Chosen
# at the usual cube-root-of-epsilon scale for a central difference, and checked against
# approx_fprime in the test suite rather than trusted.
_FD_STEP = 1e-5

# Where the weight vector sits in `_objective`'s argument tuple, after (x, y, home_i, away_i).
# Named because the block fitter splices each block's own vector into that slot, and a silent
# off-by-one there would weight the likelihood by a goal count.
_WEIGHTS_ARGUMENT_POSITION = 4

# Relative back-off from the tau validity boundary when clamping rho at prediction time, so
# the correction stays strictly positive rather than landing exactly on zero.
_RHO_CLAMP_MARGIN = 0.01


@dataclass(frozen=True)
class DixonColesFit:
    """A fitted model. Team parameters are indexed by :attr:`teams`."""

    teams: tuple[str, ...]
    attack: np.ndarray
    defence: np.ndarray
    intercept: float
    home_advantage: float
    rho: float
    half_life_days: float
    ref_date: pd.Timestamp
    max_goals: int
    n_obs: int
    effective_n: float
    neg_log_lik: float
    converged: bool
    n_iterations: int
    cold_start_teams: tuple[str, ...] = ()
    # Structural home-advantage terms (empty under the production `global` mode).
    ha_names: tuple[str, ...] = ()
    ha_params: tuple[float, ...] = ()
    ha_mode: str = HA_GLOBAL
    ha_window: tuple[str | None, str | None] = (None, None)
    # Scoreline family (model.seams.scoreline). None IS the production Poisson-and-tau
    # specification, and when it is None nothing below this line participates in the fit.
    family: CountSpec | None = None
    shape: float = UNIT_SHAPE
    kappa: float = INDEPENDENT_KAPPA
    # Match-context covariates (model.seams.covariates). None IS the production model, and when
    # it is None the design is an empty matrix product that leaves every rate untouched.
    cov_spec: CovariateSpec | None = None
    cov_names: tuple[str, ...] = ()
    cov_params: tuple[float, ...] = ()
    cov_division: str = ""
    cov_undefined: dict[str, int] = field(default_factory=dict)
    # Per-parameter decay (model.seams.decay). None IS the production single-half-life fit, and
    # when it is None nothing in decay.py participates. `half_life_days` above then carries the
    # TEAM half-life, because that is the memory governing the strength parameters every
    # downstream reader of that field cares about.
    decay: DecaySpec | None = None
    decay_diagnostics: dict[str, float] = field(default_factory=dict)
    # Promotion prior (model.seams.promotion). None IS the production model, and when
    # `promotion_prior` is None every club outside `teams` falls back to 0.0 -- the league average
    # -- exactly as it always has.
    promotion: PromotionSpec | None = None
    promotion_prior: PromotionPrior | None = None
    # Mutable on purpose: prediction-time observations that the fit itself cannot know, such
    # as how often rho had to be clamped for an extreme matchup. Surfaced in the report.
    diagnostics: dict[str, int] = field(default_factory=dict)

    def _index(self) -> dict[str, int]:
        return {team: i for i, team in enumerate(self.teams)}

    def _pinned(self, teams: pd.Series, slot: int) -> float:
        """Fallback strength for clubs outside the fit: 0, or the promotion prior where there is one.

        Returns the scalar 0.0 when the seam is off, so ``np.where`` broadcasts exactly the literal
        it broadcast before this seam existed and the off path stays byte-identical.

        One value for every unidentified club rather than a per-club lookup, and that is the design
        rather than a shortcut: a club the fit could not identify has no evidence to tell it apart
        from any other such club, so giving them different pins would be inventing information. It
        also means a club arriving with zero top-flight rows -- absent from the fit's team list
        entirely, never even reaching the cold-start test -- is covered by the same line.
        """
        if self.promotion_prior is None:
            return 0.0
        return (self.promotion_prior.attack, self.promotion_prior.defence)[slot]

    def _ha_adjustment(self, dates: pd.Series | None, n_rows: int) -> np.ndarray:
        """Structural home-advantage contribution for rows being predicted.

        Rebuilt with the SAME design used at fit time, evaluated at the barrier. The trend term is
        therefore zero — a match is zero years before its own barrier — but the empty-stadium term
        is **not**: whether an upcoming match will be played behind closed doors is public before
        kickoff, so applying it is leak-free and omitting it is a mis-specification.

        Getting this wrong is not hypothetical. An earlier version zeroed every term at prediction,
        which meant the empty-stadium arm removed the crowd effect from its historical estimate and
        then forecast the 2020-21 matches as if crowds were present. It scored *worse* in exactly
        the season it was built for, and the arm as a whole looked like a clean null.
        """
        if not self.ha_names or dates is None:
            return np.zeros(n_rows)
        matrix, names = ha_design_matrix(
            pd.Series(dates), self.ref_date, mode=self.ha_mode,
            empty_start=self.ha_window[0], empty_end=self.ha_window[1],
        )
        if names != self.ha_names:
            raise ValueError(f"prediction design {names} does not match the fit's {self.ha_names}")
        return matrix @ np.asarray(self.ha_params)

    def _cov_context(
        self, rows: pd.DataFrame, history: pd.DataFrame | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Additive log-rate contribution of the context covariates for rows being predicted.

        Rebuilt with the SAME design used at fit time, from the history behind the barrier. Unlike
        the home-advantage terms, these do NOT vanish at prediction: how long each side has rested
        and whether it is in Europe are both known before kickoff, so applying them is leak-free
        and omitting them would fit a term the forecast then refuses to use.

        The history is required rather than optional. A covariate that silently evaluated to zero
        because nobody passed the matches it counts back to would be a model that fits one thing
        and predicts another, which is the failure the home-advantage seam already paid for once.
        """
        if self.cov_spec is None or self.cov_spec.is_inert:
            return None
        if history is None:
            raise ValueError(
                "covariates are fitted but no history was given to predict with; "
                "call predict_proba(rows, history=...)"
            )
        built = cov_design_matrix(rows, history, self.cov_spec, division=self.cov_division)
        if built.names != self.cov_names:
            raise ValueError(
                f"prediction design {built.names} does not match the fit's {self.cov_names}"
            )
        params = np.asarray(self.cov_params)
        return built.lam @ params, built.mu @ params

    def rates(
        self, home: pd.Series, away: pd.Series, dates: pd.Series | None = None,
        *, context: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals for each side. Unknown teams take the league average, or the prior.

        "Unknown" here means a club the fit dropped as cold-start. It takes 0 -- the league average
        -- unless the promotion seam supplied a prior for it, which is the only way to reach the
        twelve of thirty promoted clubs that are not parameters of the fit at all.
        """
        idx = self._index()
        home_i = np.array([idx.get(t, -1) for t in home])
        away_i = np.array([idx.get(t, -1) for t in away])
        a_home = np.where(home_i >= 0, self.attack[home_i], self._pinned(home, 0))
        a_away = np.where(away_i >= 0, self.attack[away_i], self._pinned(away, 0))
        d_home = np.where(home_i >= 0, self.defence[home_i], self._pinned(home, 1))
        d_away = np.where(away_i >= 0, self.defence[away_i], self._pinned(away, 1))
        ha = self._ha_adjustment(dates, len(a_home))
        cov_lam, cov_mu = context if context is not None else (0.0, 0.0)
        log_lam = np.clip(self.intercept + self.home_advantage + ha + cov_lam + a_home - d_away,
                          _LOG_RATE_MIN, _LOG_RATE_MAX)
        log_mu = np.clip(self.intercept + cov_mu + a_away - d_home,
                         _LOG_RATE_MIN, _LOG_RATE_MAX)
        return np.exp(log_lam), np.exp(log_mu)

    def match_rates(
        self, rows: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals for a match frame, with every fitted seam applied.

        The first half of :meth:`predict_proba`, split out because the season simulator needs the
        rates rather than the three-class collapse. Sharing the step is the point: a simulator
        that rebuilt the linear predictor itself would be a second implementation of the model,
        and the two would drift apart the first time a seam changed.
        """
        return self.rates(
            rows["home_team"], rows["away_team"],
            rows["date"] if "date" in rows.columns else None,
            context=self._cov_context(rows, history),
        )

    def predict_proba(
        self, rows: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> np.ndarray:
        """(N, 3) home/draw/away probabilities for a match frame.

        rho is clamped per match into the range that keeps tau positive at *that match's* rates.
        The range is rate-dependent, so a rho that is valid across the training sample can still be
        invalid for an extreme matchup the fit never saw; without the clamp that match would get a
        negative scoreline probability. Every clamp is counted into :attr:`diagnostics`, because a
        model quietly changing its own dependence parameter is exactly the kind of thing that
        should show up in a report.
        """
        lam, mu = self.match_rates(rows, history)
        rho = self.rho
        if self.family is None or self.family.fits_rho:
            rho, n_clamped = clamp_rho_for_rates(lam, mu, self.rho, margin=_RHO_CLAMP_MARGIN)
            if n_clamped:
                self.diagnostics["rho_clamped"] = (
                    self.diagnostics.get("rho_clamped", 0) + n_clamped
                )
        if self.family is None:
            return three_class_from_rates(lam, mu, rho, self.max_goals)
        return collapse_three_class(
            joint_grid(self.family, lam, mu, shape=self.shape, rho=rho, kappa=self.kappa,
                       max_goals=self.max_goals)
        )

    def team_table(self) -> pd.DataFrame:
        """Fitted strengths, strongest attack first — the human-readable form."""
        return pd.DataFrame(
            {
                "team": self.teams,
                "attack": self.attack,
                "defence": self.defence,
                "cold_start": [t in set(self.cold_start_teams) for t in self.teams],
            }
        ).sort_values("attack", ascending=False, kind="stable").reset_index(drop=True)

    def as_dict(self) -> dict[str, object]:
        return {
            "intercept": self.intercept,
            "home_advantage": self.home_advantage,
            "rho": self.rho,
            "half_life_days": self.half_life_days,
            "ref_date": str(pd.Timestamp(self.ref_date).date()),
            "n_teams": len(self.teams),
            "n_obs": self.n_obs,
            "effective_n": self.effective_n,
            "neg_log_lik": self.neg_log_lik,
            "converged": self.converged,
            "n_iterations": self.n_iterations,
            "n_cold_start": len(self.cold_start_teams),
            "cold_start_teams": list(self.cold_start_teams),
            "home_advantage_terms": dict(zip(self.ha_names, self.ha_params)),
            "covariates": None if self.cov_spec is None else self.cov_spec.label(),
            "covariate_terms": dict(zip(self.cov_names, self.cov_params)),
            "covariate_undefined": dict(self.cov_undefined),
            "scoreline_family": None if self.family is None else self.family.label(),
            "weibull_shape": self.shape,
            "frank_kappa": self.kappa,
            "diagnostics": dict(self.diagnostics),
        }


def decay_weights(dates: pd.Series, ref_date: pd.Timestamp, half_life_days: float) -> np.ndarray:
    """``exp(-ln2 * age_days / half_life)`` — 1.0 at the barrier, 0.5 one half-life earlier."""
    if half_life_days <= 0:
        raise ValueError(f"half_life_days must be positive; got {half_life_days}")
    seconds = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.total_seconds().to_numpy()
    age = seconds / 86400.0  # MATH: seconds per day
    age = np.clip(age, 0.0, None)
    return np.exp(-math.log(2.0) * age / half_life_days)


def _unpack(
    theta: np.ndarray, n_teams: int, n_ha: int = 0, n_family: int = 0, n_cov: int = 0
) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimiser vector -> (c, h, rho, attack, defence, ha, cov) with sum-to-zero applied.

    Seam parameters are appended AFTER the team blocks, in the order the seams were built, so
    adding one cannot shift the layout of anything that existed before it — which is what lets
    each seam be byte-identical when it carries no columns.
    """
    c = theta[_INTERCEPT]
    h = theta[_HOME_ADV]
    rho = theta[_RHO]
    free = n_teams - 1
    a_free = theta[_N_GLOBAL: _N_GLOBAL + free]
    d_free = theta[_N_GLOBAL + free: _N_GLOBAL + 2 * free]
    attack = np.concatenate([a_free, [-a_free.sum()]])
    defence = np.concatenate([d_free, [-d_free.sum()]])
    ha_start = _N_GLOBAL + 2 * free
    ha = theta[ha_start: ha_start + n_ha] if n_ha else np.zeros(0)
    cov_start = ha_start + n_ha
    cov = theta[cov_start: cov_start + n_cov] if n_cov else np.zeros(0)
    return c, h, rho, attack, defence, ha, cov


def _tau_terms(
    x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """tau and its log-derivatives wrt log lam, log mu and rho, per match."""
    tau = np.ones_like(lam)
    d_log_lam = np.zeros_like(lam)
    d_log_mu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho

    if np.any(tau <= 0):
        return tau, d_log_lam, d_log_mu, d_rho  # caller checks tau and bails

    # d log tau / d log lam  (and the mirror for mu), then d log tau / d rho.
    prod00 = lam[m00] * mu[m00] * rho
    d_log_lam[m00] = -prod00 / tau[m00]
    d_log_mu[m00] = -prod00 / tau[m00]
    d_rho[m00] = -(lam[m00] * mu[m00]) / tau[m00]

    d_log_lam[m01] = (lam[m01] * rho) / tau[m01]
    d_rho[m01] = lam[m01] / tau[m01]

    d_log_mu[m10] = (mu[m10] * rho) / tau[m10]
    d_rho[m10] = mu[m10] / tau[m10]

    d_rho[m11] = -1.0 / tau[m11]
    return tau, d_log_lam, d_log_mu, d_rho


def _objective(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    home_i: np.ndarray,
    away_i: np.ndarray,
    weights: np.ndarray,
    lgamma_x: np.ndarray,
    lgamma_y: np.ndarray,
    n_teams: int,
    ha_design: np.ndarray,
    cov_lam_design: np.ndarray,
    cov_mu_design: np.ndarray,
    penalty: tuple[np.ndarray, float, float, float] | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted negative log-likelihood and its analytic gradient.

    ``penalty`` is a ridge term ``(mask, centre_attack, centre_defence, strength)``, or None.
    Two seams build it -- the promotion prior toward the promoted-club level, general shrinkage
    toward the league average -- and the likelihood does not need to know which, which is why the
    centre arrives as two floats rather than as either seam's own type.

    Last in the signature and defaulting to None so that every existing caller -- the production
    solve, the decay seam's block cycles, and the gradient tests -- is untouched, and so that with
    both seams off the only difference is one identity check against None.
    """
    n_ha = ha_design.shape[1]
    n_cov = cov_lam_design.shape[1]
    c, h, rho, attack, defence, ha, cov = _unpack(theta, n_teams, n_ha, n_cov=n_cov)

    # Structural home advantage: exactly zero when the seam carries no columns.
    ha_term = ha_design @ ha if n_ha else 0.0
    # Match context enters BOTH rates, through its own design for each.
    cov_lam_term = cov_lam_design @ cov if n_cov else 0.0
    cov_mu_term = cov_mu_design @ cov if n_cov else 0.0
    log_lam = np.clip(c + h + ha_term + cov_lam_term + attack[home_i] - defence[away_i],
                      _LOG_RATE_MIN, _LOG_RATE_MAX)
    log_mu = np.clip(c + cov_mu_term + attack[away_i] - defence[home_i],
                     _LOG_RATE_MIN, _LOG_RATE_MAX)
    lam, mu = np.exp(log_lam), np.exp(log_mu)

    tau, dt_dloglam, dt_dlogmu, dt_drho = _tau_terms(x, y, lam, mu, rho)
    if np.any(tau <= 0):
        return _INVALID_NLL, np.zeros_like(theta)

    ll_per = (
        x * log_lam - lam - lgamma_x
        + y * log_mu - mu - lgamma_y
        + np.log(tau)
    )
    total = float(np.sum(weights * ll_per))
    if not np.isfinite(total):
        return _INVALID_NLL, np.zeros_like(theta)

    # d(log-likelihood)/d(log lam) and /d(log mu), per match.
    g_lam = weights * (x - lam + dt_dloglam)
    g_mu = weights * (y - mu + dt_dlogmu)

    grad = np.zeros_like(theta)
    grad[_INTERCEPT] = g_lam.sum() + g_mu.sum()
    grad[_HOME_ADV] = g_lam.sum()
    grad[_RHO] = float(np.sum(weights * dt_drho))

    # Attack of team t appears in lam when t is home and in mu when t is away; defence of t
    # appears (negated) in lam when t is away and in mu when t is home.
    d_attack = np.bincount(home_i, weights=g_lam, minlength=n_teams) + \
        np.bincount(away_i, weights=g_mu, minlength=n_teams)
    d_defence = -(np.bincount(away_i, weights=g_lam, minlength=n_teams) +
                  np.bincount(home_i, weights=g_mu, minlength=n_teams))

    # Sum-to-zero: the last team's parameter is minus the sum of the free ones, so every free
    # parameter also carries the last team's derivative with a minus sign.
    free = n_teams - 1
    grad[_N_GLOBAL: _N_GLOBAL + free] = d_attack[:free] - d_attack[-1]
    grad[_N_GLOBAL + free: _N_GLOBAL + 2 * free] = d_defence[:free] - d_defence[-1]
    if n_ha:
        # Each design column enters the home rate additively, exactly as h does.
        grad[_N_GLOBAL + 2 * free: _N_GLOBAL + 2 * free + n_ha] = ha_design.T @ g_lam
    if n_cov:
        # One parameter, two designs: the derivative is the sum of both rates' contributions.
        start = _N_GLOBAL + 2 * free + n_ha
        grad[start: start + n_cov] = cov_lam_design.T @ g_lam + cov_mu_design.T @ g_mu

    value, out = -total, -grad
    if penalty is not None:
        mask, centre_attack, centre_defence, strength = penalty
        pen, d_pen_attack, d_pen_defence = ridge_penalty(
            attack, defence, mask, centre_attack, centre_defence, strength
        )
        if pen:
            # The penalty ADDS to the negative log-likelihood, so its gradient adds too -- no sign
            # flip here, unlike the likelihood block above which is negated on the way out.
            value += pen
            # Same sum-to-zero transform the likelihood gradient uses twenty lines up, and for the
            # same reason: the last team's parameter is minus the sum of the free ones, so a
            # penalty on the last team's strength is felt by every free parameter. Penalising a
            # promoted club that happens to land in the last slot is not a special case, it is the
            # case most likely to be got wrong, which is why the gradient test targets it.
            out[_N_GLOBAL: _N_GLOBAL + free] += d_pen_attack[:free] - d_pen_attack[-1]
            out[_N_GLOBAL + free: _N_GLOBAL + 2 * free] += d_pen_defence[:free] - d_pen_defence[-1]
    return value, out


def _family_globals(theta: np.ndarray, family: CountSpec, n_family: int) -> tuple[float, float]:
    """The family's own scalars, read off the tail of the optimiser vector."""
    tail = theta[len(theta) - n_family:] if n_family else np.zeros(0)
    cursor = 0
    shape = UNIT_SHAPE
    kappa = INDEPENDENT_KAPPA
    if family.fits_shape:
        shape = math.exp(tail[cursor])
        cursor += 1
    if family.fits_kappa:
        kappa = float(tail[cursor])
    return shape, kappa


def _family_rates(
    theta: np.ndarray,
    home_i: np.ndarray,
    away_i: np.ndarray,
    n_teams: int,
    ha_design: np.ndarray,
    n_family: int,
    cov_lam_design: np.ndarray | None = None,
    cov_mu_design: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Expected goals for the fit sample, plus rho, from an optimiser vector."""
    n_ha = ha_design.shape[1]
    n_cov = 0 if cov_lam_design is None else cov_lam_design.shape[1]
    c, h, rho, attack, defence, ha, cov = _unpack(theta, n_teams, n_ha, n_family, n_cov)
    ha_term = ha_design @ ha if n_ha else 0.0
    cov_lam_term = cov_lam_design @ cov if n_cov else 0.0
    cov_mu_term = cov_mu_design @ cov if n_cov else 0.0
    log_lam = np.clip(c + h + ha_term + cov_lam_term + attack[home_i] - defence[away_i],
                      _LOG_RATE_MIN, _LOG_RATE_MAX)
    log_mu = np.clip(c + cov_mu_term + attack[away_i] - defence[home_i],
                     _LOG_RATE_MIN, _LOG_RATE_MAX)
    return np.exp(log_lam), np.exp(log_mu), rho


def _family_nll(
    marg,
    spec: CountSpec,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    rho: float,
    kappa: float,
) -> float:
    """Weighted negative log-likelihood given marginals that are already computed."""
    if not marg.is_trustworthy:
        return _INVALID_NLL
    prob, _, _ = observed_cell_likelihood(
        spec, marg, x, y, rho=rho, kappa=kappa, want_gradient=False
    )
    if np.any(prob <= 0.0):
        return _INVALID_NLL
    total = float(np.sum(weights * log_probabilities(prob)))
    return _INVALID_NLL if not np.isfinite(total) else -total


def _objective_family(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    home_i: np.ndarray,
    away_i: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    ha_design: np.ndarray,
    family: CountSpec,
    n_family: int,
    max_goals: int,
) -> float:
    """Weighted negative log-likelihood under an alternative scoreline family. Value only.

    Kept entirely separate from :func:`_objective` rather than generalising it. The production path
    is the thing every other arm is measured against, so the cost of a shared abstraction -- a
    branch inside the hot loop that could change a rounding anywhere -- is not worth paying to save
    a page of code. When the seam is off, none of this executes.
    """
    lam, mu, rho = _family_rates(theta, home_i, away_i, n_teams, ha_design, n_family)
    shape, kappa = _family_globals(theta, family, n_family)
    try:
        marg = marginals(
            family, lam, mu, shape=shape, max_goals=max_goals, want_gradient=False
        )
    except CountFamilyError:
        return _INVALID_NLL
    return _family_nll(marg, family, x, y, weights, rho, kappa)


def _value_and_gradient_family(theta: np.ndarray, *args) -> tuple[float, np.ndarray]:
    """Both at once, from a single pass over the marginals.

    ``minimize`` calls the objective and the Jacobian at the same point but through separate
    callables, which would build the count distributions twice per iteration for no reason. This is
    what is actually handed to the optimiser; the value-only form survives for the finite-difference
    probes, which genuinely do want the cheaper build.
    """
    _gradient_family.last_value = None
    grad = _gradient_family(theta, *args)
    value = _gradient_family.last_value
    if value is None or not np.isfinite(value):
        # The gradient pass bailed on an invalid point, so there is no likelihood to report from
        # it; fall back to the value-only path, which returns the sentinel that steers L-BFGS-B
        # back out of wherever it went.
        return _objective_family(theta, *args), grad
    return value, grad


def _gradient_family(theta: np.ndarray, *args) -> np.ndarray:
    """Analytic for every parameter that enters through a rate; finite-difference for the rest.

    The team block is ~100 long, so it has to be closed-form or a walk costs hours -- and it is,
    because every one of those parameters reaches the likelihood only through ``log lam`` and
    ``log mu``, and the count family hands back the derivative with respect to exactly those.

    The remaining three scalars are differenced. rho and kappa are cheap to difference because
    **neither touches the marginals**: the same pmf serves every perturbation, so a step costs one
    rectangle difference. The Weibull shape is not, since it changes the series coefficients
    themselves, so it is the only parameter here that pays for a full rebuild -- twice, which is
    still two evaluations against the hundred a fully numerical gradient would need.
    """
    (x, y, home_i, away_i, weights, n_teams, ha_design, family, n_family, max_goals) = args
    n_ha = ha_design.shape[1]
    lam, mu, rho = _family_rates(theta, home_i, away_i, n_teams, ha_design, n_family)
    shape, kappa = _family_globals(theta, family, n_family)
    try:
        marg = marginals(family, lam, mu, shape=shape, max_goals=max_goals)
    except CountFamilyError:
        return np.zeros_like(theta)
    if not marg.is_trustworthy:
        return np.zeros_like(theta)
    prob, d_lam, d_mu = observed_cell_likelihood(
        family, marg, x, y, rho=rho, kappa=kappa
    )
    if np.any(prob <= 0.0):
        return np.zeros_like(theta)
    _gradient_family.last_value = -float(np.sum(weights * log_probabilities(prob)))

    g_lam = weights * d_lam / prob
    g_mu = weights * d_mu / prob

    grad = np.zeros_like(theta)
    grad[_INTERCEPT] = g_lam.sum() + g_mu.sum()
    grad[_HOME_ADV] = g_lam.sum()

    d_attack = np.bincount(home_i, weights=g_lam, minlength=n_teams) +         np.bincount(away_i, weights=g_mu, minlength=n_teams)
    d_defence = -(np.bincount(away_i, weights=g_lam, minlength=n_teams) +
                  np.bincount(home_i, weights=g_mu, minlength=n_teams))
    free = n_teams - 1
    grad[_N_GLOBAL: _N_GLOBAL + free] = d_attack[:free] - d_attack[-1]
    grad[_N_GLOBAL + free: _N_GLOBAL + 2 * free] = d_defence[:free] - d_defence[-1]
    if n_ha:
        grad[_N_GLOBAL + 2 * free: _N_GLOBAL + 2 * free + n_ha] = ha_design.T @ g_lam

    # rho and kappa: differenced against the marginals already in hand.
    def _dependence_step(rho_v: float, kappa_v: float) -> float:
        return _family_nll(marg, family, x, y, weights, rho_v, kappa_v)

    if family.fits_rho:
        step = _FD_STEP * max(1.0, abs(rho))
        up, down = _dependence_step(rho + step, kappa), _dependence_step(rho - step, kappa)
        if up < _INVALID_NLL and down < _INVALID_NLL:
            grad[_RHO] = -(up - down) / (2.0 * step)
    if family.fits_kappa:
        step = _FD_STEP * max(1.0, abs(kappa))
        up, down = _dependence_step(rho, kappa + step), _dependence_step(rho, kappa - step)
        if up < _INVALID_NLL and down < _INVALID_NLL:
            grad[len(theta) - 1] = -(up - down) / (2.0 * step)

    # The shape, which does change the marginals and so pays for two rebuilds.
    if family.fits_shape:
        k = len(theta) - n_family
        step = _FD_STEP * max(1.0, abs(theta[k]))
        values = []
        for offset in (step, -step):
            probe = theta.copy()
            probe[k] += offset
            try:
                probe_marg = marginals(
                    family, lam, mu, shape=math.exp(probe[k]), max_goals=max_goals,
                    want_gradient=False,
                )
            except CountFamilyError:
                values.append(_INVALID_NLL)
                continue
            values.append(_family_nll(probe_marg, family, x, y, weights, rho, kappa))
        if max(values) < _INVALID_NLL:
            grad[k] = -(values[0] - values[1]) / (2.0 * step)

    return -grad


def _starting_point(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, n_teams: int, n_ha: int = 0
) -> np.ndarray:
    """Data-driven start: the weighted mean goal rates, flat strengths, no low-score correction.

    Derived rather than configured, so there is no arbitrary starting constant to justify or to
    drift out of date as scoring rates change.
    """
    total_w = weights.sum()
    mean_home = float((weights * x).sum() / total_w)
    mean_away = float((weights * y).sum() / total_w)
    theta = np.zeros(_N_GLOBAL + 2 * (n_teams - 1) + n_ha)
    theta[_INTERCEPT] = math.log(max(mean_away, np.finfo(float).tiny))
    theta[_HOME_ADV] = math.log(max(mean_home, np.finfo(float).tiny)) - theta[_INTERCEPT]
    theta[_RHO] = 0.0
    return theta


def fit_dixon_coles(
    history: pd.DataFrame,
    *,
    half_life_days: float,
    ref_date: pd.Timestamp,
    max_goals: int,
    param_bounds: dict[str, tuple[float, float]],
    min_effective_share: float,
    warm_start: DixonColesFit | None = None,
    max_iter: int,
    ha_mode: str = HA_GLOBAL,
    ha_window: tuple[str, str] | None = None,
    family: CountSpec | None = None,
    covariates: CovariateSpec | None = None,
    cov_division: str = "",
    decay: DecaySpec | None = None,
    promotion: PromotionSpec | None = None,
    shrinkage: ShrinkageSpec | None = None,
) -> DixonColesFit:
    """Fit the model on a training frame, weighted toward ``ref_date``.

    ``history`` must already be restricted to matches strictly before the barrier — the splitter
    owns that, and this function deliberately does not re-filter, so a caller cannot accidentally
    pass unfiltered data and have it silently corrected.
    """
    if len(history) == 0:
        raise ValueError("cannot fit on an empty history")
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    if covariates is not None and not covariates.is_inert:
        # The context terms count back through the calendar, so they need the columns that say
        # which calendar a row belongs to. Asked for explicitly rather than filled in.
        required |= {"season", "division", "played"}
    if promotion is not None:
        # The prior is estimated per season, so the seam needs to know which season a row is in.
        # Asked for explicitly rather than filled in, exactly as the covariates are.
        required |= {"season"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")

    # Under per-parameter decay the TEAM half-life is what governs this vector. It is used for
    # cold-start detection, for `effective_n`, and to seed the optimiser -- all three of which are
    # questions about team strength, so the team memory is the semantically right one. The block
    # solves below immediately re-fit the level and home advantage under their own weights, so the
    # seed does not bias where they land.
    strength_half_life = float(half_life_days if decay is None else decay.team)
    weights = decay_weights(history["date"], ref_date, strength_half_life)
    teams_all = sorted(set(history["home_team"]) | set(history["away_team"]))

    # Effective, not raw, sample size per team: under a short half-life a club returning after
    # years away has a healthy match count and almost no usable information.
    idx_all = {t: i for i, t in enumerate(teams_all)}
    home_all = history["home_team"].map(idx_all).to_numpy()
    away_all = history["away_team"].map(idx_all).to_numpy()
    effective = (
        np.bincount(home_all, weights=weights, minlength=len(teams_all))
        + np.bincount(away_all, weights=weights, minlength=len(teams_all))
    )
    # Relative to the median team, so the criterion means the same thing at every decay rate.
    threshold = float(np.median(effective)) * min_effective_share
    cold = tuple(t for t, e in zip(teams_all, effective) if e < threshold)
    teams = tuple(t for t in teams_all if t not in set(cold))
    if len(teams) < 2:
        raise ValueError(
            f"only {len(teams)} team(s) clear {min_effective_share:.0%} of the median team's "
            f"effective history ({threshold:.2f}); the fit would be unidentified"
        )

    # --- promotion prior (model.seams.promotion) ---------------------------------------------
    # Estimated from the SAME decay weights the strength parameters use, so the prior carries the
    # same memory as the thing it is applied to and tracks the drift in promoted clubs' standard
    # without a second half-life to tune.
    prior = None
    penalty = None
    if promotion is not None and shrinkage is not None:
        # Both seams build a penalty, and an arm carrying both would be a two-axis test wearing a
        # one-axis name -- the same refusal this function already makes for decay with covariates.
        raise ValueError(
            "the promotion prior and general shrinkage cannot run together: both are ridge terms "
            "on the same parameters, so an arm carrying both moves two axes"
        )
    if promotion is not None:
        if family is not None:
            raise ValueError(
                "the promotion seam and an alternative scoreline family are not implemented "
                "together; _objective_family carries no penalty term"
            )
        prior = estimate_prior(history, weights, min_prior_clubs=promotion.min_prior_clubs)
    if prior is not None:
        # Site A, the pin, is carried by `prior` itself and applied in `DixonColesFit._pinned`.
        # Site B, the shrink. Only clubs that ARE parameters can be penalised.
        promoted_mask = np.array([t in set(prior.teams) for t in teams], dtype=bool)
        penalty = (promoted_mask, prior.attack, prior.defence, promotion.shrinkage)

    if shrinkage is not None and not shrinkage.is_inert:
        # General hierarchical shrinkage: every identified club, toward the league average. Cold-start
        # clubs are already at exactly 0 and are not parameters, so this moves the fitted clubs
        # toward where the dropped ones have always sat.
        if family is not None:
            raise ValueError(
                "general shrinkage and an alternative scoreline family are not implemented "
                "together; _objective_family carries no penalty term"
            )
        penalty = (np.ones(len(teams), dtype=bool), 0.0, 0.0, shrinkage.strength)

    # Cold-start teams keep the league-average strength of 0 and are dropped from the fit sample,
    # so they neither absorb nor distort the parameters of the teams that are identified.
    index = {t: i for i, t in enumerate(teams)}
    keep = history["home_team"].isin(index) & history["away_team"].isin(index)
    fit_rows = history[keep]
    weights = weights[keep.to_numpy()]
    if len(fit_rows) == 0:
        raise ValueError("no training matches remain after removing cold-start teams")

    x = fit_rows["home_goals"].to_numpy(dtype=float)
    y = fit_rows["away_goals"].to_numpy(dtype=float)
    home_i = fit_rows["home_team"].map(index).to_numpy()
    away_i = fit_rows["away_team"].map(index).to_numpy()
    lgamma_x, lgamma_y = gammaln(x + 1.0), gammaln(y + 1.0)
    n_teams = len(teams)

    ha_design, ha_names = ha_design_matrix(
        fit_rows["date"], ref_date, mode=ha_mode,
        empty_start=ha_window[0] if ha_window else None,
        empty_end=ha_window[1] if ha_window else None,
    )
    # Built on the WHOLE history and then restricted, not built on the fit sample. A team whose
    # previous match was against a cold-start club still played it: measuring rest inside the fit
    # sample alone would silently lengthen exactly the gaps that surround a promoted team.
    cov = cov_design_matrix(
        history, history.iloc[:0],
        covariates if covariates is not None else CovariateSpec(),
        division=cov_division,
    )
    if cov.n_params:
        kept = keep.to_numpy()
        cov = dataclasses.replace(cov, lam=cov.lam[kept], mu=cov.mu[kept])
    if decay is not None and (cov.n_params or family is not None):
        # The block layout covers the intercept, rho, home advantage and the team strengths. A
        # covariate or family parameter would belong to no block and be pinned for the whole
        # cycle, which is a silently wrong fit rather than a refused one.
        raise ValueError(
            "per-parameter decay cannot run with the covariate or scoreline-family seams: its "
            "block layout has nowhere to put their parameters"
        )
    if cov.n_params and family is not None:
        # Not a limitation worth working around silently: every arm in this project moves ONE
        # axis, so a run that asked for both would be a two-axis test wearing a one-axis name.
        raise ValueError("the covariate seam and the scoreline family seam cannot run together")

    family_names: list[str] = []
    if family is not None:
        if family.fits_shape:
            family_names.append("weibull_log_shape")
        if family.fits_kappa:
            family_names.append("frank_kappa")
    n_family = len(family_names)

    theta0 = _starting_point(x, y, weights, n_teams, len(ha_names) + cov.n_params + n_family)
    if warm_start is not None:
        theta0 = _warm_start_vector(
            theta0, warm_start, teams, n_family, len(ha_names), cov.names
        )

    lo_s, hi_s = param_bounds["strength"]
    # rho is pinned shut, not removed, when the family carries no tau term: keeping the layout
    # fixed means the family seam cannot shift the position of anything that existed before it,
    # which is the same discipline the home-advantage seam follows.
    rho_bounds = (
        tuple(param_bounds["rho"]) if family is None or family.fits_rho else (0.0, 0.0)
    )
    bounds = [
        tuple(param_bounds["intercept"]),
        tuple(param_bounds["home_advantage"]),
        rho_bounds,
    ] + [(lo_s, hi_s)] * (2 * (n_teams - 1)) + [
        tuple(param_bounds[name]) for name in ha_names
    ] + [
        tuple(param_bounds[covariates.bound_key(name)]) for name in cov.names
    ] + [tuple(param_bounds[name]) for name in family_names]
    theta0 = np.clip(theta0, [b[0] for b in bounds], [b[1] for b in bounds])

    decay_diagnostics: dict[str, float] = {}
    if decay is not None:
        # Block coordinate descent: each block is a proper weighted MLE under its own half-life,
        # using this same `_objective`. Nothing about the likelihood is reimplemented.
        cycle = fit_blockwise(
            theta0,
            bounds,
            decay,
            block_weights(fit_rows["date"], ref_date, decay),
            BlockLayout(n_teams=n_teams, n_ha=len(ha_names), total=len(theta0)),
            objective=_objective,
            args_without_weights=(x, y, home_i, away_i, lgamma_x, lgamma_y, n_teams,
                                  ha_design, cov.lam, cov.mu, penalty),
            weights_position=_WEIGHTS_ARGUMENT_POSITION,
            max_iter=max_iter,
        )
        result = SimpleNamespace(
            x=cycle["theta"], fun=cycle["value"],
            success=cycle["converged"], nit=cycle["n_iterations"],
        )
        decay_diagnostics = {
            "cycles": float(cycle["cycles"]),
            "final_movement": cycle["movement"],
            "hit_cycle_cap": float(cycle["hit_cycle_cap"]),
        }
    elif family is None:
        result = minimize(
            _objective,
            theta0,
            args=(x, y, home_i, away_i, weights, lgamma_x, lgamma_y, n_teams, ha_design,
                  cov.lam, cov.mu, penalty),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_iter},
        )
    else:
        args = (x, y, home_i, away_i, weights, n_teams, ha_design, family, n_family,
                max_goals)
        result = minimize(
            _value_and_gradient_family,
            theta0,
            args=args,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": max_iter},
        )
    c, h, rho, attack, defence, ha, cov_params = _unpack(
        result.x, n_teams, len(ha_names), n_family, cov.n_params
    )
    shape, kappa = (
        (UNIT_SHAPE, INDEPENDENT_KAPPA) if family is None
        else _family_globals(result.x, family, n_family)
    )
    # Diagnostics describe the SOLUTION, not the search. A running maximum over every point the
    # optimiser probed reports its worst excursion into invalid territory -- which the guards
    # already rejected -- and says nothing about the fit that came back.
    family_diagnostics: dict[str, float] = {}
    if family is not None:
        solution_lam, solution_mu, _ = _family_rates(
            result.x, home_i, away_i, n_teams, ha_design, n_family
        )
        solution = marginals(
            family, solution_lam, solution_mu, shape=shape, max_goals=max_goals,
            want_gradient=False,
        )
        family_diagnostics = {
            "tail_deficit": solution.tail_deficit,
            "series_condition": solution.condition,
        }
    return DixonColesFit(
        teams=teams,
        attack=attack,
        defence=defence,
        intercept=float(c),
        home_advantage=float(h),
        rho=float(rho),
        half_life_days=strength_half_life,
        ref_date=pd.Timestamp(ref_date),
        max_goals=max_goals,
        n_obs=int(len(fit_rows)),
        effective_n=float(weights.sum()),
        neg_log_lik=float(result.fun),
        converged=bool(result.success),
        n_iterations=int(result.nit),
        cold_start_teams=cold,
        promotion=promotion,
        promotion_prior=prior,
        ha_names=tuple(ha_names),
        ha_params=tuple(float(v) for v in ha),
        ha_mode=ha_mode,
        ha_window=(ha_window[0], ha_window[1]) if ha_window else (None, None),
        family=family,
        shape=shape,
        kappa=kappa,
        cov_spec=covariates,
        cov_names=cov.names,
        cov_params=tuple(float(v) for v in cov_params),
        cov_division=cov_division,
        cov_undefined=dict(cov.undefined),
        decay=decay,
        decay_diagnostics=decay_diagnostics,
        diagnostics=family_diagnostics,
    )


def _warm_start_vector(
    theta0: np.ndarray, previous: DixonColesFit, teams: tuple[str, ...], n_family: int = 0,
    n_ha: int = 0, cov_names: tuple[str, ...] = (),
) -> np.ndarray:
    """Seed the optimiser from a previous fit, matching teams by name.

    Consecutive barriers differ by a single matchday, so the previous solution is close. Teams the
    previous fit did not know start at the league average, which is exactly the cold-start prior.

    Context-covariate coefficients carry over **by name**, not by position, so an arm that changes
    which terms it holds cannot silently seed one term's coefficient into another's slot.
    """
    theta = theta0.copy()
    theta[_INTERCEPT] = previous.intercept
    theta[_HOME_ADV] = previous.home_advantage
    theta[_RHO] = previous.rho
    prev = {t: i for i, t in enumerate(previous.teams)}
    free = len(teams) - 1
    for k, team in enumerate(teams[:free]):
        if team in prev:
            theta[_N_GLOBAL + k] = previous.attack[prev[team]]
            theta[_N_GLOBAL + free + k] = previous.defence[prev[team]]
    if previous.cov_names:
        offset = _N_GLOBAL + 2 * free + n_ha
        for k, name in enumerate(cov_names):
            if name in previous.cov_names:
                theta[offset + k] = previous.cov_params[previous.cov_names.index(name)]
    if n_family and previous.family is not None:
        tail = []
        if previous.family.fits_shape:
            tail.append(math.log(previous.shape))
        if previous.family.fits_kappa:
            tail.append(previous.kappa)
        theta[len(theta) - n_family:] = tail[:n_family]
    return theta


def fit_summary(fit: DixonColesFit) -> dict[str, object]:
    """The fitted parameters plus the rho sign, which gates whether a copula arm is worth building.

    The build brief premises a Weibull-copula arm on *negative* low-score dependence, inherited
    from the WC2026 corpus. The research report notes a five-league study found dependence positive
    in four leagues and negative only in Ligue 1, so the Premier League's sign is an open question
    that this fit answers for free.
    """
    summary = fit.as_dict()
    summary["rho_sign"] = "negative" if fit.rho < 0 else ("positive" if fit.rho > 0 else "zero")
    return summary
