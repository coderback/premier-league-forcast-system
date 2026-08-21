"""Score-driven (GAS) dynamic team states — the ``dynamics`` seam.

The production model treats each team's attack and defence as *constant over the training window*
and handles time variation by weighting the likelihood: a match a half-life ago counts half as
much as today's. That is a blunt instrument. It forgets on a fixed calendar schedule regardless of
whether anything happened, and it cannot say "this team was different in November".

The alternative in the literature is a genuinely time-varying state. Koopman & Lit (2015) make the
strengths latent stochastic processes; the Generalised Autoregressive Score family (Creal, Koopman
& Lucas 2013) is the observation-driven counterpart, where the state is updated by the *score of
the conditional likelihood* — the model's own derivative with respect to the state, which is the
steepest-ascent direction given what just happened. For a Poisson log-rate the score is simply
``goals - expected goals``, so the recursion reads as "a team that beats its expectation gets
better, in proportion to how much it beat it".

The recursion
-------------
Each team carries a deviation from its fitted level, ``a_i`` on attack and ``d_i`` on defence,
both starting at zero. A match between H and A is forecast at

    log lam = [level: c + h + A_H - D_A] + a_H - d_A
    log mu  = [level: c     + A_A - D_H] + a_A - d_H

and afterwards the four states involved are updated

    a_H <- B*a_H + K*s_home        d_A <- B*d_A - K*s_home
    a_A <- B*a_A + K*s_away        d_H <- B*d_H - K*s_away

with ``s_home = (x - lam + dlog tau/dlog lam) / lam**e`` the scaled score. ``B`` is the
persistence, ``K`` the score loading, and ``e`` the scaling exponent (0 = unit scaling,
0.5 = inverse square-root information, the Creal-Koopman-Lucas default, 1 = inverse information,
which makes the update a *relative* surprise and so invariant to the league's scoring level).

Conceding is the mirror of scoring: the same surprise that raises the home attack lowers the away
defence. That is not an extra assumption, it is what the score says — ``a_H`` and ``d_A`` enter
``log lam`` with opposite signs, so their derivatives are equal and opposite.

Four decisions worth stating, because each could reasonably have gone the other way
-----------------------------------------------------------------------------------

**The clock is team-match time, not calendar time.** A state decays when its team plays, not when
the calendar advances. This is the Elo convention and the natural one for a rating: information
arrives in matches, so ``B = 0.95`` means "half forgotten in fourteen matches" regardless of
whether those matches were crowded into December or spread across a summer. The cost is that a
team idle through an international break carries an undecayed state into its next fixture. That is
a declared simplification, not an oversight.

**Estimation is two-stage.** The level parameters come from the ordinary weighted MLE, exactly as
the baseline fits them and unaware that any state exists; only then does the filter run. Koopman &
Lit estimate everything jointly. Two-stage is less efficient and it under-states the arm — the
level fit has already absorbed, into a constant, some of the variation the states are meant to
explain. It biases *against* accepting the arm, which is the safe direction for a bias to run, and
it is what keeps a full walk-forward affordable.

**The filter re-runs at every barrier and it is still leak-free.** At barrier ``T`` the filter
replays every match strictly before ``T`` using the level fit made at ``T``. A 1998 match is
therefore scored against a level that a forecaster standing in 1998 would not have had. Nothing
dated at or after ``T`` enters anything, so no information crosses the barrier the acceptance
instrument cares about; the states are filtered, the levels are smoothed, and the combination is
the standard practical compromise.

**Cold-start teams accumulate states.** A promoted team has no fitted level — it is pinned at the
league average by :func:`plmodel.model.dixon_coles.fit_dixon_coles` — but it does have results, and
the filter is happy to give it a deviation from that average within weeks. That is a side effect
worth watching in the promoted-team calibration slice rather than a designed feature.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from plmodel.model.counts import CountSpec
from plmodel.model.dixon_coles import DixonColesFit
from plmodel.model.scoreline import clamp_rho_for_rates, three_class_from_rates

# Relative back-off from the tau validity boundary, matching the production predictor's.
_RHO_CLAMP_MARGIN = 0.01

# Scaling exponents the recursion recognises, by the name each goes by in the GAS literature.
UNIT_SCALING = 0.0
INVERSE_SQRT_INFORMATION = 0.5
INVERSE_INFORMATION = 1.0


class DynamicsError(ValueError):
    """Raised when the dynamics seam is switched on without the parameters it needs."""


@dataclass(frozen=True)
class GasSpec:
    """The three recursion parameters plus the memory the level fit is given.

    ``half_life_days`` belongs here rather than being inherited from ``model`` because the two
    mechanisms are substitutes: dynamics exist to track time variation, and a model that tracks it
    with states may well want the likelihood to forget more slowly than one that has only decay.
    Handing this arm the production half-life would make the comparison a test of whose
    hyperparameter happened to suit whom. The same argument, and the same remedy, as the elo-dc
    arm's separate half-life.
    """

    score_loading: float
    persistence: float
    scaling_exponent: float
    half_life_days: float
    state_bound: float

    @property
    def is_inert(self) -> bool:
        """A zero loading leaves every state at zero forever, reproducing the baseline exactly."""
        return self.score_loading == 0.0

    @classmethod
    def from_seam(cls, seam: dict, *, state_bound: float, fallback_half_life: float) -> "GasSpec":
        required = ("score_loading", "persistence", "scaling_exponent")
        missing = [k for k in required if seam.get(k) is None]
        if missing:
            raise DynamicsError(
                f"model.seams.dynamics is missing {missing}; these are tuned on "
                "backtest.tuning_span and must be written into config.yaml before the seam runs"
            )
        half_life = seam.get("half_life_days")
        return cls(
            score_loading=float(seam["score_loading"]),
            persistence=float(seam["persistence"]),
            scaling_exponent=float(seam["scaling_exponent"]),
            half_life_days=float(fallback_half_life if half_life is None else half_life),
            state_bound=float(state_bound),
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "score_loading": self.score_loading,
            "persistence": self.persistence,
            "scaling_exponent": self.scaling_exponent,
            "half_life_days": self.half_life_days,
        }


@dataclass(frozen=True)
class GasStates:
    """Filtered deviations as of a barrier, plus what the filter had to do to get there."""

    teams: tuple[str, ...]
    attack: np.ndarray
    defence: np.ndarray
    n_updates: int
    n_state_clipped: int
    n_tau_invalid: int

    def deviation(self, teams: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """``(attack, defence)`` deviations for a column of team names; 0 for a team never seen."""
        index = {t: i for i, t in enumerate(self.teams)}
        idx = np.array([index.get(t, -1) for t in teams])
        attack = np.where(idx >= 0, self.attack[idx], 0.0)
        defence = np.where(idx >= 0, self.defence[idx], 0.0)
        return attack, defence

    @property
    def dispersion(self) -> float:
        """RMS state magnitude in log-goal units — how much the filter is actually saying.

        This is the arm's analogue of a pool weight: a dynamics arm whose states sit near zero is
        the baseline wearing a hat, and its null would say nothing about dynamics.
        """
        if not self.teams:
            return 0.0
        return float(np.sqrt(np.mean(np.concatenate([self.attack, self.defence]) ** 2)))

    def as_dict(self) -> dict[str, object]:
        return {
            "n_teams": len(self.teams),
            "n_updates": self.n_updates,
            "dispersion": self.dispersion,
            "max_abs_attack": float(np.abs(self.attack).max()) if len(self.teams) else 0.0,
            "max_abs_defence": float(np.abs(self.defence).max()) if len(self.teams) else 0.0,
            "n_state_clipped": self.n_state_clipped,
            "n_tau_invalid": self.n_tau_invalid,
        }


def _tau_log_derivatives(
    x: float, y: float, lam: float, mu: float, rho: float
) -> tuple[float, float, bool]:
    """``(dlog tau/dlog lam, dlog tau/dlog mu, valid)`` for one match.

    The low-score correction touches only the four cells below, so every other scoreline
    contributes nothing and the branch costs one comparison. Including it keeps the filter's score
    the true score of the model being fitted rather than of a nearby independent-Poisson one.
    """
    if x > 1 or y > 1:
        return 0.0, 0.0, True
    if x == 0 and y == 0:
        product = lam * mu * rho
        tau = 1.0 - product
        if tau <= 0:
            return 0.0, 0.0, False
        return -product / tau, -product / tau, True
    if x == 0 and y == 1:
        tau = 1.0 + lam * rho
        if tau <= 0:
            return 0.0, 0.0, False
        return (lam * rho) / tau, 0.0, True
    if x == 1 and y == 0:
        tau = 1.0 + mu * rho
        if tau <= 0:
            return 0.0, 0.0, False
        return 0.0, (mu * rho) / tau, True
    # (1, 1): tau = 1 - rho does not depend on either rate, so both derivatives vanish.
    return 0.0, 0.0, (1.0 - rho) > 0


def filter_states(history: pd.DataFrame, fit: DixonColesFit, spec: GasSpec) -> GasStates:
    """Run the score-driven recursion over ``history`` on top of a fitted level.

    ``history`` must already be restricted to matches strictly before the barrier — the splitter
    owns that, exactly as the fitter does, so this cannot silently repair a caller's mistake.
    """
    required = {"date", "home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing columns: {sorted(missing)}")

    if fit.family is not None:
        raise DynamicsError(
            "the filter's score is the score of the Poisson-and-tau likelihood; a level fitted "
            "under another scoreline family would be updated by the wrong derivative"
        )
    teams = tuple(sorted(set(history["home_team"]) | set(history["away_team"])))
    if spec.is_inert:
        zeros = np.zeros(len(teams))
        return GasStates(teams, zeros, zeros.copy(), 0, 0, 0)

    # The level rates come from the fit itself, so the filter cannot drift from the predictor's
    # own formula: home advantage, structural terms and rate clipping are all already applied.
    #
    # Through `match_rates` with an empty history, which is how `fit_dixon_coles` builds its own
    # covariate design (see the cov_design_matrix call there). An earlier version called `rates`
    # directly and passed no context, so the filter's baseline omitted the covariates while
    # prediction included them — the states would have absorbed the covariate effect during
    # filtering and it would then have been applied a second time at prediction. Byte-identical
    # while the seam ships empty, and wrong the moment it does not.
    lam0, mu0 = fit.match_rates(history, history.iloc[:0])
    base_log_lam = np.log(lam0).tolist()
    base_log_mu = np.log(mu0).tolist()

    index = {t: i for i, t in enumerate(teams)}
    home_i = history["home_team"].map(index).tolist()
    away_i = history["away_team"].map(index).tolist()
    goals_home = history["home_goals"].to_numpy(dtype=float).tolist()
    goals_away = history["away_goals"].to_numpy(dtype=float).tolist()

    dev_attack = [0.0] * len(teams)
    dev_defence = [0.0] * len(teams)
    loading, persistence = spec.score_loading, spec.persistence
    exponent, rho, bound = spec.scaling_exponent, fit.rho, spec.state_bound
    scale_by_information = exponent != UNIT_SCALING
    exp = math.exp
    n_clipped = n_tau_invalid = 0

    # Written flat rather than through a helper per update: this runs once per historical match at
    # every one of ~1,150 barriers, so a tuple allocation and four function calls per iteration is
    # the difference between a walk that takes minutes and one that takes an hour.
    for k in range(len(home_i)):
        h, a = home_i[k], away_i[k]
        attack_home, attack_away = dev_attack[h], dev_attack[a]
        defence_home, defence_away = dev_defence[h], dev_defence[a]
        lam = exp(base_log_lam[k] + attack_home - defence_away)
        mu = exp(base_log_mu[k] + attack_away - defence_home)

        x, y = goals_home[k], goals_away[k]
        score_home = x - lam
        score_away = y - mu
        if x < 2.0 and y < 2.0:  # MATH: only the four lowest cells carry a tau derivative
            d_lam, d_mu, valid = _tau_log_derivatives(x, y, lam, mu, rho)
            if not valid:
                n_tau_invalid += 1
            score_home += d_lam
            score_away += d_mu
        if scale_by_information:
            score_home /= lam ** exponent
            score_away /= mu ** exponent

        step_home = loading * score_home
        step_away = loading * score_away
        new_attack_home = persistence * attack_home + step_home
        new_defence_away = persistence * defence_away - step_home
        new_attack_away = persistence * attack_away + step_away
        new_defence_home = persistence * defence_home - step_away

        if (-bound <= new_attack_home <= bound and -bound <= new_defence_away <= bound
                and -bound <= new_attack_away <= bound and -bound <= new_defence_home <= bound):
            dev_attack[h] = new_attack_home
            dev_defence[a] = new_defence_away
            dev_attack[a] = new_attack_away
            dev_defence[h] = new_defence_home
        else:
            # The recursion is trying to run away. Clip, count, and let the report show it: a
            # non-zero count means the loading is too large for the data, not that a team is good.
            for store, team, value in (
                (dev_attack, h, new_attack_home), (dev_defence, a, new_defence_away),
                (dev_attack, a, new_attack_away), (dev_defence, h, new_defence_home),
            ):
                if value > bound:
                    value, n_clipped = bound, n_clipped + 1
                elif value < -bound:
                    value, n_clipped = -bound, n_clipped + 1
                store[team] = value

    return GasStates(
        teams=teams,
        attack=np.asarray(dev_attack),
        defence=np.asarray(dev_defence),
        n_updates=len(home_i),
        n_state_clipped=n_clipped,
        n_tau_invalid=n_tau_invalid,
    )


@dataclass(frozen=True)
class DynamicFit:
    """A level fit plus the states filtered on top of it — the thing that makes a forecast."""

    fit: DixonColesFit
    states: GasStates
    spec: GasSpec

    # Delegated so this fit can stand in for a DixonColesFit wherever the harness reports on one.
    #
    # Written out one at a time rather than through a __getattr__ that forwards everything, and
    # that is deliberate. The two members most likely to be reached for are `attack` and
    # `defence`, and those are exactly the two where forwarding to the level would be WRONG: a
    # dynamic fit's strength is the level plus its state. A blanket forward would answer them
    # quietly and wrongly. Everything below is a member where the level's answer IS the answer;
    # everything the states change is overridden further down, and anything absent from both
    # raises, which is what it should do.
    @property
    def rho(self) -> float:
        return self.fit.rho

    @property
    def converged(self) -> bool:
        return self.fit.converged

    @property
    def n_iterations(self) -> int:
        return self.fit.n_iterations

    @property
    def half_life_days(self) -> float:
        return self.fit.half_life_days

    @property
    def home_advantage(self) -> float:
        return self.fit.home_advantage

    @property
    def intercept(self) -> float:
        return self.fit.intercept

    @property
    def teams(self) -> tuple[str, ...]:
        return self.fit.teams

    @property
    def cold_start_teams(self) -> tuple[str, ...]:
        return self.fit.cold_start_teams

    @property
    def diagnostics(self) -> dict[str, int]:
        return self.fit.diagnostics

    @property
    def max_goals(self) -> int:
        return self.fit.max_goals

    @property
    def ref_date(self) -> pd.Timestamp:
        return self.fit.ref_date

    @property
    def n_obs(self) -> int:
        return self.fit.n_obs

    @property
    def effective_n(self) -> float:
        return self.fit.effective_n

    @property
    def neg_log_lik(self) -> float:
        return self.fit.neg_log_lik

    @property
    def family(self) -> CountSpec | None:
        """The level's scoreline family — None IS the production Poisson-and-tau path.

        Delegated rather than omitted because callers guard on it. The season simulator asks
        ``fit.family is not None`` before it will draw, and an absent attribute turned that
        refusal into an AttributeError raised before the guard could even be read.
        """
        return self.fit.family

    @property
    def shape(self) -> float:
        return self.fit.shape

    @property
    def kappa(self) -> float:
        return self.fit.kappa

    @property
    def cov_spec(self):
        return self.fit.cov_spec

    @property
    def cov_names(self) -> tuple[str, ...]:
        return self.fit.cov_names

    @property
    def cov_params(self) -> tuple[float, ...]:
        return self.fit.cov_params

    @property
    def ha_names(self) -> tuple[str, ...]:
        return self.fit.ha_names

    @property
    def ha_params(self) -> tuple[float, ...]:
        return self.fit.ha_params

    # --- what the states change, and therefore what this class must answer itself ---------------

    @property
    def attack(self) -> np.ndarray:
        """The LEVEL's attack, aligned to :attr:`teams` — not the level plus the state.

        Tempting to return the effective strength here, and wrong. Every array named ``attack`` in
        this codebase means the fitted level: the season simulator's drift perturbs it, the hybrid
        consumes it as a feature, and `_warm_start_vector` seeds the next optimiser run from it.
        That last one decides it — warm-starting a level fit from level-plus-state would fold the
        filtered deviations into the level and then filter on top of them again, a compounding
        corruption that would read as slow drift rather than as a bug.

        The effective strength is not hidden; it is a named column in :meth:`team_table`.
        """
        return self.fit.attack

    @property
    def defence(self) -> np.ndarray:
        """The LEVEL's defence. See :attr:`attack`."""
        return self.fit.defence

    def _shift(self, lam, mu, home: pd.Series, away: pd.Series):
        """The level's rates times each side's filtered deviation.

        Shared by :meth:`rates` and :meth:`match_rates` so the two cannot come to express the
        recursion differently. The multiplication happens *after* the level's log-rate clip rather
        than inside it, which is what the arm was measured with.
        """
        attack_home, defence_home = self.states.deviation(home)
        attack_away, defence_away = self.states.deviation(away)
        return (
            lam * np.exp(attack_home - defence_away),
            mu * np.exp(attack_away - defence_home),
        )

    def rates(
        self, home: pd.Series, away: pd.Series, dates: pd.Series | None = None,
        *, context: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals, with the same signature the level takes.

        Identical to ``DixonColesFit.rates`` on purpose. An earlier version took a DataFrame
        instead, so two classes meant to be interchangeable had a same-named method with
        incompatible shapes — which is the trap that made this whole refactor necessary.
        """
        return self._shift(*self.fit.rates(home, away, dates, context=context), home, away)

    def match_rates(
        self, rows: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals for a match frame, with every fitted seam applied, then the states.

        Delegated to the level's own ``match_rates`` rather than reimplemented, so the covariate
        context is built by the code that owns it — including its refusal to forecast a fitted
        covariate without the history it counts back through.
        """
        return self._shift(
            *self.fit.match_rates(rows, history), rows["home_team"], rows["away_team"]
        )

    def team_table(self) -> pd.DataFrame:
        """Level, state and their sum, strongest EFFECTIVE attack first.

        Three pairs of columns rather than one, because the arm's whole claim is that the level
        and the state do different jobs — long-run quality, and drift away from it — and a table
        that folded them together would hide the thing worth looking at. ``attack`` keeps the
        meaning it has everywhere else in this codebase (the level), so a reader who knows the base
        class is not ambushed; ``attack_effective`` is what the model forecasts with.

        Rows cover the level's teams **and** the filter's. A club cold-started by the fit has no
        level — it is pinned at the league average, which is 0.0, not unknown — but the filter will
        give it a state within weeks, and a table that dropped it would be silent about a club the
        model is actively forecasting.
        """
        cold = set(self.fit.cold_start_teams)
        level = dict(zip(self.fit.teams, zip(self.fit.attack, self.fit.defence)))
        teams = list(self.fit.teams) + [t for t in self.states.teams if t not in level]
        state_attack, state_defence = self.states.deviation(pd.Series(teams, dtype=object))
        attack = np.array([level.get(t, (0.0, 0.0))[0] for t in teams])
        defence = np.array([level.get(t, (0.0, 0.0))[1] for t in teams])
        return pd.DataFrame(
            {
                "team": teams,
                "attack": attack,
                "defence": defence,
                "attack_state": state_attack,
                "defence_state": state_defence,
                "attack_effective": attack + state_attack,
                "defence_effective": defence + state_defence,
                "cold_start": [t in cold or t not in level for t in teams],
            }
        ).sort_values("attack_effective", ascending=False, kind="stable").reset_index(drop=True)

    def predict_proba(
        self, rows: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> np.ndarray:
        """(N, 3) home/draw/away probabilities, with rho clamped per match as the level does.

        The clamp count lands on the level's ``diagnostics`` dict, which the arm reuses across
        barriers between refits. That is deliberate: the report wants the total over the walk, not
        a per-barrier counter that resets every time a fresh DynamicFit is wrapped around the same
        level.
        """
        lam, mu = self.match_rates(rows, history)
        rho, n_clamped = clamp_rho_for_rates(lam, mu, self.fit.rho, margin=_RHO_CLAMP_MARGIN)
        if n_clamped:
            self.fit.diagnostics["rho_clamped"] = (
                self.fit.diagnostics.get("rho_clamped", 0) + n_clamped
            )
        return three_class_from_rates(lam, mu, rho, self.fit.max_goals)

    def as_dict(self) -> dict[str, object]:
        return {**self.fit.as_dict(), "gas": {**self.spec.as_dict(), **self.states.as_dict()}}
