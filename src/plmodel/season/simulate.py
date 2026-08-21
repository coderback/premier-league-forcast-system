"""Monte Carlo of a season's remaining fixtures.

The match model produces a scoreline distribution; a season is 380 draws from 380 of them, added
into a table. Everything hard about that is in three places.

**Sampling the Dixon-Coles scoreline exactly, and cheaply.** The obvious route -- build each
match's ``(G+1, G+1)`` grid and sample a cell -- costs one grid per match *per replicate* as soon
as strengths vary between replicates, which is 19 million grids at production settings. It is not
needed. The tau correction touches only the four cells with both scores below two, and it is
mass-preserving on that block (the four deltas cancel exactly), so the corrected law is the
independent-Poisson law with the block's *internal* proportions rewritten. Sampling is therefore:
draw two independent Poissons, and for the ~37% of draws that land in the block, redraw the cell
from the four corrected weights. Exact, and O(1) per match. See
:func:`sample_scorelines`; :mod:`tests.test_season_simulate` checks the empirical joint against
:func:`plmodel.model.scoreline.scoreline_matrix` cell by cell.

**Parameter uncertainty, drawn once per replicate.** The research report is specific that
point-estimate season simulations understate the tails and that strengths must be drawn per run,
never per match -- per-match resampling averages the perturbation away inside a single season and
collapses the points spread instead of widening it. What this project draws from is not an
asymptotic covariance: under exponential decay weighting the inverse observed information is not a
valid sampling covariance, and it would in any case answer the wrong question. The quantity a
season simulation needs is *how far a club's strength moves between the barrier and the matches
being simulated*, which mixes estimation noise with genuine mid-season change and does not need
them separated. That is measured directly from the corpus on the tuning span and enters here as
:class:`DriftSpec`. Whether it improves the forecast is decided by
:mod:`plmodel.season.validate`, not assumed.

**Undefined is not zero.** A club with no usable top-flight history is cold-started by the fit at
league average, and the simulator does not pretend otherwise: it records which clubs those were so
a report can say that three of the twenty forecasts rest on a club the model has never seen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from plmodel.model.dixon_coles import (
    _LOG_RATE_MAX,
    _LOG_RATE_MIN,
    _RHO_CLAMP_MARGIN,
    DixonColesFit,
)
from plmodel.model.dynamics import DynamicFit
from plmodel.model.scoreline import clamp_rho_for_rates
from plmodel.season.table import Question, SeasonError, points_for, rank, standings

# Cell order of the four scorelines the Dixon-Coles correction rewrites: (0,0), (0,1), (1,0),
# (1,1). A cell index c decodes as home = c // 2, away = c % 2, which is why the order is fixed
# here rather than taken from scoreline._TAU_CELLS -- the decode and the order must agree.
_BLOCK_MAX_GOALS = 1
_BLOCK_CELLS = 4

UNCERTAINTY_POINT = "point"
UNCERTAINTY_DRIFT = "drift"
UNCERTAINTIES = (UNCERTAINTY_POINT, UNCERTAINTY_DRIFT)

# Sentinel making every drift setting mandatory once the mode is selected, so a half-configured
# drift cannot quietly run at zero and be reported as if it had propagated anything.
_CONFIGURED = -1.0


@dataclass(frozen=True)
class DriftSpec:
    """How far team strengths are allowed to move between the barrier and the simulated matches.

    ``attack_sd`` and ``defence_sd`` are in log-goals over a **whole season**; the scale actually
    applied is ``sd * horizon ** horizon_exponent`` where ``horizon`` is the fraction of the
    season still to play. The exponent is measured rather than assumed to be the random walk's
    one-half -- see the ledger entry for the tuning-span fit.

    The perturbation is centred across clubs on both parameters. Fitted attack and defence are
    sum-to-zero by construction, so their measured movement is movement *within* that constraint;
    an uncentred perturbation would add a league-wide scoring shift that the measurement never
    contained and that the intercept, not the team parameters, would own.

    Centring the perturbation in *log* space is not the same as leaving the expected goal count
    alone: a rate multiplied by ``exp(d)`` with ``d`` mean-zero has expectation ``exp(sd**2 / 2)``
    times its old one, so drift raises league-wide scoring by a fraction of a percent at any
    plausible ``sd``. That is a property of a multiplicative perturbation rather than a defect,
    it is far below the 2.4% goal-rate under-prediction the production model already carries, and
    it is stated here rather than corrected away because correcting it would mean shrinking every
    club's rate by a constant that the measurement never asked for.
    """

    attack_sd: float = _CONFIGURED
    defence_sd: float = _CONFIGURED
    correlation: float = _CONFIGURED
    horizon_exponent: float = _CONFIGURED

    def __post_init__(self) -> None:
        missing = [f for f in ("attack_sd", "defence_sd", "correlation", "horizon_exponent")
                   if getattr(self, f) == _CONFIGURED]
        if missing:
            raise SeasonError(f"drift spec is missing settings: {missing}")
        if self.attack_sd < 0 or self.defence_sd < 0:
            raise SeasonError("drift standard deviations cannot be negative")
        if not -1.0 <= self.correlation <= 1.0:
            raise SeasonError(f"drift correlation must be in [-1, 1]; got {self.correlation}")

    @property
    def is_inert(self) -> bool:
        return self.attack_sd == 0.0 and self.defence_sd == 0.0

    def scale(self, horizon: float) -> float:
        """Multiplier on the season-long standard deviations for a partial horizon."""
        if not 0.0 <= horizon <= 1.0:
            raise SeasonError(f"horizon must be a fraction of a season; got {horizon}")
        return float(horizon ** self.horizon_exponent) if horizon > 0.0 else 0.0


@dataclass(frozen=True)
class SeasonSpec:
    """Everything the simulator needs that is a decision rather than a measurement."""

    n_replicates: int
    chunk_size: int
    questions: tuple[Question, ...]
    uncertainty: str = UNCERTAINTY_POINT
    drift: DriftSpec | None = None

    def __post_init__(self) -> None:
        if self.n_replicates < 1:
            raise SeasonError("n_replicates must be at least 1")
        if self.chunk_size < 1:
            raise SeasonError("chunk_size must be at least 1")
        if self.uncertainty not in UNCERTAINTIES:
            raise SeasonError(f"uncertainty must be one of {UNCERTAINTIES}; got {self.uncertainty!r}")
        if self.uncertainty == UNCERTAINTY_DRIFT and self.drift is None:
            raise SeasonError("uncertainty='drift' needs a drift spec")
        names = [q.name for q in self.questions]
        if len(set(names)) != len(names):
            raise SeasonError(f"question names must be unique; got {names}")

    def label(self) -> str:
        if self.uncertainty == UNCERTAINTY_POINT or self.drift is None or self.drift.is_inert:
            return UNCERTAINTY_POINT
        return (f"drift(att {self.drift.attack_sd:.3f}, def {self.drift.defence_sd:.3f}, "
                f"r {self.drift.correlation:+.2f}, h^{self.drift.horizon_exponent:.2f})")


@dataclass(frozen=True)
class SeasonForecast:
    """What a simulated season believes, and what it had to assume to believe it."""

    season: str
    barrier: pd.Timestamp
    teams: tuple[str, ...]
    table: pd.DataFrame
    n_played: int
    n_remaining: int
    n_replicates: int
    uncertainty: str
    probabilities: pd.DataFrame
    position_counts: np.ndarray
    points_counts: np.ndarray
    points_floor: int
    diagnostics: dict[str, object] = field(default_factory=dict)

    @property
    def horizon(self) -> float:
        total = self.n_played + self.n_remaining
        return self.n_remaining / total if total else 0.0

    def question_probability(self, question: str) -> pd.Series:
        return self.probabilities.set_index("team")[question]

    def points_quantile(self, q: float) -> np.ndarray:
        """Per-team points quantile, read off the exact replicate histogram."""
        cumulative = np.cumsum(self.points_counts, axis=1) / self.n_replicates
        reached = np.argmax(cumulative >= q, axis=1)
        return reached + self.points_floor

    def to_dict(self) -> dict[str, object]:
        return {
            "season": self.season,
            "barrier": str(pd.Timestamp(self.barrier).date()),
            "n_teams": len(self.teams),
            "n_played": self.n_played,
            "n_remaining": self.n_remaining,
            "horizon": self.horizon,
            "n_replicates": self.n_replicates,
            "uncertainty": self.uncertainty,
            "table": self.table.to_dict("records"),
            "probabilities": self.probabilities.to_dict("records"),
            "diagnostics": dict(self.diagnostics),
        }


def sample_scorelines(
    lam: np.ndarray, mu: np.ndarray, rho: np.ndarray, *,
    rng: np.random.Generator, size: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``(home_goals, away_goals)`` from the Dixon-Coles law at each rate pair.

    Exact for the untruncated model. The analytic grid in
    :func:`plmodel.model.scoreline.scoreline_matrix` renormalises over a truncated grid, so the two
    disagree by the tail beyond ``max_goals`` -- about 1e-9 of the mass at football rates, and in
    the sampler's favour rather than against it.

    ``rho`` must already be clamped into each pair's valid range; a rho outside it makes one of the
    four block weights negative, and this routine raises rather than sampling from something that
    is not a distribution.

    ``size`` joins the broadcast, so ``size=(reps, n_matches)`` with a one-dimensional rate vector
    draws independent replicates of the same fixture list. Leaving it out and broadcasting the
    *result* instead would give every replicate the same season, which is a mistake that looks
    like a working simulator until someone reads the points spread.
    """
    shape = np.broadcast_shapes(
        size or (), np.shape(lam), np.shape(mu), np.shape(rho))
    lam_b = np.broadcast_to(np.asarray(lam, dtype=float), shape)
    mu_b = np.broadcast_to(np.asarray(mu, dtype=float), shape)
    rho_b = np.broadcast_to(np.asarray(rho, dtype=float), shape)

    home = rng.poisson(lam_b)
    away = rng.poisson(mu_b)
    block = (home <= _BLOCK_MAX_GOALS) & (away <= _BLOCK_MAX_GOALS)
    if not block.any():
        return home, away

    l, m, r = lam_b[block], mu_b[block], rho_b[block]
    # tau * Poisson on the four cells, with the shared exp(-lam-mu) factor divided out. Their sum
    # is exactly (1+lam)(1+mu) -- the same block mass the independent law already gave these
    # draws, which is why redrawing *within* the block is all the correction needs.
    weights = np.empty((_BLOCK_CELLS,) + l.shape)
    weights[0] = 1.0 - l * m * r
    weights[1] = m * (1.0 + l * r)
    weights[2] = l * (1.0 + m * r)
    weights[3] = l * m * (1.0 - r)
    if np.any(weights < 0.0):
        raise SeasonError("negative scoreline weight: rho is outside its valid range for a rate pair")

    draw = rng.random(l.shape) * ((1.0 + l) * (1.0 + m))
    cumulative = np.cumsum(weights, axis=0)
    cell = np.zeros(l.shape, dtype=np.int64)
    for edge in cumulative[:-1]:
        cell += draw > edge
    home[block] = cell // 2  # MATH: cell index c decodes as (c // 2, c % 2) over the 2x2 block
    away[block] = cell % 2   # MATH: cell index c decodes as (c // 2, c % 2) over the 2x2 block
    return home, away


def _drift_deltas(
    spec: DriftSpec, *, n_replicates: int, n_teams: int, horizon: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """``(n_replicates, n_teams)`` attack and defence perturbations, centred across clubs."""
    scale = spec.scale(horizon)
    z_attack = rng.standard_normal((n_replicates, n_teams))
    z_other = rng.standard_normal((n_replicates, n_teams))
    z_defence = spec.correlation * z_attack + np.sqrt(1.0 - spec.correlation ** 2) * z_other
    attack = spec.attack_sd * scale * z_attack
    defence = spec.defence_sd * scale * z_defence
    return (attack - attack.mean(axis=1, keepdims=True),
            defence - defence.mean(axis=1, keepdims=True))


def _team_index(teams: tuple[str, ...], names: pd.Series) -> np.ndarray:
    index = {team: i for i, team in enumerate(teams)}
    missing = sorted(set(names) - set(index))
    if missing:
        raise SeasonError(f"fixture involves clubs outside the season's roster: {missing}")
    return names.map(index).to_numpy()


def simulate_season(
    fit: DixonColesFit | DynamicFit,
    played: pd.DataFrame,
    remaining: pd.DataFrame,
    *,
    spec: SeasonSpec,
    seed: int,
    history: pd.DataFrame | None = None,
    deductions: dict[str, int] | None = None,
    season: str = "",
    barrier: pd.Timestamp | None = None,
) -> SeasonForecast:
    """Simulate the remaining fixtures and summarise where each club finishes.

    ``played`` and ``remaining`` partition one season's fixture list at the barrier. Both are
    needed: the table so far is not derivable from the fixtures still to come, and the fixtures
    still to come are not derivable from the table.
    """
    if fit.family is not None:
        raise SeasonError(
            "the season simulator draws from the production Poisson-and-tau scoreline; a fit "
            "carrying a scoreline family needs its own sampler, and silently using this one "
            "would simulate a different model from the one that was fitted"
        )
    if len(remaining) == 0 and len(played) == 0:
        raise SeasonError("a season with no fixtures cannot be simulated")

    teams = tuple(sorted(
        set(played["home_team"]) | set(played["away_team"])
        | set(remaining["home_team"]) | set(remaining["away_team"])
    ))
    n_teams = len(teams)
    ranked = standings(played, teams=teams, deductions=deductions)
    ranked.insert(0, "position", np.arange(len(ranked)) + 1)
    base = ranked.set_index("team").reindex(list(teams)).reset_index()

    base_points = base["points"].to_numpy()
    base_gf = base["goals_for"].to_numpy()
    base_ga = base["goals_against"].to_numpy()

    n_remaining = len(remaining)
    if n_remaining:
        home_idx = _team_index(teams, remaining["home_team"])
        away_idx = _team_index(teams, remaining["away_team"])
        lam, mu = fit.match_rates(remaining, history)
        log_lam, log_mu = np.log(lam), np.log(mu)
    else:
        home_idx = away_idx = np.zeros(0, dtype=int)
        log_lam = log_mu = np.zeros(0)

    horizon = n_remaining / (len(played) + n_remaining)
    drift = spec.drift if spec.uncertainty == UNCERTAINTY_DRIFT else None

    points_floor = int(min(0, base_points.min())) if n_teams else 0
    points_span = int(base_points.max() + 1 + n_remaining * 3 - points_floor) if n_teams else 1
    points_counts = np.zeros((n_teams, points_span), dtype=np.int64)
    position_counts = np.zeros((n_teams, n_teams), dtype=np.int64)
    question_counts = {q.name: np.zeros(n_teams, dtype=np.int64) for q in spec.questions}
    boundary_ties = {q.name: 0 for q in spec.questions}
    any_tie = 0
    rho_clamped = 0

    rng = np.random.default_rng(seed)
    done = 0
    while done < spec.n_replicates:
        reps = min(spec.chunk_size, spec.n_replicates - done)
        points = np.broadcast_to(base_points, (reps, n_teams)).copy()
        goals_for = np.broadcast_to(base_gf, (reps, n_teams)).copy()
        goals_against = np.broadcast_to(base_ga, (reps, n_teams)).copy()

        if n_remaining:
            if drift is None or drift.is_inert:
                lam_rep = np.exp(np.clip(log_lam, _LOG_RATE_MIN, _LOG_RATE_MAX))
                mu_rep = np.exp(np.clip(log_mu, _LOG_RATE_MIN, _LOG_RATE_MAX))
                shared_rates = True
            else:
                shared_rates = False
                d_attack, d_defence = _drift_deltas(
                    drift, n_replicates=reps, n_teams=n_teams, horizon=horizon, rng=rng
                )
                lam_rep = np.exp(np.clip(
                    log_lam + d_attack[:, home_idx] - d_defence[:, away_idx],
                    _LOG_RATE_MIN, _LOG_RATE_MAX))
                mu_rep = np.exp(np.clip(
                    log_mu + d_attack[:, away_idx] - d_defence[:, home_idx],
                    _LOG_RATE_MIN, _LOG_RATE_MAX))
            rho, clamped = clamp_rho_for_rates(lam_rep, mu_rep, fit.rho, margin=_RHO_CLAMP_MARGIN)
            # Counted per fixture, not per fixture per replicate. Under shared rates every chunk
            # produces the same count, so it is assigned rather than accumulated; under drift the
            # rates differ by replicate, so the per-chunk total is divided back down. Reporting
            # the raw figure would multiply one clamped fixture by fifty thousand.
            rho_clamped = clamped if shared_rates else rho_clamped + clamped // reps
            home_goals, away_goals = sample_scorelines(
                lam_rep, mu_rep, rho, rng=rng, size=(reps, n_remaining)
            )

            flat = (np.arange(reps)[:, None] * n_teams).astype(np.int64)
            for side, scored, conceded in (
                (home_idx, home_goals, away_goals),
                (away_idx, away_goals, home_goals),
            ):
                target = (flat + side[None, :]).ravel()
                size = reps * n_teams
                points += np.bincount(
                    target, weights=points_for(scored, conceded).ravel(), minlength=size
                ).reshape(reps, n_teams).astype(np.int64)
                goals_for += np.bincount(
                    target, weights=scored.ravel(), minlength=size
                ).reshape(reps, n_teams).astype(np.int64)
                goals_against += np.bincount(
                    target, weights=conceded.ravel(), minlength=size
                ).reshape(reps, n_teams).astype(np.int64)

        goal_difference = goals_for - goals_against
        positions, level = rank(points, goal_difference, goals_for, rng=rng)
        any_tie += int(level.any(axis=1).sum())
        for question in spec.questions:
            mask = question.satisfied(positions, n_teams)
            question_counts[question.name] += mask.sum(axis=0)
            boundary_ties[question.name] += int(level[:, question.cut(n_teams)].sum())

        rows = np.arange(n_teams)[None, :]
        position_counts += np.bincount(
            (rows * n_teams + positions).ravel(), minlength=n_teams * n_teams
        ).reshape(n_teams, n_teams)
        points_counts += np.bincount(
            (rows * points_span + (points - points_floor)).ravel(),
            minlength=n_teams * points_span,
        ).reshape(n_teams, points_span)
        done += reps

    probabilities = pd.DataFrame({"team": list(teams)})
    for question in spec.questions:
        probabilities[question.name] = question_counts[question.name] / spec.n_replicates
    mean_points = (points_counts @ (np.arange(points_span) + points_floor)) / spec.n_replicates
    mean_position = (position_counts @ (np.arange(n_teams) + 1)) / spec.n_replicates
    probabilities["mean_points"] = mean_points
    probabilities["mean_position"] = mean_position
    probabilities = probabilities.sort_values("mean_position", kind="stable").reset_index(drop=True)

    cold = tuple(t for t in teams if t in set(fit.cold_start_teams) or t not in set(fit.teams))
    diagnostics: dict[str, object] = {
        "cold_start_teams": list(cold),
        "n_cold_start": len(cold),
        "replicates_with_a_level_pair": any_tie,
        "boundary_ties": dict(boundary_ties),
        "rho_clamped": rho_clamped,
        "questions": {q.name: q.label() for q in spec.questions},
        "spec": spec.label(),
        "horizon": horizon,
        "seed": int(seed),
    }
    return SeasonForecast(
        season=season or (str(played["season"].iloc[0]) if len(played) else
                          str(remaining["season"].iloc[0])),
        barrier=pd.Timestamp(barrier) if barrier is not None else (
            pd.Timestamp(remaining["date"].min()) if n_remaining
            else pd.Timestamp(played["date"].max())),
        teams=teams,
        table=ranked,
        n_played=len(played),
        n_remaining=n_remaining,
        n_replicates=spec.n_replicates,
        uncertainty=spec.label(),
        probabilities=probabilities,
        position_counts=position_counts,
        points_counts=points_counts,
        points_floor=points_floor,
        diagnostics=diagnostics,
    )
