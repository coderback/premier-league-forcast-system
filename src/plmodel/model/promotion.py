"""A prior for newly promoted clubs, estimated from what promoted clubs have actually been worth.

The production fit has almost nothing to say about a club in its first top-flight season, and what
it does say is wrong in a measurable direction. Below ``min_effective_share`` of the median team's
effective history a club is dropped from the fit entirely and scored at attack = defence = 0 --
"unknown teams take the league average", :meth:`DixonColesFit.rates`. **Twelve of the thirty
promoted clubs since 2016-17 are pinned there**, and the other eighteen are fitted on a median of
26.7 effective matches against an established club's 102.7.

League average is not a neutral prior for these clubs. Over the test decade a promoted club's first
season returns 34.4 points against 52.6, 0.996 goals for against 1.348 and 1.763 against against
1.348 -- roughly -0.30 in log attack and -0.27 in log defensive solidity, on 28% of every season's
fixtures. Four separate arms have pointed here: the Elo synthesis (arm 1), the European flag that
turned out to be group shrinkage (arm 6), the multi-tier fit that made promoted clubs *worse*
(arm 8), and the season simulator, which puts their preseason relegation forecast at 0.446 against
0.533 observed.

What this module does NOT do
----------------------------
**It does not read the Championship.** Arm 8 did, with real E1 results, and failed backwards -- it
damaged exactly the fixtures it was built to fix. A pooled fit learns the gap between two divisions
from the clubs that have played in both, and those are overwhelmingly recently-relegated clubs,
whom parachute payments make the strongest side of the lower division. The bridge is made of the
wrong clubs and has been getting wronger for thirty years. So the prior here is estimated from E0
alone, where a promoted club's own top-flight results are the only evidence used.

**It does not hardcode the gap.** The -0.30/-0.27 figures above were measured on the *test decade*;
freezing them into config would be leakage wearing a constant's clothing. They also drift -- promoted
clubs went from 0.219 below the clubs they face (2006-16) to 0.352 below (2016-26) -- so a frozen
value would be wrong in a known direction and getting wronger.

**It does not add a memory to tune.** The estimate is decay-weighted at whatever half-life the fit
is already using, so it tracks that drift for free. A second, separately tuned memory would be a
half-life selected for a quantity this window may not move enough to inform, which is what the
stationarity rule exists to refuse.

Identifying a promoted club without looking forward
---------------------------------------------------
Everything here is computed from training rows: no fixture list, no current-season results, nothing
dated on or after the barrier. Two different questions get two different answers, and conflating
them would be a bug:

* **Which clubs does the penalty apply to?** Those in their first season of the current spell,
  read off the latest season present in the training window -- :func:`promoted_at_barrier`. See its
  docstring for the one approximation this makes at a season-opening barrier.
* **Which clubs get the pin?** Every club the fit could not identify. A club with no usable
  top-flight history is, epistemically, in the promoted club's position whether it arrived by
  promotion or by returning after years away, and the prior is a far better description of it than
  the league average is. This needs no season logic at all, and it is the only thing that can reach
  a club arriving with zero E0 rows -- which is not merely cold-start but absent from the fit's
  team list entirely.

The two sites the prior is applied at
--------------------------------------
=========  ==============================================  ================================
site       reaches                                         mechanism
=========  ==============================================  ================================
pin        cold-start clubs, which are not parameters      scored at the prior, not at 0
shrink     promoted clubs that ARE fitted, on thin data    ridge penalty toward the prior
=========  ==============================================  ================================

The pin is the only thing that can reach the twelve, because a club dropped from the fit has no
parameter to shrink. The penalty is the only thing that can reach the eighteen, because they are
fitted freely today. One prior, two sites, and the pre-registered slices report them separately.

Inertness
---------
With the seam off, :func:`Config.promotion_spec` returns ``None``, no penalty is constructed and no
pinned map exists. The fit takes the same code path it takes today and the walk is byte-identical --
the seam is absent from the call graph rather than reproducing the same answer by a longer route.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plmodel.model.shrinkage import ridge_penalty


class PromotionError(ValueError):
    """Raised when a promotion-prior configuration cannot be honoured."""


@dataclass(frozen=True)
class PromotionSpec:
    """How hard to pull a promoted club toward the estimated promoted level.

    ``shrinkage`` is the ridge coefficient on the penalty; it is NOT weighted per team, because the
    likelihood's own curvature already scales with a club's effective sample size. A club with a
    full history barely moves and a thin one moves a lot, which is the behaviour wanted, and it
    costs one knob rather than two.
    """

    shrinkage: float
    min_prior_clubs: int

    def __post_init__(self) -> None:
        if self.shrinkage < 0:
            raise PromotionError(f"shrinkage must be non-negative; got {self.shrinkage}")
        if self.min_prior_clubs < 1:
            raise PromotionError(
                f"min_prior_clubs must be at least 1; got {self.min_prior_clubs}"
            )

    # No `is_inert` property, deliberately. The scoreline family has one because its off state is a
    # readable pair of VALUES (poisson + tau); this seam's off state is the switch, exactly as for
    # dynamics and decay. And it could not be `shrinkage == 0`: at zero shrinkage the penalty
    # vanishes but a cold-start club is still scored at the prior rather than at league average,
    # which is a different model. A property that always returned False would be a trap, so
    # inertness is decided in one place -- `Config.promotion_spec` returning None.


@dataclass(frozen=True)
class PromotionPrior:
    """The estimated promoted-club level, in log-goal units, plus what it was estimated from."""

    attack: float
    defence: float
    n_clubs: int
    effective_n: float
    teams: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "prior_attack": self.attack,
            "prior_defence": self.defence,
            "n_prior_clubs": self.n_clubs,
            "prior_effective_n": self.effective_n,
            "n_promoted_now": len(self.teams),
            "promoted_now": list(self.teams),
        }


def _season_teams(history: pd.DataFrame) -> dict[str, set[str]]:
    return {
        season: set(g["home_team"]) | set(g["away_team"])
        for season, g in history.groupby("season", sort=True)
    }


def promoted_at_barrier(history: pd.DataFrame) -> tuple[str, ...]:
    """Clubs playing their first season of this spell, in the latest season ``history`` contains.

    Training rows only -- no fixture list, no current-season results, nothing dated on or after the
    barrier. This is the set the ridge penalty applies to.

    **The one approximation in this seam, stated rather than hidden.** At a mid-season barrier the
    latest season in ``history`` is the season in progress, and this returns exactly the clubs
    currently in their first season up -- the intended target, and where the great majority of
    promoted-club fixtures live. At a *season-opening* barrier no match of the new season has been
    played yet, so the latest season in ``history`` is the one just finished and this returns the
    PREVIOUS intake: clubs now starting their second season, which the penalty will pull toward a
    prior that is wrong for them by roughly however much they improved over a season.

    Separating the two cases needs the barrier's season label, and the only honest sources for it
    are the fixture list (test data) or a hardcoded Aug-May calendar (a magic number in ``model/``).
    Neither is worth it here, because the mis-targeted group is the survivors of the previous
    intake -- the second-weakest clubs in the division -- so the prior is wrong for them in size
    rather than in direction. It is recorded in the pre-registration as a known approximation, and
    :func:`test_promotion.test_season_opening_barrier_carries_the_previous_intake` pins the
    behaviour so it cannot change silently.
    """
    if history.empty:
        return ()
    latest = sorted(history["season"].unique())[-1]
    return tuple(sorted(first_season_clubs(history).get(latest, set())))


def first_season_clubs(history: pd.DataFrame) -> dict[str, set[str]]:
    """Season -> clubs playing their first season of this spell, over the whole training window.

    The pool the prior is estimated from. The first season in ``history`` has no predecessor and so
    contributes nobody: reporting its whole membership as promoted would be an artefact of where
    the corpus starts rather than a fact about football. Same reasoning, and same shape, as
    :func:`plmodel.eval.slices.promoted_teams`.
    """
    by_season = _season_teams(history)
    seasons = sorted(by_season)
    return {
        season: (set() if i == 0 else by_season[season] - by_season[seasons[i - 1]])
        for i, season in enumerate(seasons)
    }


def estimate_prior(
    history: pd.DataFrame,
    weights: np.ndarray,
    *,
    min_prior_clubs: int,
) -> PromotionPrior | None:
    """The promoted level in log-goal units, from decay-weighted goal rates in ``history``.

    ``prior_attack  =  log(GF_promoted / GF_league)``
    ``prior_defence = -log(GA_promoted / GA_league)``

    The sign on defence is the model's convention, not a choice: ``defence`` is solidity and enters
    the rate as ``- defence[opponent]``, so a club that concedes more has a *lower* defence. Both
    priors come out negative for a promoted club, which is the point.

    ``weights`` are the fit's own decay weights, aligned to ``history``, so the estimate carries the
    same memory as the parameters it will be applied to and tracks the drift in promoted clubs'
    standard without a second half-life to tune.

    Returns ``None`` when the window holds fewer than ``min_prior_clubs`` promoted clubs to learn
    from -- early in a walk there is genuinely nothing to estimate, and a prior built from one club
    is worse than no prior. The caller then behaves exactly as the baseline does.
    """
    pool = first_season_clubs(history)
    flat = {t for teams in pool.values() for t in teams}
    if len(flat) < min_prior_clubs:
        return None

    home = history["home_team"].to_numpy()
    away = history["away_team"].to_numpy()
    season = history["season"].to_numpy()
    hg = history["home_goals"].to_numpy(dtype=float)
    ag = history["away_goals"].to_numpy(dtype=float)

    # A row counts toward the promoted pool once per SIDE that was promoted that season, so a
    # promoted-vs-promoted fixture contributes both of its sides rather than being counted once or
    # dropped. Scored and conceded are tracked separately because attack and defence are.
    is_home_promoted = np.array(
        [t in pool.get(s, ()) for t, s in zip(home, season)], dtype=bool
    )
    is_away_promoted = np.array(
        [t in pool.get(s, ()) for t, s in zip(away, season)], dtype=bool
    )

    w_promoted = weights * is_home_promoted + weights * is_away_promoted
    scored_promoted = weights * is_home_promoted * hg + weights * is_away_promoted * ag
    conceded_promoted = weights * is_home_promoted * ag + weights * is_away_promoted * hg

    # The league baseline is every side of every match, promoted or not -- the same denominator the
    # league-average strength of 0 refers to, so the ratio is the gap the model would otherwise
    # assume away.
    #
    # One baseline serves both priors, and that is arithmetic rather than an approximation: every
    # goal scored in a match is also conceded in it, so the league's goals-for per side and its
    # goals-against per side are the SAME number. `gf_league` below is therefore the correct
    # denominator for the conceded ratio too, and computing a separate `ga_league` would produce a
    # second name for one quantity.
    w_league = 2.0 * weights.sum()
    scored_league = (weights * (hg + ag)).sum()

    eff = float(w_promoted.sum())
    if eff <= 0.0 or w_league <= 0.0:
        return None

    gf_promoted = float(scored_promoted.sum()) / eff
    ga_promoted = float(conceded_promoted.sum()) / eff
    gf_league = float(scored_league) / float(w_league)
    if min(gf_promoted, ga_promoted, gf_league) <= 0.0:
        return None

    return PromotionPrior(
        attack=float(np.log(gf_promoted / gf_league)),
        defence=float(-np.log(ga_promoted / gf_league)),
        n_clubs=len(flat),
        effective_n=eff,
        teams=promoted_at_barrier(history),
    )


def penalty_and_gradient(
    attack: np.ndarray,
    defence: np.ndarray,
    promoted_mask: np.ndarray,
    prior: PromotionPrior,
    shrinkage: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Ridge penalty toward the promoted-club prior, and its gradient.

    A thin adapter over :func:`plmodel.model.shrinkage.ridge_penalty`, which is the same quadratic
    with a different centre. Kept as a named function because "pull promoted clubs toward the
    promoted level" is what this seam means, and because the general seam and this one must not grow
    two copies of one derivative.
    """
    return ridge_penalty(
        attack, defence, promoted_mask, prior.attack, prior.defence, shrinkage
    )
