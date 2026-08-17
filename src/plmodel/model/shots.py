"""A process-based variant: the same machinery fitted to shots on target instead of goals.

Goals are a sparse realisation of chance creation, so a rating fitted to them inherits finishing
noise. Expected goals is the standard remedy but is proprietary and, for us, has one live provider
and no coverage before 2014. Shots on target is the pre-xG proxy, present in the corpus from
2000/01 — free, complete, and already ingested.

The specification follows Pitcan (2026) §5.3 exactly, so that reproducing his pooling result on
Premier League data tests his claim rather than a variant of it:

1. Fit the identical Dixon-Coles machinery to shots on target, giving shot rates ``lam_S, mu_S``.
2. Convert to goal rates with a **single league-wide finishing factor** per side, estimated on the
   same window under the same decay weights:

       kappa_home = sum_m w_m * goals_m^home / sum_m w_m * sot_m^home

   and analogously for away. Deliberately league-wide rather than club-specific: club-level
   finishing variation is precisely the noise the substitution exists to remove. (Corroborated by
   the finishing-skill literature — shots-over-expected has year-over-year stability r ~ 0.63 while
   goals-over-expected is r ~ 0.12.)
3. **The low-score correction is not carried across.** Its parameter would have been estimated
   against shot counts, where the (0,0) and (1,1) cells mean something entirely different. The
   variant is therefore independent-Poisson on converted rates — a real difference from the goals
   model, reported rather than concealed.

This is also the groundwork for the xG-channel arm: swap shots on target for xG and the same
conversion and pooling apply.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plmodel.model.dixon_coles import DixonColesFit, decay_weights, fit_dixon_coles
from plmodel.model.scoreline import three_class_from_rates

# Shot counts have no meaningful low-score correction, so the variant is independent Poisson.
_NO_LOW_SCORE_CORRECTION = 0.0

SOT_COLUMNS: tuple[str, str] = ("home_sot", "away_sot")


@dataclass(frozen=True)
class ShotsModelFit:
    """A shots-on-target fit plus the finishing factors that convert it to goal rates."""

    shot_fit: DixonColesFit
    kappa_home: float
    kappa_away: float
    max_goals: int

    def rates(self, home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Expected *goals*, obtained by scaling the fitted shot rates."""
        lam_s, mu_s = self.shot_fit.rates(home, away)
        return self.kappa_home * lam_s, self.kappa_away * mu_s

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        lam, mu = self.rates(rows["home_team"], rows["away_team"])
        return three_class_from_rates(lam, mu, _NO_LOW_SCORE_CORRECTION, self.max_goals)

    def as_dict(self) -> dict[str, object]:
        return {
            "kappa_home": self.kappa_home,
            "kappa_away": self.kappa_away,
            "rho": _NO_LOW_SCORE_CORRECTION,
            "shot_model": self.shot_fit.as_dict(),
        }


def finishing_factors(
    history: pd.DataFrame, ref_date: pd.Timestamp, half_life_days: float
) -> tuple[float, float]:
    """League-wide weighted goals-per-shot-on-target, per side.

    Weighted by the same decay as the fit, so the conversion reflects the same period the ratings
    do — a factor estimated on the full history would misstate the current finishing rate exactly
    where the ratings are most current.
    """
    covered = history.dropna(subset=list(SOT_COLUMNS))
    if covered.empty:
        raise ValueError("no rows with shots-on-target coverage")
    weights = decay_weights(covered["date"], ref_date, half_life_days)
    kappas = []
    for goals_col, sot_col in (("home_goals", "home_sot"), ("away_goals", "away_sot")):
        shots = float(np.sum(weights * covered[sot_col].to_numpy(dtype=float)))
        if shots <= 0:
            raise ValueError(f"no weighted {sot_col} to convert from")
        kappas.append(float(np.sum(weights * covered[goals_col].to_numpy(dtype=float))) / shots)
    return kappas[0], kappas[1]


def fit_shots_model(
    history: pd.DataFrame,
    *,
    half_life_days: float,
    ref_date: pd.Timestamp,
    max_goals: int,
    param_bounds: dict[str, tuple[float, float]],
    min_effective_share: float,
    max_iter: int,
    warm_start: ShotsModelFit | None = None,
) -> ShotsModelFit:
    """Fit the shots-on-target variant on rows that carry shot coverage."""
    covered = history.dropna(subset=list(SOT_COLUMNS))
    if covered.empty:
        raise ValueError(
            "no shots-on-target coverage in the training window "
            "(the source carries it only from 2000/01)"
        )

    # The same fitter, handed shot counts in place of goals. Rho is fitted here against shot counts
    # and then discarded — see the module docstring.
    as_shots = covered.assign(
        home_goals=covered["home_sot"].to_numpy(dtype=float),
        away_goals=covered["away_sot"].to_numpy(dtype=float),
    )
    shot_fit = fit_dixon_coles(
        as_shots,
        half_life_days=half_life_days,
        ref_date=ref_date,
        max_goals=max_goals,
        param_bounds=param_bounds,
        min_effective_share=min_effective_share,
        max_iter=max_iter,
        warm_start=warm_start.shot_fit if warm_start is not None else None,
    )
    kappa_home, kappa_away = finishing_factors(covered, ref_date, half_life_days)
    return ShotsModelFit(
        shot_fit=shot_fit,
        kappa_home=kappa_home,
        kappa_away=kappa_away,
        max_goals=max_goals,
    )
