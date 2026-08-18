"""A second observation channel: the same machinery fitted to a chance-creation signal.

Goals are a sparse realisation of chance creation, so a rating fitted to them inherits finishing
noise. The remedy is to fit the identical Dixon-Coles machinery to a denser signal and convert the
resulting rate back into goals with a single league-wide finishing factor.

The channel is a **parameter**, not a hardcoded column. Which signal fills it is a data-availability
question that has changed twice during this project already, and the modelling question — does a
chance-creation channel carry information the goals model lacks? — is identical whichever fills it:

* ``sot`` — shots on target. Complete from 2000/01 in our own corpus, and the substitution Pitcan
  makes in his §5.3 for exactly the same reason.
* ``xg`` — expected goals. The signal the build plan asks for, currently unobtainable (see
  NOTES.md, 2026-08-18) and wired here so that acquiring it is a configuration change rather than
  a rewrite.

Specification, following Pitcan (2026) §5.3
-------------------------------------------
1. Fit the Dixon-Coles machinery to the channel counts, giving channel rates ``lam_c, mu_c``.
2. Convert to goal rates with a **league-wide** finishing factor per side, estimated on the same
   window under the same decay weights::

       kappa_home = sum_m w_m * goals_m^home / sum_m w_m * channel_m^home

   Deliberately league-wide rather than club-specific: club-level finishing variation is precisely
   the noise the substitution exists to remove. Corroborated by the finishing-skill literature —
   shots-over-expected has year-over-year stability r ~ 0.63 against r ~ 0.12 for
   goals-over-expected.
3. **The low-score correction is not carried across.** Its parameter would have been estimated
   against channel counts, where the (0,0) and (1,1) cells mean something entirely different. The
   variant is therefore independent-Poisson on converted rates — a real difference from the goals
   model, reported rather than concealed.

An xG channel needs one extra thought that shots on target does not: xG is continuous, so the
Poisson likelihood is being applied to a non-count. That is the same approximation the literature
makes when fitting rating models to xG, but it is an approximation, and it belongs in the ledger
when the arm eventually runs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plmodel.model.dixon_coles import DixonColesFit, decay_weights, fit_dixon_coles
from plmodel.model.scoreline import three_class_from_rates

# Channel counts carry no meaningful low-score correction, so the variant is independent Poisson.
_NO_LOW_SCORE_CORRECTION = 0.0


@dataclass(frozen=True)
class ChannelSpec:
    """Which columns carry the chance-creation signal."""

    name: str
    home_column: str
    away_column: str
    continuous: bool = False

    @property
    def columns(self) -> tuple[str, str]:
        return (self.home_column, self.away_column)


# Shots on target: complete from 2000/01, already ingested, no external dependency.
SHOTS_ON_TARGET = ChannelSpec("sot", "home_sot", "away_sot")
# Expected goals: the build plan's intended channel. Continuous, so the Poisson likelihood is an
# approximation rather than an exact model — flagged here so the ledger records it when it runs.
EXPECTED_GOALS = ChannelSpec("xg", "home_xg", "away_xg", continuous=True)

CHANNELS: dict[str, ChannelSpec] = {c.name: c for c in (SHOTS_ON_TARGET, EXPECTED_GOALS)}


def get_channel(name: str) -> ChannelSpec:
    if name not in CHANNELS:
        raise ValueError(f"unknown channel {name!r}; known: {sorted(CHANNELS)}")
    return CHANNELS[name]


@dataclass(frozen=True)
class ChannelModelFit:
    """A channel fit plus the finishing factors that convert it back to goal rates."""

    channel: ChannelSpec
    channel_fit: DixonColesFit
    kappa_home: float
    kappa_away: float
    max_goals: int

    def rates(self, home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Expected *goals*, obtained by scaling the fitted channel rates."""
        lam_c, mu_c = self.channel_fit.rates(home, away)
        return self.kappa_home * lam_c, self.kappa_away * mu_c

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        lam, mu = self.rates(rows["home_team"], rows["away_team"])
        return three_class_from_rates(lam, mu, _NO_LOW_SCORE_CORRECTION, self.max_goals)

    def as_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel.name,
            "kappa_home": self.kappa_home,
            "kappa_away": self.kappa_away,
            "rho": _NO_LOW_SCORE_CORRECTION,
            "channel_model": self.channel_fit.as_dict(),
        }


def finishing_factors(
    history: pd.DataFrame, ref_date: pd.Timestamp, half_life_days: float, channel: ChannelSpec
) -> tuple[float, float]:
    """League-wide weighted goals-per-channel-unit, per side.

    Weighted by the same decay as the fit, so the conversion reflects the same period the ratings
    do — a factor estimated on the full history would misstate the current finishing rate exactly
    where the ratings are most current.
    """
    covered = history.dropna(subset=list(channel.columns))
    if covered.empty:
        raise ValueError(f"no rows with {channel.name} coverage")
    weights = decay_weights(covered["date"], ref_date, half_life_days)
    kappas = []
    for goals_col, channel_col in (
        ("home_goals", channel.home_column), ("away_goals", channel.away_column)
    ):
        denominator = float(np.sum(weights * covered[channel_col].to_numpy(dtype=float)))
        if denominator <= 0:
            raise ValueError(f"no weighted {channel_col} to convert from")
        numerator = float(np.sum(weights * covered[goals_col].to_numpy(dtype=float)))
        kappas.append(numerator / denominator)
    return kappas[0], kappas[1]


def fit_channel_model(
    history: pd.DataFrame,
    *,
    channel: ChannelSpec = SHOTS_ON_TARGET,
    half_life_days: float,
    ref_date: pd.Timestamp,
    max_goals: int,
    param_bounds: dict[str, tuple[float, float]],
    min_effective_share: float,
    max_iter: int,
    warm_start: ChannelModelFit | None = None,
) -> ChannelModelFit:
    """Fit the second-channel variant on rows that carry the channel's coverage."""
    missing = [c for c in channel.columns if c not in history.columns]
    if missing:
        raise ValueError(
            f"channel {channel.name!r} needs columns {missing}, absent from the training frame"
        )
    covered = history.dropna(subset=list(channel.columns))
    if covered.empty:
        raise ValueError(
            f"no {channel.name} coverage in the training window; check the ingest's coverage report"
        )

    # The same fitter, handed channel counts in place of goals. Rho is fitted here against those
    # counts and then discarded — see the module docstring.
    as_channel = covered.assign(
        home_goals=covered[channel.home_column].to_numpy(dtype=float),
        away_goals=covered[channel.away_column].to_numpy(dtype=float),
    )
    channel_fit = fit_dixon_coles(
        as_channel,
        half_life_days=half_life_days,
        ref_date=ref_date,
        max_goals=max_goals,
        param_bounds=param_bounds,
        min_effective_share=min_effective_share,
        max_iter=max_iter,
        warm_start=warm_start.channel_fit if warm_start is not None else None,
    )
    kappa_home, kappa_away = finishing_factors(covered, ref_date, half_life_days, channel)
    return ChannelModelFit(
        channel=channel,
        channel_fit=channel_fit,
        kappa_home=kappa_home,
        kappa_away=kappa_away,
        max_goals=max_goals,
    )
