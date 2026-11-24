"""Team ratings from bookmakers' prices of earlier matches: the market-informed track's model.

**What it is.** Every priced past match is inverted to the pair of goal rates whose independent
Poisson home/draw/away probabilities reproduce its de-vigged price. Attack and defence are then
fitted to those implied rates by weighted least squares, with exponential decay and a ridge on the
team terms, exactly as the production model fits them to goals. The forecast is independent Poisson
from the fitted rates.

**What it never reads.** The forecast match's own price. Ratings at a barrier come only from matches
strictly before it, so every forecast is still made before its own market; what it borrows is the
market's view of each team's strength last week. That makes it a market-informed model under the
acceptance rule's two tracks: it can never become production, and it is reported under its own
name. See NOTES.md, 2026-11-18.

**Why a price and not a goal.** A goal count is a very noisy reading of strength (per-match
signal-to-noise about 0.13); a price is a near noise-free one that already contains who was
injured, signed or sacked at the time. The independent review measured the gap to the close this
closes at about 0.0015 of 0.0050, and the tuning span confirmed a gain of −0.0016 before either
evaluation decade was walked.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plmodel.data.odds import devig_shin, resolve_family


class OddsRatingsError(ValueError):
    """Raised when the ratings cannot be formed honestly (no prices, an inversion that failed)."""


# --- prices ------------------------------------------------------------------------------------

def price_table(frame: pd.DataFrame, families: tuple[str, ...]) -> np.ndarray:
    """(N, 3) de-vigged pre-close probabilities, the first family that priced each row winning.

    NaN on rows no family priced. Shin de-vig, the project's primary method, family by family.
    """
    out = np.full((len(frame), 3), np.nan)
    for name in families:
        if not all(c in frame.columns for c in _family_columns(name)):
            continue
        odds, covered, _ = resolve_family(frame, name)
        take = covered & np.isnan(out).any(axis=1)
        if take.any():
            out[take], _ = devig_shin(odds[take])
    return out


def _family_columns(name: str) -> tuple[str, ...]:
    from plmodel.data.odds import get_family

    return get_family(name).columns


# --- the Poisson three-way and its inversion ---------------------------------------------------

def _marginals(rate: np.ndarray, max_goals: int) -> np.ndarray:
    """(N, G) Poisson probabilities of 0..max_goals, renormalised over the truncated grid."""
    k = np.arange(max_goals + 1)
    log_pmf = k[None, :] * np.log(rate)[:, None] - rate[:, None] - _log_factorial(k)[None, :]
    pmf = np.exp(log_pmf)
    return pmf / pmf.sum(axis=1, keepdims=True)


def _log_factorial(k: np.ndarray) -> np.ndarray:
    from scipy.special import gammaln

    return gammaln(k + 1.0)


def _three_way_parts(lam: np.ndarray, mu: np.ndarray, max_goals: int):
    """Home, draw, away, and the two off-by-one masses the analytic Jacobian needs."""
    p = _marginals(lam, max_goals)
    q = _marginals(mu, max_goals)
    # P(home goals > j), for each j, from the renormalised home marginal.
    above = 1.0 - np.cumsum(p, axis=1)
    home = (q * above).sum(axis=1)
    draw = (p * q).sum(axis=1)
    away = 1.0 - home - draw
    home_one_more = (p[:, 1:] * q[:, :-1]).sum(axis=1)    # P(X = Y + 1)
    away_one_more = (q[:, 1:] * p[:, :-1]).sum(axis=1)    # P(Y = X + 1)
    return home, draw, away, home_one_more, away_one_more


def three_way(lam: np.ndarray, mu: np.ndarray, max_goals: int) -> np.ndarray:
    """(N, 3) independent-Poisson home/draw/away probabilities."""
    home, draw, away, _, _ = _three_way_parts(np.asarray(lam, float), np.asarray(mu, float), max_goals)
    return np.c_[home, draw, away]


def implied_rates(
    probs: np.ndarray, *, max_goals: int, tolerance: float, max_iter: int
) -> np.ndarray:
    """(N, 2) goal rates whose independent-Poisson home and away probabilities equal ``probs``'.

    Newton on log rates for every row at once, with the exact derivatives of the untruncated model:
    dP(home)/dlam = P(draw), dP(home)/dmu = -P(X = Y + 1), and symmetrically for away. The residual
    uses the truncated, renormalised probabilities, so the fixed point is exact for the model the
    forecast uses; the Jacobian only sets the pace. Raises rather than returns a row that did not
    converge: a rate fitted to the wrong price is worse than a missing one.
    """
    target = np.asarray(probs, dtype=float)
    z = np.zeros((len(target), 2))                            # log rates, starting at 1 goal each
    # Each row stops on its OWN convergence. Iterating every row until the slowest converges would
    # let one match's price change how many steps another row takes, and so its last bits: a
    # numerical coupling between matches that the leakage test exists to rule out.
    active = np.ones(len(target), dtype=bool)
    for _ in range(max_iter):
        if not active.any():
            return np.exp(z)
        za = z[active]
        lam, mu = np.exp(za[:, 0]), np.exp(za[:, 1])
        home, draw, away, home_one_more, away_one_more = _three_way_parts(lam, mu, max_goals)
        r_home, r_away = home - target[active, 0], away - target[active, 2]
        done = np.maximum(np.abs(r_home), np.abs(r_away)) < tolerance
        # Jacobian of (home, away) with respect to (log lam, log mu).
        a, b = lam * draw, -mu * home_one_more
        c, d = -lam * away_one_more, mu * draw
        det = a * d - b * c
        step_lam = (d * r_home - b * r_away) / det
        step_mu = (a * r_away - c * r_home) / det
        # A full Newton step from a poor start can overshoot into a region where the draw mass,
        # and so the Jacobian, vanishes; one unit of log rate per step keeps every row on track.
        za[:, 0] -= np.where(done, 0.0, np.clip(step_lam, -1.0, 1.0))
        za[:, 1] -= np.where(done, 0.0, np.clip(step_mu, -1.0, 1.0))
        z[active] = za
        active[np.flatnonzero(active)[done]] = False
    if not active.any():
        return np.exp(z)
    raise OddsRatingsError(
        f"price inversion did not converge in {max_iter} steps for {int(active.sum())} row(s)"
    )


# --- the two-market inversion (`odds-ratings-ou`) ----------------------------------------------

def over_under_prices(frame: pd.DataFrame) -> np.ndarray:
    """(N,) proportionally de-vigged P(over 2.5) from the first pre-close source that priced each
    row; NaN where none did."""
    from plmodel.data.odds import OVER_UNDER_PRECLOSE

    out = np.full(len(frame), np.nan)
    for over, under in OVER_UNDER_PRECLOSE:
        if over not in frame.columns or under not in frame.columns:
            continue
        odds = frame[[over, under]].to_numpy(dtype=float)
        take = np.isfinite(odds).all(axis=1) & (odds > 1.0).all(axis=1) & np.isnan(out)
        implied = 1.0 / odds[take]
        out[take] = implied[:, 0] / implied.sum(axis=1)
    return out


def _over_parts(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(total goals > 2) and P(total goals == 2) from the two marginals."""
    total_two = p[:, 0] * q[:, 2] + p[:, 1] * q[:, 1] + p[:, 2] * q[:, 0]
    at_most_one = p[:, 0] * q[:, 0] + p[:, 0] * q[:, 1] + p[:, 1] * q[:, 0]
    return 1.0 - at_most_one - total_two, total_two


def two_market_rates(
    supremacy: np.ndarray, over: np.ndarray, *, max_goals: int, tolerance: float, max_iter: int
) -> np.ndarray:
    """(N, 2) rates with P(home) - P(away) = ``supremacy`` and P(over 2.5) = ``over``.

    The 1X2 price says who is better; the over/under price says how many goals. Fitting both rates
    to the 1X2 price alone, as :func:`implied_rates` does, also forces the draw to match, and an
    independent Poisson model can only produce the market's draw rate by pulling both rates down:
    the 9-13% goal shortfall the goal-market yardstick measured. Here the draw falls where the model
    puts it. Newton on log rates with exact untruncated derivatives, each row stopping on its own
    convergence; raises rather than returning a row that did not converge.
    """
    target_s = np.asarray(supremacy, dtype=float)
    target_o = np.asarray(over, dtype=float)
    z = np.zeros((len(target_s), 2))
    active = np.ones(len(target_s), dtype=bool)
    for _ in range(max_iter):
        if not active.any():
            return np.exp(z)
        za = z[active]
        lam, mu = np.exp(za[:, 0]), np.exp(za[:, 1])
        home, draw, away, home_one_more, away_one_more = _three_way_parts(lam, mu, max_goals)
        over_p, total_two = _over_parts(_marginals(lam, max_goals), _marginals(mu, max_goals))
        r_s = (home - away) - target_s[active]
        r_o = over_p - target_o[active]
        done = np.maximum(np.abs(r_s), np.abs(r_o)) < tolerance
        # d(home - away)/d log rate, and d P(over)/d log rate = rate * P(total == 2).
        a, b = lam * (draw + away_one_more), -mu * (draw + home_one_more)
        c, d = lam * total_two, mu * total_two
        det = a * d - b * c
        step_lam = (d * r_s - b * r_o) / det
        step_mu = (a * r_o - c * r_s) / det
        za[:, 0] -= np.where(done, 0.0, np.clip(step_lam, -1.0, 1.0))
        za[:, 1] -= np.where(done, 0.0, np.clip(step_mu, -1.0, 1.0))
        z[active] = za
        active[np.flatnonzero(active)[done]] = False
    if not active.any():
        return np.exp(z)
    raise OddsRatingsError(
        f"two-market inversion did not converge in {max_iter} steps for {int(active.sum())} row(s)"
    )


def market_rates(
    frame: pd.DataFrame, families: tuple[str, ...], *, use_over_under: bool, max_goals: int,
    tolerance: float, max_iter: int,
) -> np.ndarray:
    """(N, 2) implied rates for every priced row of ``frame``; NaN where unpriced.

    With ``use_over_under``, rows priced on both markets use the two-market inversion and rows
    priced on 1X2 only fall back to the 1X2 inversion. Each row depends on its own prices alone.
    """
    prices = price_table(frame, families)
    rates = np.full((len(frame), 2), np.nan)
    has_1x2 = ~np.isnan(prices).any(axis=1)
    both = np.zeros(len(frame), dtype=bool)
    if use_over_under:
        over = over_under_prices(frame)
        both = has_1x2 & ~np.isnan(over)
        if both.any():
            rates[both] = two_market_rates(
                prices[both, 0] - prices[both, 2], over[both],
                max_goals=max_goals, tolerance=tolerance, max_iter=max_iter,
            )
    only = has_1x2 & ~both
    if only.any():
        rates[only] = implied_rates(prices[only], max_goals=max_goals, tolerance=tolerance,
                                    max_iter=max_iter)
    return rates


# --- the ratings -------------------------------------------------------------------------------

@dataclass(frozen=True)
class OddsRatings:
    """Fitted ratings. A club absent from the fit sits at 0: the league average."""

    teams: tuple[str, ...]
    intercept: float
    home_advantage: float
    attack: np.ndarray
    defence: np.ndarray
    n_obs: int

    def rates(self, home: pd.Series, away: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        idx = {t: i for i, t in enumerate(self.teams)}

        def strength(clubs: pd.Series, table: np.ndarray) -> np.ndarray:
            return np.array([table[idx[c]] if c in idx else 0.0 for c in clubs], dtype=float)

        a_h, a_a = strength(home, self.attack), strength(away, self.attack)
        d_h, d_a = strength(home, self.defence), strength(away, self.defence)
        lam = np.exp(self.intercept + self.home_advantage + a_h - d_a)
        mu = np.exp(self.intercept + a_a - d_h)
        return lam, mu

    def predict_proba(self, rows: pd.DataFrame, max_goals: int) -> np.ndarray:
        lam, mu = self.rates(rows["home_team"], rows["away_team"])
        return three_way(lam, mu, max_goals)


def fit_odds_ratings(
    history: pd.DataFrame,
    rates: np.ndarray,
    *,
    ref_date: pd.Timestamp,
    half_life_days: float,
    ridge: float,
    lookback_days: float,
) -> OddsRatings:
    """Weighted least squares of log implied rates on intercept, home advantage, attack, defence.

    ``history`` and ``rates`` are aligned; rows with NaN rates are unpriced and skipped. Only rows
    dated strictly before ``ref_date`` and within ``lookback_days`` of it enter, so a caller cannot
    leak the forecast date's own prices by passing too wide a frame.
    """
    dates = pd.to_datetime(history["date"])
    ref = pd.Timestamp(ref_date)
    age = (ref - dates).dt.days.to_numpy(dtype=float)
    use = np.isfinite(rates).all(axis=1) & (age > 0) & (age <= lookback_days)
    if not use.any():
        raise OddsRatingsError(f"no priced match in the {lookback_days:.0f} days before {ref.date()}")

    rows = history.loc[use]
    teams = tuple(sorted(set(rows["home_team"]) | set(rows["away_team"])))
    index = {t: i for i, t in enumerate(teams)}
    n, n_teams = int(use.sum()), len(teams)
    h = rows["home_team"].map(index).to_numpy()
    a = rows["away_team"].map(index).to_numpy()
    weight = np.sqrt(np.exp(-np.log(2.0) * age[use] / half_life_days))

    # Columns: intercept, home advantage, attack[teams], defence[teams].
    design = np.zeros((2 * n, 2 + 2 * n_teams))
    r = np.arange(n)
    design[r, 0] = 1.0
    design[r, 1] = 1.0
    design[r, 2 + h] = 1.0
    design[r, 2 + n_teams + a] = -1.0
    design[n + r, 0] = 1.0
    design[n + r, 2 + a] = 1.0
    design[n + r, 2 + n_teams + h] = -1.0
    target = np.r_[np.log(rates[use, 0]), np.log(rates[use, 1])]
    w = np.r_[weight, weight]

    penalty = np.sqrt(ridge) * np.eye(2 + 2 * n_teams)
    penalty[0, 0] = penalty[1, 1] = 0.0                  # the level and home advantage are free
    beta = np.linalg.lstsq(
        np.vstack([design * w[:, None], penalty]),
        np.r_[target * w, np.zeros(2 + 2 * n_teams)],
        rcond=None,
    )[0]
    return OddsRatings(
        teams=teams,
        intercept=float(beta[0]),
        home_advantage=float(beta[1]),
        attack=beta[2:2 + n_teams],
        defence=beta[2 + n_teams:],
        n_obs=n,
    )
