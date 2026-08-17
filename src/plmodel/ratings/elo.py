"""Elo ratings — the single-strength-scalar alternative to per-team attack and defence.

    We = 1 / (1 + 10 ** (-(R_home + home_advantage - R_away) / scale))
    R_home' = R_home + K * G * (W - We)

with ``W`` = 1 for a home win, 0.5 for a draw, 0 for a loss, and ``G`` a goal-difference multiplier
that makes a heavier win move the ratings further. Away ratings move by exactly the opposite
amount, so total rating is conserved.

This exists only as a comparison arm. The WC2026 project parameterised its Dixon-Coles by a single
Elo difference **because international football is too sparse for 2N attack/defence parameters** —
teams play ~10 matches a year and many pairs never meet. A 20-team league playing 38 rounds does
not have that constraint, so the choice should be reversed here. That is a hypothesis, and this
module exists so it can be measured rather than assumed. The research report notes no clean
published head-to-head isolates the two parameterisations on league data.

One replay, computed once
-------------------------
The replay is a single forward pass: a team's rating before match *m* depends only on matches
before *m*, so the pass is causal by construction and one global replay serves every barrier. This
is both faster than replaying per barrier and safer — there is no window for an off-by-one to let a
result inform its own prediction.

**The replay sees only the prediction division.** Ratings could equally be carried across the
promotion boundary from the lower tiers, and Club Elo does exactly that, which would hand promoted
teams a real rating instead of a cold start. That is a genuine advantage — and it belongs to the
multi-tier arm, not this one. Mixing it in here would confound "per-team versus scalar" with
"one tier versus four" and neither result would be interpretable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The logistic scale of the rating difference. 400 is the defining constant of the Elo system: a
# 400-point lead corresponds to 10:1 expected odds. Not a tunable — changing it merely rescales
# every rating and K in step.
ELO_SCALE = 400.0
_LOG_BASE = 10.0

# Outcome values for the update.
_WIN, _DRAW, _LOSS = 1.0, 0.5, 0.0


@dataclass
class EloConfig:
    """Elo scheme parameters. K and the home advantage are tuned; the rest are structural."""

    initial_rating: float
    k: float
    home_advantage: float
    gd_two_goal: float
    gd_slope_offset: float
    gd_slope_divisor: float


def goal_difference_multiplier(gd: np.ndarray, cfg: EloConfig) -> np.ndarray:
    """The World Football Elo weighting: 1 for a one-goal margin, more for heavier wins.

    Without it a 5-0 and a 1-0 move the ratings identically, which discards the clearest available
    signal about how far apart two sides actually are.
    """
    margin = np.abs(np.asarray(gd, dtype=float))
    out = np.ones_like(margin)
    out[margin == 2] = cfg.gd_two_goal
    big = margin >= 3
    out[big] = (cfg.gd_slope_offset + margin[big]) / cfg.gd_slope_divisor
    return out


def expected_home_score(rating_diff: np.ndarray | float) -> np.ndarray | float:
    """Expected score for the home side given the home-advantage-adjusted rating difference."""
    return 1.0 / (1.0 + _LOG_BASE ** (-np.asarray(rating_diff, dtype=float) / ELO_SCALE))


@dataclass
class EloReplay:
    """A completed forward pass: per-match pre-ratings, plus a rating lookup at any barrier."""

    history: pd.DataFrame = field(repr=False)
    final_ratings: dict[str, float]
    config: EloConfig

    def __post_init__(self) -> None:
        """Index each team's rating trajectory once, so a barrier lookup is a binary search.

        Rescanning the history at every barrier is O(matches) per call, and across a thousand
        barriers inside a tuning sweep that dominates everything else — the four-parameter fit it
        feeds takes milliseconds by comparison. Indexing once turns each lookup into one binary
        search per team, of which there are only ever about fifty.
        """
        rows: dict[str, list[int]] = {}
        values: dict[str, list[float]] = {}
        for column, rating_column in (("home_team", "home_elo_post"), ("away_team", "away_elo_post")):
            teams = self.history[column].to_numpy()
            ratings = self.history[rating_column].to_numpy()
            for position, (team, rating) in enumerate(zip(teams, ratings)):
                rows.setdefault(team, []).append(position)
                values.setdefault(team, []).append(rating)
        # Each side was appended in row order, so merging the two lists needs a stable re-sort.
        self._team_rows = {}
        self._team_ratings = {}
        for team, positions in rows.items():
            order = np.argsort(np.asarray(positions), kind="stable")
            self._team_rows[team] = np.asarray(positions)[order]
            self._team_ratings[team] = np.asarray(values[team])[order]
        self._dates = self.history["date"].to_numpy()

    def ratings_asof(self, barrier: pd.Timestamp) -> dict[str, float]:
        """Each team's rating using only matches strictly before ``barrier``.

        Reads the *post-match* rating of each team's last prior match, which is exactly the state
        an online rating would be in at that moment. Teams that have not yet played are absent
        rather than defaulted, so a caller must decide explicitly what an unrated team means.
        """
        end = int(np.searchsorted(self._dates, np.datetime64(pd.Timestamp(barrier)), side="left"))
        if end == 0:
            return {}
        out: dict[str, float] = {}
        for team, positions in self._team_rows.items():
            k = int(np.searchsorted(positions, end, side="left"))
            if k:
                out[team] = float(self._team_ratings[team][k - 1])
        return out

    def rating_diff(
        self, home: pd.Series, away: pd.Series, ratings: dict[str, float]
    ) -> np.ndarray:
        """Home-minus-away rating difference; an unrated team takes the initial rating."""
        base = self.config.initial_rating
        return np.array(
            [ratings.get(h, base) - ratings.get(a, base) for h, a in zip(home, away)],
            dtype=float,
        )


def compute_elo(matches: pd.DataFrame, cfg: EloConfig) -> EloReplay:
    """Replay Elo forward over a date-sorted match frame.

    Returns the frame with ``home_elo_pre``/``away_elo_pre`` and the post-match ratings attached,
    so a downstream model can read the pre-match difference without recomputing anything.
    """
    if not matches["date"].is_monotonic_increasing:
        raise ValueError("Elo replay needs a date-sorted frame; ordering IS the model here")

    ratings: dict[str, float] = {}
    base = cfg.initial_rating
    n = len(matches)
    home_pre = np.empty(n)
    away_pre = np.empty(n)
    home_post = np.empty(n)
    away_post = np.empty(n)

    home_goals = matches["home_goals"].to_numpy(dtype=float)
    away_goals = matches["away_goals"].to_numpy(dtype=float)
    multiplier = goal_difference_multiplier(home_goals - away_goals, cfg)

    for i, row in enumerate(matches.itertuples(index=False)):
        r_home = ratings.get(row.home_team, base)
        r_away = ratings.get(row.away_team, base)
        home_pre[i], away_pre[i] = r_home, r_away

        expected = expected_home_score(r_home + cfg.home_advantage - r_away)
        if home_goals[i] > away_goals[i]:
            actual = _WIN
        elif home_goals[i] < away_goals[i]:
            actual = _LOSS
        else:
            actual = _DRAW

        change = cfg.k * multiplier[i] * (actual - expected)
        ratings[row.home_team] = r_home + change
        ratings[row.away_team] = r_away - change   # zero-sum: total rating is conserved
        home_post[i], away_post[i] = ratings[row.home_team], ratings[row.away_team]

    out = matches.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["home_elo_post"] = home_post
    out["away_elo_post"] = away_post
    out["elo_diff_pre"] = home_pre - away_pre
    return EloReplay(history=out, final_ratings=dict(ratings), config=cfg)
