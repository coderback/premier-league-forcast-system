"""League tables: how results become standings, and how standings break ties.

Two jobs, kept apart because they fail in different ways.

**Accumulating results into a table** is arithmetic and is shared by the simulator (which
accumulates simulated scorelines) and the validator (which accumulates the ones that happened).
Both go through :func:`standings` so a discrepancy between "what we forecast" and "what we score
against" cannot come from two different ideas of what a league table is.

**Ordering the table** is a rule, and the Premier League's rule is narrower than most:

    points, then goal difference, then goals scored, and if clubs are still equal they occupy the
    same position -- unless the position decides the title, relegation or qualification for another
    competition, in which case they play off at a neutral venue.

Head-to-head is **not** a Premier League tiebreak, unlike La Liga and Serie A, and encoding one
here would silently change every simulated season. The playoff clause is the reason this module
takes a random generator: a one-off match on neutral ground between two clubs that finished level
on points, goal difference and goals scored is as close to a coin flip as this project can
justify, so the tie is broken uniformly at random *within the replicate* and the number of
replicates where that happened is reported rather than hidden. A simulator that instead broke such
ties alphabetically, or by array order, would hand a systematic advantage to whichever club the
corpus happened to list first.

**Points deductions are not derivable from results** and therefore are not computed here. They are
supplied by the caller from :mod:`plmodel.config` (see the ``season.points_deductions`` block), and
:func:`standings` applies them as an explicit adjustment so a table that carries one says so.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Three points for a win, one for a draw. The Premier League has used this since its founding, so
# it is a property of the competition rather than a tunable, and it belongs in the code that
# defines what a league table is.
WIN_POINTS = 3
DRAW_POINTS = 1
LOSS_POINTS = 0

# The ordering keys, most significant first. Named so the ordering rule is readable at the call
# site and so a future league with a different rule has one obvious place to disagree.
ORDER_KEYS = ("points", "goal_difference", "goals_for")

TOP = "top"
BOTTOM = "bottom"
ENDS = (TOP, BOTTOM)


class SeasonError(ValueError):
    """Raised when a frame cannot be read as a season: duplicated fixtures, missing scores."""


@dataclass(frozen=True)
class Question:
    """A positional question about a final table, e.g. "finish in the top 4".

    Positional on purpose. Which positions earn the title, relegation or a European place is a
    season-specific rule the corpus does not carry -- England had three Champions League places
    before 2002-03 and five in 2023-24 and 2024-25, and 1994-95 relegated four clubs to shrink the
    division from 22 to 20. The simulator answers "where does this club finish"; mapping a finish
    to a competition is a separate claim, made in config.yaml where it can be dated.
    """

    name: str
    end: str
    places: int

    def __post_init__(self) -> None:
        if self.end not in ENDS:
            raise SeasonError(f"question {self.name!r}: end must be one of {ENDS}, got {self.end!r}")
        if self.places < 1:
            raise SeasonError(f"question {self.name!r}: places must be at least 1")

    def cut(self, n_teams: int) -> int:
        """The 0-based index of the first position *outside* this question's set.

        For "top 4" that is 4; for "bottom 3" in a 20-team league it is 17. The boundary between
        this index and the one before it is where the playoff clause can fire.
        """
        if self.places >= n_teams:
            raise SeasonError(
                f"question {self.name!r} asks for {self.places} of {n_teams} places, which every "
                "club satisfies; that is not a question"
            )
        return self.places if self.end == TOP else n_teams - self.places

    def satisfied(self, positions: np.ndarray, n_teams: int) -> np.ndarray:
        """Boolean mask over 0-based finishing positions."""
        cut = self.cut(n_teams)
        return positions < cut if self.end == TOP else positions >= cut

    def label(self) -> str:
        return f"{self.end} {self.places}"


def points_for(scored: np.ndarray, conceded: np.ndarray) -> np.ndarray:
    """Match points from a scoreline, for either side."""
    return np.where(scored > conceded, WIN_POINTS,
                    np.where(scored == conceded, DRAW_POINTS, LOSS_POINTS))


def standings(
    results: pd.DataFrame,
    *,
    teams: tuple[str, ...] | None = None,
    deductions: dict[str, int] | None = None,
) -> pd.DataFrame:
    """A league table from played results, ordered as the league orders it.

    ``teams`` pins the roster, so a club that has played no matches yet still appears on zero
    points rather than vanishing from its own division's table. Without it a table built before a
    ball is kicked would be empty, which is not the same statement as "everyone is on nothing".
    """
    frame = results
    if teams is None:
        teams = tuple(sorted(set(frame["home_team"]) | set(frame["away_team"])))
    if len(frame) and frame[["home_goals", "away_goals"]].isna().to_numpy().any():
        raise SeasonError("standings() was given a match with no score; filter to played first")

    home_goals = frame["home_goals"].to_numpy(dtype=float)
    away_goals = frame["away_goals"].to_numpy(dtype=float)
    rows = pd.concat(
        [
            pd.DataFrame({
                "team": frame["home_team"].to_numpy(),
                "played": 1,
                "points": points_for(home_goals, away_goals),
                "goals_for": home_goals,
                "goals_against": away_goals,
            }),
            pd.DataFrame({
                "team": frame["away_team"].to_numpy(),
                "played": 1,
                "points": points_for(away_goals, home_goals),
                "goals_for": away_goals,
                "goals_against": home_goals,
            }),
        ],
        ignore_index=True,
    )
    table = (
        rows.groupby("team", sort=False, observed=True).sum()
        .reindex(list(teams))
        .fillna(0)
    )
    applied = {t: int((deductions or {}).get(t, 0)) for t in teams}
    table["deduction"] = [applied[t] for t in teams]
    table["points"] = table["points"] - table["deduction"]
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table = table.reset_index().rename(columns={"index": "team"})
    for column in ("played", "points", "goals_for", "goals_against", "goal_difference"):
        table[column] = table[column].astype(int)
    return table.sort_values(list(ORDER_KEYS), ascending=False, kind="stable").reset_index(drop=True)


def rank(
    points: np.ndarray,
    goal_difference: np.ndarray,
    goals_for: np.ndarray,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """0-based finishing positions for a batch of tables, with ties broken at random.

    Every input is ``(n_replicates, n_teams)``. Returns the positions in the same shape and a
    boolean ``(n_replicates, n_teams)`` mask marking, for each *position*, whether the club that
    took it was level with the club immediately above it on all three ordering keys -- which is
    exactly the situation the competition's playoff clause covers, and which the caller needs so
    it can report how often a reported probability rests on a coin flip rather than on football.
    """
    keys = (points, goal_difference, goals_for)
    shapes = {np.shape(k) for k in keys}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise SeasonError(f"rank() needs matching 2-D arrays; got {[np.shape(k) for k in keys]}")

    coin = rng.random(np.shape(points))
    # lexsort's LAST key is the primary one and it sorts ascending, so the keys go least
    # significant first and the result is reversed to put the champion at position 0.
    order = np.lexsort((coin, goals_for, goal_difference, points), axis=-1)[:, ::-1]
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.arange(order.shape[1]), axis=-1)

    level = np.zeros(order.shape, dtype=bool)
    if order.shape[1] > 1:
        ordered = [np.take_along_axis(k, order, axis=-1) for k in keys]
        same = np.ones(ordered[0][:, 1:].shape, dtype=bool)
        for values in ordered:
            same &= values[:, 1:] == values[:, :-1]
        level[:, 1:] = same
    return positions, level
