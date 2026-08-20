"""Match-context covariates — the seam for terms that describe the fixture, not the teams.

Everything the production model knows about a match is *who is playing*. This module lets a match
also carry properties of its **place in the calendar**: how long each side has rested, how many
matches each has just played, and whether each is carrying a European midweek commitment the
league fixture list cannot see.

Three shapes are available, and they are three different constructs rather than three encodings of
one:

* **rest** — days since that team's previous league match, within the season, clipped.
* **congestion** — league matches that team played in a trailing window. A club that played twice
  in eight days and one that played once are both "four days rested" by the first measure.
* **euro** — whether that club qualified for European football this season, restricted to the
  weeks UEFA actually plays. Its job is not to measure strength; see below.

How a covariate enters the rates
--------------------------------
A context term is a *temporary strength adjustment*, so it enters exactly where strength does::

    diff  (one parameter b, the encoding the research report recommends)
        log lam += b * (v_home - v_away)
        log mu  -= b * (v_home - v_away)

    split (two parameters, attack and defence free)
        log lam += b_att * v_home - b_def * v_away
        log mu  += b_att * v_away - b_def * v_home

``diff`` is ``split`` with ``b_att == b_def``, which is worth seeing plainly: the differential form
is not a separate model, it is the restriction that a tired side scores less by exactly as much as
it concedes more. Under that restriction ``log lam + log mu`` is untouched, so the term moves *who
wins* without moving *how many goals* — and the ``split`` arm exists to test whether that
restriction is one the data accepts.

Rest is computed from league matches only, which is a measurement error, not an approximation
---------------------------------------------------------------------------------------------
The corpus is football-data.co.uk's E0-E3. Cup ties and European matches are not in it and cannot
be derived from it. So a club that played in Europe on Thursday and in the league on Sunday appears
here as having rested seven days when it rested three. **The error is not random: it falls on
exactly the clubs whose congestion the arm is trying to measure**, and it always points the same
way — European clubs look more rested than they are.

That is what the ``euro`` term is for. It is a *qualification* proxy, derived from the previous
season's final table of the prediction division, and restricted to the weeks UEFA plays. It cannot
observe whether a specific midweek match happened; what it can do is give the fit somewhere to put
the systematic part of the error.

Two consequences are stated here because they belong to the term rather than to any one result.
First, misclassification attenuates: a club that qualified through a cup the corpus cannot see is
scored as not-in-Europe, which pushes the fitted coefficient toward zero rather than away from it.
Second, and more dangerous, **the flag is correlated with being good**, because the clubs that
qualify for Europe are the clubs that finished high. A term that fires for strong clubs and not
weak ones can absorb strength rather than fatigue.

The window is what makes that testable. European football is played from mid-September to the end
of May; the term is defined to be zero outside it. A fatigue effect must therefore be absent in
August, and a strength effect will not be — so the same coefficient fitted on the out-of-window
matches alone is a placebo, and a real one. Nothing here is gated on it; it is reported.

Undefined is a value
--------------------
A team's first league match of a season has no previous league match, so its rest is not a small
number or a large one — it is undefined. It is never filled in. The whole term switches off for
that match: every column is zero, the match contributes nothing to identifying the coefficient, and
the count is reported.

On this corpus that case is entirely symmetric — measured on the tuning span, all 100 matches with
undefined rest have it undefined on *both* sides, because the opening round is the opening round
for both clubs. A one-sided case would be a genuinely different question (it would mean choosing
what an unknown rest is worth relative to a known one) and is counted separately rather than
quietly averaged in.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TERM_REST = "rest"
TERM_CONGESTION = "congestion"
TERM_EURO = "euro"
TERMS: tuple[str, ...] = (TERM_REST, TERM_CONGESTION, TERM_EURO)

MODE_DIFF = "diff"
MODE_SPLIT = "split"
MODES: tuple[str, ...] = (MODE_DIFF, MODE_SPLIT)

# Suffixes for the two parameters a term carries under `split`. Under `diff` a term carries one
# parameter and the name is the term itself.
_ATTACK, _DEFENCE = "attack", "defence"

# Sentinel making every setting mandatory, so a caller cannot get a plausible-looking default that
# was never justified anywhere. Mirrors CountSpec.n_series_terms.
_CONFIGURED = -1


class CovariateError(ValueError):
    """Raised when a covariate is asked for that this module cannot build."""


@dataclass(frozen=True)
class CovariateSpec:
    """Which context terms are active and how each one is measured.

    ``terms`` empty IS the production model. Nothing below is read in that case, which is what
    makes the seam byte-identical rather than merely numerically equal when off.
    """

    terms: tuple[str, ...] = ()
    mode: str = MODE_DIFF
    rest_clip_days: int = _CONFIGURED
    rest_reference_days: int = _CONFIGURED
    congestion_window_days: int = _CONFIGURED
    euro_top_k: int = _CONFIGURED
    euro_window: tuple[str, str] = ("", "")

    def __post_init__(self) -> None:
        unknown = [t for t in self.terms if t not in TERMS]
        if unknown:
            raise CovariateError(f"unknown covariate(s) {unknown}; expected some of {TERMS}")
        if len(set(self.terms)) != len(self.terms):
            raise CovariateError(f"repeated covariate in {self.terms}")
        if self.mode not in MODES:
            raise CovariateError(f"unknown covariate mode {self.mode!r}; expected one of {MODES}")
        if self.is_inert:
            return
        if TERM_REST in self.terms and (
            self.rest_clip_days == _CONFIGURED or self.rest_reference_days == _CONFIGURED
        ):
            raise CovariateError("the rest term needs rest_clip_days and rest_reference_days")
        if TERM_CONGESTION in self.terms and self.congestion_window_days == _CONFIGURED:
            raise CovariateError("the congestion term needs congestion_window_days")
        if TERM_EURO in self.terms and (
            self.euro_top_k == _CONFIGURED or not all(self.euro_window)
        ):
            raise CovariateError("the euro term needs euro_top_k and euro_window")

    @property
    def is_inert(self) -> bool:
        return not self.terms

    def names(self) -> tuple[str, ...]:
        """Parameter names, in the order the design matrix produces them."""
        if self.mode == MODE_DIFF:
            return tuple(f"cov_{t}" for t in self.terms)
        return tuple(f"cov_{t}_{part}" for t in self.terms for part in (_ATTACK, _DEFENCE))

    def bound_key(self, name: str) -> str:
        """The ``param_bounds`` key a parameter name reads.

        Both halves of a ``split`` term share the term's box: attack and defence are the same
        quantity measured on the same scale, and giving them different boxes would be a modelling
        claim nobody has made.
        """
        for term in self.terms:
            if name == f"cov_{term}" or name.startswith(f"cov_{term}_"):
                return f"cov_{term}"
        raise CovariateError(f"{name!r} is not a parameter of {self.terms}")

    def label(self) -> str:
        return f"{'+'.join(self.terms)} ({self.mode})" if self.terms else "none"


# --- the measurements ---------------------------------------------------------------------------

def _appearances(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One entry per team-appearance, sorted by club-season and then date.

    Returned as four aligned arrays rather than a frame because both backward-looking terms are
    rebuilt at every barrier of a walk — a thousand times per arm — and a pandas groupby over
    string keys at that cadence costs more than the fit it decorates.

    ``group`` is an integer id for (club, season) and ``day`` is a day number, so a gap between
    two matches is a subtraction and a trailing window is a searchsorted.
    """
    n = len(frame)
    if n == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, np.zeros(0, dtype=bool), empty, empty
    teams = np.concatenate([frame["home_team"].to_numpy(), frame["away_team"].to_numpy()])
    seasons = np.tile(np.asarray(frame["season"]), 2)
    day = np.tile(
        pd.to_datetime(frame["date"]).to_numpy("datetime64[D]").astype(np.int64), 2
    )
    rows = np.tile(np.arange(n), 2)
    is_home = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])
    team_code = pd.factorize(teams)[0]
    season_code = pd.factorize(seasons)[0]
    group = team_code * (int(season_code.max()) + 1) + season_code
    order = np.lexsort((day, group))
    return rows[order], is_home[order], group[order], day[order]


def rest_days(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Days since each side's previous league match in the same season; NaN at a season's start.

    Within-season, because the summer break is not rest in any sense the term means: a club's
    first match of a season follows an eleven-week gap that says nothing about fatigue, and letting
    it in would make the covariate's largest values the ones that carry no information.
    """
    rows, is_home, group, day = _appearances(frame)
    gap = np.full(len(rows), np.nan)
    if len(rows) > 1:
        same = group[1:] == group[:-1]
        gap[1:][same] = (day[1:] - day[:-1])[same].astype(float)
    return _scatter(len(frame), rows, is_home, gap)


def congestion_count(frame: pd.DataFrame, *, window_days: int) -> tuple[np.ndarray, np.ndarray]:
    """League matches each side played in the ``window_days`` before this one, same season.

    Strictly before: a match never counts itself. Season-bounded for the same reason rest is.
    """
    rows, is_home, group, day = _appearances(frame)
    if len(rows) == 0:
        return _scatter(len(frame), rows, is_home, np.zeros(0))
    # One globally sorted key per appearance, so the window search is a single searchsorted over
    # the whole corpus instead of one pass per club-season. The day number occupies the low digits
    # and the club-season the high ones, which keeps the array sorted overall AND makes a search
    # for "this day minus the window" unable to reach into a neighbouring club's block.
    span = int(day.max() - day.min()) + int(window_days) + 1
    key = group.astype(np.int64) * span + (day - day.min())
    left = np.searchsorted(key, key - int(window_days), side="left")
    counts = (np.arange(len(rows)) - left).astype(float)
    return _scatter(len(frame), rows, is_home, counts)


def european_qualification(
    frame: pd.DataFrame, history: pd.DataFrame, *, top_k: int, division: str
) -> dict[tuple[str, str], bool]:
    """``(season, team) -> qualified for Europe this season``, from the previous season's table.

    Derived, not looked up. The corpus has no European fixtures and no cup results, so what can be
    computed is where a club finished in the league the season before, and English UEFA entry is
    mostly decided exactly there.

    ``top_k`` is a **causal** definition and is deliberately not tuned — the same discipline the
    empty-stadium window follows. Fitting the cut-off to whichever value scores best would be
    fitting the definition to the outcome, and the resulting term would look better than the
    evidence deserves.

    Only seasons already complete behind the barrier can produce a table, so the first season of
    any corpus flags nobody. That is the correct answer rather than a gap to fill: at that point
    the information genuinely does not exist behind the barrier.
    """
    done = history[(history["division"] == division) & history["played"]]
    if len(done) == 0:
        return {}
    table = _final_table(done)
    ranked = {season: list(group.index.get_level_values("team"))
              for season, group in table.groupby(level="season", sort=False)}
    qualified: dict[tuple[str, str], bool] = {}
    for season in sorted(set(frame["season"].to_numpy()) | set(ranked)):
        earlier = [s for s in ranked if s < season]
        if not earlier:
            continue
        for team in ranked[max(earlier)][:top_k]:
            qualified[(season, team)] = True
    return qualified


def _final_table(done: pd.DataFrame) -> pd.DataFrame:
    """League table per season, ordered as the league orders it: points, goal difference, goals."""
    draw = done["home_goals"] == done["away_goals"]
    home_pts = np.where(done["home_goals"] > done["away_goals"], 3, np.where(draw, 1, 0))
    away_pts = np.where(done["away_goals"] > done["home_goals"], 3, np.where(draw, 1, 0))
    rows = pd.concat([
        pd.DataFrame({"season": done["season"].to_numpy(), "team": done["home_team"].to_numpy(),
                      "pts": home_pts,
                      "gd": (done["home_goals"] - done["away_goals"]).to_numpy(),
                      "gf": done["home_goals"].to_numpy()}),
        pd.DataFrame({"season": done["season"].to_numpy(), "team": done["away_team"].to_numpy(),
                      "pts": away_pts,
                      "gd": (done["away_goals"] - done["home_goals"]).to_numpy(),
                      "gf": done["away_goals"].to_numpy()}),
    ], ignore_index=True)
    totals = rows.groupby(["season", "team"], sort=False, observed=True).sum()
    return totals.sort_values(["season", "pts", "gd", "gf"],
                              ascending=[True, False, False, False], kind="stable")


def in_european_window(dates: pd.Series, window: tuple[str, str]) -> np.ndarray:
    """1.0 for match dates inside UEFA's playing season, which wraps the calendar year.

    Given as ``("MM-DD", "MM-DD")`` because the window is a property of the football calendar and
    repeats every year; a list of absolute dates would have to be extended by hand every August.
    A window whose start is after its end wraps the year end, which is what UEFA's season does.
    """
    start, end = _month_day(window[0]), _month_day(window[1])
    stamps = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    inside = stamps.dt.month.to_numpy() * 100 + stamps.dt.day.to_numpy()  # MATH: MMDD encoding
    if start <= end:
        # A window that does not wrap the year end. UEFA's own season does wrap, so this branch
        # exists for its complement: the placebo needs to name the summer months that the real
        # window excludes, and it cannot do that with a wrapping interval.
        return ((inside >= start) & (inside <= end)).astype(float)
    return ((inside >= start) | (inside <= end)).astype(float)


def _month_day(value: str) -> int:
    """``"09-14"`` -> 914, so a date inside the window is an integer comparison.

    Formatting every match date as a string instead costs a fifth of a second per call, and this
    is called twice at every barrier of a walk.
    """
    month, day = value.split("-")
    return int(month) * 100 + int(day)  # MATH: MMDD encoding


def _scatter(
    n_rows: int, rows: np.ndarray, is_home: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-appearance values back to (home, away) arrays in the frame's own row order."""
    home = np.full(n_rows, np.nan)
    away = np.full(n_rows, np.nan)
    home[rows[is_home]] = values[is_home]
    away[rows[~is_home]] = values[~is_home]
    return home, away


# --- the design ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class CovariateDesign:
    """The two design matrices and the bookkeeping a report needs to read them."""

    lam: np.ndarray
    mu: np.ndarray
    names: tuple[str, ...]
    undefined: dict[str, int] = field(default_factory=dict)
    one_sided: dict[str, int] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        return len(self.names)


def per_side_values(
    frame: pd.DataFrame, history: pd.DataFrame, term: str, spec: CovariateSpec, *, division: str
) -> tuple[np.ndarray, np.ndarray]:
    """(home, away) values of one term for every row of ``frame``, NaN where undefined.

    ``history`` supplies everything the term needs from behind the barrier — the previous match
    dates that rest counts back to, and the completed seasons the European table is read from. It
    must already be truncated; this function does not re-filter, for the same reason
    ``fit_dixon_coles`` does not.
    """
    if term == TERM_REST:
        home, away = rest_days(_combine(history, frame))
        home, away = home[-len(frame):], away[-len(frame):]
        clip = float(spec.rest_clip_days)
        reference = float(spec.rest_reference_days)
        return np.clip(home, None, clip) - reference, np.clip(away, None, clip) - reference
    if term == TERM_CONGESTION:
        home, away = congestion_count(
            _combine(history, frame), window_days=spec.congestion_window_days
        )
        return home[-len(frame):], away[-len(frame):]
    if term == TERM_EURO:
        qualified = european_qualification(
            frame, _completed(history, frame), top_k=spec.euro_top_k, division=division
        )
        window = in_european_window(frame["date"], spec.euro_window)
        seasons = frame["season"].to_numpy()
        home = np.array([qualified.get((s, t), False)
                         for s, t in zip(seasons, frame["home_team"])], dtype=float)
        away = np.array([qualified.get((s, t), False)
                         for s, t in zip(seasons, frame["away_team"])], dtype=float)
        return home * window, away * window
    raise CovariateError(f"no measurement defined for covariate {term!r}")


_TABLE_COLUMNS = ("season", "division", "played", "home_team", "away_team",
                  "home_goals", "away_goals")


def _completed(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """Every result available behind the barrier, for reading final league tables from.

    At fit time the frame IS the history and this is a no-op; at prediction time the tables live
    in the history and the frame contributes nothing, because a season cannot rank itself before
    it has finished. Including both is safe in either direction for exactly that reason: the term
    for season S reads season S-1, every match of which is behind any barrier inside S.
    """
    columns = list(_TABLE_COLUMNS)
    if history is None or len(history) == 0:
        return frame[columns]
    return pd.concat([history[columns], frame[columns]], ignore_index=True)


def _combine(history: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    """History followed by the rows being described, with a fresh index.

    The backward-looking terms need each row's predecessors, and for a row being *forecast* those
    predecessors live in the history rather than in the frame. Concatenating is safe because every
    term looks strictly backwards and a club plays at most once on a matchday, so no row of
    ``frame`` can ever be another row's predecessor.
    """
    columns = ["date", "season", "home_team", "away_team"]
    if history is None or len(history) == 0:
        return frame[columns].reset_index(drop=True)
    return pd.concat([history[columns], frame[columns]], ignore_index=True)


def design(
    frame: pd.DataFrame, history: pd.DataFrame, spec: CovariateSpec, *, division: str
) -> CovariateDesign:
    """Design matrices for the home rate and the away rate, sharing one parameter vector.

    Both are ``(N, K)`` with the same ``K``: a term contributes to both rates, and a parameter that
    entered only one of them would be a claim that fatigue affects scoring but not conceding —
    which is what ``split`` measures rather than assumes.

    ``K`` is zero when the seam is off, so the contribution is an empty matrix product and the fit
    is byte-identical to one that never heard of covariates.
    """
    n = len(frame)
    if spec.is_inert:
        return CovariateDesign(np.zeros((n, 0)), np.zeros((n, 0)), ())

    lam_columns: list[np.ndarray] = []
    mu_columns: list[np.ndarray] = []
    undefined: dict[str, int] = {}
    one_sided: dict[str, int] = {}
    for term in spec.terms:
        home, away = per_side_values(frame, history, term, spec, division=division)
        missing_home, missing_away = np.isnan(home), np.isnan(away)
        blank = missing_home | missing_away
        undefined[term] = int(blank.sum())
        one_sided[term] = int((missing_home ^ missing_away).sum())
        # Undefined switches the term off for that match rather than filling a value in.
        home = np.where(blank, 0.0, np.nan_to_num(home))
        away = np.where(blank, 0.0, np.nan_to_num(away))
        if spec.mode == MODE_DIFF:
            differential = home - away
            lam_columns.append(differential)
            mu_columns.append(-differential)
        else:
            lam_columns.extend([home, -away])
            mu_columns.extend([away, -home])
    return CovariateDesign(
        np.column_stack(lam_columns),
        np.column_stack(mu_columns),
        spec.names(),
        undefined=undefined,
        one_sided=one_sided,
    )
