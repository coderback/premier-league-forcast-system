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

A fourth term breaks the pattern on purpose, and lives here because the machinery is identical:

* **sot_form** — that side's mean shots-on-target differential over its previous ``window``
  recorded league matches. It describes the TEAM, not the calendar. It is the external-regressor
  construction of the shots hypothesis: shots sit on top of the goal-based strengths as one
  coefficient, rather than inside the likelihood as a second observation channel sharing the same
  strengths, which is what Arm 4 (``dc+sot``) did and was null at +0.00003. Read
  ``trailing_differential`` for what counts as a recorded match.

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
TERM_SOT_FORM = "sot_form"
# The same form term with a recency bound, and the bounded term also fed by Championship matches.
# Both exist because `sot_form` reads E0 history only and has no time limit, so a promoted club's
# form comes from its last Premier League spell however long ago that was (Coventry's 2026-27 form
# opened on its 2000-01 shots). See NOTES.md, 2026-11-17, for the pre-registration.
TERM_SOT_FORM_RECENT = "sot_form_recent"
TERM_SOT_FORM_E1 = "sot_form_e1"
TERMS: tuple[str, ...] = (
    TERM_REST, TERM_CONGESTION, TERM_EURO, TERM_SOT_FORM, TERM_SOT_FORM_RECENT, TERM_SOT_FORM_E1,
)
_FORM_TERMS: tuple[str, ...] = (TERM_SOT_FORM, TERM_SOT_FORM_RECENT, TERM_SOT_FORM_E1)
_BOUNDED_FORM_TERMS: tuple[str, ...] = (TERM_SOT_FORM_RECENT, TERM_SOT_FORM_E1)

# The source columns the shots-form term reads, home then away.
SOT_COLUMNS: tuple[str, str] = ("home_sot", "away_sot")

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
    sot_form_window: int = _CONFIGURED
    sot_form_recency_days: int = _CONFIGURED
    sot_form_e1_min_promotions: int = _CONFIGURED
    # Championship matches for `sot_form_e1`, already truncated at the barrier by the walk. Data
    # rather than a setting, so it is excluded from equality and repr: two specs that say the same
    # thing about what a term MEANS compare equal whatever rows they were handed.
    e1_rows: pd.DataFrame | None = field(default=None, compare=False, repr=False)

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
        if any(t in self.terms for t in _FORM_TERMS) and self.sot_form_window < 1:
            raise CovariateError("a shots-form term needs a sot_form_window of at least 1")
        if any(t in self.terms for t in _BOUNDED_FORM_TERMS) and self.sot_form_recency_days < 1:
            raise CovariateError("a bounded form term needs sot_form_recency_days of at least 1")
        if TERM_SOT_FORM_E1 in self.terms and self.sot_form_e1_min_promotions < 1:
            raise CovariateError("the sot_form_e1 term needs sot_form_e1_min_promotions")

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


def trailing_differential(
    frame: pd.DataFrame, home_col: str, away_col: str, *, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Each side's mean ``own - opponent`` count over its previous ``window`` recorded matches.

    Strictly before: a match never counts itself. NOT season-bounded, unlike rest and congestion.
    Those measure a gap in the calendar, which the summer resets; this measures how well a side is
    playing, which the summer does not, and a window of twenty spans a season boundary for half of
    every season's matches.

    A match whose counts are missing is skipped rather than read as zero: before 2000-01 the source
    records no shots at all, and a zero there would be a claim that two sides produced nothing.
    A side with no recorded match behind it is undefined, NaN, and the design switches the term off.

    Vectorised over one sort of the appearances, for the same reason ``_appearances`` is: the term
    is rebuilt at every barrier of a walk.
    """
    n = len(frame)
    if n == 0:
        return np.zeros(0), np.zeros(0)
    home_v = pd.to_numeric(frame[home_col], errors="coerce").to_numpy(dtype=float)
    away_v = pd.to_numeric(frame[away_col], errors="coerce").to_numpy(dtype=float)
    team = pd.factorize(np.concatenate([frame["home_team"].to_numpy(),
                                        frame["away_team"].to_numpy()]))[0]
    day = np.tile(pd.to_datetime(frame["date"]).to_numpy("datetime64[D]").astype(np.int64), 2)
    diff = np.concatenate([home_v - away_v, away_v - home_v])
    rows = np.tile(np.arange(n), 2)
    is_home = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])
    order = np.lexsort((day, team))
    team, diff, rows, is_home = team[order], diff[order], rows[order], is_home[order]

    values = np.full(len(team), np.nan)
    valid = ~np.isnan(diff)
    if not valid.any():
        return _scatter(n, rows, is_home, values)

    # Rolling mean over the recorded entries only, each one INCLUDING itself ...
    v_team, v_diff = team[valid], diff[valid]
    k = np.arange(len(v_diff))
    starts = np.r_[0, np.flatnonzero(v_team[1:] != v_team[:-1]) + 1]
    group_start = np.repeat(starts, np.diff(np.r_[starts, len(k)]))
    cum = np.r_[0.0, np.cumsum(v_diff)]
    first = np.maximum(k + 1 - window, group_start)
    through = (cum[k + 1] - cum[first]) / (k + 1 - first)

    # ... then each appearance reads the latest such mean strictly before it, for the same side.
    last = np.cumsum(valid) - valid - 1
    same_side = (last >= 0) & (v_team[np.clip(last, 0, None)] == team)
    values[same_side] = through[last[same_side]]
    return _scatter(n, rows, is_home, values)


def _side_appearances(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """One entry per side of every row: team, day number, own-minus-opponent SoT, row, is_home."""
    n = len(frame)
    home_v = pd.to_numeric(frame[SOT_COLUMNS[0]], errors="coerce").to_numpy(dtype=float)
    away_v = pd.to_numeric(frame[SOT_COLUMNS[1]], errors="coerce").to_numpy(dtype=float)
    day = pd.to_datetime(frame["date"]).to_numpy("datetime64[D]").astype(np.int64)
    return {
        "team": np.concatenate([frame["home_team"].to_numpy(), frame["away_team"].to_numpy()]),
        "day": np.tile(day, 2),
        "diff": np.concatenate([home_v - away_v, away_v - home_v]),
        "row": np.tile(np.arange(n), 2),
        "is_home": np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)]),
    }


def trailing_mean_bounded(
    team: np.ndarray, day: np.ndarray, diff: np.ndarray, *, window: int,
    max_age_days: int | None,
) -> np.ndarray:
    """Per entry: the mean ``diff`` of that team's previous ``window`` recorded entries.

    Recorded means ``diff`` is not NaN. Previous means an EARLIER DAY, so an entry never reads
    itself. With ``max_age_days`` set, only entries played within that many days before count; a
    team with none is NaN, which the design turns into "term off for this match".

    Long-format on purpose: the Championship arm feeds E0 and E1 entries through the same window,
    and the only thing that marks an E1 entry is that nothing reads a value AT it.

    With ``max_age_days=None`` this is ``trailing_differential``'s definition exactly, which a unit
    test asserts, so the bounded arm differs from production in the bound and in nothing else.
    """
    out = np.full(len(team), np.nan)
    recorded = ~np.isnan(diff)
    if not recorded.any():
        return out
    codes = pd.factorize(team)[0].astype(np.int64)
    base = int(day.min())
    reach = int(max_age_days or 0)
    # One sorted key per entry: the team in the high digits, the day in the low ones, with room
    # for a lookback of `reach` days, so a window search can never cross into another team's block.
    span = int(day.max()) - base + reach + 1
    key = codes * span + (day - base)
    order = np.argsort(key[recorded], kind="stable")
    rec_key = key[recorded][order]
    cum = np.r_[0.0, np.cumsum(diff[recorded][order])]

    upto = np.searchsorted(rec_key, key, side="left")  # recorded entries on an earlier day
    start = np.maximum(upto - int(window), np.searchsorted(rec_key, codes * span, side="left"))
    if max_age_days is not None:
        start = np.maximum(start, np.searchsorted(rec_key, key - reach, side="left"))
    count = upto - start
    ok = count > 0
    out[ok] = (cum[upto[ok]] - cum[start[ok]]) / count[ok]
    return out


def promotion_map(
    e0: pd.DataFrame, e1: pd.DataFrame, *, window: int, min_promotions: int
) -> tuple[float, float] | None:
    """``(a, b)`` mapping a Championship SoT differential onto the Premier League's scale.

    Fitted by OLS over every COMPLETED promotion in the rows given: a club in an E0 season that was
    not in the previous E0 season, whose first ``window`` E0 matches that season all have recorded
    shots (y, their mean differential), and which has ``window`` recorded E1 matches before its
    first E0 match (x, their mean). Shots only, never results. The rows are what the walk handed
    over, already behind the barrier, so "completed" is decided by what exists rather than by date.

    A level shift as well as a slope, because a promoted side out-shoots Championship opponents and
    is out-shot in the Premier League; no pure scale turns a positive into a negative.

    Returns None below ``min_promotions`` pairs, and the arm then reads E0 only.
    """
    if e0 is None or e1 is None or len(e0) == 0 or len(e1) == 0 or "season" not in e0:
        return None
    # Called at fit time and again at prediction time on the same rows, at every barrier of a walk.
    # Keyed on what determines the answer, so a different frame can never read a stale map.
    memo_key = (_content_digest(e0, ("season", *_FORM_COLUMNS)),
                _content_digest(e1, _FORM_COLUMNS), int(window), int(min_promotions))
    if memo_key in _MAP_MEMO:
        return _MAP_MEMO[memo_key]
    if len(_MAP_MEMO) >= MAP_MEMO_LIMIT:
        _MAP_MEMO.clear()
    _MAP_MEMO[memo_key] = result = _fit_promotion_map(
        e0, e1, window=window, min_promotions=min_promotions)
    return result


def _content_digest(frame: pd.DataFrame, columns: tuple[str, ...]) -> tuple[int, int]:
    """Row count and an order-sensitive content hash of the columns the map reads."""
    hashed = pd.util.hash_pandas_object(frame[list(columns)], index=False).to_numpy()
    return len(frame), int(np.bitwise_xor.reduce(hashed * np.arange(1, len(hashed) + 1,
                                                                    dtype=np.uint64)))


# Entries kept before the memo is emptied. A walk asks for one map per barrier, twice, and over at
# most ~1,200 barriers; this is a memory ceiling, not a tuning choice, and any value above one
# barrier's worth of calls gives the same answers.
MAP_MEMO_LIMIT = 4096
_MAP_MEMO: dict[tuple, tuple[float, float] | None] = {}


def _fit_promotion_map(
    e0: pd.DataFrame, e1: pd.DataFrame, *, window: int, min_promotions: int
) -> tuple[float, float] | None:
    seasons = sorted(e0["season"].unique())
    members = {s: set(e0.loc[e0["season"] == s, "home_team"]) for s in seasons}
    side0 = _side_appearances(e0)
    side0["season"] = np.tile(e0["season"].to_numpy(), 2)
    long0 = pd.DataFrame(side0).sort_values(["team", "day"], kind="stable")
    side1 = _side_appearances(e1)
    long1 = pd.DataFrame(side1)
    long1 = long1[~np.isnan(long1["diff"])].sort_values(["team", "day"], kind="stable")
    xs, ys = [], []
    for previous, season in zip(seasons[:-1], seasons[1:]):
        for club in members[season] - members[previous]:
            first = long0[(long0["team"] == club) & (long0["season"] == season)].head(window)
            if len(first) < window or first["diff"].isna().any():
                continue
            before = long1[(long1["team"] == club) & (long1["day"] < first["day"].min())]
            before = before.tail(window)
            if len(before) < window:
                continue
            xs.append(float(before["diff"].mean()))
            ys.append(float(first["diff"].mean()))
    if len(xs) < min_promotions:
        return None
    b, a = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return float(a), float(b)


def _bounded_form(
    frame: pd.DataFrame, history: pd.DataFrame, spec: CovariateSpec, *, with_e1: bool
) -> tuple[np.ndarray, np.ndarray]:
    """(home, away) bounded form for the rows of ``frame``; optionally fed by mapped E1 entries."""
    predicting = history is not None and len(history) > 0
    rows = frame.assign(**{c: np.nan for c in SOT_COLUMNS}) if predicting else frame
    e0 = _combine(history, rows, columns=_FORM_COLUMNS)
    side = _side_appearances(e0)
    team, day, diff = side["team"], side["day"], side["diff"]
    if with_e1:
        if spec.e1_rows is None:
            raise CovariateError(
                "sot_form_e1 was asked for but no Championship rows were handed to the spec; "
                "refusing to fall back to E0 only, which would silently make this arm sot_form_recent"
            )
        # The map is fitted on what is behind the barrier: the fit's own frame at fit time, the
        # history at prediction time -- the same rows either way.
        mapping = promotion_map(
            history if predicting else frame, spec.e1_rows,
            window=spec.sot_form_window, min_promotions=spec.sot_form_e1_min_promotions,
        )
        if mapping is not None:
            extra = _side_appearances(spec.e1_rows[list(_FORM_COLUMNS)])
            a, b = mapping
            team = np.concatenate([team, extra["team"]])
            day = np.concatenate([day, extra["day"]])
            diff = np.concatenate([diff, a + b * extra["diff"]])
    values = trailing_mean_bounded(
        team, day, diff, window=spec.sot_form_window, max_age_days=spec.sot_form_recency_days
    )[: len(side["team"])]
    home, away = _scatter(len(e0), side["row"], side["is_home"], values)
    return home[-len(frame):], away[-len(frame):]


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
    if term == TERM_SOT_FORM:
        # The rows being FORECAST never contribute their own shots: those are outcomes. Within a
        # single-date walk block a club cannot precede itself anyway, but a caller predicting
        # several dates at once would otherwise read the earlier ones' shots into the later.
        predicting = history is not None and len(history) > 0
        rows = frame.assign(**{c: np.nan for c in SOT_COLUMNS}) if predicting else frame
        home, away = trailing_differential(
            _combine(history, rows, columns=_FORM_COLUMNS), *SOT_COLUMNS,
            window=spec.sot_form_window,
        )
        return home[-len(frame):], away[-len(frame):]
    if term == TERM_SOT_FORM_RECENT:
        return _bounded_form(frame, history, spec, with_e1=False)
    if term == TERM_SOT_FORM_E1:
        return _bounded_form(frame, history, spec, with_e1=True)
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


# The calendar terms need the season; the form term does not, and must not demand it: `pl predict`
# and `pl fit` describe fixtures typed at the command line, which carry a date and two teams only.
_CALENDAR_COLUMNS: tuple[str, ...] = ("date", "season", "home_team", "away_team")
FORM_COLUMNS: tuple[str, ...] = ("date", "home_team", "away_team", *SOT_COLUMNS)
_FORM_COLUMNS = FORM_COLUMNS


def _combine(
    history: pd.DataFrame, frame: pd.DataFrame, *,
    columns: tuple[str, ...] = _CALENDAR_COLUMNS,
) -> pd.DataFrame:
    """History followed by the rows being described, with a fresh index.

    The backward-looking terms need each row's predecessors, and for a row being *forecast* those
    predecessors live in the history rather than in the frame. Concatenating is safe because every
    term looks strictly backwards and a club plays at most once on a matchday, so no row of
    ``frame`` can ever be another row's predecessor.
    """
    columns = list(columns)
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
