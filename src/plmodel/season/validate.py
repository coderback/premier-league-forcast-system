"""Does the season simulator's long-horizon forecast actually happen?

Almost every published football model *reports* title and relegation probabilities; the research
report's survey of this literature names multi-season validation of those probabilities as an open
gap, because it needs many season-years and nobody has them. This module is this project's attempt
at it, and the first thing it has to say is how thin the evidence necessarily is:
**a decade is ten independent season-years.** Twenty clubs give twenty rows per season, but exactly
one of them wins the title, so those rows carry nothing like twenty seasons of information. Every
interval here is therefore a **cluster bootstrap over seasons** -- see
:func:`plmodel.eval.metrics.paired_delta_clustered` -- and the row count is reported beside the
season count so a reader can see which of the two is the real sample size.

**What is scored.** Each question in the spec ("finish in the top 4") becomes a binary forecast per
club per barrier, scored by Brier and by log loss. Brier is the headline: it is bounded, so a
confident forecast of an event that then happens cannot dominate the mean, whereas log loss on a
50,000-replicate estimate of a probability the model puts near zero is mostly a report on the
floor applied to it. The floor is a decision and lives in config.yaml.

**The acceptance rule is not applied here, and the report says so.** That rule governs match
forecasts scored by RPS against a de-vigged market; a season forecast is a different quantity on a
different unit, and this corpus carries no outright market to be a second gate. What is reused is
the rule's *construction* -- a paired bootstrap, and a delta that must have a 95% interval
excluding zero or P(better) >= 0.95 -- which is what makes a comparison between two simulator
settings decidable rather than a matter of taste.

**Points deductions are applied to the actual table and to nothing else.** A deduction is a
disciplinary fact the model cannot forecast and the corpus does not record. Scoring against a
table that ignores one would credit the model for a relegation that the results did not cause;
scoring against a table that includes one charges the model for something it could not know. The
figures come from :mod:`plmodel.config`, and which seasons they reach is recorded rather than
assumed: on this corpus exactly one deduction changes any question's outcome, and it falls in the
tuning span.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from plmodel.eval.backtest import training_frame
from plmodel.eval.metrics import paired_delta_clustered
from plmodel.model.production import production_fit
from plmodel.season.simulate import SeasonForecast, SeasonSpec, simulate_season
from plmodel.season.table import Question, SeasonError, standings

# Column set a season validation needs. Named here so the loader can be narrowed to it: the corpus
# carries two hundred columns and this reads eight of them.
MATCH_COLUMNS = ("date", "division", "season", "played",
                 "home_team", "away_team", "home_goals", "away_goals")


@dataclass(frozen=True)
class SeasonBarrier:
    """A point inside one season at which a forecast of the rest of it is made."""

    season: str
    week: int
    date: pd.Timestamp
    n_played: int
    n_remaining: int

    @property
    def horizon(self) -> float:
        return self.n_remaining / (self.n_played + self.n_remaining)

    def label(self) -> str:
        return f"{self.season} w{self.week:02d}"


def matchweek_barriers(
    rows: pd.DataFrame, *, weeks: tuple[int, ...], fixtures_per_week: int
) -> list[SeasonBarrier]:
    """Barriers after the given number of completed matchweeks, snapped to a date boundary.

    A Premier League matchweek is ten fixtures, but they are not all played on one day and a
    barrier that fell inside a date would put two halves of the same afternoon on opposite sides
    of it. So the target index picks a fixture and the barrier is that fixture's *date*, which
    moves the boundary by at most a matchday and keeps the split honest.
    """
    if rows.empty:
        raise SeasonError("cannot place barriers in a season with no fixtures")
    dates = pd.to_datetime(rows["date"]).to_numpy()
    out: list[SeasonBarrier] = []
    for week in weeks:
        target = week * fixtures_per_week
        if target >= len(rows):
            raise SeasonError(
                f"matchweek {week} is past the end of a {len(rows)}-fixture season"
            )
        barrier = pd.Timestamp(dates[target])
        played = int((dates < barrier.to_datetime64()).sum())
        out.append(SeasonBarrier(
            season=str(rows["season"].iloc[0]), week=week, date=barrier,
            n_played=played, n_remaining=len(rows) - played,
        ))
    return out


def actual_table(
    rows: pd.DataFrame, *, teams: tuple[str, ...], deductions: dict[str, int] | None = None
) -> pd.DataFrame:
    """The season's real final table, with 1-based positions and a flag for unresolved ties.

    The Premier League's rule leaves clubs level on points, goal difference and goals scored in the
    same position unless the position decides something, in which case they play off. No such
    playoff has been needed, but the flag exists rather than a quiet stable sort so that if one
    ever is, this reports an ambiguous outcome instead of inventing a winner.
    """
    table = standings(rows, teams=teams, deductions=deductions)
    table.insert(0, "position", np.arange(len(table)) + 1)
    keys = table[["points", "goal_difference", "goals_for"]].to_numpy()
    level = np.zeros(len(table), dtype=bool)
    level[1:] = (keys[1:] == keys[:-1]).all(axis=1)
    table["level_with_the_club_above"] = level
    return table


def promoted_clubs(matches: pd.DataFrame, season: str) -> frozenset[str]:
    """Clubs in this season's division that were not in the previous season's.

    The season-level analogue of the promoted-team calibration slice the playbook asks for. It is
    where a per-team-parameter model is weakest -- three clubs arrive each year with no top-flight
    history, or with a stale one -- and where the market's information advantage is largest, so a
    season forecast that is badly calibrated overall may be badly calibrated only here.

    The first season in the corpus has no predecessor and therefore no promoted clubs. That is a
    missing value, not an empty set, but the two coincide in what they license: nothing about that
    season's promoted clubs is claimed, because none can be identified.
    """
    seasons = sorted(matches["season"].unique())
    if season not in seasons or seasons.index(season) == 0:
        return frozenset()
    previous = seasons[seasons.index(season) - 1]
    rows = matches[matches["season"] == season]
    before = matches[matches["season"] == previous]
    here = set(rows["home_team"]) | set(rows["away_team"])
    then = set(before["home_team"]) | set(before["away_team"])
    return frozenset(here - then)


def score_forecast(
    forecast: SeasonForecast,
    actual: pd.DataFrame,
    *,
    questions: tuple[Question, ...],
    prob_floor: float,
    seed: int,
    promoted: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """One row per club per question, plus the club's points PIT, with losses attached."""
    n_teams = len(forecast.teams)
    positions = actual.set_index("team")["position"].reindex(list(forecast.teams)).to_numpy() - 1
    if np.isnan(positions.astype(float)).any():
        raise SeasonError("the actual table is missing a club the forecast covers")
    points = actual.set_index("team")["points"].reindex(list(forecast.teams)).to_numpy()

    probabilities = forecast.probabilities.set_index("team")
    cold = set(forecast.diagnostics.get("cold_start_teams", ()))
    rows: list[dict[str, object]] = []
    for question in questions:
        happened = question.satisfied(positions, n_teams).astype(float)
        p = probabilities[question.name].reindex(list(forecast.teams)).to_numpy()
        clipped = np.clip(p, prob_floor, 1.0 - prob_floor)
        for i, team in enumerate(forecast.teams):
            rows.append({
                "season": forecast.season,
                "barrier": forecast.barrier,
                "horizon": forecast.horizon,
                "uncertainty": forecast.uncertainty,
                "team": team,
                "question": question.name,
                "probability": float(p[i]),
                "outcome": float(happened[i]),
                "brier": float((p[i] - happened[i]) ** 2),
                "log_loss": float(-(happened[i] * np.log(clipped[i])
                                    + (1.0 - happened[i]) * np.log(1.0 - clipped[i]))),
                "floored": bool(p[i] < prob_floor or p[i] > 1.0 - prob_floor),
                "cold_start": team in cold,
                "promoted": team in promoted,
                "final_position": int(positions[i]) + 1,
            })

    frame = pd.DataFrame(rows)
    frame["points_pit"] = frame["team"].map(
        dict(zip(forecast.teams, _points_pit(forecast, points, seed=seed)))
    )
    frame["points_log_score"] = frame["team"].map(
        dict(zip(forecast.teams, points_log_score(forecast, points, floor=prob_floor)))
    )
    return frame


def points_log_score(
    forecast: SeasonForecast, actual_points: np.ndarray, *, floor: float
) -> np.ndarray:
    """``-log P(actual final points)`` under each club's simulated points distribution.

    The PIT says whether the predictive distribution is the right *width*; this scores it. It is
    strictly proper, it reads the whole distribution rather than a binary slice of it, and unlike
    the Kolmogorov-Smirnov distance it is one number per club-season, which is what lets the
    clustered bootstrap put an interval on a difference between two settings.

    The floor bites far less here than on a question probability: final points spread over some
    forty values, so a club's actual total typically carries a few percent of the replicates rather
    than a few ten-thousandths. Where it does bite the club finished somewhere the simulation never
    reached in ten thousand tries, which is a real statement about the forecast and not an artefact
    of the replicate count.
    """
    counts = forecast.points_counts
    index = np.clip(np.asarray(actual_points) - forecast.points_floor, 0, counts.shape[1] - 1)
    mass = counts[np.arange(counts.shape[0]), index] / forecast.n_replicates
    return -np.log(np.clip(mass, floor, 1.0))


def _points_pit(forecast: SeasonForecast, actual_points: np.ndarray, *, seed: int) -> np.ndarray:
    """Randomised probability integral transform of each club's actual final points.

    ``P(sim < actual) + V * P(sim == actual)`` with ``V`` uniform. The randomisation is what makes
    the transform exactly uniform for a *discrete* forecast; without it a points distribution
    concentrated on a few values would look miscalibrated however right it was. This is the
    sharpest single check on the simulator, because it reads the whole predictive distribution
    rather than one binary slice of it, and it is the quantity the drift setting is calibrated on.

    The generator is seeded the same way for every forecast, so two settings compared on the same
    club-season see the same uniform draw. The randomisation is then paired out of the comparison
    instead of adding noise to it.
    """
    counts = forecast.points_counts
    cumulative = np.cumsum(counts, axis=1) / forecast.n_replicates
    index = np.clip(np.asarray(actual_points) - forecast.points_floor, 0, counts.shape[1] - 1)
    rows = np.arange(counts.shape[0])
    at_or_below = cumulative[rows, index]
    exactly = counts[rows, index] / forecast.n_replicates
    below = at_or_below - exactly
    return below + np.random.default_rng(seed).random(counts.shape[0]) * exactly


def pit_summary(values: np.ndarray) -> dict[str, float]:
    """How far a set of PIT values is from uniform, and in which direction.

    ``ks`` is the Kolmogorov-Smirnov distance from ``U(0,1)``. ``tail_mass`` is the share falling
    in the outer tenth at either end, which is 0.2 under a calibrated forecast, above it when the
    predictive distribution is too narrow, and below it when it is too wide -- the direction is the
    part that matters, because an over-confident season simulator and an over-cautious one both
    show up as a large ``ks``.
    """
    u = np.sort(np.asarray(values, dtype=float))
    n = len(u)
    if n == 0:
        raise SeasonError("cannot summarise an empty PIT sample")
    grid = np.arange(1, n + 1) / n
    ks = float(np.max(np.abs(np.concatenate([grid - u, u - (grid - 1.0 / n)]))))
    outer = 0.1  # MATH: the outer tenth at each end, so a calibrated sample puts 0.2 here
    return {
        "n": n,
        "ks": ks,
        "tail_mass": float(np.mean((u < outer) | (u > 1.0 - outer))),
        "mean": float(u.mean()),
    }


def run_span(
    matches: pd.DataFrame,
    cfg,
    *,
    seasons: tuple[str, ...],
    specs: dict[str, SeasonSpec],
    weeks: tuple[int, ...],
    fixtures_per_week: int,
    prob_floor: float,
    deductions: dict[str, dict[str, int]] | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Forecast every barrier of every season under every spec, and score it.

    One fit per barrier, shared by every spec: the specs differ in how they propagate uncertainty
    from the fit, and giving each its own fit would let a difference in the optimiser's path show
    up as a difference in calibration.
    """
    if not specs:
        raise SeasonError("run_span needs at least one spec")
    questions = {tuple(s.questions) for s in specs.values()}
    if len(questions) != 1:
        raise SeasonError("every spec in a run must ask the same questions to be comparable")

    frames: list[pd.DataFrame] = []
    for season in seasons:
        rows = matches[matches["season"] == season].reset_index(drop=True)
        teams = tuple(sorted(set(rows["home_team"]) | set(rows["away_team"])))
        season_deductions = (deductions or {}).get(season, {})
        final = actual_table(rows, teams=teams, deductions=season_deductions)
        newcomers = promoted_clubs(matches, season)
        for barrier in matchweek_barriers(
            rows, weeks=weeks, fixtures_per_week=fixtures_per_week
        ):
            fit = production_fit(cfg, matches, barrier.date)
            played = rows[rows["date"] < barrier.date]
            remaining = rows[rows["date"] >= barrier.date]
            for name, spec in specs.items():
                forecast = simulate_season(
                    fit, played, remaining, spec=spec, seed=cfg.seed,
                    history=training_frame(matches, barrier.date),
                    deductions=season_deductions, season=season, barrier=barrier.date,
                )
                scored = score_forecast(
                    forecast, final, questions=spec.questions,
                    prob_floor=prob_floor, seed=cfg.seed, promoted=newcomers,
                )
                scored["spec"] = name
                scored["week"] = barrier.week
                scored["n_cold_start"] = forecast.diagnostics["n_cold_start"]
                frames.append(scored)
            if progress:
                print(f"  {barrier.label()}  horizon {barrier.horizon:.2f}  "
                      f"{len(rows) - len(played)} fixtures to play", flush=True)
    return pd.concat(frames, ignore_index=True)


def summarise(scored: pd.DataFrame, *, n_boot: int, seed: int,
              baseline: str | None = None) -> dict[str, object]:
    """Per-spec scores, PIT calibration, and each spec's clustered delta against the baseline."""
    specs = list(dict.fromkeys(scored["spec"]))
    reference = baseline if baseline is not None else specs[0]
    if reference not in specs:
        raise SeasonError(f"baseline {reference!r} is not among the specs {specs}")

    out: dict[str, object] = {"baseline": reference, "specs": {}}
    base = scored[scored["spec"] == reference].reset_index(drop=True)
    for name in specs:
        block = scored[scored["spec"] == name].reset_index(drop=True)
        pit = block.drop_duplicates(["season", "week", "team"])["points_pit"].to_numpy()
        entry: dict[str, object] = {
            "n_rows": int(len(block)),
            "n_seasons": int(block["season"].nunique()),
            "brier": float(block["brier"].mean()),
            "log_loss": float(block["log_loss"].mean()),
            "share_floored": float(block["floored"].mean()),
            "points_log_score": float(
                block.drop_duplicates(["season", "week", "team"])["points_log_score"].mean()),
            "points_pit": pit_summary(pit),
            "by_question": {
                q: {"brier": float(g["brier"].mean()), "log_loss": float(g["log_loss"].mean())}
                for q, g in block.groupby("question", sort=False)
            },
            "by_week": {
                int(w): {"brier": float(g["brier"].mean()),
                         "points_pit": pit_summary(
                             g.drop_duplicates(["season", "team"])["points_pit"].to_numpy())}
                for w, g in block.groupby("week", sort=False)
            },
            # Diagnostic slices, never gates -- the same standing rule the match harness uses.
            # A model calibrated overall and wrong on promoted clubs is not failing; it is saying
            # where its next improvement is.
            "by_promotion": {
                ("promoted" if flag else "established"): {
                    "n": int(len(g)),
                    "brier": float(g["brier"].mean()),
                    "points_pit": pit_summary(
                        g.drop_duplicates(["season", "week", "team"])["points_pit"].to_numpy()),
                    "by_question": {
                        q: float(sub["brier"].mean())
                        for q, sub in g.groupby("question", sort=False)
                    },
                }
                for flag, g in block.groupby("promoted", sort=False)
            },
        }
        if name != reference:
            if not block[["season", "week", "team", "question"]].equals(
                base[["season", "week", "team", "question"]]
            ):
                raise SeasonError(f"spec {name!r} is not aligned row-for-row with the baseline")
            entry["vs_baseline"] = paired_delta_clustered(
                block["brier"].to_numpy(), base["brier"].to_numpy(),
                block["season"].to_numpy(), n_boot=n_boot, seed=seed,
            )
            # The same construction on the whole predictive distribution rather than on three
            # binary slices of it. One row per club-season, so the questions do not triple-count.
            one_per_club = block.drop_duplicates(["season", "week", "team"])
            base_per_club = base.drop_duplicates(["season", "week", "team"])
            entry["vs_baseline_points"] = paired_delta_clustered(
                one_per_club["points_log_score"].to_numpy(),
                base_per_club["points_log_score"].to_numpy(),
                one_per_club["season"].to_numpy(), n_boot=n_boot, seed=seed,
            )
        out["specs"][name] = entry
    return out
