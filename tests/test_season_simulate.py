"""The season Monte Carlo.

The load-bearing claim is that :func:`sample_scorelines` draws from the same distribution the
match model reports, by a route that never builds the grid. That is checked against
:func:`plmodel.model.scoreline.scoreline_matrix` cell by cell rather than against a second
sampler, because two samplers written by the same author fail the same way.

Everything else here is structural: a position is taken by exactly one club, goals conceded equal
goals scored, and a simulator handed no fixtures reports the table it was given.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from plmodel.model.counts import CountSpec
from plmodel.model.dixon_coles import DixonColesFit
from plmodel.model.scoreline import (
    clamp_rho_for_rates,
    scoreline_matrix,
    three_class_from_rates,
)
from plmodel.season.simulate import (
    UNCERTAINTY_DRIFT,
    DriftSpec,
    SeasonSpec,
    _drift_deltas,
    sample_scorelines,
    simulate_season,
)
from plmodel.season.table import Question, SeasonError

MAX_GOALS = 12
QUESTIONS = (Question("title", "top", 1), Question("top_four", "top", 4),
             Question("relegation", "bottom", 3))


def _fit(teams: tuple[str, ...], attack=None, defence=None, *, rho=-0.05,
         intercept=0.1, home=0.25) -> DixonColesFit:
    n = len(teams)
    attack = np.zeros(n) if attack is None else np.asarray(attack, dtype=float)
    defence = np.zeros(n) if defence is None else np.asarray(defence, dtype=float)
    return DixonColesFit(
        teams=teams, attack=attack, defence=defence, intercept=intercept,
        home_advantage=home, rho=rho, half_life_days=730.0,
        ref_date=pd.Timestamp("2020-08-01"), max_goals=MAX_GOALS, n_obs=1, effective_n=1.0,
        neg_log_lik=0.0, converged=True, n_iterations=1,
    )


def _round_robin(teams: tuple[str, ...], *, start="2020-08-08") -> pd.DataFrame:
    rows = []
    day = pd.Timestamp(start)
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if i == j:
                continue
            rows.append({"date": day, "season": "2020-21", "home_team": home,
                         "away_team": away, "home_goals": np.nan, "away_goals": np.nan})
            day += pd.Timedelta(days=1)
    return pd.DataFrame(rows)


def _spec(**kwargs) -> SeasonSpec:
    settings = {"n_replicates": 2000, "chunk_size": 500, "questions": QUESTIONS}
    settings.update(kwargs)
    return SeasonSpec(**settings)


# --- the sampler ------------------------------------------------------------------------------

def test_the_sampler_reproduces_the_analytic_scoreline_grid() -> None:
    """The whole point of the block trick: exact, without ever building a grid."""
    lam = np.array([1.7, 0.8, 2.4, 1.1])
    mu = np.array([1.1, 1.6, 0.6, 1.1])
    rho = np.full(4, -0.12)
    n = 400_000
    home, away = sample_scorelines(lam, mu, rho, rng=np.random.default_rng(11), size=(n, 4))
    analytic = scoreline_matrix(lam, mu, rho, MAX_GOALS)
    # Three standard errors on a share of n; the corner cells carry most of the mass.
    tolerance = 3.0 * np.sqrt(0.25 / n)
    for match in range(4):
        for x in range(4):
            for y in range(4):
                got = float(np.mean((home[:, match] == x) & (away[:, match] == y)))
                assert abs(got - analytic[match, x, y]) < tolerance


def test_the_corrected_cells_are_the_only_ones_the_correction_moves() -> None:
    """A strong rho must change the four low-score cells and leave everything else alone."""
    lam, mu = np.array([1.4]), np.array([1.2])
    n = 300_000
    kwargs = dict(rng=np.random.default_rng(5), size=(n, 1))
    plain, _ = sample_scorelines(lam, mu, np.array([0.0]), **kwargs)
    plain_away = sample_scorelines(lam, mu, np.array([0.0]),
                                   rng=np.random.default_rng(5), size=(n, 1))[1]
    corrected = sample_scorelines(lam, mu, np.array([-0.25]),
                                  rng=np.random.default_rng(5), size=(n, 1))
    share = lambda h, a, x, y: float(np.mean((h[:, 0] == x) & (a[:, 0] == y)))
    # 0-0 lifts under a negative rho ...
    assert share(*corrected, 0, 0) > share(plain, plain_away, 0, 0) + 0.005
    # ... and the cells outside the block are untouched beyond sampling noise.
    for x, y in ((3, 2), (2, 3), (4, 0)):
        assert abs(share(*corrected, x, y) - share(plain, plain_away, x, y)) < 0.004


def test_a_rho_of_zero_is_two_independent_poissons() -> None:
    lam, mu = np.array([1.6]), np.array([1.3])
    n = 200_000
    home, away = sample_scorelines(lam, mu, np.zeros(1), rng=np.random.default_rng(2),
                                   size=(n, 1))
    assert abs(np.corrcoef(home[:, 0], away[:, 0])[0, 1]) < 0.01
    assert home.mean() == pytest.approx(1.6, abs=0.02)
    assert away.mean() == pytest.approx(1.3, abs=0.02)


def test_an_out_of_range_rho_raises_rather_than_sampling_a_non_distribution() -> None:
    with pytest.raises(SeasonError, match="negative scoreline weight"):
        sample_scorelines(np.array([2.0]), np.array([2.0]), np.array([0.9]),
                          rng=np.random.default_rng(0), size=(10, 1))


def test_the_size_argument_draws_independent_replicates() -> None:
    """Guards the mistake of drawing once and broadcasting the result across replicates."""
    home, _ = sample_scorelines(np.array([1.5]), np.array([1.2]), np.zeros(1),
                                rng=np.random.default_rng(0), size=(500, 1))
    assert home.shape == (500, 1)
    assert len(np.unique(home)) > 1


# --- the season engine ------------------------------------------------------------------------

def test_a_single_fixture_reproduces_the_match_model() -> None:
    """The simulator's home/draw/away must be predict_proba's, or it is a different model."""
    teams = ("A", "B")
    fit = _fit(teams, attack=[0.3, -0.3], defence=[0.2, -0.2])
    fixtures = pd.DataFrame([{"date": pd.Timestamp("2020-08-08"), "season": "2020-21",
                              "home_team": "A", "away_team": "B",
                              "home_goals": np.nan, "away_goals": np.nan}])
    spec = _spec(n_replicates=200_000, chunk_size=50_000,
                 questions=(Question("winner", "top", 1),))
    forecast = simulate_season(fit, fixtures.iloc[:0], fixtures, spec=spec, seed=3)

    lam, mu = fit.match_rates(fixtures)
    rho, _ = clamp_rho_for_rates(lam, mu, fit.rho, margin=0.01)
    analytic = three_class_from_rates(lam, mu, rho, MAX_GOALS)[0]
    # A wins the group iff it wins the match; a draw sends the position to the coin.
    expected = analytic[0] + analytic[1] / 2.0
    assert forecast.question_probability("winner")["A"] == pytest.approx(expected, abs=0.005)


def test_every_position_is_filled_in_every_replicate() -> None:
    teams = tuple("ABCDEF")
    forecast = simulate_season(_fit(teams), _round_robin(teams).iloc[:0], _round_robin(teams),
                               spec=_spec(), seed=1)
    assert forecast.position_counts.sum(axis=0).tolist() == [2000] * 6
    assert forecast.position_counts.sum(axis=1).tolist() == [2000] * 6
    assert forecast.points_counts.sum(axis=1).tolist() == [2000] * 6


def test_the_probabilities_of_a_question_sum_to_the_places_it_asks_about() -> None:
    """Four clubs finish in the top four of every replicate, so the column must sum to four."""
    teams = tuple("ABCDEFGHIJ")
    forecast = simulate_season(_fit(teams), _round_robin(teams).iloc[:0], _round_robin(teams),
                               spec=_spec(), seed=4)
    assert forecast.probabilities["title"].sum() == pytest.approx(1.0)
    assert forecast.probabilities["top_four"].sum() == pytest.approx(4.0)
    assert forecast.probabilities["relegation"].sum() == pytest.approx(3.0)


def test_a_finished_season_forecasts_what_already_happened() -> None:
    """With nothing left to play the simulator is a league table with a coin for exact ties."""
    teams = tuple("ABCDEF")
    played = _round_robin(teams)
    rng = np.random.default_rng(9)
    played["home_goals"] = rng.integers(0, 4, len(played))
    played["away_goals"] = rng.integers(0, 4, len(played))
    forecast = simulate_season(_fit(teams), played, played.iloc[:0], spec=_spec(), seed=2)
    champion = forecast.table["team"].iloc[0]
    assert forecast.question_probability("title")[champion] == pytest.approx(1.0)
    assert forecast.n_remaining == 0 and forecast.horizon == 0.0


def test_stronger_clubs_win_more_often() -> None:
    teams = tuple("ABCDEF")
    fit = _fit(teams, attack=[0.5, 0.3, 0.1, -0.1, -0.3, -0.5],
               defence=[0.5, 0.3, 0.1, -0.1, -0.3, -0.5])
    forecast = simulate_season(fit, _round_robin(teams).iloc[:0], _round_robin(teams),
                               spec=_spec(), seed=6)
    ordered = forecast.probabilities.set_index("team").loc[list(teams), "title"].to_numpy()
    # Non-increasing throughout, and strictly decreasing where the replicate count can resolve it:
    # the bottom clubs all sit at zero, and a tie between two zeros is the sample size talking.
    assert (np.diff(ordered) <= 0).all()
    assert ordered[0] > ordered[1] > ordered[2]


def test_points_accumulate_onto_the_table_already_played() -> None:
    """A club that has banked points keeps them: the simulation adds, it does not replace."""
    teams = tuple("ABCDEF")
    fixtures = _round_robin(teams)
    played = fixtures.iloc[:2].copy()
    played["home_goals"], played["away_goals"] = [5, 5], [0, 0]
    remaining = fixtures.iloc[2:]
    forecast = simulate_season(_fit(teams), played, remaining, spec=_spec(), seed=7)
    banked = forecast.table.set_index("team").loc["A", "points"]
    assert banked == 6   # A hosts the first two fixtures of the round robin and wins both
    assert forecast.points_counts[forecast.teams.index("A")][:banked - forecast.points_floor].sum() == 0


def test_a_deduction_moves_the_starting_table_and_nothing_else() -> None:
    teams = tuple("ABCDEF")
    fixtures = _round_robin(teams)
    played = fixtures.iloc[:2].copy()
    played["home_goals"], played["away_goals"] = [5, 5], [0, 0]
    plain = simulate_season(_fit(teams), played, fixtures.iloc[2:], spec=_spec(), seed=7)
    docked = simulate_season(_fit(teams), played, fixtures.iloc[2:], spec=_spec(), seed=7,
                             deductions={"A": 6})
    assert plain.table.set_index("team").loc["A", "points"] == 6
    assert docked.table.set_index("team").loc["A", "points"] == 0
    assert docked.question_probability("title")["A"] < plain.question_probability("title")["A"]


# --- uncertainty ------------------------------------------------------------------------------

def test_the_drift_perturbation_is_centred_across_clubs() -> None:
    """Sum-to-zero in, sum-to-zero out: drift moves clubs relative to each other, not the league."""
    spec = DriftSpec(attack_sd=0.2, defence_sd=0.2, correlation=0.3, horizon_exponent=0.5)
    attack, defence = _drift_deltas(spec, n_replicates=500, n_teams=20, horizon=1.0,
                                    rng=np.random.default_rng(0))
    assert np.abs(attack.sum(axis=1)).max() < 1e-10
    assert np.abs(defence.sum(axis=1)).max() < 1e-10
    assert attack.std() == pytest.approx(0.2, rel=0.1)
    assert np.corrcoef(attack.ravel(), defence.ravel())[0, 1] == pytest.approx(0.3, abs=0.05)


def test_the_horizon_shrinks_the_perturbation_as_a_square_root() -> None:
    spec = DriftSpec(attack_sd=0.2, defence_sd=0.2, correlation=0.0, horizon_exponent=0.5)
    assert spec.scale(1.0) == pytest.approx(1.0)
    assert spec.scale(0.25) == pytest.approx(0.5)
    assert spec.scale(0.0) == 0.0


def test_drift_widens_the_points_distribution() -> None:
    """The claim the mode exists for. A point-estimate season is narrower than a drifting one."""
    teams = tuple("ABCDEFGH")
    fit = _fit(teams, attack=np.linspace(0.4, -0.4, 8), defence=np.linspace(0.4, -0.4, 8))
    fixtures = _round_robin(teams)
    common = dict(n_replicates=4000, chunk_size=1000)
    point = simulate_season(fit, fixtures.iloc[:0], fixtures, spec=_spec(**common), seed=8)
    drift = simulate_season(
        fit, fixtures.iloc[:0], fixtures, seed=8,
        spec=_spec(uncertainty=UNCERTAINTY_DRIFT, drift=DriftSpec(
            attack_sd=0.2, defence_sd=0.2, correlation=0.0, horizon_exponent=0.5), **common),
    )
    spread = lambda f: float((f.points_quantile(0.95) - f.points_quantile(0.05)).mean())
    assert spread(drift) > spread(point) * 1.1
    # And the title race is less settled: the favourite's probability falls.
    assert drift.probabilities["title"].max() < point.probabilities["title"].max()


def test_a_zero_drift_spec_is_the_point_estimate() -> None:
    """The inertness check every seam in this project carries."""
    teams = tuple("ABCDE")
    fixtures = _round_robin(teams)
    fit = _fit(teams, attack=np.linspace(0.3, -0.3, 5))
    point = simulate_season(fit, fixtures.iloc[:0], fixtures, spec=_spec(), seed=5)
    inert = simulate_season(
        fit, fixtures.iloc[:0], fixtures, seed=5,
        spec=_spec(uncertainty=UNCERTAINTY_DRIFT, drift=DriftSpec(
            attack_sd=0.0, defence_sd=0.0, correlation=0.0, horizon_exponent=0.5)),
    )
    assert inert.uncertainty == "point"
    assert np.array_equal(point.position_counts, inert.position_counts)
    assert np.array_equal(point.points_counts, inert.points_counts)


def test_the_chunk_size_changes_the_noise_and_not_the_distribution() -> None:
    """Chunking is a memory setting. It moves the random stream, so it is checked statistically."""
    teams = tuple("ABCDEF")
    fixtures = _round_robin(teams)
    fit = _fit(teams, attack=np.linspace(0.3, -0.3, 6))
    n = 20_000
    a = simulate_season(fit, fixtures.iloc[:0], fixtures, seed=5,
                        spec=_spec(n_replicates=n, chunk_size=1000))
    b = simulate_season(fit, fixtures.iloc[:0], fixtures, seed=5,
                        spec=_spec(n_replicates=n, chunk_size=n))
    # Three standard errors on a probability estimated from n replicates.
    assert np.abs(a.probabilities.set_index("team")["title"]
                  - b.probabilities.set_index("team")["title"]).max() < 3.0 * np.sqrt(0.25 / n)


def test_a_fixed_seed_and_chunk_size_reproduce_exactly() -> None:
    teams = tuple("ABCDEF")
    fixtures = _round_robin(teams)
    runs = [simulate_season(_fit(teams), fixtures.iloc[:0], fixtures, spec=_spec(), seed=12)
            for _ in range(2)]
    assert np.array_equal(runs[0].position_counts, runs[1].position_counts)
    assert np.array_equal(runs[0].points_counts, runs[1].points_counts)


# --- refusals ---------------------------------------------------------------------------------

def test_a_fit_carrying_a_scoreline_family_is_refused() -> None:
    """The sampler draws the production scoreline; anything else must say so rather than pretend."""
    teams = tuple("ABC")
    fit = dataclasses.replace(
        _fit(teams), family=CountSpec(marginal="weibull", n_series_terms=60))
    fixtures = _round_robin(teams)
    with pytest.raises(SeasonError, match="scoreline family"):
        simulate_season(fit, fixtures.iloc[:0], fixtures, spec=_spec(), seed=1)


def test_a_fixture_outside_the_roster_is_refused() -> None:
    teams = tuple("ABC")
    fixtures = _round_robin(teams)
    # Build the roster from a subset, then hand the simulator a fixture it does not cover.
    with pytest.raises(SeasonError, match="outside the season's roster"):
        from plmodel.season.simulate import _team_index

        _team_index(("A", "B"), fixtures["home_team"])


def test_spec_validation() -> None:
    with pytest.raises(SeasonError, match="at least 1"):
        _spec(n_replicates=0)
    with pytest.raises(SeasonError, match="must be one of"):
        _spec(uncertainty="posterior")
    with pytest.raises(SeasonError, match="needs a drift spec"):
        _spec(uncertainty=UNCERTAINTY_DRIFT)
    with pytest.raises(SeasonError, match="unique"):
        _spec(questions=(Question("a", "top", 1), Question("a", "top", 4)))
    with pytest.raises(SeasonError, match="missing settings"):
        DriftSpec(attack_sd=0.1)
    with pytest.raises(SeasonError, match="in \\[-1, 1\\]"):
        DriftSpec(attack_sd=0.1, defence_sd=0.1, correlation=2.0, horizon_exponent=0.5)


# --- standing in for a DixonColesFit -------------------------------------------------------------
#
# `dc-gas` is accepted and unwired. When it is wired the simulator will be handed a DynamicFit, so
# these are the prerequisite proofs: it accepts one, the states actually reach the sampler, and a
# fit it cannot draw from is refused rather than crashed.

def _dynamic(teams, results, *, loading=0.15):
    """A dynamic fit over a short history, so the states are non-zero and known to be so."""
    from plmodel.model.dynamics import DynamicFit, GasSpec, filter_states

    level = _fit(teams)
    history = pd.DataFrame({
        "date": pd.to_datetime([r[0] for r in results]),
        "home_team": [r[1] for r in results], "away_team": [r[2] for r in results],
        "home_goals": [float(r[3]) for r in results],
        "away_goals": [float(r[4]) for r in results],
    })
    spec = GasSpec(score_loading=loading, persistence=0.95, scaling_exponent=1.0,
                   half_life_days=730.0, state_bound=1.5)
    return level, DynamicFit(level, filter_states(history, level, spec), spec)


def test_the_simulator_accepts_a_dynamic_fit_and_forecasts_from_its_states() -> None:
    """The regression for a crash, and the proof the states are not silently dropped.

    `simulate_season` guards on `fit.family is not None`. A DynamicFit exposed no `family`, so the
    guard raised AttributeError before it could be read — a crash where a refusal was intended.
    Now it is accepted, and the load-bearing assertion is the last one: if the states failed to
    reach the sampler the two forecasts would be identical.
    """
    teams = tuple("ABCDEF")
    beatings = [(f"2020-01-{d:02d}", "A", "B", 4, 0) for d in range(1, 9)]
    level, dynamic = _dynamic(teams, beatings)
    fixtures = _round_robin(teams)

    from_level = simulate_season(level, fixtures.iloc[:0], fixtures, spec=_spec(), seed=5)
    from_states = simulate_season(dynamic, fixtures.iloc[:0], fixtures, spec=_spec(), seed=5)

    assert from_states.probabilities["title"].sum() == pytest.approx(1.0)
    assert from_states.position_counts.sum(axis=0).tolist() == [2000] * len(teams)
    assert (from_states.question_probability("title")["A"]
            > from_level.question_probability("title")["A"])


def test_a_dynamic_fit_with_no_states_simulates_exactly_what_its_level_does() -> None:
    """The season-shaped inertness contract: byte-identical, not merely close."""
    teams = tuple("ABCDEF")
    level, inert = _dynamic(teams, [("2020-01-01", "A", "B", 1, 0)], loading=0.0)
    fixtures = _round_robin(teams)
    plain = simulate_season(level, fixtures.iloc[:0], fixtures, spec=_spec(), seed=9)
    dynamic = simulate_season(inert, fixtures.iloc[:0], fixtures, spec=_spec(), seed=9)
    assert np.array_equal(plain.position_counts, dynamic.position_counts)
    assert np.array_equal(plain.points_counts, dynamic.points_counts)


def test_a_dynamic_fit_over_a_scoreline_family_is_refused_rather_than_crashed() -> None:
    """Refused, not crashed — the distinction this whole delegation exists to restore."""
    from plmodel.model.dynamics import DynamicFit, GasStates

    teams = tuple("ABCDEF")
    level = dataclasses.replace(
        _fit(teams), family=CountSpec(marginal="weibull", n_series_terms=60)
    )
    zeros = np.zeros(len(teams))
    states = GasStates(teams, zeros, zeros.copy(), 0, 0, 0)
    spec = _dynamic(teams, [("2020-01-01", "A", "B", 1, 0)])[1].spec
    fixtures = _round_robin(teams)
    with pytest.raises(SeasonError, match="scoreline family"):
        simulate_season(DynamicFit(level, states, spec), fixtures.iloc[:0], fixtures,
                        spec=_spec(), seed=1)
