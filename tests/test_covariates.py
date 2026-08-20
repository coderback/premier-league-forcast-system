"""Match-context covariates: what they measure, and that measuring them changes nothing when off.

Two properties carry most of the weight. The measurements must be *definitionally* right — a rest
term that is off by a day, or a congestion count that includes the match it describes, would be a
covariate whose null says nothing about rest — so both are checked against a brute-force reading of
their own definition rather than against a remembered number. And the seam must be exactly inert:
the whole arm is a claim about what a term adds to the production model, which is only meaningful
if not having the term leaves that model untouched.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from plmodel.model import covariates as cv
from plmodel.model.dixon_coles import fit_dixon_coles


def _frame(rows: list[tuple[str, str, str, str, int, int]]) -> pd.DataFrame:
    """(date, season, home, away, home_goals, away_goals) -> a corpus-shaped frame."""
    return pd.DataFrame(
        rows, columns=["date", "season", "home_team", "away_team", "home_goals", "away_goals"]
    ).assign(date=lambda f: pd.to_datetime(f["date"]), division="E0", played=True)


# --- the spec's guards ----------------------------------------------------------------------------

def test_an_unknown_term_is_refused_rather_than_ignored() -> None:
    with pytest.raises(cv.CovariateError):
        cv.CovariateSpec(terms=("rest", "weather"))


def test_a_repeated_term_is_refused() -> None:
    """Two identical columns would make the design singular and the coefficient meaningless."""
    with pytest.raises(cv.CovariateError):
        cv.CovariateSpec(terms=("rest", "rest"))


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(cv.CovariateError):
        cv.CovariateSpec(terms=("rest",), mode="ridge", rest_clip_days=14, rest_reference_days=7)


@pytest.mark.parametrize("terms", [("rest",), ("congestion",), ("euro",)])
def test_a_term_without_its_settings_is_refused(terms: tuple[str, ...]) -> None:
    """No plausible default. A clip or a window that nobody justified is a magic number."""
    with pytest.raises(cv.CovariateError):
        cv.CovariateSpec(terms=terms)


def test_the_empty_spec_is_inert_and_needs_no_settings() -> None:
    spec = cv.CovariateSpec()
    assert spec.is_inert and spec.names() == () and spec.label() == "none"


# --- the measurements -----------------------------------------------------------------------------

def test_rest_counts_days_back_to_the_previous_league_match() -> None:
    frame = _frame([
        ("2020-08-01", "2020-21", "A", "B", 1, 0),
        ("2020-08-08", "2020-21", "B", "A", 2, 2),   # both sides: 7 days
        ("2020-08-11", "2020-21", "A", "C", 0, 0),   # A: 3 days; C: first match
    ])
    home, away = cv.rest_days(frame)
    assert np.isnan(home[0]) and np.isnan(away[0])
    assert (home[1], away[1]) == (7.0, 7.0)
    assert home[2] == 3.0 and np.isnan(away[2])


def test_rest_does_not_reach_across_the_summer() -> None:
    """A club's first match of a season follows an eleven-week gap that is not rest."""
    frame = _frame([
        ("2020-05-01", "2019-20", "A", "B", 1, 0),
        ("2020-08-15", "2020-21", "A", "B", 1, 0),
    ])
    home, away = cv.rest_days(frame)
    assert np.isnan(home[1]) and np.isnan(away[1])


def test_congestion_matches_a_brute_force_reading_of_its_definition() -> None:
    """Written out the slow way and compared, because an off-by-one here is invisible in a fit."""
    rng = np.random.default_rng(11)
    teams = list("ABCDEF")
    days = np.sort(rng.choice(np.arange(120), size=60, replace=False))
    rows = []
    for day in days:
        home, away = rng.choice(teams, size=2, replace=False)
        rows.append((str(pd.Timestamp("2021-08-01") + pd.Timedelta(days=int(day))), "2021-22",
                     home, away, 1, 1))
    frame = _frame(rows)
    window = 14

    day_number = frame["date"].to_numpy("datetime64[D]").astype(np.int64)
    expected_home, expected_away = np.zeros(len(frame)), np.zeros(len(frame))
    for column, out in (("home_team", expected_home), ("away_team", expected_away)):
        for i in range(len(frame)):
            team = frame[column].iloc[i]
            plays = ((frame["home_team"] == team) | (frame["away_team"] == team)).to_numpy()
            earlier = day_number[plays]
            out[i] = int(((earlier >= day_number[i] - window) & (earlier < day_number[i])).sum())

    home, away = cv.congestion_count(frame, window_days=window)
    assert np.array_equal(home, expected_home)
    assert np.array_equal(away, expected_away)


def test_congestion_never_counts_the_match_it_describes() -> None:
    frame = _frame([("2021-01-01", "2020-21", "A", "B", 0, 0)])
    home, away = cv.congestion_count(frame, window_days=14)
    assert (home[0], away[0]) == (0.0, 0.0)


def test_european_qualification_reads_the_PREVIOUS_season_table() -> None:
    """Two seasons, one result each way, so the ordering is decided by points and then goals."""
    frame = _frame([
        ("2019-09-01", "2019-20", "A", "B", 3, 0),
        ("2019-09-08", "2019-20", "C", "D", 1, 1),
        ("2020-09-01", "2020-21", "A", "B", 0, 0),
    ])
    qualified = cv.european_qualification(frame, frame, top_k=2, division="E0")
    # A won and B lost, so A is first and B last; C and D drew and sit between them.
    assert qualified.get(("2020-21", "A")) is True
    assert ("2020-21", "B") not in qualified
    # The first season of the corpus has nothing behind it and flags nobody.
    assert not any(season == "2019-20" for season, _ in qualified)


def test_the_european_window_wraps_the_year_and_its_complement_does_not() -> None:
    dates = pd.Series(pd.to_datetime(
        ["2020-01-15", "2020-05-20", "2020-06-15", "2020-08-20", "2020-09-20"]
    ))
    uefa = cv.in_european_window(dates, ("09-14", "05-31"))
    summer = cv.in_european_window(dates, ("06-01", "09-13"))
    assert list(uefa) == [1.0, 1.0, 0.0, 0.0, 1.0]
    assert list(summer) == [1.0 - v for v in uefa], "the placebo window must be the complement"


# --- the design -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_split(cfg, corpus):
    """A history and a later frame to describe with it, from the real corpus."""
    history = corpus[corpus["season"] <= "2014-15"].reset_index(drop=True)
    frame = corpus[corpus["season"] == "2015-16"].head(120).reset_index(drop=True)
    return history, frame


@pytest.fixture(scope="module")
def full_spec(cfg):
    return cfg.model.covariate_spec(terms=("rest", "congestion", "euro"))


def test_an_off_seam_produces_an_empty_design(corpus_split) -> None:
    history, frame = corpus_split
    built = cv.design(frame, history, cv.CovariateSpec(), division="E0")
    assert built.lam.shape == (len(frame), 0) and built.mu.shape == (len(frame), 0)
    assert built.n_params == 0


def test_the_differential_form_moves_the_balance_and_not_the_total(corpus_split, full_spec) -> None:
    """``diff`` is the restriction that log lam + log mu is untouched; that is checkable."""
    history, frame = corpus_split
    built = cv.design(frame, history, full_spec, division="E0")
    assert built.names == ("cov_rest", "cov_congestion", "cov_euro")
    assert np.array_equal(built.lam, -built.mu)


def test_the_split_form_frees_the_two_halves(corpus_split, cfg) -> None:
    history, frame = corpus_split
    spec = cfg.model.covariate_spec(terms=("rest",), mode="split")
    built = cv.design(frame, history, spec, division="E0")
    assert built.names == ("cov_rest_attack", "cov_rest_defence")
    # The home side's own value drives the home rate's attack column and the away rate's defence
    # column, so the two designs are each other's mirror rather than each other's negative.
    assert np.array_equal(built.lam[:, 0], -built.mu[:, 1])
    assert not np.array_equal(built.lam, -built.mu)


def test_the_split_form_collapses_to_the_differential_when_its_halves_are_equal(
    corpus_split, cfg
) -> None:
    """The claim the module's docstring makes, asserted rather than described."""
    history, frame = corpus_split
    diff = cv.design(frame, history, cfg.model.covariate_spec(terms=("rest",)), division="E0")
    split = cv.design(
        frame, history, cfg.model.covariate_spec(terms=("rest",), mode="split"), division="E0"
    )
    one = np.ones(2)
    assert np.allclose(split.lam @ one, diff.lam @ np.ones(1))
    assert np.allclose(split.mu @ one, diff.mu @ np.ones(1))


def test_an_undefined_term_switches_off_rather_than_being_filled_in(cfg) -> None:
    """Non-negotiable #6 at the covariate: a season's opening round has no rest, and gets none."""
    frame = _frame([
        ("2020-08-15", "2020-21", "A", "B", 1, 0),   # opening round: no rest for either side
        ("2020-08-19", "2020-21", "C", "B", 1, 0),   # B again after 4 days
        ("2020-08-22", "2020-21", "A", "B", 1, 0),   # A rested 7, B rested 3
    ])
    spec = cfg.model.covariate_spec(terms=("rest",))
    built = cv.design(frame, frame.iloc[:0], spec, division="E0")
    assert built.undefined["rest"] == 2 and built.one_sided["rest"] == 1
    assert built.lam[0, 0] == 0.0 and built.mu[0, 0] == 0.0
    # The last match has rest on both sides, so the term is live and carries the difference.
    assert built.lam[2, 0] == pytest.approx(4.0)
    assert built.mu[2, 0] == pytest.approx(-4.0)


def test_a_one_sided_undefined_value_is_counted_separately(cfg) -> None:
    """A different question from the symmetric case, so it is never quietly averaged in."""
    frame = _frame([
        ("2020-08-15", "2020-21", "A", "B", 1, 0),
        ("2020-08-22", "2020-21", "A", "C", 1, 0),   # C has not played yet
    ])
    built = cv.design(frame, frame.iloc[:0], cfg.model.covariate_spec(terms=("rest",)),
                      division="E0")
    assert built.one_sided["rest"] == 1
    assert built.lam[1, 0] == 0.0


def test_a_truncated_history_gives_the_same_values_as_the_whole_corpus(cfg, corpus) -> None:
    """Structural leak-freedom: every term looks strictly backwards, so a barrier cannot matter.

    If a covariate computed behind a barrier ever disagreed with the same covariate computed with
    the whole corpus available, something in it would be reading forwards. Asserting equality is
    what makes the walk's per-barrier rebuild trustworthy rather than merely conventional.
    """
    spec = cfg.model.covariate_spec(terms=("rest", "congestion", "euro"))
    whole = cv.design(corpus, corpus.iloc[:0], spec, division="E0")

    barrier = pd.Timestamp("2015-08-20")
    history = corpus[corpus["date"] < barrier].reset_index(drop=True)
    ahead = corpus[corpus["date"] >= barrier].head(80).reset_index(drop=True)
    truncated = cv.design(ahead, history, spec, division="E0")

    rows = np.flatnonzero((corpus["date"] >= barrier).to_numpy())[:80]
    assert np.array_equal(truncated.lam, whole.lam[rows])
    assert np.array_equal(truncated.mu, whole.mu[rows])


# --- through the fit --------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def training(corpus):
    return corpus[corpus["season"].isin(["2012-13", "2013-14", "2014-15"])].reset_index(drop=True)


def _fit(frame, cfg, spec):
    model = cfg.model
    return fit_dixon_coles(
        frame, half_life_days=model.decay_half_life_days,
        ref_date=frame["date"].max() + pd.Timedelta(days=1), max_goals=model.max_goals,
        param_bounds=model.param_bounds, min_effective_share=model.min_effective_share,
        max_iter=model.max_iter, covariates=spec, cov_division="E0",
    )


def test_an_empty_spec_fits_byte_identically_to_no_spec_at_all(training, cfg) -> None:
    """Non-negotiable #4 at the parameter vector, not merely at the forecast."""
    without = _fit(training, cfg, None)
    empty = _fit(training, cfg, cv.CovariateSpec())
    assert without.neg_log_lik == empty.neg_log_lik
    assert np.array_equal(without.attack, empty.attack)
    assert np.array_equal(without.defence, empty.defence)
    assert (without.intercept, without.home_advantage, without.rho) == (
        empty.intercept, empty.home_advantage, empty.rho
    )


def test_the_analytic_gradient_matches_a_numerical_one(training, cfg) -> None:
    """The whole walk depends on this: a wrong covariate gradient converges somewhere plausible."""
    from scipy.special import gammaln

    from plmodel.model.dixon_coles import _objective, _starting_point, decay_weights

    model = cfg.model
    spec = cfg.model.covariate_spec(terms=("rest", "congestion", "euro"), mode="split")
    built = cv.design(training, training.iloc[:0], spec, division="E0")
    teams = sorted(set(training["home_team"]) | set(training["away_team"]))
    index = {t: i for i, t in enumerate(teams)}
    x = training["home_goals"].to_numpy(dtype=float)
    y = training["away_goals"].to_numpy(dtype=float)
    weights = decay_weights(training["date"], training["date"].max(), model.decay_half_life_days)
    args = (x, y, training["home_team"].map(index).to_numpy(),
            training["away_team"].map(index).to_numpy(), weights,
            gammaln(x + 1.0), gammaln(y + 1.0), len(teams),
            np.zeros((len(training), 0)), built.lam, built.mu)

    rng = np.random.default_rng(3)
    theta = _starting_point(x, y, weights, len(teams), built.n_params)
    theta[3:] = rng.normal(0.0, 0.15, len(theta) - 3)
    theta[2] = -0.04
    theta[len(theta) - built.n_params:] = rng.normal(0.0, 0.01, built.n_params)

    _, analytic = _objective(theta, *args)
    numeric = approx_fprime(theta, lambda t: _objective(t, *args)[0], 1e-6)
    assert np.abs(analytic - numeric).max() < 1e-3 * max(1.0, np.abs(numeric).max())


def test_predicting_without_the_history_raises_rather_than_silently_dropping_the_term(
    training, corpus, cfg
) -> None:
    """A model that fits a term and then forecasts without it is the failure ha-empty already paid
    for once; here it is an error rather than a quiet zero."""
    fit = _fit(training, cfg, cfg.model.covariate_spec(terms=("rest",)))
    ahead = corpus[corpus["season"] == "2015-16"].head(20)
    with pytest.raises(ValueError, match="history"):
        fit.predict_proba(ahead)
    assert fit.predict_proba(ahead, training).shape == (len(ahead), 3)


def test_the_covariate_and_family_seams_refuse_to_run_together(training, cfg) -> None:
    """One axis per arm. Two would be a two-axis test wearing a one-axis name."""
    from plmodel.model.counts import CountSpec

    family = CountSpec(marginal="weibull", dependence="frank", n_series_terms=40)
    with pytest.raises(ValueError, match="cannot run together"):
        fit_dixon_coles(
            training, half_life_days=cfg.model.decay_half_life_days,
            ref_date=training["date"].max(), max_goals=cfg.model.max_goals,
            param_bounds=cfg.model.param_bounds,
            min_effective_share=cfg.model.min_effective_share, max_iter=cfg.model.max_iter,
            covariates=cfg.model.covariate_spec(terms=("rest",)), cov_division="E0",
            family=family,
        )


def test_a_fitted_term_changes_the_forecast(training, corpus, cfg) -> None:
    """The arm must do something, or its null is a statement about the wiring."""
    ahead = corpus[corpus["season"] == "2015-16"].head(120)
    plain = _fit(training, cfg, None).predict_proba(ahead)
    with_rest = _fit(training, cfg, cfg.model.covariate_spec(terms=("rest",)))
    moved = with_rest.predict_proba(ahead, training)
    assert np.abs(moved - plain).max() > 1e-3
    assert np.abs(moved.sum(axis=1) - 1.0).max() < 1e-12


def test_the_summary_names_the_spec_and_its_coefficients(training, cfg) -> None:
    fit = _fit(training, cfg, cfg.model.covariate_spec(terms=("rest", "euro")))
    summary = fit.as_dict()
    assert summary["covariates"] == "rest+euro (diff)"
    assert set(summary["covariate_terms"]) == {"cov_rest", "cov_euro"}
    assert summary["covariate_undefined"]["rest"] > 0


# --- through the harness ------------------------------------------------------------------------

@pytest.mark.integration
def test_the_context_arms_run_the_walk_and_each_one_moves(cfg, corpus) -> None:
    import hashlib

    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, _fit_summary, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )

    def digest(probs: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()

    baseline, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    for name in ("rest", "rest-euro", "rest-split", "congestion"):
        probs, state = run_arm(ArmSpec.parse(name), corpus, splits, cfg)
        assert probs.shape == baseline.shape
        assert np.abs(probs - baseline).max() > 1e-3, f"{name} is indistinguishable from baseline"
        assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12
        block = _fit_summary(state)["covariates"]
        assert block["n_fits"] == len(splits)
        assert set(block["terms"]) == set(state["fit"].cov_names)

    again, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    assert digest(again) == digest(baseline), "the baseline moved while the context arms ran"


@pytest.mark.integration
def test_the_report_carries_no_covariate_block_for_the_baseline(cfg, corpus) -> None:
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, _fit_summary, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )
    _, plain = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    assert "covariates" not in _fit_summary(plain)


def test_the_configured_settings_are_the_ones_the_arms_use(cfg) -> None:
    """The seam says which terms are on; the settings block says what they mean. Both, once."""
    spec = cfg.model.covariate_spec(terms=("rest", "congestion", "euro"))
    settings = cfg.model.context
    assert spec.rest_clip_days == settings["rest_clip_days"]
    assert spec.congestion_window_days == settings["congestion_window_days"]
    assert spec.euro_top_k == settings["euro_top_k"]
    assert spec.euro_window == tuple(settings["euro_window"])
    # The shipped configuration has the seam off, so the production model asks for no spec at all.
    assert cfg.model.covariate_spec() is None
    assert dataclasses.replace(spec, terms=()).is_inert
