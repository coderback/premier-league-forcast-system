"""Structural home-advantage terms: the trend and the empty-stadium dummy.

Two properties carry the arm. The design must vanish at prediction — otherwise a forecast would
use the *average* home advantage over its training window rather than the current one, which is
the whole thing the arm exists to fix. And the seam must be exactly inert when off, since it lives
inside the production fitter.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime
from scipy.special import gammaln

from plmodel.config import load_config
from plmodel.model.dixon_coles import _objective, fit_dixon_coles
from plmodel.model.home_advantage import (
    MODES, design, empty_stadium_flag, parse_mode, prediction_design,
)

BOUNDS = {
    "intercept": (-2.0, 2.0), "home_advantage": (-1.0, 1.0), "rho": (-0.2, 0.2),
    "strength": (-3.0, 3.0), "ha_trend": (-0.2, 0.2), "ha_empty": (-1.0, 1.0),
}
WINDOW = ("2020-06-17", "2021-05-23")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _dates(*values: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(values)))


# --- modes -----------------------------------------------------------------------------------

def test_mode_parsing() -> None:
    assert parse_mode("global") == (False, False)
    assert parse_mode("trend") == (True, False)
    assert parse_mode("empty") == (False, True)
    assert parse_mode("trend+empty") == (True, True)


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown home_advantage mode"):
        parse_mode("sometimes")


def test_every_mode_builds_a_design() -> None:
    dates = _dates("2020-10-01", "2023-01-01")
    for mode in MODES:
        matrix, names = design(dates, pd.Timestamp("2024-01-01"), mode=mode,
                               empty_start=WINDOW[0], empty_end=WINDOW[1])
        assert matrix.shape == (2, len(names))


# --- the global mode is exactly empty -----------------------------------------------------------

def test_global_mode_adds_no_columns() -> None:
    """Zero columns is what makes the seam byte-identical rather than merely nearly identical."""
    matrix, names = design(_dates("2020-10-01"), pd.Timestamp("2024-01-01"), mode="global")
    assert matrix.shape == (1, 0) and names == ()


# --- the trend term ------------------------------------------------------------------------------

def test_trend_is_years_before_the_barrier() -> None:
    matrix, names = design(
        _dates("2023-01-01", "2022-01-01"), pd.Timestamp("2024-01-01"), mode="trend"
    )
    assert names == ("ha_trend",)
    assert matrix[0, 0] == pytest.approx(1.0, abs=0.01)
    assert matrix[1, 0] == pytest.approx(2.0, abs=0.01)


def test_trend_is_zero_at_the_barrier() -> None:
    """The forecast must use the CURRENT home advantage, not the window average."""
    matrix, _ = design(_dates("2024-01-01"), pd.Timestamp("2024-01-01"), mode="trend")
    assert matrix[0, 0] == 0.0


def test_trend_never_goes_negative() -> None:
    """A stray future row must not be given a negative age and pull the trend the wrong way."""
    matrix, _ = design(_dates("2025-01-01"), pd.Timestamp("2024-01-01"), mode="trend")
    assert matrix[0, 0] == 0.0


# --- the empty-stadium term ------------------------------------------------------------------------

def test_empty_flag_covers_the_causal_window() -> None:
    flags = empty_stadium_flag(
        _dates("2020-03-01", "2020-06-17", "2020-07-20", "2021-01-15", "2021-05-23", "2021-08-14"),
        *WINDOW,
    )
    assert flags.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_the_2020_restart_is_inside_the_window() -> None:
    """Defined causally, not empirically.

    The June-July 2020 restart was played behind closed doors and shows NO loss of home advantage
    (home 46.7%, home:away goals 1.315, against a pre-COVID 45.7% and 1.279), while 2020-21
    collapses to 37.9% and 1.008. Excluding the restart because it fails to show the expected
    effect would be fitting the window to the outcome, so it stays in and the arm is allowed to be
    diluted by it.
    """
    assert empty_stadium_flag(_dates("2020-07-01"), *WINDOW)[0] == 1.0


def test_empty_mode_requires_a_window() -> None:
    with pytest.raises(ValueError, match="needs a window"):
        design(_dates("2020-10-01"), pd.Timestamp("2024-01-01"), mode="empty")


def test_config_window_matches_the_known_regime(cfg) -> None:
    window = cfg.model.seams["home_advantage"]
    assert pd.Timestamp(window["empty_start"]) == pd.Timestamp("2020-06-17")
    assert pd.Timestamp(window["empty_end"]) == pd.Timestamp("2021-05-23")


# --- prediction ------------------------------------------------------------------------------------

def test_prediction_design_is_all_zeros() -> None:
    """A match being forecast is neither in the past nor behind closed doors."""
    assert prediction_design(5, ("ha_trend", "ha_empty")).tolist() == [[0.0, 0.0]] * 5


# --- the gradient of the new parameters --------------------------------------------------------------

@pytest.mark.parametrize("n_ha", [1, 2])
def test_design_parameter_gradients(n_ha: int) -> None:
    """The new parameters enter the likelihood, so they need the same check the rest got."""
    rng = np.random.default_rng(7)
    n_teams, n = 6, 300
    home = rng.integers(0, n_teams, n)
    away = (home + 1 + rng.integers(0, n_teams - 1, n)) % n_teams
    x = rng.poisson(1.5, n).astype(float)
    y = rng.poisson(1.1, n).astype(float)
    weights = rng.uniform(0.3, 1.0, n)
    ha_design = rng.uniform(0.0, 3.0, (n, n_ha))
    args = (x, y, home, away, weights, gammaln(x + 1), gammaln(y + 1), n_teams, ha_design,
            np.zeros((n, 0)), np.zeros((n, 0)))

    theta = np.concatenate([
        [0.1, 0.25, -0.05], rng.uniform(-0.4, 0.4, 2 * (n_teams - 1)),
        rng.uniform(-0.1, 0.1, n_ha),
    ])
    numeric = approx_fprime(theta, lambda t: _objective(t, *args)[0], 1e-7)
    assert np.allclose(numeric, _objective(theta, *args)[1], rtol=1e-3, atol=1e-4)


# --- inertness and recovery on real fits ----------------------------------------------------------

def _history(
    *, start: str = "2016-08-01", end: str = "2023-06-01",
    empty_from: str | None = None, empty_to: str | None = None, empty_effect: float = 0.0,
    trend_per_year: float = 0.0, ref: str | None = None, base_ha: float = 0.30, seed: int = 3,
) -> pd.DataFrame:
    """A synthetic league spanning a real date range.

    Generated by walking the calendar rather than by counting rounds, because these tests are about
    date windows: a fixture generator that produces the right number of matches over the wrong span
    silently leaves the tested window empty.
    """
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    pairs = [(h, a) for h in teams for a in teams if h != a]
    reference = pd.Timestamp(ref) if ref else pd.Timestamp(end)

    rows, date, i = [], pd.Timestamp(start), 0
    while date <= pd.Timestamp(end):
        home, away = pairs[i % len(pairs)]
        ha = base_ha
        if trend_per_year:
            ha += trend_per_year * max((reference - date).days, 0) / 365.25
        if empty_from is not None and pd.Timestamp(empty_from) <= date <= pd.Timestamp(
            empty_to or end
        ):
            ha += empty_effect
        rows.append({
            "date": date, "home_team": home, "away_team": away,
            "home_goals": float(rng.poisson(np.exp(0.1 + ha))),
            "away_goals": float(rng.poisson(np.exp(0.1))),
        })
        date += pd.Timedelta(days=3)
        i += 1
    return pd.DataFrame(rows)


def _fit(history, **kw):
    base = {
        "half_life_days": 100_000.0,
        "ref_date": history["date"].max() + pd.Timedelta(days=1),
        "max_goals": 12, "param_bounds": BOUNDS, "min_effective_share": 0.15, "max_iter": 500,
    }
    return fit_dixon_coles(history, **{**base, **kw})


def test_global_mode_is_byte_identical_to_no_mode_argument() -> None:
    """The seam lives inside the production fitter, so its off state must change nothing."""
    history = _history()
    default = _fit(history)
    explicit = _fit(history, ha_mode="global")
    assert np.array_equal(default.attack, explicit.attack)
    assert default.home_advantage == explicit.home_advantage
    assert default.neg_log_lik == explicit.neg_log_lik
    assert default.ha_names == () and default.ha_params == ()


def test_empty_term_recovers_a_planted_effect() -> None:
    """Plant a known home-advantage collapse over a window and read it back."""
    history = _history(empty_from="2020-06-17", empty_to="2021-05-23", empty_effect=-0.30)
    fit = _fit(history, ha_mode="empty", ha_window=("2020-06-17", "2021-05-23"))
    terms = dict(zip(fit.ha_names, fit.ha_params))
    assert terms["ha_empty"] == pytest.approx(-0.30, abs=0.12)


def test_empty_term_is_near_zero_when_nothing_happened() -> None:
    """A term that always finds an effect would be fitting noise."""
    fit = _fit(_history(), ha_mode="empty", ha_window=("2020-06-17", "2021-05-23"))
    terms = dict(zip(fit.ha_names, fit.ha_params))
    assert abs(terms["ha_empty"]) < 0.12


def test_trend_term_recovers_a_planted_decline() -> None:
    """A home advantage declining through the sample shows a POSITIVE trend coefficient, because
    the design counts years *before* the barrier — and the reported h is the current value."""
    ref = "2023-06-01"
    history = _history(start="2008-01-01", end=ref, base_ha=0.10, trend_per_year=0.02, ref=ref)
    fit = _fit(history, ha_mode="trend", ref_date=pd.Timestamp(ref))
    terms = dict(zip(fit.ha_names, fit.ha_params))
    assert terms["ha_trend"] == pytest.approx(0.02, abs=0.008)
    # The forecast uses the CURRENT home advantage, not the fifteen-year average.
    assert fit.home_advantage == pytest.approx(0.10, abs=0.05)


def test_trend_term_is_near_zero_when_home_advantage_is_stable() -> None:
    """A trend that always finds a decline would be fitting drift into noise."""
    fit = _fit(_history(start="2008-01-01", end="2023-06-01"), ha_mode="trend",
               ref_date=pd.Timestamp("2023-06-01"))
    terms = dict(zip(fit.ha_names, fit.ha_params))
    assert abs(terms["ha_trend"]) < 0.008


def test_empty_term_is_applied_when_forecasting_a_crowdless_match() -> None:
    """Regression: the term must be applied at PREDICTION, not only removed at fit time.

    An earlier version zeroed every structural term when forecasting. The empty-stadium arm
    therefore cleaned the crowd effect out of its historical estimate and then predicted the
    2020-21 matches as if crowds were present — it scored *worse* in the one season it was built
    for (+0.0006 RPS), and the arm as a whole looked like a clean null. Whether an upcoming match
    is behind closed doors is public before kickoff, so using it is leak-free.
    """
    history = _history(empty_from="2020-06-17", empty_to="2021-05-23", empty_effect=-0.30)
    fit = fit_dixon_coles(
        history, half_life_days=100_000.0, ref_date=pd.Timestamp("2021-01-01"),
        max_goals=12, param_bounds=BOUNDS, min_effective_share=0.15, max_iter=500,
        ha_mode="empty", ha_window=("2020-06-17", "2021-05-23"),
    )
    fixture = pd.DataFrame({"home_team": ["A"], "away_team": ["B"]})
    in_window = fixture.assign(date=pd.Timestamp("2021-01-05"))     # behind closed doors
    after = fixture.assign(date=pd.Timestamp("2021-09-05"))         # crowds back

    lam_empty, _ = fit.rates(in_window["home_team"], in_window["away_team"], in_window["date"])
    lam_normal, _ = fit.rates(after["home_team"], after["away_team"], after["date"])
    assert lam_empty[0] < lam_normal[0], "the crowdless forecast must carry the penalty"
    assert np.log(lam_normal[0] / lam_empty[0]) == pytest.approx(0.30, abs=0.12)


def test_trend_term_stays_zero_at_prediction() -> None:
    """The trend must NOT be applied: a match is zero years before its own barrier, so the
    forecast uses the current home advantage rather than the window average."""
    history = _history(start="2008-01-01", end="2023-06-01", base_ha=0.10, trend_per_year=0.02,
                       ref="2023-06-01")
    fit = fit_dixon_coles(
        history, half_life_days=100_000.0, ref_date=pd.Timestamp("2023-06-01"),
        max_goals=12, param_bounds=BOUNDS, min_effective_share=0.15, max_iter=500,
        ha_mode="trend",
    )
    rows = pd.DataFrame({"home_team": ["A"], "away_team": ["B"],
                         "date": [pd.Timestamp("2023-06-01")]})
    with_dates, _ = fit.rates(rows["home_team"], rows["away_team"], rows["date"])
    without, _ = fit.rates(rows["home_team"], rows["away_team"])
    assert with_dates[0] == pytest.approx(without[0])


def test_prediction_without_dates_falls_back_to_no_adjustment() -> None:
    """A caller with no date column gets the base home advantage rather than a crash."""
    fit = _fit(_history(), ha_mode="empty", ha_window=("2020-06-17", "2021-05-23"))
    lam, _ = fit.rates(pd.Series(["A"]), pd.Series(["B"]))
    assert np.isfinite(lam[0])


def test_all_arms_are_registered() -> None:
    from plmodel.eval.compare import registered_arms

    assert {"ha-trend", "ha-empty", "ha-both"} <= set(registered_arms())
