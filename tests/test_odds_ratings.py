"""The market-informed `odds-ratings` arm: price inversion, the ratings fit, and the leakage guard.

The test that matters most is the leakage one. This arm reads bookmakers' prices, and the whole
claim that it is still a forecast rests on it never reading the price of the match it forecasts, or
of anything on or after that date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.model.odds_ratings import (
    OddsRatingsError, fit_odds_ratings, implied_rates, price_table, three_way,
)

MAX_GOALS = 12


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_the_inversion_recovers_the_rates_that_made_the_price() -> None:
    rng = np.random.default_rng(0)
    lam, mu = rng.uniform(0.3, 3.5, 500), rng.uniform(0.2, 3.0, 500)
    probs = three_way(lam, mu, MAX_GOALS)
    rates = implied_rates(probs, max_goals=MAX_GOALS, tolerance=1e-12, max_iter=100)
    np.testing.assert_allclose(rates[:, 0], lam, rtol=1e-8)
    np.testing.assert_allclose(rates[:, 1], mu, rtol=1e-8)


def test_an_inversion_that_does_not_converge_raises() -> None:
    probs = three_way(np.array([3.4]), np.array([0.3]), MAX_GOALS)
    with pytest.raises(OddsRatingsError, match="did not converge"):
        implied_rates(probs, max_goals=MAX_GOALS, tolerance=1e-12, max_iter=1)


def test_bet365_is_preferred_and_william_hill_fills_its_gaps() -> None:
    frame = pd.DataFrame({
        "B365H": [2.0, np.nan], "B365D": [3.4, np.nan], "B365A": [3.8, np.nan],
        "WHH": [5.0, 1.8], "WHD": [3.6, 3.5], "WHA": [1.7, 4.5],
    })
    prices = price_table(frame, ("bet365", "william_hill"))
    assert prices[0, 0] > prices[0, 2]          # bet365's home favourite, not William Hill's away one
    assert prices[1, 0] > prices[1, 2]          # William Hill's price where bet365 has none
    assert np.isnan(price_table(frame.iloc[:, :0], ("bet365",))).all()


def _league(n_rounds: int = 30) -> tuple[pd.DataFrame, dict[str, float]]:
    """Six clubs whose implied rates come exactly from known strengths."""
    attack = {"A": 0.4, "B": 0.2, "C": 0.0, "D": 0.0, "E": -0.2, "F": -0.4}
    rows = []
    day = pd.Timestamp("2020-08-01")
    clubs = list(attack)
    for r in range(n_rounds):
        for i, h in enumerate(clubs):
            a = clubs[(i + r + 1) % len(clubs)]
            if a == h:
                continue
            rows.append({"date": day, "home_team": h, "away_team": a,
                         "lam": np.exp(0.1 + 0.25 + attack[h]), "mu": np.exp(0.1 + attack[a])})
        day += pd.Timedelta(days=4)
    return pd.DataFrame(rows), attack


def test_the_fit_recovers_known_strengths() -> None:
    history, attack = _league()
    rates = history[["lam", "mu"]].to_numpy()
    ratings = fit_odds_ratings(history, rates, ref_date=history["date"].max() + pd.Timedelta(days=1),
                               half_life_days=10_000.0, ridge=1e-9, lookback_days=10_000.0)
    fitted = dict(zip(ratings.teams, ratings.attack))
    centred = {k: v - np.mean(list(fitted.values())) for k, v in fitted.items()}
    for club, value in attack.items():
        assert centred[club] == pytest.approx(value, abs=1e-6)
    assert ratings.home_advantage == pytest.approx(0.25, abs=1e-6)


def test_nothing_on_or_after_the_barrier_is_read() -> None:
    """Change every price dated on or after the barrier to nonsense: the fit must not move."""
    history, _ = _league()
    rates = history[["lam", "mu"]].to_numpy()
    barrier = history["date"].iloc[len(history) // 2]
    kwargs = dict(ref_date=barrier, half_life_days=30.0, ridge=0.05, lookback_days=1500.0)
    clean = fit_odds_ratings(history, rates, **kwargs)
    poisoned = rates.copy()
    poisoned[(history["date"] >= barrier).to_numpy()] = [[9.0, 0.1]]
    dirty = fit_odds_ratings(history, poisoned, **kwargs)
    assert clean.teams == dirty.teams
    np.testing.assert_array_equal(clean.attack, dirty.attack)
    assert clean.n_obs == int((history["date"] < barrier).sum())


def test_no_priced_history_is_refused_rather_than_guessed() -> None:
    history, _ = _league(2)
    with pytest.raises(OddsRatingsError, match="no priced match"):
        fit_odds_ratings(history, np.full((len(history), 2), np.nan),
                         ref_date=history["date"].max(), half_life_days=30.0, ridge=0.05,
                         lookback_days=1500.0)


def test_the_arm_never_reads_a_later_price_in_a_walk(cfg) -> None:
    """End to end through the harness: poisoning the prices of the last matchday's matches must
    leave every earlier forecast bit-for-bit unchanged."""
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, arm_track, run_arm

    rng = np.random.default_rng(3)
    clubs = ["Arsenal", "Chelsea", "Everton", "Fulham"]
    rows = []
    for d in range(16):
        day = pd.Timestamp("2024-08-01") + pd.Timedelta(days=7 * d)
        for k in range(2):
            h, a = clubs[(d + k) % 4], clubs[(d + k + 1 + d % 2) % 4]
            if h == a:
                a = clubs[(d + k + 2) % 4]
            rows.append({"date": day, "season": "2024-25", "division": "E0", "home_team": h,
                         "away_team": a, "result": rng.choice(["H", "D", "A"]),
                         "home_goals": 1.0, "away_goals": 1.0,
                         "B365H": rng.uniform(1.6, 3.5), "B365D": 3.4, "B365A": rng.uniform(2.0, 5.0)})
    matches = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    splits = walk_forward(matches, min_train_matches=4)
    clean, _ = run_arm(ArmSpec.parse("odds-ratings"), matches, splits, cfg)
    poisoned = matches.copy()
    last = poisoned["date"] == poisoned["date"].max()
    poisoned.loc[last, ["B365H", "B365D", "B365A"]] = [1.05, 15.0, 30.0]
    dirty, _ = run_arm(ArmSpec.parse("odds-ratings"), poisoned, splits, cfg)
    np.testing.assert_array_equal(clean, dirty)

    track = arm_track("odds-ratings")
    assert track.track == "market-informed" and track.reads_match_prices is False
