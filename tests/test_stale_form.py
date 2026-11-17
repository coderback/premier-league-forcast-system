"""The two bounded shots-form terms: `sot_form_recent` and `sot_form_e1`.

Pre-registered 2026-11-17 (NOTES.md). Production's `sot_form` reads E0 history with no time limit,
so a promoted club's form comes from its previous Premier League spell. These tests pin the five
properties the pre-registration names: the 365-day bound, that no club continuously in E0 is
touched, the E1 mapping, the fallback below the promotion minimum, and that a forecast row never
contributes its own shots.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from plmodel.model import covariates as cv

WINDOW = 3


def _spec(term: str, *, recency: int = 365, min_promotions: int = 2,
          e1: pd.DataFrame | None = None) -> cv.CovariateSpec:
    return cv.CovariateSpec(terms=(term,), sot_form_window=WINDOW, sot_form_recency_days=recency,
                            sot_form_e1_min_promotions=min_promotions, e1_rows=e1)


def _matches(rows, division: str = "E0") -> pd.DataFrame:
    """(date, season, home, away, home_sot, away_sot)."""
    return pd.DataFrame(rows, columns=["date", "season", "home_team", "away_team",
                                       "home_sot", "away_sot"]).assign(
        date=lambda f: pd.to_datetime(f["date"]), division=division, played=True,
        home_goals=0, away_goals=0)


# --- 1. the bound ----------------------------------------------------------------------------------

def test_the_bounded_mean_matches_a_brute_force_reading_of_its_definition() -> None:
    """Mean of the previous `window` recorded entries on EARLIER days within `max_age` days."""
    rng = np.random.default_rng(23)
    n = 400
    team = rng.choice(list("ABCDEFG"), n)
    day = np.sort(rng.integers(0, 2000, n))
    diff = np.where(rng.random(n) < 0.2, np.nan, rng.integers(-6, 7, n).astype(float))
    for max_age in (None, 30, 365):
        got = cv.trailing_mean_bounded(team, day, diff, window=WINDOW, max_age_days=max_age)
        for i in range(n):
            prior = [j for j in range(n) if team[j] == team[i] and day[j] < day[i]
                     and not np.isnan(diff[j])
                     and (max_age is None or day[i] - day[j] <= max_age)]
            prior = sorted(prior, key=lambda j: (day[j], j))[-WINDOW:]
            want = np.mean(diff[prior]) if prior else np.nan
            np.testing.assert_allclose(got[i], want, equal_nan=True, err_msg=f"entry {i}")


def test_unbounded_it_is_production_s_definition_exactly() -> None:
    """With no bound the new mean IS trailing_differential, so the arm differs only in the bound."""
    rng = np.random.default_rng(5)
    rows = []
    for k in range(120):
        h, a = rng.choice(list("ABCDEF"), 2, replace=False)
        hs, as_ = (np.nan, np.nan) if rng.random() < 0.15 else rng.integers(0, 10, 2)
        rows.append((str((pd.Timestamp("2020-08-01") + pd.Timedelta(days=3 * k)).date()),
                     "2020-21", h, a, hs, as_))
    frame = _matches(rows)
    home, away = cv.trailing_differential(frame, "home_sot", "away_sot", window=WINDOW)
    side = cv._side_appearances(frame)
    got = cv.trailing_mean_bounded(side["team"], side["day"], side["diff"],
                                   window=WINDOW, max_age_days=None)
    n = len(frame)
    np.testing.assert_allclose(got[:n], home, equal_nan=True)
    np.testing.assert_allclose(got[n:], away, equal_nan=True)


def test_a_club_back_after_a_long_gap_starts_with_its_term_off() -> None:
    """Old-spell form is dropped; the new spell refills the window as it is played."""
    old = _matches([("2015-01-10", "2014-15", "P", "X", 9, 0),
                    ("2015-02-10", "2014-15", "X", "P", 0, 9)])
    new = _matches([("2026-08-15", "2026-27", "P", "Y", 2, 4),
                    ("2026-08-22", "2026-27", "Y", "P", 5, 1)])
    frame = pd.concat([old, new], ignore_index=True)
    bounded, _ = cv.per_side_values(frame, frame.iloc[:0], "sot_form_recent",
                                    _spec("sot_form_recent"), division="E0")
    stale, _ = cv.per_side_values(frame, frame.iloc[:0], "sot_form", _spec("sot_form"),
                                  division="E0")
    assert stale[2] == 9.0                 # production reads 2015's shots in 2026
    assert np.isnan(bounded[2])            # the bounded term does not
    _, away_bounded = cv.per_side_values(frame, frame.iloc[:0], "sot_form_recent",
                                         _spec("sot_form_recent"), division="E0")
    assert away_bounded[3] == -2.0         # and refills from the new spell


# --- 2. no other club ----------------------------------------------------------------------------

@pytest.mark.integration
def test_no_club_continuously_in_the_premier_league_has_its_term_changed(cfg, corpus) -> None:
    """The bound is 365 days because no continuous club's window reaches back that far."""
    e0 = corpus[(corpus["division"] == "E0") & corpus["played"]].sort_values(
        "date", kind="stable").reset_index(drop=True)
    e0 = e0[e0["season"] >= "2000-01"].reset_index(drop=True)
    spec = dataclasses.replace(cfg.model.covariate_spec(terms=("sot_form_recent",)))
    plain = cfg.model.covariate_spec(terms=("sot_form",))
    bh, ba = cv.per_side_values(e0, e0.iloc[:0], "sot_form_recent", spec, division="E0")
    ph, pa = cv.per_side_values(e0, e0.iloc[:0], "sot_form", plain, division="E0")
    seasons = sorted(e0["season"].unique())
    previous = dict(zip(seasons[1:], seasons[:-1]))
    members = {s: set(g["home_team"]) for s, g in e0.groupby("season")}

    def continuous(team_col: str) -> np.ndarray:
        return np.array([s in previous and t in members[previous[s]]
                         for s, t in zip(e0["season"], e0[team_col])])

    for values, plain_values, col in ((bh, ph, "home_team"), (ba, pa, "away_team")):
        keep = continuous(col) & ~np.isnan(plain_values)
        np.testing.assert_array_equal(values[keep], plain_values[keep])
    # ...and it does change some promoted clubs, or the arm would be the baseline.
    assert (np.isnan(bh) & ~np.isnan(ph)).sum() + (np.isnan(ba) & ~np.isnan(pa)).sum() > 0


# --- 3. the E1 map -------------------------------------------------------------------------------

def _promotion_world():
    """Three clubs promoted into 2021-22 after a Championship season, with known shot means."""
    e1_rows, e0_rows = [], []
    start = pd.Timestamp("2020-09-01")
    targets = {"P1": (4.0, -1.0), "P2": (2.0, -2.0), "P3": (0.0, -3.0)}  # (x_E1, y_E0)
    k = 0
    for club, (x, _) in targets.items():
        for i in range(WINDOW):
            d = (start + pd.Timedelta(days=7 * k)).date(); k += 1
            e1_rows.append((str(d), "2020-21", club, "C", 5 + x, 5))
    # E0 2020-21: the clubs that are NOT promoted.
    for i in range(4):
        d = (pd.Timestamp("2020-10-01") + pd.Timedelta(days=7 * i)).date()
        e0_rows.append((str(d), "2020-21", "S1", "S2", 3, 3))
    k = 0
    for club, (_, y) in targets.items():
        for i in range(WINDOW):
            d = (pd.Timestamp("2021-08-14") + pd.Timedelta(days=7 * k)).date(); k += 1
            e0_rows.append((str(d), "2021-22", club, "S1", 4 + y, 4))
    return _matches(e0_rows), _matches(e1_rows, division="E1")


def test_the_map_is_the_ols_line_through_completed_promotions() -> None:
    e0, e1 = _promotion_world()
    a, b = cv.promotion_map(e0, e1, window=WINDOW, min_promotions=3)
    # y = -3 + 0.5 x through (4,-1), (2,-2), (0,-3).
    assert a == pytest.approx(-3.0) and b == pytest.approx(0.5)


def test_e1_entries_enter_the_window_mapped_onto_the_e0_scale() -> None:
    """A promoted club's first E0 match reads its Championship form through the map."""
    e0, e1 = _promotion_world()
    spec = _spec("sot_form_e1", min_promotions=3, e1=e1)
    home, _ = cv.per_side_values(e0, e0.iloc[:0], "sot_form_e1", spec, division="E0")
    first_p1 = e0.index[(e0["home_team"] == "P1")][0]
    assert home[first_p1] == pytest.approx(-3.0 + 0.5 * 4.0)
    recent, _ = cv.per_side_values(e0, e0.iloc[:0], "sot_form_recent",
                                   _spec("sot_form_recent"), division="E0")
    assert np.isnan(recent[first_p1])      # without E1 the term is off there


def test_the_map_is_identical_whatever_the_process_hash_seed() -> None:
    """Promoted clubs were once walked in set order, which Python randomises per process.

    The OLS then summed in a different order each run, and identical `pl compare` runs disagreed in
    P(better). Twelve clubs with irrational-looking shot means make the sums order-sensitive.
    """
    import subprocess
    import sys

    code = (
        "import numpy as np, pandas as pd\n"
        "from plmodel.model import covariates as cv\n"
        "rng = np.random.default_rng(3); W = 3; e0 = []; e1 = []\n"
        "clubs = [f'P{i:02d}' for i in range(12)]\n"
        "for k in range(4): e0.append(('2020-10-0%d' % (k + 1), '2020-21', 'S1', 'S2', 3.0, 3.0))\n"
        "for c in clubs:\n"
        "    for i in range(W):\n"
        "        e1.append((str((pd.Timestamp('2020-09-01') + pd.Timedelta(days=len(e1))).date()), '2020-21',"
        " c, 'C', float(rng.random() * 7), float(rng.random() * 3)))\n"
        "        e0.append((str((pd.Timestamp('2021-08-14') + pd.Timedelta(days=len(e0))).date()), '2021-22',"
        " c, 'S1', float(rng.random() * 5), float(rng.random() * 6)))\n"
        "f = lambda rows, d: pd.DataFrame(rows, columns=['date','season','home_team','away_team',"
        "'home_sot','away_sot']).assign(date=lambda x: pd.to_datetime(x['date']), division=d)\n"
        "print(repr(cv.promotion_map(f(e0,'E0'), f(e1,'E1'), window=W, min_promotions=3)))\n"
    )
    import os

    outs = set()
    for seed in ("1", "2", "3", "4"):
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              env={**os.environ, "PYTHONHASHSEED": seed})
        assert done.returncode == 0, done.stderr
        outs.add(done.stdout)
    assert len(outs) == 1, outs


# --- 4. the fallback -----------------------------------------------------------------------------

def test_below_the_promotion_minimum_the_e1_term_is_the_bounded_term() -> None:
    e0, e1 = _promotion_world()
    assert cv.promotion_map(e0, e1, window=WINDOW, min_promotions=4) is None
    spec = _spec("sot_form_e1", min_promotions=4, e1=e1)
    with_e1, _ = cv.per_side_values(e0, e0.iloc[:0], "sot_form_e1", spec, division="E0")
    recent, _ = cv.per_side_values(e0, e0.iloc[:0], "sot_form_recent",
                                   _spec("sot_form_recent"), division="E0")
    np.testing.assert_array_equal(with_e1, recent)


def test_the_e1_term_refuses_to_run_without_championship_rows() -> None:
    e0, _ = _promotion_world()
    with pytest.raises(cv.CovariateError, match="no Championship rows"):
        cv.per_side_values(e0, e0.iloc[:0], "sot_form_e1", _spec("sot_form_e1"), division="E0")


# --- 5. no forecast row reads its own shots ----------------------------------------------------

@pytest.mark.parametrize("term", ["sot_form_recent", "sot_form_e1"])
def test_a_forecast_row_never_reads_its_own_or_a_later_rows_shots(term: str) -> None:
    e0, e1 = _promotion_world()
    history, ahead = e0.iloc[:-2], e0.iloc[-2:]
    spec = _spec(term, min_promotions=3, e1=e1)
    first, _ = cv.per_side_values(ahead, history, term, spec, division="E0")
    flipped = ahead.assign(home_sot=[0, 0], away_sot=[99, 99])
    second, _ = cv.per_side_values(flipped, history, term, spec, division="E0")
    np.testing.assert_array_equal(first, second)
    # Both forecast rows read the history alone, so a block cannot see its own earlier match.
    assert first[0] == first[1]
