"""Walk-forward splitting and the leakage invariant.

Leak-freedom is the one property that, if broken, makes every downstream number flattering and
wrong at the same time — and a leaking backtest does not look broken, it looks *good*. So the
tests here spend most of their effort trying to build a split that leaks and asserting it is
refused.
"""
from __future__ import annotations

import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval.backtest import (
    LeakageError, Split, assert_no_leakage, barrier_dates, iter_splits, split_summary,
    training_frame, validate_splits, walk_forward,
)


def _matches(dates: list[str], *, season: str = "2024-25", division: str = "E0") -> pd.DataFrame:
    """A minimal date-sorted frame; two teams per date so counts are unambiguous."""
    stamps = pd.to_datetime(sorted(dates))
    return pd.DataFrame(
        {
            "date": stamps,
            "season": season,
            "division": division,
            "home_team": [f"H{i}" for i in range(len(stamps))],
            "away_team": [f"A{i}" for i in range(len(stamps))],
        }
    )


# --- barriers ----------------------------------------------------------------------------------

def test_one_barrier_per_distinct_date() -> None:
    df = _matches(["2024-08-17", "2024-08-17", "2024-08-24", "2024-08-31"])
    assert barrier_dates(df) == [
        pd.Timestamp("2024-08-17"), pd.Timestamp("2024-08-24"), pd.Timestamp("2024-08-31")
    ]


def test_barriers_are_restricted_to_the_span() -> None:
    df = pd.concat([
        _matches(["2015-08-17"], season="2015-16"),
        _matches(["2016-08-17", "2016-08-24"], season="2016-17"),
    ], ignore_index=True)
    assert barrier_dates(df, first_season="2016-17") == [
        pd.Timestamp("2016-08-17"), pd.Timestamp("2016-08-24")
    ]


def test_unsorted_frame_is_refused() -> None:
    """The prefix/slice representation is only valid on a sorted frame; an unsorted one would be
    silently wrong rather than loudly wrong."""
    df = _matches(["2024-08-17", "2024-08-24"]).iloc[::-1]
    with pytest.raises(ValueError, match="sorted by date"):
        walk_forward(df)


# --- the walk ----------------------------------------------------------------------------------

def test_train_is_everything_strictly_before_the_barrier() -> None:
    df = _matches(["2024-08-17", "2024-08-24", "2024-08-24", "2024-08-31"])
    splits = walk_forward(df)
    second = splits[1]
    assert second.barrier == pd.Timestamp("2024-08-24")
    assert second.n_train == 1                      # only the 17th
    assert second.n_test == 2                       # both matches on the 24th
    assert (second.train(df)["date"] < second.barrier).all()
    assert (second.test(df)["date"] == second.barrier).all()


def test_first_barrier_has_no_training_data() -> None:
    splits = walk_forward(_matches(["2024-08-17", "2024-08-24"]))
    assert splits[0].n_train == 0


def test_min_train_matches_skips_the_thin_start() -> None:
    df = _matches(["2024-08-%02d" % d for d in range(1, 11)])
    splits = walk_forward(df, min_train_matches=5)
    assert splits[0].n_train >= 5
    assert all(s.n_train >= 5 for s in splits)


def test_splits_partition_the_test_span() -> None:
    """Every match in the span is predicted exactly once — no gaps, no double counting."""
    df = _matches(["2024-08-17", "2024-08-17", "2024-08-24", "2024-08-31", "2024-08-31"])
    splits = walk_forward(df)
    assert sum(s.n_test for s in splits) == len(df)
    covered = [i for s in splits for i in range(s.test_start, s.test_stop)]
    assert covered == list(range(len(df)))


def test_walk_is_deterministic() -> None:
    """The splits are the yardstick; a non-deterministic walk makes every backtest irreproducible."""
    df = _matches(["2024-08-17", "2024-08-24", "2024-08-31"])
    assert walk_forward(df) == walk_forward(df)


def test_empty_span_raises() -> None:
    df = _matches(["2024-08-17"])
    with pytest.raises(ValueError, match="no splits produced"):
        walk_forward(df, first_season="2030-31")


# --- the leakage invariant ----------------------------------------------------------------------

def test_leakage_error_fires_on_a_deliberately_corrupted_split() -> None:
    """The definition-of-done check: hand the assertion a split that leaks and watch it refuse.

    Built by moving the barrier forward one matchday while leaving the training prefix in place,
    which is exactly the shape of an off-by-one in a splitter.
    """
    df = _matches(["2024-08-17", "2024-08-24", "2024-08-31"])
    splits = walk_forward(df)
    corrupted = Split(
        index=0,
        barrier=splits[1].barrier,
        fit_barrier=splits[1].barrier,
        train_end=splits[2].train_end,      # trains through the 24th...
        test_start=splits[1].test_start,    # ...but predicts the 24th
        test_stop=splits[1].test_stop,
        is_refit=True,
    )
    with pytest.raises(LeakageError, match="temporal leakage"):
        assert_no_leakage(corrupted.train(df), corrupted.test(df), corrupted.barrier)


def test_leakage_error_fires_when_training_reaches_the_barrier() -> None:
    train = _matches(["2024-08-17", "2024-08-24"])
    test = _matches(["2024-08-24"])
    with pytest.raises(LeakageError):
        assert_no_leakage(train, test, pd.Timestamp("2024-08-24"))


def test_leakage_error_fires_when_test_does_not_start_at_its_barrier() -> None:
    """A split can be internally consistent and still be built at the wrong barrier."""
    train = _matches(["2024-08-17"])
    test = _matches(["2024-08-31"])
    with pytest.raises(LeakageError, match="does not start at its barrier"):
        assert_no_leakage(train, test, pd.Timestamp("2024-08-24"))


def test_empty_test_set_is_refused() -> None:
    with pytest.raises(LeakageError, match="empty test set"):
        assert_no_leakage(_matches(["2024-08-17"]), _matches([]).iloc[:0],
                          pd.Timestamp("2024-08-24"))


def test_validate_splits_catches_out_of_order_barriers() -> None:
    df = _matches(["2024-08-17", "2024-08-24", "2024-08-31"])
    splits = walk_forward(df)
    with pytest.raises(LeakageError, match="out of order"):
        validate_splits(df, [splits[1], splits[0]])


def test_validate_splits_catches_a_fit_from_the_future() -> None:
    df = _matches(["2024-08-17", "2024-08-24", "2024-08-31"])
    splits = walk_forward(df)
    ahead = Split(**{**splits[1].__dict__, "fit_barrier": splits[2].barrier})
    with pytest.raises(LeakageError, match="after its barrier"):
        validate_splits(df, [ahead])


def test_validate_splits_accepts_the_real_walk() -> None:
    df = _matches(["2024-08-%02d" % d for d in range(1, 15)])
    validate_splits(df, walk_forward(df))


def test_iter_splits_asserts_as_it_goes() -> None:
    df = _matches(["2024-08-17", "2024-08-24", "2024-08-31"])
    produced = list(iter_splits(df, walk_forward(df)))
    assert len(produced) == 3
    for split, train, test in produced:
        assert len(train) == split.n_train and len(test) == split.n_test


# --- extra training data cannot bypass the barrier ----------------------------------------------

def test_training_frame_filters_a_second_source() -> None:
    """The multi-tier fit joins another division's rows; they face the same barrier."""
    other = _matches(["2024-08-10", "2024-08-24", "2024-09-01"], division="E1")
    out = training_frame(other, pd.Timestamp("2024-08-24"))
    assert list(out["date"]) == [pd.Timestamp("2024-08-10")]


def test_training_frame_on_an_empty_source() -> None:
    empty = _matches([]).iloc[:0]
    assert len(training_frame(empty, pd.Timestamp("2024-08-24"))) == 0


# --- the refit cadence seam ---------------------------------------------------------------------

def test_refit_every_one_refits_at_every_barrier() -> None:
    """The seam's off position: every split fits at its own barrier."""
    df = _matches(["2024-08-%02d" % d for d in range(1, 9)])
    splits = walk_forward(df, refit_every=1)
    assert all(s.is_refit for s in splits)
    assert all(s.fit_barrier == s.barrier for s in splits)


def test_refit_cadence_reuses_the_most_recent_fit() -> None:
    df = _matches(["2024-08-%02d" % d for d in range(1, 9)])
    splits = walk_forward(df, refit_every=3)
    assert [s.is_refit for s in splits] == [True, False, False, True, False, False, True, False]
    # Splits 1 and 2 reuse split 0's fit; 4 and 5 reuse split 3's.
    assert splits[1].fit_barrier == splits[0].barrier
    assert splits[2].fit_barrier == splits[0].barrier
    assert splits[4].fit_barrier == splits[3].barrier


def test_a_stale_fit_is_never_from_the_future() -> None:
    """The cadence stays leak-free either way: a reused fit saw strictly less data, never more."""
    df = _matches(["2024-08-%02d" % d for d in range(1, 15)])
    for cadence in (1, 2, 5, 100):
        splits = walk_forward(df, refit_every=cadence)
        assert all(s.fit_barrier <= s.barrier for s in splits)
        validate_splits(df, splits)


def test_cadence_does_not_change_the_prediction_units() -> None:
    """Fitting less often must not change *what* is predicted, only with which parameters."""
    df = _matches(["2024-08-%02d" % d for d in range(1, 15)])
    base = walk_forward(df, refit_every=1)
    sparse = walk_forward(df, refit_every=4)
    assert [(s.barrier, s.test_start, s.test_stop) for s in base] == \
           [(s.barrier, s.test_start, s.test_stop) for s in sparse]


def test_invalid_cadence_raises() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        walk_forward(_matches(["2024-08-17"]), refit_every=0)


# --- the real corpus ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus() -> pd.DataFrame:
    cfg = load_config()
    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    df = pd.read_parquet(path)
    return df[df["division"] == cfg.backtest.prediction_division].reset_index(drop=True)


@pytest.mark.integration
def test_real_walk_matches_the_planned_yardstick(corpus: pd.DataFrame) -> None:
    """The committed figures: 3,800 matches over 1,153 barriers in the test decade."""
    cfg = load_config()
    splits = walk_forward(
        corpus,
        first_season=cfg.backtest.test_span.first_season,
        last_season=cfg.backtest.test_span.last_season,
        refit_every=cfg.backtest.refit_every,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    summary = split_summary(splits)
    assert summary["n_test_matches"] == 3800
    assert summary["n_splits"] == 1153
    assert summary["first_barrier"].startswith("2016-08")
    assert summary["max_test_per_barrier"] <= 10   # a 20-team round is ten matches


@pytest.mark.integration
def test_real_walk_is_leak_free(corpus: pd.DataFrame) -> None:
    """The invariant, run over every one of the real splits rather than a sample."""
    cfg = load_config()
    splits = walk_forward(
        corpus,
        first_season=cfg.backtest.test_span.first_season,
        last_season=cfg.backtest.test_span.last_season,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    validate_splits(corpus, splits)


@pytest.mark.integration
def test_real_walk_trains_on_the_full_history(corpus: pd.DataFrame) -> None:
    """A test span opening in 2016/17 must still see the 23 seasons behind it."""
    cfg = load_config()
    splits = walk_forward(
        corpus,
        first_season=cfg.backtest.test_span.first_season,
        last_season=cfg.backtest.test_span.last_season,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    assert splits[0].n_train > 8_000
    assert splits[-1].n_train == len(corpus) - splits[-1].n_test


@pytest.mark.integration
def test_sensitivity_span_is_a_usable_second_decade(corpus: pd.DataFrame) -> None:
    cfg = load_config()
    splits = walk_forward(
        corpus,
        first_season=cfg.backtest.sensitivity_span.first_season,
        last_season=cfg.backtest.sensitivity_span.last_season,
        min_train_matches=cfg.backtest.min_train_matches,
    )
    assert split_summary(splits)["n_test_matches"] == 3800
    validate_splits(corpus, splits)
