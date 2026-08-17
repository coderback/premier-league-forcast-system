"""Rolling-origin walk-forward splitting, with leak-freedom enforced by assertion.

Protocol: for each **matchday** in the test span — one distinct match date — train on everything
dated strictly before it, predict that day's matches, roll forward. A hard :class:`LeakageError`
fires on every split, not on a sample of them.

Why a date and not a round
--------------------------
football-data.co.uk carries no round column, and a reconstructed one is not monotonic in time: a
fixture postponed in December and replayed in March still belongs to round 17, so "train strictly
before round 17" would train on data from after part of round 17 was played. A date has no such
ambiguity. Kickoff times only exist from 2019/20, so the barrier is date-granular; every match
sharing a date shares a barrier.

Why splits are integers
-----------------------
The corpus is sorted by date, so a barrier makes the training set a *prefix* of the frame and the
test set a *contiguous slice*. A split is therefore three integers plus a timestamp, not two
materialised frames — the whole ten-season walk (1,153 splits) costs almost nothing to hold, and
every arm in a comparison replays the identical splits by construction rather than by convention.

**No k-fold anywhere, ever.** It leaks future into past for time series.

The refit cadence seam
----------------------
Prediction always happens at every matchday. *Fitting* may happen less often: with
``refit_every = n``, a split reuses the most recent refit barrier at or before its own. That keeps
the protocol leak-free either way (a stale fit saw strictly less data, never more) and makes "does
weekly refitting actually matter?" a measurable question rather than an assumption — the WC2026
project found its live-update channel null three separate times. ``refit_every = 1`` is the
default and is byte-identical to having no seam at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import pandas as pd


class LeakageError(AssertionError):
    """Raised when any training row is dated on or after the first test row.

    An AssertionError subclass by intent: this is a violated invariant, not a recoverable
    condition, and nothing in the codebase may catch it.
    """


@dataclass(frozen=True)
class Split:
    """One rolling-origin split: train on ``[0, train_end)``, predict ``[test_start, test_stop)``.

    ``fit_barrier`` is the date whose fitted parameters this split uses. It equals ``barrier``
    unless a refit cadence is in force, in which case it is the most recent refit at or before the
    barrier — always in the past, never ahead of it.
    """

    index: int
    barrier: pd.Timestamp
    fit_barrier: pd.Timestamp
    train_end: int
    test_start: int
    test_stop: int
    is_refit: bool

    @property
    def n_train(self) -> int:
        return self.train_end

    @property
    def n_test(self) -> int:
        return self.test_stop - self.test_start

    def train(self, matches: pd.DataFrame) -> pd.DataFrame:
        return matches.iloc[: self.train_end]

    def test(self, matches: pd.DataFrame) -> pd.DataFrame:
        return matches.iloc[self.test_start : self.test_stop]


def _check_sorted(matches: pd.DataFrame) -> None:
    """The integer-slice representation is only valid on a date-sorted frame."""
    if "date" not in matches.columns:
        raise ValueError("matches frame needs a 'date' column")
    dates = matches["date"]
    if not dates.is_monotonic_increasing:
        raise ValueError(
            "matches must be sorted by date before splitting; an unsorted frame would make the "
            "prefix/slice representation silently wrong"
        )


def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, barrier: pd.Timestamp) -> None:
    """The standing invariant, checked on every split.

    Structural rather than procedural: the check runs whether or not anyone remembered to think
    about it, and it also verifies the barrier itself rather than only the two frames' relative
    order — a split can be internally consistent and still be built at the wrong barrier.
    """
    if len(test) == 0:
        raise LeakageError(f"empty test set at barrier {barrier.date()}")
    if len(train) == 0:
        return
    train_max = train["date"].max()
    test_min = test["date"].min()
    if train_max >= test_min:
        raise LeakageError(
            f"temporal leakage at barrier {barrier.date()}: "
            f"max train date {train_max.date()} >= min test date {test_min.date()}"
        )
    if train_max >= barrier:
        raise LeakageError(
            f"training data at or after the barrier {barrier.date()}: max train date {train_max.date()}"
        )
    if test_min != barrier:
        raise LeakageError(
            f"test set does not start at its barrier {barrier.date()}: first test date {test_min.date()}"
        )


def training_frame(source: pd.DataFrame, barrier: pd.Timestamp) -> pd.DataFrame:
    """Rows of ``source`` usable for training at ``barrier`` — strictly before it.

    The canonical way to pull *additional* training data (a second division for the multi-tier
    fit, an external feed) into a split. Routing every such join through here means extra data
    cannot bypass the barrier just because it did not come from the prediction frame.
    """
    out = source[source["date"] < barrier]
    if len(out) and out["date"].max() >= barrier:
        raise LeakageError(f"training_frame produced rows at or after {barrier.date()}")
    return out


def barrier_dates(
    matches: pd.DataFrame, *, first_season: str | None = None, last_season: str | None = None
) -> list[pd.Timestamp]:
    """The distinct match dates in the test span, in order — one barrier each."""
    _check_sorted(matches)
    span = matches
    if first_season is not None:
        span = span[span["season"] >= first_season]
    if last_season is not None:
        span = span[span["season"] <= last_season]
    return [pd.Timestamp(d) for d in sorted(span["date"].unique())]


def walk_forward(
    matches: pd.DataFrame,
    *,
    first_season: str | None = None,
    last_season: str | None = None,
    refit_every: int = 1,
    min_train_matches: int = 0,
) -> list[Split]:
    """Build the rolling-origin splits over the test span.

    ``matches`` is the prediction universe (typically one division), sorted by date. Training uses
    every row of it dated before each barrier — including rows outside the test span, so a test
    span starting in 2016/17 still trains on the full history behind it.

    Returns a list rather than a generator: the splits are the yardstick, and every arm in a
    comparison must replay exactly the same ones. Materialising them once makes that structural.
    """
    if refit_every < 1:
        raise ValueError(f"refit_every must be at least 1; got {refit_every}")
    _check_sorted(matches)

    dates = matches["date"].to_numpy()
    splits: list[Split] = []
    fit_barrier: pd.Timestamp | None = None
    for i, barrier in enumerate(barrier_dates(matches, first_season=first_season,
                                              last_season=last_season)):
        # searchsorted on a sorted date column: the prefix before the barrier is the training set,
        # the run of rows equal to it is the test set.
        train_end = int(dates.searchsorted(barrier.to_datetime64(), side="left"))
        test_stop = int(dates.searchsorted(barrier.to_datetime64(), side="right"))
        if train_end < min_train_matches:
            continue
        is_refit = (len(splits) % refit_every) == 0
        if is_refit or fit_barrier is None:
            fit_barrier = barrier
        splits.append(
            Split(
                index=len(splits),
                barrier=barrier,
                fit_barrier=fit_barrier,
                train_end=train_end,
                test_start=train_end,
                test_stop=test_stop,
                is_refit=is_refit,
            )
        )
    if not splits:
        raise ValueError(
            "no splits produced — check the test span against the corpus and min_train_matches"
        )
    return splits


def validate_splits(matches: pd.DataFrame, splits: Sequence[Split]) -> None:
    """Run the leakage assertion over every split, plus the walk's own structural invariants.

    Called by the harness before any arm runs. The per-split check is the non-negotiable; the
    cross-split checks catch a splitter bug that leaves each split individually valid — barriers
    out of order, or a test set that overlaps its neighbour, would both do that.
    """
    previous: Split | None = None
    for split in splits:
        assert_no_leakage(split.train(matches), split.test(matches), split.barrier)
        if split.fit_barrier > split.barrier:
            raise LeakageError(
                f"split {split.index} fits at {split.fit_barrier.date()}, after its barrier "
                f"{split.barrier.date()}"
            )
        if previous is not None:
            if split.barrier <= previous.barrier:
                raise LeakageError(
                    f"barriers out of order: {previous.barrier.date()} then {split.barrier.date()}"
                )
            if split.test_start < previous.test_stop:
                raise LeakageError(
                    f"split {split.index} overlaps split {previous.index}"
                )
        previous = split


def iter_splits(matches: pd.DataFrame, splits: Sequence[Split]) -> Iterator[tuple[Split, pd.DataFrame, pd.DataFrame]]:
    """Yield ``(split, train, test)``, asserting leak-freedom on each one as it is produced."""
    for split in splits:
        train, test = split.train(matches), split.test(matches)
        assert_no_leakage(train, test, split.barrier)
        yield split, train, test


def split_summary(splits: Sequence[Split]) -> dict[str, object]:
    """A JSON-serialisable description of the walk, for the harness report."""
    return {
        "n_splits": len(splits),
        "n_refits": sum(1 for s in splits if s.is_refit),
        "n_test_matches": sum(s.n_test for s in splits),
        "first_barrier": str(splits[0].barrier.date()),
        "last_barrier": str(splits[-1].barrier.date()),
        "min_train_matches": min(s.n_train for s in splits),
        "max_train_matches": max(s.n_train for s in splits),
        "max_test_per_barrier": max(s.n_test for s in splits),
    }
