"""The joint multi-division fit: what it may see, and that it actually uses it.

Two properties carry this arm. The lower divisions must reach the fit **only** through the
splitter's own truncation, so a wider corpus cannot smuggle a result across the barrier that the
prediction frame would have been stopped at; and the arm must genuinely fit on them, because an
arm that silently falls back to the prediction division is the baseline wearing a different name
and its null would say nothing.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval.backtest import walk_forward
from plmodel.eval.compare import ArmSpec, _tier_identity, run_arm


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def corpus(cfg):
    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    frame = pd.read_parquet(path, columns=["date", "division", "season", "played", "result",
                                           "home_team", "away_team", "home_goals", "away_goals"])
    frame = frame[frame["played"]].sort_values("date", kind="stable").reset_index(drop=True)
    return frame


@pytest.fixture(scope="module")
def e0(cfg, corpus):
    top = corpus[corpus["division"] == cfg.backtest.prediction_division]
    return top.sort_values("date", kind="stable").reset_index(drop=True)


def _digest(probs: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()


def _splits(cfg, e0):
    return walk_forward(e0, first_season="2024-25", last_season="2024-25",
                        min_train_matches=cfg.backtest.min_train_matches)


# --- the arm refuses to run without what it needs -------------------------------------------------

def test_the_arm_refuses_to_run_without_lower_divisions(cfg, e0) -> None:
    """Falling back to E0 alone would make this arm the baseline and its null uninformative.

    This is the false-null trap in its most tempting form: the fallback is one line, it never
    crashes, and the resulting "no effect" is indistinguishable from an honest one.
    """
    splits = _splits(cfg, e0)
    with pytest.raises(ValueError, match="multi-division history"):
        run_arm(ArmSpec.parse("dc-tiers"), e0, splits, cfg)


# --- the cache cannot confuse two corpora ----------------------------------------------------------

def test_the_tier_frame_is_part_of_the_cache_identity(corpus) -> None:
    two = corpus[corpus["division"].isin(["E0", "E1"])]
    three = corpus[corpus["division"].isin(["E0", "E1", "E2"])]
    assert _tier_identity(None) != _tier_identity(two)
    assert _tier_identity(two) != _tier_identity(three)


def test_the_tier_identity_is_stable_for_the_same_frame(corpus) -> None:
    two = corpus[corpus["division"].isin(["E0", "E1"])]
    assert _tier_identity(two) == _tier_identity(two.copy())


# --- leak freedom -----------------------------------------------------------------------------------

@pytest.mark.integration
def test_no_lower_division_row_reaches_a_fit_at_or_after_its_barrier(cfg, corpus, e0) -> None:
    """The load-bearing guard, checked on the frame the arm is actually handed.

    E1 plays on days E0 does not, so a barrier taken from the E0 calendar sits in the middle of the
    E1 fixture list far more often than it sits in the middle of E0's. If the truncation were ever
    done on the prediction frame's index rather than on dates, this is where it would show.
    """
    from plmodel.eval.backtest import training_frame

    tiers = corpus[corpus["division"].isin(["E0", "E1"])]
    for split in _splits(cfg, e0):
        rows = training_frame(tiers, split.barrier)
        assert rows["date"].max() < split.barrier
        # And it must not be empty of the lower division, or the arm is quietly the baseline.
        assert (rows["division"] == "E1").any()


@pytest.mark.integration
def test_the_arm_differs_from_the_baseline_and_uses_more_teams(cfg, corpus, e0) -> None:
    """Switched on, it must do something — and the something must be the extra division."""
    tiers = corpus[corpus["division"].isin(["E0", "E1"])]
    splits = _splits(cfg, e0)
    baseline, base_state = run_arm(ArmSpec.parse("dixon-coles"), e0, splits, cfg)
    joint, joint_state = run_arm(ArmSpec.parse("dc-tiers"), e0, splits, cfg, tiers=tiers)

    assert _digest(joint) != _digest(baseline)
    assert len(joint_state["fits"][-1].teams) > len(base_state["fits"][-1].teams)


@pytest.mark.integration
def test_the_joint_fit_rescues_promoted_teams_from_the_league_average(cfg, corpus, e0) -> None:
    """The whole point of the arm, stated as a measurement rather than a hope.

    A promoted club is pinned at the league average by the production model because its top-flight
    history has decayed away; the joint fit gives it parameters from the forty-six matches it just
    played. Measured at the 2024-25 opening barrier the production fit pins Ipswich and the joint
    fit pins no top-flight club at all.
    """
    tiers = corpus[corpus["division"].isin(["E0", "E1"])]
    splits = _splits(cfg, e0)
    _, base_state = run_arm(ArmSpec.parse("dixon-coles"), e0, splits, cfg)
    _, joint_state = run_arm(ArmSpec.parse("dc-tiers"), e0, splits, cfg, tiers=tiers)

    season = e0[e0["season"] == "2024-25"]
    playing = set(season["home_team"]) | set(season["away_team"])
    pinned_base = playing & set(base_state["fits"][0].cold_start_teams)
    pinned_joint = playing & set(joint_state["fits"][0].cold_start_teams)
    assert pinned_base, "the baseline is expected to pin at least one top-flight club here"
    assert len(pinned_joint) < len(pinned_base)
