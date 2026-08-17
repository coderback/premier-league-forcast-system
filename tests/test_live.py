"""The live ledger: forecasts frozen before kickoff, scored after.

A frozen forecast that can be rewritten after the fact is not frozen, so most of these tests are
about refusing to rewrite one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval.live import freeze_matchday, load_ledger, next_barrier, score_ledger


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01", "2026-08-08"]),
            "season": "2026-27", "division": "E0",
            "home_team": ["Arsenal", "Chelsea"], "away_team": ["Chelsea", "Everton"],
            "result": ["H", "D"],
        }
    )


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-22", "2026-08-22", "2026-08-29"]),
            "season": "2026-27", "division": "E0",
            "home_team": ["Everton", "Fulham", "Arsenal"],
            "away_team": ["Arsenal", "Chelsea", "Fulham"],
            "result": [None, None, None],
        }
    )


def test_next_barrier_is_the_earliest_unplayed_date() -> None:
    assert next_barrier(_fixtures()) == pd.Timestamp("2026-08-22")


def test_no_fixtures_is_not_an_error() -> None:
    assert next_barrier(_fixtures().iloc[:0]) is None


def test_freeze_writes_only_the_next_matchday(cfg, tmp_path: Path) -> None:
    """One barrier at a time: freezing the whole season up front would forecast the later
    matchdays without the results of the earlier ones, which is not what will happen live."""
    path, block = freeze_matchday(
        _fixtures(), _history(), cfg, tmp_path, arm_names=["uniform", "home-always"]
    )
    assert block["barrier"] == "2026-08-22"
    assert block["n_fixtures"] == 2            # the 29th is a separate barrier
    assert path.name == "2026-08-22.json"
    assert json.loads(path.read_text(encoding="utf-8"))["n_fixtures"] == 2


def test_frozen_block_embeds_the_acceptance_rule(cfg, tmp_path: Path) -> None:
    _, block = freeze_matchday(_fixtures(), _history(), cfg, tmp_path, arm_names=["uniform"])
    assert block["acceptance_rule"] == cfg.acceptance_rule


def test_frozen_forecasts_are_well_formed(cfg, tmp_path: Path) -> None:
    _, block = freeze_matchday(
        _fixtures(), _history(), cfg, tmp_path, arm_names=["uniform", "home-always"]
    )
    for fixture in block["fixtures"]:
        for arm, probs in fixture["forecasts"].items():
            assert len(probs) == 3
            assert sum(probs) == pytest.approx(1.0)


def test_refreezing_the_same_barrier_is_refused(cfg, tmp_path: Path) -> None:
    """The whole point of the ledger: a forecast that can be rewritten proves nothing."""
    freeze_matchday(_fixtures(), _history(), cfg, tmp_path, arm_names=["uniform"])
    with pytest.raises(FileExistsError, match="not rewritten"):
        freeze_matchday(_fixtures(), _history(), cfg, tmp_path, arm_names=["uniform"])


def test_same_day_results_are_excluded_from_the_forecast(cfg, tmp_path: Path) -> None:
    """The same barrier discipline as the backtest, on the live path too.

    An early kickoff on the barrier date is excluded even though its result exists: barriers are
    date-granular (kickoff times only exist from 2019/20), so letting a same-day result inform a
    later same-day forecast would break the symmetry with the backtest that validates this path.
    The exclusion is counted in the frozen block rather than silently applied.
    """
    same_day = pd.concat([_history(), _fixtures().iloc[:1].assign(result="H")])
    _, block = freeze_matchday(_fixtures(), same_day, cfg, tmp_path, arm_names=["uniform"])
    assert block["n_train_matches"] == len(_history())
    assert block["n_excluded_same_day"] == 1


def test_scoring_an_empty_ledger(cfg, tmp_path: Path) -> None:
    assert score_ledger(tmp_path, _history()).empty


def test_scoring_skips_fixtures_without_a_result(cfg, tmp_path: Path) -> None:
    """Unplayed fixtures are not scorable yet; they must not be dropped or counted as wrong."""
    freeze_matchday(_fixtures(), _history(), cfg, tmp_path, arm_names=["uniform"])
    assert score_ledger(tmp_path, _history()).empty


def test_scoring_a_played_matchday(cfg, tmp_path: Path) -> None:
    freeze_matchday(
        _fixtures(), _history(), cfg, tmp_path, arm_names=["uniform", "home-always"]
    )
    results = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-22", "2026-08-22"]),
            "home_team": ["Everton", "Fulham"], "away_team": ["Arsenal", "Chelsea"],
            "result": ["A", "H"],
        }
    )
    scored = score_ledger(tmp_path, results).set_index("arm")
    assert scored.loc["uniform", "n"] == 2
    # Uniform scores 5/18 on a home or away win, whatever happens.
    assert scored.loc["uniform", "rps"] == pytest.approx(5 / 18)
    # home-always is perfect on the home win and maximally wrong on the away win.
    assert scored.loc["home-always", "rps"] == pytest.approx(0.5)


def test_ledger_round_trips(cfg, tmp_path: Path) -> None:
    freeze_matchday(_fixtures(), _history(), cfg, tmp_path, arm_names=["uniform"])
    blocks = load_ledger(tmp_path)
    assert len(blocks) == 1 and blocks[0]["barrier"] == "2026-08-22"


def test_load_ledger_on_a_missing_directory(tmp_path: Path) -> None:
    assert load_ledger(tmp_path / "nope") == []
