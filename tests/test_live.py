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


# --- the round, and what became of each frozen fixture ----------------------------------------

def _round() -> pd.DataFrame:
    """Friday to Monday with eight different clubs, then a Tuesday that brings Arsenal back."""
    rows = [
        ("2026-08-21", "Arsenal", "Chelsea"), ("2026-08-22", "Everton", "Fulham"),
        ("2026-08-22", "Brentford", "Burnley"), ("2026-08-23", "Leeds", "Wolves"),
        ("2026-08-24", "Spurs", "Hull"), ("2026-08-25", "Arsenal", "Everton"),
    ]
    return pd.DataFrame({
        "date": pd.to_datetime([r[0] for r in rows]), "season": "2026-27", "division": "E0",
        "home_team": [r[1] for r in rows], "away_team": [r[2] for r in rows],
        "result": [None] * len(rows),
    })


def test_the_round_runs_until_a_club_would_play_twice() -> None:
    from plmodel.eval.live import round_dates

    assert round_dates(_round()) == [pd.Timestamp(d) for d in
                                     ("2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24")]
    assert round_dates(_round().iloc[:0]) == []


def test_a_later_date_of_the_round_records_what_it_could_not_know(cfg, tmp_path: Path) -> None:
    _, block = freeze_matchday(_round(), _history(), cfg, tmp_path, arm_names=["uniform"],
                               barrier=pd.Timestamp("2026-08-23"))
    assert block["barrier"] == "2026-08-23" and block["n_fixtures"] == 1
    assert block["n_pending_before_barrier"] == 3       # Friday's and Saturday's three fixtures
    first = freeze_matchday(_round(), _history(), cfg, tmp_path / "f", arm_names=["uniform"])[1]
    assert first["n_pending_before_barrier"] == 0


def test_every_frozen_fixture_is_accounted_for(cfg, tmp_path: Path) -> None:
    """Scored, still to come, or past its date with a reason: a postponed match must not vanish."""
    from plmodel.eval.live import ledger_resolution

    for date in ("2026-08-21", "2026-08-22", "2026-08-23", "2026-08-24"):
        freeze_matchday(_round(), _history(), cfg, tmp_path, arm_names=["uniform"],
                        barrier=pd.Timestamp(date))
    results = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-21", "2026-08-22", "2026-10-01"]),
        "home_team": ["Arsenal", "Everton", "Brentford"],
        "away_team": ["Chelsea", "Fulham", "Burnley"],
        "result": ["H", "D", "A"],
    })
    fate = ledger_resolution(tmp_path, results, today=pd.Timestamp("2026-08-24"))
    assert (fate["n_frozen"], fate["n_scored"], fate["n_to_come"]) == (5, 2, 1)
    status = {(u["home_team"], u["away_team"]): u["status"] for u in fate["unresolved"]}
    assert status[("Brentford", "Burnley")].startswith("rescheduled: played 2026-10-01")
    assert status[("Leeds", "Wolves")].startswith("no result yet")


def test_the_friday_market_is_frozen_beside_the_forecasts_and_scored(cfg, tmp_path: Path) -> None:
    """The gate is the Friday line, and the only Friday line that matches a live forecast exactly
    is the one the feed carried at the freeze. A fixture the feed did not price is left out."""
    fixtures = _fixtures().assign(B365H=[2.0, None, 2.5], B365D=[3.4, None, 3.3],
                                  B365A=[3.8, None, 2.9])
    _, block = freeze_matchday(fixtures, _history(), cfg, tmp_path, arm_names=["uniform"])
    assert block["tracks"] == {"uniform": "independent"}
    priced, unpriced = block["fixtures"]                  # Everton v Arsenal, Fulham v Chelsea
    assert priced["market_at_freeze"]["bet365"]["odds"] == [2.0, 3.4, 3.8]
    assert sum(priced["market_at_freeze"]["bet365"]["probs"]) == pytest.approx(1.0)
    assert unpriced["market_at_freeze"] == {}

    results = pd.DataFrame({
        "date": pd.to_datetime(["2026-08-22", "2026-08-22"]),
        "home_team": ["Everton", "Fulham"], "away_team": ["Arsenal", "Chelsea"],
        "result": ["H", "D"],
    })
    scored = score_ledger(tmp_path, results).set_index("arm")
    assert scored.loc["uniform", "n"] == 2
    assert scored.loc["market:bet365 (at freeze)", "n"] == 1


def test_a_failing_market_informed_arm_never_costs_the_production_freeze(cfg, tmp_path: Path) -> None:
    """No prices in the history: `odds-ratings` cannot fit. Production must still freeze, and the
    file must say why the market-informed arm is missing."""
    _, block = freeze_matchday(_fixtures(), _history(), cfg, tmp_path,
                               arm_names=["uniform", "odds-ratings"])
    assert block["arms"] == ["uniform"]
    assert "odds-ratings" in block["arm_errors"]
    assert "no priced match" in block["arm_errors"]["odds-ratings"]


def test_the_market_informed_model_is_frozen_by_default() -> None:
    from plmodel.cli import build_parser

    args = build_parser().parse_args(["live"])
    arms = args.arms.split(",")
    assert arms[0] == "dc+sot-form" and "odds-ratings" in arms
