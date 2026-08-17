"""The live ledger: forecasts frozen before kickoff, scored after the result lands.

A backtest can always be re-run; a live forecast cannot. Writing the probabilities to a dated file
*before* the matches are played is what makes the running record a pre-registration rather than a
reconstruction.

An honest note on how much that buys, for the model-free arms specifically. Uniform, home-always
and the de-vigged market have no fitted parameters, and football-data.co.uk publishes closing odds
in the season file after the fact — so a ledger rebuilt in September is numerically identical to
one frozen in August. There is nothing to peek at yet. The discipline becomes load-bearing the
moment a *fitted* model is added, because then the frozen file is the only evidence of what the
model believed before it saw the result. Building it now is operational rehearsal, and it means
the habit exists before it matters.

The barrier is the same one the backtest uses: the next unplayed match date. Forecasts are made
from data strictly before it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from plmodel.config import Config

LEDGER_SUFFIX = ".json"


def next_barrier(fixtures: pd.DataFrame) -> pd.Timestamp | None:
    """The earliest unplayed match date, or None when the fixture list is empty."""
    if len(fixtures) == 0:
        return None
    return pd.Timestamp(fixtures["date"].min())


def freeze_matchday(
    fixtures: pd.DataFrame,
    history: pd.DataFrame,
    cfg: Config,
    ledger_dir: Path,
    *,
    arm_names: list[str],
) -> tuple[Path, dict] | None:
    """Write the next unplayed matchday's forecasts to a dated file.

    Returns ``(path, block)``, or None when there is nothing to freeze. Refuses to overwrite an
    existing file for the same barrier: a frozen forecast that can be rewritten after the fact is
    not frozen, and silently replacing one would destroy the only thing this file is for.
    """
    from plmodel.eval.compare import ArmSpec, _REGISTRY

    barrier = next_barrier(fixtures)
    if barrier is None:
        return None

    day = fixtures[fixtures["date"] == barrier].sort_values("home_team", kind="stable")
    # Strictly before the barrier, exactly as the backtest splits. Same-day results are excluded
    # even when they exist: with date-granular barriers (kickoff times only exist from 2019/20)
    # an early kickoff cannot inform a later one on the same day without breaking the symmetry
    # between the live path and the backtest it is validated by. The count is recorded rather than
    # asserted away — asserting on the *filtered* frame would be vacuous, since the filter is what
    # makes the assertion true.
    train = history[history["date"] < barrier]
    n_excluded_same_day = int((history["date"] == barrier).sum())

    specs = [ArmSpec.parse(name) for name in arm_names]
    forecasts: dict[str, np.ndarray] = {}
    for spec in specs:
        probs = np.asarray(_REGISTRY[spec.forecaster](day, train, cfg), dtype=float)
        forecasts[spec.name] = probs

    block = {
        "barrier": str(barrier.date()),
        "frozen_at": pd.Timestamp.now("UTC").isoformat(),
        "division": cfg.backtest.prediction_division,
        "n_fixtures": int(len(day)),
        "n_train_matches": int(len(train)),
        "n_excluded_same_day": n_excluded_same_day,
        "arms": list(forecasts),
        "acceptance_rule": cfg.acceptance_rule,
        "fixtures": [
            {
                "date": str(pd.Timestamp(row.date).date()),
                "home_team": row.home_team,
                "away_team": row.away_team,
                "forecasts": {
                    arm: [float(x) for x in probs[i]] for arm, probs in forecasts.items()
                },
            }
            for i, row in enumerate(day.itertuples(index=False))
        ],
    }

    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / f"{barrier.date()}{LEDGER_SUFFIX}"
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; a frozen forecast is not rewritten. Delete it deliberately "
            "if it was written in error, and record why."
        )
    path.write_text(json.dumps(block, indent=2), encoding="utf-8")
    return path, block


def load_ledger(ledger_dir: Path) -> list[dict]:
    """Every frozen block, oldest first."""
    if not ledger_dir.exists():
        return []
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(ledger_dir.glob(f"*{LEDGER_SUFFIX}"))
    ]


def score_ledger(ledger_dir: Path, results: pd.DataFrame) -> pd.DataFrame:
    """Score every frozen forecast whose match now has a result.

    Fixtures still unplayed are skipped rather than dropped from the ledger — they are simply not
    scorable yet, and will be next time this runs.
    """
    from plmodel.eval import metrics

    played = {
        (str(pd.Timestamp(r.date).date()), r.home_team, r.away_team): r.result
        for r in results.itertuples(index=False)
    }
    codes = {"H": metrics.HOME, "D": metrics.DRAW, "A": metrics.AWAY}

    rows: list[dict[str, object]] = []
    for block in load_ledger(ledger_dir):
        for fixture in block["fixtures"]:
            key = (fixture["date"], fixture["home_team"], fixture["away_team"])
            if key not in played:
                continue
            outcome = np.array([codes[played[key]]])
            for arm, probs in fixture["forecasts"].items():
                p = np.asarray([probs], dtype=float)
                rows.append(
                    {
                        "barrier": block["barrier"],
                        "home_team": fixture["home_team"],
                        "away_team": fixture["away_team"],
                        "result": played[key],
                        "arm": arm,
                        "rps": float(metrics.rps(p, outcome)[0]),
                    }
                )
    if not rows:
        return pd.DataFrame()

    scored = pd.DataFrame(rows)
    return (
        scored.groupby("arm", sort=True)
        .agg(n=("rps", "size"), rps=("rps", "mean"))
        .reset_index()
    )
