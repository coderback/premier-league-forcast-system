"""The fit production uses, in one place.

Three commands need "the model, fitted as of a barrier": `pl fit`, `pl predict` and `pl simulate`,
plus the season validation sweep. Until now each built its own, and two of them —
``cli._season_fit`` and ``season.validate._fit_at`` — were near-identical bodies that had already
drifted apart in which exception they raised.

That duplication is cheap while there is one way to fit. It stops being cheap the moment there are
two: `dc-gas` is accepted and unwired, and promoting it means every one of those call sites has to
learn about the dynamics seam. **This module exists so that is one edit rather than four**, and so
that a command cannot quietly keep using the old model because nobody remembered it existed.

It deliberately does **not** branch on the seam today. The production model is plain Dixon-Coles
and `model.seams.dynamics.enabled` is false; this is the seam where a branch will go, not the
branch itself. Adding the branch before the configuration is accepted would be wiring by stealth.
"""
from __future__ import annotations

import pandas as pd

from plmodel.config import Config
from plmodel.eval.backtest import training_frame
from plmodel.model.dixon_coles import DixonColesFit, fit_dixon_coles


def production_fit(
    cfg: Config,
    matches: pd.DataFrame,
    barrier: pd.Timestamp,
    *,
    half_life_days: float | None = None,
) -> DixonColesFit:
    """The production model fitted on everything strictly before ``barrier``.

    The strictly-before rule is enforced by :func:`~plmodel.eval.backtest.training_frame` rather
    than re-implemented, so a live forecast and a backtest split cannot disagree about what "before"
    means.

    ``half_life_days`` overrides the configured memory. It exists for `pl fit --half-life`, which
    is a diagnostic knob for looking at what a different memory believes, and for nothing else —
    production reads the configured value.
    """
    train = training_frame(matches, barrier)
    if train.empty:
        raise ValueError(f"no matches before {pd.Timestamp(barrier).date()} to fit on")
    return fit_dixon_coles(
        train,
        half_life_days=half_life_days if half_life_days is not None
        else cfg.model.decay_half_life_days,
        ref_date=barrier,
        max_goals=cfg.model.max_goals,
        param_bounds=cfg.model.param_bounds,
        min_effective_share=cfg.model.min_effective_share,
        max_iter=cfg.model.max_iter,
    )
