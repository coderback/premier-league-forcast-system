"""The fit production uses, in one place.

Three commands need "the model, fitted as of a barrier": `pl fit`, `pl predict` and `pl simulate`,
plus the season validation sweep. Until now each built its own, and two of them —
``cli._season_fit`` and ``season.validate._fit_at`` — were near-identical bodies that had already
drifted apart in which exception they raised.

That duplication is cheap while there is one way to fit. It stops being cheap the moment there are
two: `dc-gas` is accepted and unwired, and promoting it means every one of those call sites has to
learn about the dynamics seam. **This module exists so that is one edit rather than four**, and so
that a command cannot quietly keep using the old model because nobody remembered it existed.

The first seam to come through here is the covariate seam, after `dc+sot-form` passed all four
gates on 2026-09-23: production is Dixon-Coles plus a trailing shots-on-target form term. It is
read from `model.seams.covariates` rather than hardcoded, so the configuration stays the one place
that says what production is, and a fit made with the seam off is the plain model byte for byte.

**A covariate fit needs its history at prediction time.** The form term is rebuilt from the
matches behind the barrier, so every caller of ``predict_proba`` / ``match_rates`` on a production
fit passes the training frame; the fit raises rather than silently dropping the term without it.
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
        covariates=cfg.model.covariate_spec(),
        cov_division=cfg.backtest.prediction_division,
    )
