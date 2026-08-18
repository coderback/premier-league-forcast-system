"""Shots on target as the second observation channel.

Kept as a named entry point onto :mod:`plmodel.model.channels`, which holds the general
implementation. The channel is a parameter there rather than a hardcoded column, because which
signal fills it is a data-availability question that has already changed twice on this project,
while the modelling question is identical whichever does.

The specification is Pitcan (2026) §5.3 — see the channels module for the full statement.
"""
from __future__ import annotations

from plmodel.model.channels import (
    SHOTS_ON_TARGET,
    ChannelModelFit,
    ChannelSpec,
    finishing_factors as _channel_finishing_factors,
    fit_channel_model,
)

SOT_COLUMNS: tuple[str, str] = SHOTS_ON_TARGET.columns

# The reproduction and its tests refer to this name; it is the general fit bound to one channel.
ShotsModelFit = ChannelModelFit


def finishing_factors(history, ref_date, half_life_days):
    """League-wide weighted goals per shot on target, per side."""
    return _channel_finishing_factors(history, ref_date, half_life_days, SHOTS_ON_TARGET)


def fit_shots_model(history, **kwargs) -> ChannelModelFit:
    """Fit the shots-on-target channel. See :func:`plmodel.model.channels.fit_channel_model`."""
    return fit_channel_model(history, channel=SHOTS_ON_TARGET, **kwargs)


__all__ = [
    "ChannelSpec", "ShotsModelFit", "SHOTS_ON_TARGET", "SOT_COLUMNS",
    "finishing_factors", "fit_shots_model",
]
