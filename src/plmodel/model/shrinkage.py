"""General hierarchical shrinkage: pull every club's attack and defence toward the league average.

Two arms pointed here and neither was built to.

**Arm 1 (elo-dc)** fought the production model to a statistical draw with **4 parameters against
67**, and the entry drew the conclusion out loud: per-team attack and defence are "over-parameterised
relative to the information available", with the Elo scalar's sequential update "acting as a very
effective regulariser". The recorded lead was per-team strengths shrunk toward a prior.

**Arm 6 (rest-euro)** is the direct evidence, and unusually it is evidence of a *gain* rather than of
a gap. The only context covariate that looked alive turned out not to be measuring fatigue at all:
the fit was moving strength **out of the six flagged clubs, attack and defence together by 0.024
log-goals each**, and handing part of it back as a season-constant bonus inside UEFA's calendar. It
helped most where the flag differential was zero -- big-six against big-six, both ratings moved and
no bonus on either side -- and did nothing in the decade when the big four were not a stable
over-rated group. The fit found that trade worth making even though it had to pay for it with a
spurious calendar term. This seam is the same move without the disguise.

The penalty
-----------
::

    penalty = strength * sum_i ( A_i^2 + D_i^2 )

over the clubs the fit identified. No per-club weight, and that is a decision rather than an
omission: the likelihood's own curvature already scales with a club's effective sample size, so a
club with a full history barely moves and a thin one moves a long way. That is exactly the
hierarchical behaviour wanted, and it costs one knob instead of two.

Cold-start clubs are already at 0 -- dropped from the fit and scored at the league average -- so they
are *fully* shrunk before this seam does anything. What this seam does is move the fitted clubs
toward where the cold-start clubs have always been.

Why the ridge lives here and the promotion prior borrows it
------------------------------------------------------------
:func:`ridge_penalty` is one function with two callers. The promotion seam pulls promoted clubs
toward an estimated promoted-club level; this seam pulls every club toward zero. Same quadratic, same
gradient, different centre and different mask -- so there is one implementation and one derivative to
get right, rather than two that can drift apart. The same discipline the decay seam follows when it
calls the production ``_objective`` instead of writing a second likelihood.

The gradient is returned against the FULL attack and defence vectors, not the free ones. Pushing it
through the sum-to-zero reparameterisation is the caller's job, because that subtlety belongs next to
the packing code that creates it.

Inertness
---------
With the seam off, :func:`Config.shrinkage_spec` returns ``None``, no penalty is constructed, and the
fit takes exactly the path it takes today. Unlike the promotion seam -- whose pin still moved at zero
coefficient, so that ``shrinkage = 0`` was a different model from the seam being absent -- this seam
has a single site, so **``strength = 0`` IS the baseline**, bit for bit. That makes the tuning grid's
own zero point the incumbent the resolution rule compares against, rather than an approximation of
it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ShrinkageError(ValueError):
    """Raised when a shrinkage configuration cannot be honoured."""


@dataclass(frozen=True)
class ShrinkageSpec:
    """How hard to pull every club's attack and defence toward the league average.

    Not to be confused with ``model.seams.promotion.shrinkage``, which is the *promotion* seam's
    coefficient toward the promoted-club prior. Different centre, different mask, different arm --
    they share only the quadratic in :func:`ridge_penalty`.
    """

    strength: float

    def __post_init__(self) -> None:
        if self.strength < 0:
            raise ShrinkageError(f"strength must be non-negative; got {self.strength}")

    @property
    def is_inert(self) -> bool:
        """Zero strength really is the baseline here, so unlike the promotion seam this can say so.

        The promotion seam deliberately has no such property, because its pin still fires at zero
        coefficient. This seam has one site and nothing else to switch, so a zero strength and an
        absent seam are the same model and the property is honest rather than a trap.
        """
        return self.strength == 0.0


def ridge_penalty(
    attack: np.ndarray,
    defence: np.ndarray,
    mask: np.ndarray,
    centre_attack: float,
    centre_defence: float,
    strength: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Quadratic penalty pulling the masked clubs toward a centre, and its gradient.

    ``mask`` selects which clubs are penalised -- every fitted club for this seam, only the promoted
    ones for the promotion seam. The centre is a scalar pair rather than a per-club vector because
    neither caller needs per-club centres: a club the fit could not tell apart from another has no
    evidence to justify a different target.

    Returned against the FULL attack/defence vectors. The caller pushes them through the sum-to-zero
    reparameterisation -- see ``dixon_coles._objective``, where a penalty on the constrained last
    club has to be felt by every free parameter.
    """
    if strength <= 0.0 or not mask.any():
        return 0.0, np.zeros_like(attack), np.zeros_like(defence)
    da = np.where(mask, attack - centre_attack, 0.0)
    dd = np.where(mask, defence - centre_defence, 0.0)
    value = strength * float(da @ da + dd @ dd)
    return value, 2.0 * strength * da, 2.0 * strength * dd
