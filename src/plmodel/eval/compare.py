"""The acceptance harness — the standing instrument every candidate change is judged by.

One arm changes one axis. Every arm replays the identical walk-forward splits, is scored on the
identical matches, and is compared to the baseline with the paired bootstrap and a HAC-corrected
Diebold-Mariano. The acceptance rule is read from config and embedded verbatim in the report, so a
report can never claim a rule the project does not hold.

Three guards make this a falsification machine rather than a scoreboard:

**The alignment guard.** Arms are compared per match, so a silent reindex would destroy the
pairing that makes the comparison sensitive in the first place. Row identities are asserted equal
across arms and the run fails loudly rather than reindexing.

**The does-it-do-anything guard.** ``assert_arms_differ`` requires every non-baseline arm's
probability vector to differ from the baseline's, and the baseline to be reproducible bit for bit.
This exists because of the WC2026 false-null trap: a broken experiment returning "no effect" looks
exactly like a correct experiment returning "no effect", and nothing else in a harness can tell
them apart.

**The market gate.** Gate 1 is scored on the full pool; gate 2 on the odds-covered subset, because
the acceptance rule scopes it there. An arm that improves the pooled number while degrading against
the market has optimised the instrument rather than the target — precisely what WC2026's ``rsfit``
arm did, and the reason there are two gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from plmodel.config import Config
from plmodel.eval import metrics
from plmodel.eval.backtest import Split, training_frame, validate_splits
from plmodel.eval.calibration import calibration_report
from plmodel.eval.slices import add_slice_columns, all_slices
from plmodel.model.counts import INDEPENDENT_KAPPA, UNIT_SHAPE

# A pool weight this size or larger counts as the channel materially contributing, for
# reporting only — nothing is gated on it. Matches the threshold the reproduction used when
# calling a fitted weight "near zero".
_MATERIAL_POOL_WEIGHT = 0.05

# Columns that identify a match. Two arms disagreeing on these are not comparable.
ALIGN_COLUMNS: tuple[str, ...] = ("date", "division", "home_team", "away_team")

@dataclass(frozen=True)
class ArmContext:
    """Everything a forecaster is allowed to see at one barrier.

    ``state`` is a per-arm scratch dict that persists across the walk — the seam a fitted model
    uses to warm-start from its previous solution and to honour the refit cadence. Model-free arms
    ignore it. Nothing here reaches past the barrier: ``train`` is already filtered by the
    splitter, and ``split.is_refit`` is decided from the barrier's position, never from results.
    """

    split: Split
    test: pd.DataFrame
    train: pd.DataFrame
    cfg: Config
    state: dict
    # Multi-division history for the joint-tier fit, ALREADY truncated at the barrier by
    # ``backtest.training_frame`` -- the same guard the prediction frame goes through, so extra
    # divisions cannot bypass the barrier just because they did not come from ``train``. None for
    # every arm that does not ask for it.
    tiers: pd.DataFrame | None = None


# A forecaster returns (n, 3) probabilities for the context's test rows.
Forecaster = Callable[[ArmContext], np.ndarray]

_REGISTRY: dict[str, Forecaster] = {}


def register(name: str) -> Callable[[Forecaster], Forecaster]:
    """Register a forecaster under an arm name."""

    def deco(fn: Forecaster) -> Forecaster:
        if name in _REGISTRY:
            raise ValueError(f"forecaster {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def registered_arms() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


@register("uniform")
def _uniform(ctx: ArmContext) -> np.ndarray:
    from plmodel.model.baselines import uniform

    return uniform(ctx.test)


@register("home-always")
def _home_always(ctx: ArmContext) -> np.ndarray:
    from plmodel.model.baselines import home_always

    return home_always(ctx.test)


@register("home-rate")
def _home_rate(ctx: ArmContext) -> np.ndarray:
    """The league's long-run home/draw/away split, estimated from data before the barrier."""
    from plmodel.model.baselines import empirical_base_rate, home_rate

    return home_rate(ctx.test, rate=empirical_base_rate(ctx.train))


@register("dixon-coles")
def _dixon_coles(ctx: ArmContext) -> np.ndarray:
    """Per-team Dixon-Coles, warm-started along the walk and refit on the configured cadence."""
    from plmodel.model.dixon_coles import fit_dixon_coles

    model = ctx.cfg.model
    previous = ctx.state.get("fit")
    if previous is None or ctx.split.is_refit:
        previous = fit_dixon_coles(
            ctx.train,
            half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("fit"),
            max_iter=model.max_iter,
        )
        ctx.state["fit"] = previous
        ctx.state.setdefault("fits", []).append(previous)
    return previous.predict_proba(ctx.test)


def _dc_arm(ctx: "ArmContext", *, ha_mode: str) -> np.ndarray:
    """Shared body for the production model and its home-advantage variants.

    One function so the variants cannot drift from the baseline in any way other than the seam —
    which is what makes the paired delta attributable to the seam alone.
    """
    from plmodel.model.dixon_coles import fit_dixon_coles

    model = ctx.cfg.model
    window = (model.seams.get("home_advantage") or {})
    previous = ctx.state.get("fit")
    if previous is None or ctx.split.is_refit:
        previous = fit_dixon_coles(
            ctx.train,
            half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("fit"),
            max_iter=model.max_iter,
            ha_mode=ha_mode,
            ha_window=(window.get("empty_start"), window.get("empty_end")),
        )
        ctx.state["fit"] = previous
        ctx.state.setdefault("fits", []).append(previous)
    return previous.predict_proba(ctx.test)


@register("ha-trend")
def _ha_trend(ctx: ArmContext) -> np.ndarray:
    """Home advantage carries a linear trend, so the forecast uses its CURRENT value."""
    return _dc_arm(ctx, ha_mode="trend")


@register("ha-empty")
def _ha_empty(ctx: ArmContext) -> np.ndarray:
    """Home advantage carries an explicit behind-closed-doors term."""
    return _dc_arm(ctx, ha_mode="empty")


@register("ha-both")
def _ha_both(ctx: ArmContext) -> np.ndarray:
    return _dc_arm(ctx, ha_mode="trend+empty")


def _tier_arm(ctx: "ArmContext", *, divisions: tuple[str, ...]) -> np.ndarray:
    """Shared body for the joint multi-division fits.

    Every parameter is estimated on the pooled frame and the forecast is made for the prediction
    division only. There is no division term: a team's attack and defence mean the same thing
    wherever it plays, and the opponent's parameters carry the rest, so the lower division's lower
    scoring rate falls out of its teams having weaker attack and defence rather than needing to be
    stated. Promotion and relegation link the two scales -- clubs that have played in both
    divisions are the bridge, and thirty-three seasons supply plenty of them.

    The one thing a pooled fit is *forced* to share is home advantage, and that is the assumption
    worth checking rather than asserting. Measured on 2010-11 onward, the home:away goal ratio is
    1.248 in E0 and 1.248 in E1, agreeing to four decimal places in logs (+0.2216 against +0.2219).
    E2 is materially different at +0.1925, which is why adding it is a separate arm rather than a
    free extension of this one.

    The likelihood half-life stays at the production value. Unlike the Elo-scalar and score-driven
    arms, more divisions is not a substitute for time decay -- it adds teams, not recency -- so
    there is no fairness argument for giving this arm its own memory, and giving it one anyway
    would make the delta a two-axis change. If the arm is accepted, the acceptance rule's mandated
    retune is where the half-life gets to move.
    """
    from plmodel.model.dixon_coles import fit_dixon_coles

    model = ctx.cfg.model
    if ctx.tiers is None:
        raise ValueError(
            f"arm needs multi-division history for {list(divisions)}; pass `tiers=` to run_arm. "
            "Refusing to fall back to the prediction division, which would silently make this arm "
            "the baseline and turn its null into an uninformative one."
        )
    history = ctx.tiers[ctx.tiers["division"].isin(divisions)]
    previous = ctx.state.get("fit")
    if previous is None or ctx.split.is_refit:
        previous = fit_dixon_coles(
            history,
            half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("fit"),
            max_iter=model.max_iter,
        )
        ctx.state["fit"] = previous
        ctx.state.setdefault("fits", []).append(previous)
    return previous.predict_proba(ctx.test)


@register("dc-tiers")
def _dc_tiers(ctx: ArmContext) -> np.ndarray:
    """The plan's Arm 8: fit the top two divisions jointly, forecast the top one.

    A promoted club arrives with no top-flight form and is pinned at the league average by the
    production model. That is not a missing feature but a wrong one: the club has just played
    forty-six competitive matches the model refuses to look at.
    """
    return _tier_arm(ctx, divisions=("E0", "E1"))


@register("dc-tiers3")
def _dc_tiers3(ctx: ArmContext) -> np.ndarray:
    """The same fit one division further down, where home advantage stops matching."""
    return _tier_arm(ctx, divisions=("E0", "E1", "E2"))


@register("dc-gas")
def _dc_gas(ctx: ArmContext) -> np.ndarray:
    """The plan's Arm 3: score-driven dynamic team states on top of the fitted level.

    Two things happen at every barrier and only one of them is a refit. The level parameters honour
    the ordinary refit cadence and warm-start from the previous solution, exactly as the baseline
    does. The **filter always re-runs**, because its whole job is to be current: a state refreshed
    only on refit barriers would be a stale rating pretending to be a dynamic one.

    The arm carries its own half-life, tuned on the tuning span. Decay and dynamics are substitutes
    for one another, so handing this arm the production value would make the comparison a test of
    whose hyperparameter happened to suit whom.
    """
    from plmodel.model.dixon_coles import fit_dixon_coles
    from plmodel.model.dynamics import DynamicFit, GasSpec, filter_states

    model = ctx.cfg.model
    spec = ctx.state.get("spec")
    if spec is None:
        spec = GasSpec.from_seam(
            model.seams.get("dynamics") or {},
            state_bound=model.param_bounds["gas_state"][1],
            fallback_half_life=model.decay_half_life_days,
        )
        ctx.state["spec"] = spec

    level = ctx.state.get("level")
    if level is None or ctx.split.is_refit:
        level = fit_dixon_coles(
            ctx.train,
            half_life_days=spec.half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("level"),
            max_iter=model.max_iter,
        )
        ctx.state["level"] = level

    dynamic = DynamicFit(level, filter_states(ctx.train, level, spec), spec)
    ctx.state.setdefault("fits", []).append(dynamic)
    return dynamic.predict_proba(ctx.test)


def _gbm_spec(cfg: Config):
    """The tree settings, read from config rather than defaulted anywhere in code."""
    from plmodel.model.hybrid import GbmSpec

    gbm = cfg.model.gbm
    return GbmSpec(
        n_estimators=int(gbm["n_estimators"]),
        learning_rate=float(gbm["learning_rate"]),
        num_leaves=int(gbm["num_leaves"]),
        min_data_in_leaf=int(gbm["min_data_in_leaf"]),
        seed=int(cfg.seed),
    )


def _hybrid_parents(ctx: "ArmContext") -> tuple[np.ndarray, np.ndarray]:
    """``(dc_probs, gbm_probs)`` at this barrier, from one shared Dixon-Coles fit.

    Shared on purpose. The tree model consumes the level fit's abilities, so refitting the level
    separately for each parent would let them drift apart in ways that have nothing to do with the
    axis under test.
    """
    from plmodel.model.dixon_coles import fit_dixon_coles
    from plmodel.model.hybrid import fit_hybrid

    model = ctx.cfg.model
    if ctx.state.get("level") is None or ctx.split.is_refit:
        ctx.state["level"] = fit_dixon_coles(
            ctx.train,
            half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("level"),
            max_iter=model.max_iter,
        )
        ctx.state["hybrid"] = fit_hybrid(
            ctx.train, ctx.state["level"], spec=_gbm_spec(ctx.cfg)
        )
        ctx.state.setdefault("fits", []).append(ctx.state["hybrid"])
    return (
        ctx.state["level"].predict_proba(ctx.test),
        ctx.state["hybrid"].predict_proba(ctx.test),
    )


@register("gbm")
def _gbm(ctx: ArmContext) -> np.ndarray:
    """The gradient-boosted parent on its own.

    Run because an ensemble result is uninterpretable without it. "The blend beat the baseline" and
    "the blend beat both its parents" are different claims, and only the second is an ensemble
    result rather than a better single model wearing a blend's clothes.
    """
    return _hybrid_parents(ctx)[1]


def _blend_arm(ctx: "ArmContext", *, fixed_weight: float | None) -> np.ndarray:
    """Logarithmic pool of the two parents, at a fixed weight or one fitted online.

    With ``fixed_weight=None`` the weight is chosen at every barrier by minimising log loss over
    every EARLIER barrier's already-resolved forecasts -- matches that are all strictly before this
    barrier, so the choice is leak-free by construction. Until enough resolved history has
    accumulated the weight is pinned at zero, which makes the arm exactly the goals baseline rather
    than a guess.
    """
    from plmodel.model.baselines import outcomes_of
    from plmodel.reproduce.pooling import fit_pair_weight, log_pool

    seam = ctx.cfg.model.seams.get("ensemble") or {}
    dc_probs, gbm_probs = _hybrid_parents(ctx)

    weight = fixed_weight
    if weight is None:
        weight = 0.0
        resolved = ctx.state.setdefault("resolved", [])
        if resolved:
            past_outcomes = np.concatenate([r[2] for r in resolved])
            if len(past_outcomes) >= int(seam["min_pool_history"]):
                weight = fit_pair_weight(
                    np.vstack([r[1] for r in resolved]),
                    np.vstack([r[0] for r in resolved]),
                    past_outcomes,
                    n_grid=int(seam["pool_weight_grid"]),
                )["weight"]
        resolved.append((dc_probs, gbm_probs, outcomes_of(ctx.test)))

    ctx.state.setdefault("weights", []).append(float(weight))
    return log_pool([gbm_probs, dc_probs], [weight, 1.0 - weight])


@register("ens-gbm")
def _ens_gbm(ctx: ArmContext) -> np.ndarray:
    """The blend with its weight fitted by predictive likelihood -- the arm the plan specifies."""
    return _blend_arm(ctx, fixed_weight=None)


@register("ens-gbm-half")
def _ens_gbm_half(ctx: ArmContext) -> np.ndarray:
    """The blend at the fixed even split the WC2026 project shipped."""
    seam = ctx.cfg.model.seams.get("ensemble") or {}
    return _blend_arm(ctx, fixed_weight=float(seam["even_split_weight"]))


def _family_arm(ctx: "ArmContext", *, marginal: str, dependence: str) -> np.ndarray:
    """Shared body for the alternative scoreline families.

    One function so the four cells of the marginal-by-dependence square cannot drift from each
    other, and so the only thing that separates any of them from the production model is the family
    itself: same half-life, same cold-start rule, same warm-started refit cadence.

    The half-life is deliberately NOT re-tuned per family, unlike the score-driven and Elo arms.
    Those two replace the mechanism that tracks change over time, so handing them the production
    memory would have been a handicap. A count law and a dependence device say nothing about time
    at all -- they change the shape of a single match's scoreline distribution -- so there is no
    fairness argument here, and giving one anyway would turn a one-axis test into a two-axis one.
    """
    from plmodel.model.counts import CountSpec
    from plmodel.model.dixon_coles import fit_dixon_coles

    model = ctx.cfg.model
    spec = CountSpec(
        marginal=marginal,
        dependence=dependence,
        n_series_terms=int(model.seams["scoreline"]["n_series_terms"]),
    )
    previous = ctx.state.get("fit")
    if previous is None or ctx.split.is_refit:
        previous = fit_dixon_coles(
            ctx.train,
            half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("fit"),
            max_iter=model.max_iter,
            family=spec,
        )
        ctx.state["fit"] = previous
        ctx.state.setdefault("fits", []).append(previous)
    return previous.predict_proba(ctx.test)


@register("dc-copula")
def _dc_copula(ctx: ArmContext) -> np.ndarray:
    """Poisson margins, Frank copula: the dependence device changed and nothing else."""
    return _family_arm(ctx, marginal="poisson", dependence="frank")


@register("dc-weibull")
def _dc_weibull(ctx: ArmContext) -> np.ndarray:
    """Weibull counts, tau correction: the marginal law changed and nothing else."""
    return _family_arm(ctx, marginal="weibull", dependence="tau")


@register("dc-weibull-copula")
def _dc_weibull_copula(ctx: ArmContext) -> np.ndarray:
    """Both at once -- the specification the literature actually proposes."""
    return _family_arm(ctx, marginal="weibull", dependence="frank")


def _context_arm(ctx: "ArmContext", *, terms: tuple[str, ...], mode: str) -> np.ndarray:
    """Shared body for the match-context arms.

    One function so the four combinations cannot drift from each other, and so the only thing
    separating any of them from the production model is which context terms the design carries:
    same half-life, same cold-start rule, same tau correction with the same fitted rho, same
    warm-started refit cadence.

    The history is handed to ``predict_proba`` as well as to the fit, because unlike the
    home-advantage terms these do NOT vanish at prediction time. How long each side has rested and
    whether it is in Europe are both known before kickoff, so using them is leak-free — and the
    history reaches no further than the barrier, because the splitter truncated it.
    """
    from plmodel.model.dixon_coles import fit_dixon_coles

    model = ctx.cfg.model
    spec = model.covariate_spec(terms=terms, mode=mode)
    previous = ctx.state.get("fit")
    if previous is None or ctx.split.is_refit:
        previous = fit_dixon_coles(
            ctx.train,
            half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("fit"),
            max_iter=model.max_iter,
            covariates=spec,
            cov_division=ctx.cfg.backtest.prediction_division,
        )
        ctx.state["fit"] = previous
        ctx.state.setdefault("fits", []).append(previous)
    return previous.predict_proba(ctx.test, ctx.train)


@register("rest")
def _rest(ctx: ArmContext) -> np.ndarray:
    """Differential days of rest, one parameter. The plan's primary term."""
    return _context_arm(ctx, terms=("rest",), mode="diff")


@register("rest-euro")
def _rest_euro(ctx: ArmContext) -> np.ndarray:
    """Differential rest plus the European-commitment flag — the arm the plan specifies."""
    return _context_arm(ctx, terms=("rest", "euro"), mode="diff")


@register("rest-split")
def _rest_split(ctx: ArmContext) -> np.ndarray:
    """Differential rest with its attack and defence halves free rather than tied.

    Tests the restriction the differential encoding imposes: that a tired side scores less by
    exactly as much as it concedes more. Same information, one more parameter.
    """
    return _context_arm(ctx, terms=("rest",), mode="split")


@register("congestion")
def _congestion(ctx: ArmContext) -> np.ndarray:
    """Matches already played in a trailing window, instead of the gap since the last one.

    A club that played twice in eight days and one that played once are both "four days rested";
    this is the measurement that can tell them apart.
    """
    return _context_arm(ctx, terms=("congestion",), mode="diff")


def _pooled_channel_arm(ctx: "ArmContext", *, channel_name: str) -> np.ndarray:
    """Goals model logarithmically pooled with a chance-creation channel.

    This is the arm the plan actually specifies. Running the channel *instead of* goals is a
    different and much weaker proposition — Pitcan's shots model loses to his goals model outright,
    yet still earns 0.35 weight beside it, because the question is whether it carries information
    the goals model lacks, not whether it is better on its own.

    The weight is fitted **online, on forecasts this walk has already made and whose results have
    since landed**. At each barrier the pool weight minimises log loss over every earlier barrier's
    out-of-sample forecasts. That is leak-free by construction — those matches are all strictly
    before the current barrier — and it is the only way to fit the weight on genuinely
    out-of-sample predictions without surrendering a season of the test pool to a validation
    window. Until enough resolved history has accumulated the weight is pinned at zero, which makes
    the arm exactly the goals baseline rather than a guess.
    """
    from plmodel.model.channels import fit_channel_model, get_channel
    from plmodel.model.dixon_coles import fit_dixon_coles
    from plmodel.reproduce.pooling import fit_pair_weight, log_pool

    from plmodel.model.baselines import outcomes_of

    model = ctx.cfg.model
    observation = model.seams.get("observation") or {}
    min_history = int(observation.get("min_pool_history", 0))
    grid = int(observation.get("pool_weight_grid", 0))
    channel = get_channel(channel_name)

    if previous_needs_refit := (ctx.state.get("goals_fit") is None or ctx.split.is_refit):
        ctx.state["goals_fit"] = fit_dixon_coles(
            ctx.train, half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier, max_goals=model.max_goals,
            param_bounds=model.param_bounds, min_effective_share=model.min_effective_share,
            warm_start=ctx.state.get("goals_fit"), max_iter=model.max_iter,
        )
        try:
            ctx.state["channel_fit"] = fit_channel_model(
                ctx.train, channel=channel, half_life_days=model.decay_half_life_days,
                ref_date=ctx.split.fit_barrier, max_goals=model.max_goals,
                param_bounds=model.param_bounds,
                min_effective_share=model.min_effective_share,
                max_iter=model.max_iter, warm_start=ctx.state.get("channel_fit"),
            )
        except ValueError:
            # The channel has no history yet. Recorded, and the pool falls back to goals alone
            # rather than inventing a forecast.
            ctx.state["channel_fit"] = None
            ctx.state["n_barriers_without_channel"] = (
                ctx.state.get("n_barriers_without_channel", 0) + 1
            )

    goals_probs = ctx.state["goals_fit"].predict_proba(ctx.test)
    channel_fit = ctx.state.get("channel_fit")
    if channel_fit is None:
        ctx.state.setdefault("weights", []).append(0.0)
        return goals_probs
    channel_probs = channel_fit.predict_proba(ctx.test)

    # Fit the weight on everything this walk has already forecast and seen resolved.
    resolved = ctx.state.setdefault("resolved", [])
    weight = 0.0
    if resolved:
        past_goals = np.vstack([r[0] for r in resolved])
        past_channel = np.vstack([r[1] for r in resolved])
        past_outcomes = np.concatenate([r[2] for r in resolved])
        if len(past_outcomes) >= min_history:
            weight = fit_pair_weight(
                past_channel, past_goals, past_outcomes, n_grid=grid
            )["weight"]

    ctx.state.setdefault("weights", []).append(weight)
    resolved.append((goals_probs, channel_probs, outcomes_of(ctx.test)))
    return log_pool([channel_probs, goals_probs], [weight, 1.0 - weight])


@register("dc+xg")
def _dc_plus_xg(ctx: ArmContext) -> np.ndarray:
    """The plan's Arm 4: expected goals as a second observation channel, log-pooled."""
    return _pooled_channel_arm(ctx, channel_name="xg")


@register("dc+sot")
def _dc_plus_sot(ctx: ArmContext) -> np.ndarray:
    """The same pool with shots on target, so the two channels can be compared directly."""
    return _pooled_channel_arm(ctx, channel_name="sot")


@register("elo-dc")
def _elo_dixon_coles(ctx: ArmContext) -> np.ndarray:
    """Dixon-Coles parameterised by one Elo rating difference — the single-scalar comparison.

    The Elo replay is computed once over the whole prediction frame and cached in the arm state: a
    forward pass is causal by construction, so one global replay is both faster and safer than
    replaying per barrier. The replay reads only the prediction division, so this arm faces exactly
    the same promoted-team cold start the production model does — carrying ratings across the
    promotion boundary is the multi-tier arm's question, not this one's.
    """
    from plmodel.model.elo_dc import fit_elo_dixon_coles

    cfg, model = ctx.cfg, ctx.cfg.model
    replay = ctx.state.get("replay")
    if replay is None:
        from plmodel.ratings.elo import compute_elo

        replay = compute_elo(ctx.state["matches"], cfg.elo.to_scheme())
        ctx.state["replay"] = replay

    previous = ctx.state.get("fit")
    if previous is None or ctx.split.is_refit:
        history = replay.history.iloc[: ctx.split.train_end]
        previous = fit_elo_dixon_coles(
            history,
            replay,
            half_life_days=cfg.elo.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            max_iter=model.max_iter,
        )
        ctx.state["fit"] = previous
        ctx.state.setdefault("fits", []).append(previous)
    return previous.predict_proba(ctx.test)


def _channel_arm(ctx: "ArmContext", *, channel_name: str) -> np.ndarray:
    """Shared body for every second-observation-channel arm.

    One code path so shots on target and expected goals differ only in which columns they read —
    which is what makes a paired delta between them attributable to the signal rather than to the
    implementation.
    """
    from plmodel.model.channels import fit_channel_model, get_channel

    model = ctx.cfg.model
    channel = get_channel(channel_name)
    previous = ctx.state.get("fit")
    if previous is None or ctx.split.is_refit:
        previous = fit_channel_model(
            ctx.train,
            channel=channel,
            half_life_days=model.decay_half_life_days,
            ref_date=ctx.split.fit_barrier,
            max_goals=model.max_goals,
            param_bounds=model.param_bounds,
            min_effective_share=model.min_effective_share,
            max_iter=model.max_iter,
            warm_start=ctx.state.get("fit"),
        )
        ctx.state["fit"] = previous
        ctx.state.setdefault("fits", []).append(previous.channel_fit)
        ctx.state.setdefault("kappas", []).append((previous.kappa_home, previous.kappa_away))
    return previous.predict_proba(ctx.test)


@register("dixon-coles-sot")
def _dixon_coles_sot(ctx: ArmContext) -> np.ndarray:
    """Shots on target as the chance-creation channel: same machinery, no tau correction."""
    return _channel_arm(ctx, channel_name="sot")


@register("dixon-coles-xg")
def _dixon_coles_xg(ctx: ArmContext) -> np.ndarray:
    """Expected goals as the chance-creation channel.

    Note the asymmetry with the goals model: xG exists only from 2015/16, so this arm trains on a
    far shorter history than its baseline, and after 2023/24 its most recent xG is stale. Both are
    properties of the source, not of the method, and both are reported.
    """
    return _channel_arm(ctx, channel_name="xg")


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the comparison: exactly one axis changed from the baseline."""

    name: str
    forecaster: str

    @classmethod
    def parse(cls, name: str) -> "ArmSpec":
        if name not in _REGISTRY:
            raise ValueError(f"unknown arm {name!r}; registered: {registered_arms()}")
        return cls(name=name, forecaster=name)


@dataclass
class ArmResult:
    name: str
    probs: np.ndarray = field(repr=False)
    pooled: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, object] = field(default_factory=dict, repr=False)
    slices: pd.DataFrame | None = field(default=None, repr=False)
    fit_summary: dict[str, object] | None = None
    vs_baseline: dict[str, float] | None = None
    vs_baseline_dm: dict[str, float] | None = None
    vs_baseline_logloss: dict[str, float] | None = None
    vs_market: dict[str, float] | None = None


@dataclass
class CompareReport:
    acceptance_rule: str
    arms: list[ArmResult]
    matches: pd.DataFrame = field(repr=False)
    outcomes: np.ndarray = field(repr=False)
    splits: dict[str, object] = field(default_factory=dict)
    market: dict[str, object] | None = None
    fdr: dict[str, object] | None = None


def _pooled_block(probs: np.ndarray, outcomes: np.ndarray) -> dict[str, float]:
    pooled = metrics.summary(probs, outcomes)
    pooled["rps_uniform"] = metrics.mean_rps(metrics.uniform_baseline(len(outcomes)), outcomes)
    pooled["skill"] = metrics.skill(pooled["rps"], pooled["rps_uniform"])
    return pooled


def run_arm(
    spec: ArmSpec,
    matches: pd.DataFrame,
    splits: Sequence[Split],
    cfg: Config,
    *,
    cache_dir: "Path | None" = None,
    tiers: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict]:
    """Run one arm over the walk.

    Returns the probabilities aligned to the concatenated test rows, and the arm's end state (a
    fitted arm leaves its fits there for the report).

    With ``cache_dir`` set, a walk already run under the identical arm, splits, matches and
    configuration is read back instead of refitted — see :mod:`plmodel.eval.cache` for the
    fingerprint that makes "identical" checkable rather than assumed. This exists so the statistics
    can be re-run without the fits, which is what makes a sub-analysis thought of after the fact
    affordable. **Delete the cache when model or eval code changes**: the fingerprint covers
    configuration, not source.
    """
    if cache_dir is not None:
        from plmodel.eval import cache

        key = cache.fingerprint(
            spec.name, build_pool(matches, splits), splits, cfg,
            extra=_tier_identity(tiers),
        )
        hit = cache.load(cache_dir, spec.name, key)
        if hit is not None:
            probs, fit_summary = hit
            return probs, {"from_cache": True, "fit_summary": fit_summary}

    forecaster = _REGISTRY[spec.forecaster]
    # The full prediction frame, for arms that need a single pass over all of it (an Elo replay is
    # causal by construction, so computing it once is both faster and safer than per barrier).
    state: dict = {"matches": matches}
    blocks = []
    for split in splits:
        test = split.test(matches)
        # Extra divisions go through the splitter's own truncation, never around it.
        tier_rows = None if tiers is None else training_frame(tiers, split.barrier)
        probs = np.asarray(
            forecaster(ArmContext(split, test, split.train(matches), cfg, state, tier_rows)),
            dtype=float,
        )
        if probs.shape != (len(test), metrics.AWAY + 1):
            raise ValueError(
                f"arm {spec.name!r} returned {probs.shape} for {len(test)} test rows"
            )
        blocks.append(probs)
    stacked = np.vstack(blocks)
    if cache_dir is not None:
        from plmodel.eval import cache

        cache.save(cache_dir, spec.name, key, stacked, _fit_summary(state))
    return stacked, state


def _tier_identity(tiers: pd.DataFrame | None) -> str:
    """What the cache must know about the multi-division frame handed to an arm.

    Two runs of the same arm on the same splits can still differ if one was given a wider corpus,
    and a cache that could not tell them apart would serve one run's forecasts under the other's
    name. Divisions, row count and last date pin it without hashing thirty thousand rows.
    """
    if tiers is None:
        return "no-tiers"
    divisions = ",".join(sorted(tiers["division"].unique()))
    return f"{divisions}|{len(tiers)}|{tiers['date'].max()}"


def build_pool(matches: pd.DataFrame, splits: Sequence[Split]) -> pd.DataFrame:
    """The concatenated test rows, in split order — the pool every arm is scored on."""
    frames = [split.test(matches).assign(_split=split.index, _barrier=split.barrier)
              for split in splits]
    return pd.concat(frames, ignore_index=True)


def assert_aligned(frames: Sequence[pd.DataFrame]) -> None:
    """Every arm must be scored on identical match rows, in identical order."""
    if not frames:
        return
    base = frames[0][list(ALIGN_COLUMNS)].reset_index(drop=True)
    for i, other in enumerate(frames[1:], start=1):
        if not base.equals(other[list(ALIGN_COLUMNS)].reset_index(drop=True)):
            raise ValueError(
                f"arm {i} match rows do not align with the baseline; refusing to reindex — "
                "a paired comparison on mismatched rows is meaningless"
            )


def assert_arms_differ(results: Sequence[ArmResult]) -> None:
    """Every non-baseline arm must actually change the forecast.

    The WC2026 false-null trap: an arm that silently reproduces the baseline returns "no effect",
    which is indistinguishable from a correct experiment returning "no effect". Without this check
    a broken harness looks exactly like an honest null.
    """
    if not results:
        return
    baseline = results[0]
    identical = [
        arm.name for arm in results[1:] if np.array_equal(arm.probs, baseline.probs)
    ]
    if identical:
        raise ValueError(
            f"arm(s) {identical} produced probabilities identical to the baseline "
            f"{baseline.name!r} — the arm is not doing anything, so its null is uninformative"
        )


def run_compare(
    matches: pd.DataFrame,
    splits: Sequence[Split],
    cfg: Config,
    arm_names: Sequence[str],
    *,
    history: pd.DataFrame | None = None,
    n_bins: int,
    big_six: tuple[str, ...],
    cache_dir: "Path | None" = None,
    tiers: pd.DataFrame | None = None,
) -> CompareReport:
    """Run every arm over the walk and pair-compare them. The first arm is the baseline.

    ``cache_dir`` reuses forecasts from an identical earlier walk; every guard below still runs on
    them, so a cached comparison is checked exactly as hard as a fresh one.
    """
    if len(arm_names) < 1:
        raise ValueError("need at least one arm")
    specs = [ArmSpec.parse(name) for name in arm_names]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError(f"duplicate arms requested: {list(arm_names)}")

    validate_splits(matches, splits)
    rows = build_pool(matches, splits)
    assert_aligned([rows] * len(specs))  # identical by construction; asserted anyway

    from plmodel.model.baselines import market, outcomes_of

    outcomes = outcomes_of(rows)

    market_probs = market(
        rows, family=cfg.odds.gate_benchmark, method=cfg.odds.devig_primary,
        sum_tolerance=cfg.odds.sum_tolerance,
    )
    covered = ~np.isnan(market_probs).any(axis=1)

    rows = add_slice_columns(
        rows,
        history=history if history is not None else matches,
        division=cfg.backtest.prediction_division,
        big_six=big_six,
        market_probs=market_probs,
    )

    results: list[ArmResult] = []
    for spec in specs:
        probs, state = run_arm(
            spec, matches, splits, cfg, cache_dir=cache_dir, tiers=tiers
        )
        if np.isnan(probs).any():
            raise ValueError(
                f"arm {spec.name!r} produced NaN probabilities; an arm must cover the whole pool. "
                "A partially-covered forecaster is a benchmark, not an arm."
            )
        results.append(
            ArmResult(
                name=spec.name,
                probs=probs,
                pooled=_pooled_block(probs, outcomes),
                calibration=calibration_report(probs, outcomes, n_bins=n_bins),
                slices=all_slices(rows, probs, outcomes, n_bins=n_bins),
                fit_summary=_fit_summary(state),
            )
        )
    assert_arms_differ(results)

    baseline = results[0]
    for arm in results[1:]:
        arm.vs_baseline = metrics.paired_delta(
            arm.probs, baseline.probs, outcomes,
            n_boot=cfg.backtest.n_boot, seed=cfg.seed,
        )
        arm.vs_baseline_dm = metrics.diebold_mariano(
            metrics.rps(arm.probs, outcomes), metrics.rps(baseline.probs, outcomes)
        )
        # The same pairing applied to log loss. An arm that moves one rule and not the other is
        # saying something about where in the distribution it changed the forecast, and the
        # acceptance rule scores RPS only — so this is reported, never gated on.
        arm.vs_baseline_logloss = metrics.paired_delta_losses(
            metrics.log_loss(arm.probs, outcomes), metrics.log_loss(baseline.probs, outcomes),
            n_boot=cfg.backtest.n_boot, seed=cfg.seed,
        )

    market_block = _market_block(results, market_probs, covered, outcomes, cfg)
    fdr = _fdr_block(results, cfg)

    return CompareReport(
        acceptance_rule=cfg.acceptance_rule,
        arms=results,
        matches=rows,
        outcomes=outcomes,
        splits={
            "n_splits": len(splits),
            "n_refits": sum(1 for s in splits if s.is_refit),
            "first_barrier": str(splits[0].barrier.date()),
            "last_barrier": str(splits[-1].barrier.date()),
            "refit_every": cfg.backtest.refit_every,
        },
        market=market_block,
        fdr=fdr,
    )


def _fit_summary(state: dict) -> dict[str, object] | None:
    """Aggregate what a fitted arm's walk produced. None for model-free arms.

    A walk read from cache carries its summary rather than its fits — the fits are the expensive
    thing the cache exists to avoid keeping — so that is returned unchanged.

    The rho block is the one that matters beyond diagnostics: its sign over the walk decides
    whether a negative-dependence copula arm is worth building at all.
    """
    if state.get("from_cache"):
        return state.get("fit_summary")
    fits = state.get("fits")
    if not fits:
        # A pooled arm keeps its own state shape; its weight trajectory is the whole diagnostic,
        # because a pool whose weight sat at zero is indistinguishable from the baseline and its
        # null says nothing about the channel.
        weights = state.get("weights")
        if weights:
            w = np.asarray(weights, dtype=float)
            return {
                "kind": "pool",
                "n_barriers": int(w.size),
                "weight_on_channel": {
                    "mean": float(w.mean()), "min": float(w.min()), "max": float(w.max()),
                    "final": float(w[-1]),
                    "share_zero": float((w == 0.0).mean()),
                    f"share_above_{_MATERIAL_POOL_WEIGHT}": float(
                        (w > _MATERIAL_POOL_WEIGHT).mean()
                    ),
                },
                "barriers_without_channel": int(state.get("n_barriers_without_channel", 0)),
            }
        return None
    rho = np.array([f.rho for f in fits])

    # Arms return different fit types — the production model, the Elo-scalar comparison, the
    # shots variant — so read only what every fit is guaranteed to carry and probe for the rest.
    # The Elo fit names its home advantage `h`, in the same log-goal units.
    def home_adv(fit) -> float:
        return float(getattr(fit, "home_advantage", None) or getattr(fit, "h", 0.0))

    return {
        "n_fits": len(fits),
        "n_converged": int(sum(f.converged for f in fits)),
        "half_life_days": fits[-1].half_life_days,
        "mean_iterations": float(np.mean([f.n_iterations for f in fits])),
        "home_advantage": {
            "mean": float(np.mean([home_adv(f) for f in fits])),
            "first": home_adv(fits[0]),
            "last": home_adv(fits[-1]),
        },
        "rho": {
            "mean": float(rho.mean()),
            "min": float(rho.min()),
            "max": float(rho.max()),
            "share_negative": float((rho < 0).mean()),
        },
        "n_params": int(getattr(fits[-1], "n_params", 0)) or _parameter_count(fits[-1]),
        "cold_start": {
            "mean_per_fit": float(np.mean([len(getattr(f, "cold_start_teams", ())) for f in fits])),
            "max_per_fit": int(max(len(getattr(f, "cold_start_teams", ())) for f in fits)),
        },
        # Non-zero means rho had to be pulled into its valid range for an extreme matchup, which
        # in practice signals a degenerate half-life rather than an unusual fixture.
        "rho_clamped": int(
            sum(getattr(f, "diagnostics", {}).get("rho_clamped", 0) for f in fits)
        ),
        **_dynamics_block(fits),
        **_family_block(fits),
        **_context_block(fits),
        **_blend_block(state),
    }


def _dynamics_block(fits: Sequence[object]) -> dict[str, object]:
    """State dispersion over the walk — the dynamics arm's analogue of a pool weight.

    Empty for every arm that carries no states. For the one that does, this is what separates "the
    dynamics were tried and did nothing" from "the dynamics were never switched on" — the same
    distinction ``assert_arms_differ`` exists to protect, and the one a null is worthless without.
    """
    states = [getattr(f, "states", None) for f in fits]
    if not any(s is not None for s in states):
        return {}
    dispersion = np.array([s.dispersion for s in states if s is not None])
    return {
        "dynamics": {
            "spec": fits[-1].spec.as_dict(),
            "state_dispersion": {
                "mean": float(dispersion.mean()),
                "min": float(dispersion.min()),
                "max": float(dispersion.max()),
                "final": float(dispersion[-1]),
            },
            "n_state_clipped": int(sum(s.n_state_clipped for s in states if s is not None)),
            "n_tau_invalid": int(sum(s.n_tau_invalid for s in states if s is not None)),
        }
    }


def _blend_block(state: dict) -> dict[str, object]:
    """The pool weight over the walk -- the headline parameter of a blend arm.

    Empty for arms that carry no pool. For the ones that do, this separates "the blend was tried
    and the weight settled near zero" from "the blend was never given any weight to begin with",
    which are different findings and only one of them says anything about the second parent.
    """
    weights = state.get("weights")
    if not weights:
        return {}
    w = np.asarray(weights, dtype=float)
    return {
        "blend": {
            "weight_on_gbm": {
                "mean": float(w.mean()), "min": float(w.min()), "max": float(w.max()),
                "final": float(w[-1]),
                "share_zero": float((w == 0.0).mean()),
                f"share_above_{_MATERIAL_POOL_WEIGHT}": float((w > _MATERIAL_POOL_WEIGHT).mean()),
            },
            "n_barriers": int(w.size),
        }
    }


def _family_block(fits: Sequence[object]) -> dict[str, object]:
    """The alternative scoreline family's own two parameters over the walk.

    Empty for every arm on the production family, so a baseline report is unchanged by this seam
    existing. For the arms that carry one, the trajectories ARE the result: a shape that never left
    1 and a dependence parameter that never left 0 would mean the family was switched on and found
    nothing to do, which is a different finding from the family not working.

    ``series_condition`` is the cancellation loss of the Weibull count's alternating sum, worst over
    the walk. It is reported rather than merely asserted because it is the one number that says
    whether these fits are numerics or noise.
    """
    families = [getattr(f, "family", None) for f in fits]
    if not any(f is not None for f in families):
        return {}
    shape = np.array([f.shape for f in fits])
    kappa = np.array([f.kappa for f in fits])
    conditions = [getattr(f, "diagnostics", {}).get("series_condition", 0.0) for f in fits]
    deficits = [getattr(f, "diagnostics", {}).get("tail_deficit", 0.0) for f in fits]
    return {
        "scoreline_family": {
            "family": next(f.label() for f in families if f is not None),
            "weibull_shape": {
                "mean": float(shape.mean()), "min": float(shape.min()),
                "max": float(shape.max()), "final": float(shape[-1]),
                "share_above_one": float((shape > UNIT_SHAPE).mean()),
            },
            "frank_kappa": {
                "mean": float(kappa.mean()), "min": float(kappa.min()),
                "max": float(kappa.max()), "final": float(kappa[-1]),
                "share_negative": float((kappa < INDEPENDENT_KAPPA).mean()),
            },
            "worst_series_condition": float(max(conditions)),
            "worst_tail_deficit": float(max(deficits)),
        }
    }


def _context_block(fits: Sequence[object]) -> dict[str, object]:
    """What the match-context covariates were fitted to, over the walk. Empty when the seam is off.

    Every coefficient is reported with its whole trajectory rather than only its last value,
    because a context term that changes sign along the walk is measuring something other than the
    thing it is named after — and a term pinned at a bound is reporting that its box, not the
    data, chose it.

    ``undefined`` is the count of matches where a term had no value and was switched off rather
    than filled in. It belongs in the report for the same reason cold starts do: a covariate that
    was silently absent for a tenth of the sample is a different covariate.
    """
    named = [f for f in fits if getattr(f, "cov_names", ())]
    if not named:
        return {}
    last = named[-1]
    terms: dict[str, object] = {}
    for k, name in enumerate(last.cov_names):
        series = np.array([f.cov_params[k] for f in named])
        terms[name] = {
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            "final": float(series[-1]),
            "share_negative": float((series < 0.0).mean()),
        }
    undefined: dict[str, int] = {}
    for fit in named:
        for term, count in getattr(fit, "cov_undefined", {}).items():
            undefined[term] = max(undefined.get(term, 0), int(count))
    return {
        "covariates": {
            "spec": last.cov_spec.label(),
            "n_fits": len(named),
            "terms": terms,
            "max_matches_with_term_undefined": undefined,
        }
    }


def _parameter_count(fit) -> int:
    """How many free parameters a fit carries — the axis Arm 1 is actually about.

    The per-team model grows with the league; the Elo-scalar model does not. Reporting it makes the
    parsimony side of the comparison visible next to the accuracy side.
    """
    teams = getattr(fit, "teams", None)
    if teams is None:
        return int(fit.as_dict().get("n_params", 0))
    # intercept, home advantage, rho, plus (n_teams - 1) free attack and defence parameters.
    count = 3 + 2 * (len(teams) - 1)
    # An alternative scoreline family swaps rho for its own scalars rather than adding to it: a
    # copula arm has no rho at all, so counting both would overstate what it spends.
    family = getattr(fit, "family", None)
    if family is not None:
        count += int(family.fits_shape) + int(family.fits_kappa) - int(not family.fits_rho)
    # Context covariates are pure additions: they buy an extra term without giving anything up.
    count += len(getattr(fit, "cov_names", ()))
    return count


def _market_block(
    results: Sequence[ArmResult],
    market_probs: np.ndarray,
    covered: np.ndarray,
    outcomes: np.ndarray,
    cfg: Config,
) -> dict[str, object] | None:
    """Gate 2: every arm against the de-vigged market on the odds-covered subset."""
    block: dict[str, object] = {
        "benchmark": cfg.odds.gate_benchmark,
        "devig": cfg.odds.devig_primary,
        "n_covered": int(covered.sum()),
        "n_total": int(len(outcomes)),
    }
    if not covered.any():
        return block
    covered_market = market_probs[covered]
    covered_outcomes = outcomes[covered]
    block["rps_market"] = metrics.mean_rps(covered_market, covered_outcomes)
    block["log_loss_market"] = metrics.mean_log_loss(covered_market, covered_outcomes)
    for arm in results:
        arm.vs_market = metrics.paired_delta(
            arm.probs[covered], covered_market, covered_outcomes,
            n_boot=cfg.backtest.n_boot, seed=cfg.seed,
        )
    return block


def _fdr_block(results: Sequence[ArmResult], cfg: Config) -> dict[str, object] | None:
    """Benjamini-Hochberg across the family of arms tested in this run.

    Uses the DM p-value, which is a genuine two-sided p-value; the bootstrap's ``p_a_better`` is a
    one-sided posterior-style quantity and is not an input to FDR control.
    """
    p_values = {
        arm.name: arm.vs_baseline_dm["p_value"]
        for arm in results[1:]
        if arm.vs_baseline_dm is not None
    }
    if not p_values:
        return None
    return metrics.benjamini_hochberg(p_values, alpha=cfg.backtest.fdr_alpha)


def gate_verdicts(
    report: CompareReport, *, sensitivity: dict[str, dict[str, float]] | None = None
) -> dict[str, dict[str, object]]:
    """Apply the acceptance rule's four gates to each non-baseline arm.

    Gate 1 — the paired delta is favourable: 95% CI excluding 0, or P(better) >= 0.95.
    Gate 2 — the delta against the market does not degrade. Where the market covers no match this
             is NOT EVALUABLE, which is neither a pass nor a failure; the sensitivity span has no
             market coverage before 2019/20 and every arm run there returns None here.
    Gate 3 — Benjamini-Hochberg across this run's family of arms rejects the null for this arm.
             Reported since the first arm; binding since 2026-08-21, because `dc-copula` passed a
             rule that printed "BH rejects 0 of 3" beside its own acceptance.
    Gate 4 — the sensitivity span does not contradict: P(better) there >= 0.5. Deliberately weak.
             It cannot be computed from a single run, so ``sensitivity`` must be supplied, and
             **a run without it accepts nothing** — which is the point, since the alternative is
             remembering to look.

    The verdict is reported, never acted on automatically.
    """
    baseline = report.arms[0]
    rejected = ((report.fdr or {}).get("rejected") or {}) if report.fdr else {}
    verdicts: dict[str, dict[str, object]] = {}
    for arm in report.arms[1:]:
        delta = arm.vs_baseline or {}
        ci_excludes_zero = bool(delta.get("ci_high", 0.0) < 0.0)
        p_better = float(delta.get("p_a_better", 0.0))
        gate1 = ci_excludes_zero or p_better >= _P_BETTER_BAR

        gate2 = None
        if arm.vs_market is not None and baseline.vs_market is not None:
            gate2 = bool(arm.vs_market["delta_rps"] <= baseline.vs_market["delta_rps"])

        gate3 = bool(rejected.get(arm.name, False)) if report.fdr else None

        other = (sensitivity or {}).get(arm.name)
        gate4 = None if other is None else bool(
            float(other.get("p_a_better", 0.0)) >= _SENSITIVITY_LEAN_BAR
        )

        verdicts[arm.name] = {
            "gate1_vs_baseline": gate1,
            "gate1_reason": (
                "95% CI excludes 0" if ci_excludes_zero
                else f"P(better) = {p_better:.3f}"
            ),
            "gate2_vs_market": gate2,
            "gate2_reason": (
                None if gate2 is None else
                f"arm gap {arm.vs_market['delta_rps']:+.5f} vs baseline gap "
                f"{baseline.vs_market['delta_rps']:+.5f}"
            ),
            "gate3_family_wise": gate3,
            "gate3_reason": (
                None if gate3 is None else
                f"BH-adjusted p = {(report.fdr or {}).get('p_adjusted', {}).get(arm.name, float('nan')):.3f} "
                f"at alpha {(report.fdr or {}).get('alpha', float('nan')):.2f} "
                f"across {(report.fdr or {}).get('n_tests', 0)} arm(s)"
            ),
            "gate4_sensitivity": gate4,
            "gate4_reason": (
                "not evaluated: no sensitivity-span result supplied" if gate4 is None
                else f"sensitivity P(better) = {float(other.get('p_a_better', 0.0)):.3f}"
            ),
            "accepted": bool(
                gate1 and (gate2 is not False) and bool(gate3) and bool(gate4)
            ),
        }
    return verdicts


# The sensitivity span is asked only to lean the same way, not to reach significance: the two
# evaluation decades are different eras and a real effect can be genuinely smaller on one.
_SENSITIVITY_LEAN_BAR = 0.5

# The acceptance rule's one-sided bar: P(better) >= 0.95. Stated in config.yaml's rule text; kept
# here as a named constant so the code and the rule cannot drift apart silently.
_P_BETTER_BAR = 0.95


def report_json(
    report: CompareReport, *, sensitivity: dict[str, dict[str, float]] | None = None
) -> dict[str, object]:
    """The JSON report. The acceptance rule is embedded verbatim, as the brief requires."""
    return {
        "acceptance_rule": report.acceptance_rule,
        "splits": report.splits,
        "market": report.market,
        "fdr": report.fdr,
        "verdicts": gate_verdicts(report, sensitivity=sensitivity),
        "arms": {
            arm.name: {
                "pooled": arm.pooled,
                "vs_baseline": arm.vs_baseline,
                "vs_baseline_dm": arm.vs_baseline_dm,
                "vs_baseline_logloss": arm.vs_baseline_logloss,
                "vs_market": arm.vs_market,
                "calibration": {
                    outcome: block["decomposition"]
                    for outcome, block in arm.calibration.items()
                },
                "fit": arm.fit_summary,
                "slices": (
                    arm.slices.to_dict("records") if arm.slices is not None else None
                ),
            }
            for arm in report.arms
        },
    }
