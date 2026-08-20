"""A gradient-boosted per-team goals regressor that consumes the Dixon-Coles fit.

The Groll, Ley, Schauberger & Van Eetvelde (2019) pattern: give a tree model the fitted ability
parameters of a Poisson ranking model as covariates, and let it find whatever the ranking model's
functional form cannot express. In their World Cup work the hybrid "improved the predictive power
substantially" over both parents, and it is the one change that ever passed the bar in the WC2026
project (-0.0021 RPS, P=0.999).

What this arm does and does not test
------------------------------------
It holds the **information set fixed** and varies only the **functional form**. Every feature below
is derived from the Dixon-Coles fit at the same barrier or from the fixture's position in the
season; nothing here knows anything the production model does not already know. Rest and congestion
belong to a later arm and are deliberately absent, so are availability signals, and rating
trajectories are out of scope for the project as a whole.

That makes this a narrower test than the World Cup version, which fed its trees genuinely extra
covariates. The question here is sharper for being narrower: **does Dixon-Coles' log-additive map
from abilities to goal rates leave anything on the table that a tree can find?**

There is a specific reason to think it might. The model asserts

    log lam = c + h + Attack[home] - Defence[away]

so a strong attack meeting a weak defence produces exactly the sum of the two effects. Two arms
have now independently pointed at that additivity as the suspect: the Elo-scalar comparison wanted
per-team strengths shrunk toward a prior, and the multi-tier fit measured a promoted club's rating
as badly overstated relative to the clubs it actually faces. A tree fed attack and defence
*separately* alongside their combination can express a shrinkage the sum cannot.

Design decisions
----------------
**Per-team rows, not per-match.** Each match contributes two training rows, one per side, with the
target being that side's goals. It doubles the sample, shares structure between home and away
performances, and is what "per-team goals regression" means.

**The scoreline family is held fixed.** The regressor produces rates; those rates go through the
*same* tau correction with the *same* fitted rho as the production model, clamped per match by the
same guard. Arm 9 established the scoreline family as its own axis, so borrowing it here keeps this
arm to one axis. The alternative -- independent Poisson on the tree's rates, as the chance-creation
channel does -- would have confounded a rate-map test with a dependence-device test.

**The same decay weights.** The trees are fitted under the identical exponential decay the
production model uses, so the comparison is between two maps applied to the same weighted data
rather than between two memories. Decay and functional form are not substitutes for one another,
which is what distinguished the score-driven arm and earned it a half-life of its own. If this arm
is accepted, the acceptance rule's mandated retune is where the half-life gets to move.

**Missing data is never imputed.** A team with no fitted parameters carries NaN into the feature
matrix and LightGBM sends it down a learned default direction. That is the whole reason a tree
model is allowed to see cold-start clubs at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from plmodel.model.dixon_coles import DixonColesFit, decay_weights
from plmodel.model.scoreline import clamp_rho_for_rates, three_class_from_rates

# Column order of the design matrix. Fixed and asserted rather than inferred, because a tree model
# silently reading the wrong column is the kind of bug that produces a plausible null.
FEATURE_NAMES: tuple[str, ...] = (
    "own_attack",
    "own_defence",
    "opp_attack",
    "opp_defence",
    "is_home",
    "log_rate_dc",
    "own_effective_n",
    "opp_effective_n",
    "own_match_index",
    "opp_match_index",
)

# Relative back-off from the tau validity boundary, matching the production model's own clamp.
_RHO_CLAMP_MARGIN = 0.01


class HybridError(ValueError):
    """Raised when the gradient-boosted parent cannot be fitted or applied."""


@dataclass(frozen=True)
class GbmSpec:
    """LightGBM settings. Every value is configured; none is inherited from another project.

    ``deterministic`` and ``num_threads`` are not tuning knobs and are not optional. The WC2026
    project was bitten by a forest predicting in nondeterministic chunk order, and this project
    asserts byte-identical output in several places, so the model is pinned to one thread and to
    LightGBM's deterministic mode. That costs speed and buys a result that is the same twice.
    """

    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    seed: int

    def params(self) -> dict[str, object]:
        return {
            "objective": "poisson",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "seed": self.seed,
            "bagging_seed": self.seed,
            "feature_fraction_seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "verbose": -1,
        }


@dataclass(frozen=True)
class HybridFit:
    """A fitted gradient-boosted rate model, paired with the Dixon-Coles fit it consumed."""

    booster: object
    level: DixonColesFit
    spec: GbmSpec
    n_rows: int
    # Decay-weighted history per team, as it stood when the trees were trained. Carried explicitly
    # rather than recomputed at prediction time: a fixture's features must describe the history the
    # model was fitted against, not a history recomputed from a different frame.
    effective: dict[str, float] = field(default_factory=dict)
    feature_importance: tuple[float, ...] = ()
    diagnostics: dict[str, float] = field(default_factory=dict)

    # Delegated so a HybridFit can stand where a DixonColesFit does in the report.
    @property
    def teams(self) -> tuple[str, ...]:
        return self.level.teams

    @property
    def rho(self) -> float:
        return self.level.rho

    @property
    def half_life_days(self) -> float:
        return self.level.half_life_days

    @property
    def converged(self) -> bool:
        return self.level.converged

    @property
    def n_iterations(self) -> int:
        return self.level.n_iterations

    @property
    def home_advantage(self) -> float:
        return self.level.home_advantage

    @property
    def cold_start_teams(self) -> tuple[str, ...]:
        return self.level.cold_start_teams

    @property
    def max_goals(self) -> int:
        return self.level.max_goals

    def rates(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Expected goals for each side, from the trees rather than from the linear predictor."""
        home_x, away_x = design_matrix(rows, self.level, self.effective)
        lam = np.asarray(self.booster.predict(home_x), dtype=float)
        mu = np.asarray(self.booster.predict(away_x), dtype=float)
        return lam, mu

    def predict_proba(self, rows: pd.DataFrame) -> np.ndarray:
        """(N, 3) home/draw/away, using the production scoreline family on the tree's rates."""
        lam, mu = self.rates(rows)
        rho, n_clamped = clamp_rho_for_rates(lam, mu, self.level.rho, margin=_RHO_CLAMP_MARGIN)
        if n_clamped:
            self.diagnostics["rho_clamped"] = (
                self.diagnostics.get("rho_clamped", 0) + n_clamped
            )
        return three_class_from_rates(lam, mu, rho, self.level.max_goals)

    def as_dict(self) -> dict[str, object]:
        summary = self.level.as_dict()
        summary["gbm"] = {
            "n_estimators": self.spec.n_estimators,
            "learning_rate": self.spec.learning_rate,
            "num_leaves": self.spec.num_leaves,
            "min_data_in_leaf": self.spec.min_data_in_leaf,
            "n_training_rows": self.n_rows,
            "feature_importance": dict(zip(FEATURE_NAMES, self.feature_importance)),
        }
        return summary


def _team_lookup(level: DixonColesFit) -> dict[str, int]:
    return {team: i for i, team in enumerate(level.teams)}


def _side_features(
    rows: pd.DataFrame,
    level: DixonColesFit,
    *,
    attacking: str,
    defending: str,
    is_home: bool,
    log_rate: np.ndarray,
    effective: dict[str, float],
) -> np.ndarray:
    """One side's design matrix. Unknown teams carry NaN rather than a substituted value."""
    index = _team_lookup(level)
    own = rows[attacking].to_numpy()
    opp = rows[defending].to_numpy()

    def pick(values: np.ndarray, teams: np.ndarray) -> np.ndarray:
        out = np.full(len(teams), np.nan)
        for i, team in enumerate(teams):
            slot = index.get(team)
            if slot is not None:
                out[i] = values[slot]
        return out

    def history(teams: np.ndarray) -> np.ndarray:
        return np.array([effective.get(team, np.nan) for team in teams], dtype=float)

    own_index_col = "home_match_index" if is_home else "away_match_index"
    opp_index_col = "away_match_index" if is_home else "home_match_index"
    columns = {
        "own_attack": pick(level.attack, own),
        "own_defence": pick(level.defence, own),
        "opp_attack": pick(level.attack, opp),
        "opp_defence": pick(level.defence, opp),
        "is_home": np.full(len(rows), 1.0 if is_home else 0.0),
        "log_rate_dc": log_rate,
        "own_effective_n": history(own),
        "opp_effective_n": history(opp),
        "own_match_index": _optional_column(rows, own_index_col),
        "opp_match_index": _optional_column(rows, opp_index_col),
    }
    return np.column_stack([columns[name] for name in FEATURE_NAMES])


def _optional_column(rows: pd.DataFrame, name: str) -> np.ndarray:
    """A column the corpus may not carry, as NaN rather than as a fabricated zero."""
    if name not in rows.columns:
        return np.full(len(rows), np.nan)
    return pd.to_numeric(rows[name], errors="coerce").to_numpy(dtype=float)


def effective_history(history: pd.DataFrame, level: DixonColesFit) -> dict[str, float]:
    """Decay-weighted match count per team, on the same weights the level fit used.

    This is what separates "a club the model knows nothing about" from "a club the model knows a
    great deal about", and it is the feature that lets the trees treat a promoted side differently
    without anyone hard-coding what promotion means.
    """
    weights = decay_weights(history["date"], level.ref_date, level.half_life_days)
    totals: dict[str, float] = {}
    for column in ("home_team", "away_team"):
        for team, weight in zip(history[column].to_numpy(), weights):
            totals[team] = totals.get(team, 0.0) + float(weight)
    return totals


def design_matrix(
    rows: pd.DataFrame, level: DixonColesFit, effective: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """``(home_rows, away_rows)`` design matrices for a match frame."""
    dates = rows["date"] if "date" in rows.columns else None
    lam, mu = level.rates(rows["home_team"], rows["away_team"], dates)
    home = _side_features(
        rows, level, attacking="home_team", defending="away_team", is_home=True,
        log_rate=np.log(lam), effective=effective,
    )
    away = _side_features(
        rows, level, attacking="away_team", defending="home_team", is_home=False,
        log_rate=np.log(mu), effective=effective,
    )
    return home, away


def fit_hybrid(
    history: pd.DataFrame,
    level: DixonColesFit,
    *,
    spec: GbmSpec,
) -> HybridFit:
    """Fit the tree model on the same history, under the same decay weights, as the level fit.

    ``history`` must already be truncated at the barrier -- the splitter owns that, and this
    function deliberately does not re-filter, so a caller cannot pass unfiltered data and have it
    silently corrected.
    """
    import lightgbm as lgb

    if len(history) == 0:
        raise HybridError("cannot fit the hybrid on an empty history")

    effective = effective_history(history, level)
    home_x, away_x = design_matrix(history, level, effective)
    features = np.vstack([home_x, away_x])
    target = np.concatenate([
        history["home_goals"].to_numpy(dtype=float),
        history["away_goals"].to_numpy(dtype=float),
    ])
    weights = decay_weights(history["date"], level.ref_date, level.half_life_days)
    sample_weight = np.concatenate([weights, weights])

    dataset = lgb.Dataset(
        features, label=target, weight=sample_weight,
        feature_name=list(FEATURE_NAMES), free_raw_data=False,
    )
    booster = lgb.train(spec.params(), dataset, num_boost_round=spec.n_estimators)
    return HybridFit(
        booster=booster,
        level=level,
        spec=spec,
        n_rows=int(len(target)),
        effective=effective,
        feature_importance=tuple(
            float(v) for v in booster.feature_importance(importance_type="gain")
        ),
    )
