"""The gradient-boosted parent and the pool it feeds.

Two properties carry most of the weight here. The pool must be **exactly** the baseline at weight
zero and **exactly** the tree model at weight one, because a blend arm's whole result is a
statement about where between those two ends the data puts the weight -- and this arm's answer
turned out to be "at zero, for 884 consecutive barriers", which is only meaningful if zero really
means untouched. And the fit must be reproducible, because a tree model that predicts differently
on a second run would quietly break every byte-identity assertion in the suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plmodel.eval import metrics
from plmodel.model.dixon_coles import fit_dixon_coles
from plmodel.model.hybrid import (
    FEATURE_NAMES,
    GbmSpec,
    HybridError,
    design_matrix,
    effective_history,
    fit_hybrid,
)
from plmodel.reproduce.pooling import log_pool

pytest.importorskip("lightgbm", reason="the hybrid arms need the `ml` extra installed")


@pytest.fixture(scope="module")
def parts(cfg, corpus):
    """A level fit, a hybrid fit on top of it, and a held-out frame -- built once."""
    model = cfg.model
    train = corpus[corpus["season"].isin(["2012-13", "2013-14", "2014-15"])].reset_index(drop=True)
    held_out = corpus[corpus["season"] == "2015-16"].head(80).reset_index(drop=True)
    level = fit_dixon_coles(
        train, half_life_days=model.decay_half_life_days,
        ref_date=train["date"].max() + pd.Timedelta(days=1), max_goals=model.max_goals,
        param_bounds=model.param_bounds, min_effective_share=model.min_effective_share,
        max_iter=model.max_iter,
    )
    spec = GbmSpec(
        n_estimators=int(model.gbm["n_estimators"]),
        learning_rate=float(model.gbm["learning_rate"]),
        num_leaves=int(model.gbm["num_leaves"]),
        min_data_in_leaf=int(model.gbm["min_data_in_leaf"]),
        seed=int(cfg.seed),
    )
    return train, held_out, level, fit_hybrid(train, level, spec=spec), spec


# --- the pool's two ends ---------------------------------------------------------------------

def test_a_pool_at_weight_zero_is_exactly_the_baseline(parts) -> None:
    """The claim the whole arm rests on: a weight of zero leaves the production model untouched."""
    _, held_out, level, hybrid, _ = parts
    dc = level.predict_proba(held_out)
    pooled = log_pool([hybrid.predict_proba(held_out), dc], [0.0, 1.0])
    assert np.abs(pooled - dc).max() < 1e-12


def test_a_pool_at_weight_one_is_exactly_the_tree_model(parts) -> None:
    _, held_out, level, hybrid, _ = parts
    gbm = hybrid.predict_proba(held_out)
    pooled = log_pool([gbm, level.predict_proba(held_out)], [1.0, 0.0])
    assert np.abs(pooled - gbm).max() < 1e-12


def test_a_pool_averages_its_parents_on_the_log_odds_scale(parts) -> None:
    """The property a logarithmic pool actually has, which is not the obvious one.

    A pooled probability need NOT lie between its two parents' probabilities for that class: the
    pool is a normalised geometric mean, and the renormalisation can push a component outside the
    interval its parents span. What *is* exact is the pool's defining property -- every pairwise
    log-odds ratio is the weighted average of the parents' log-odds ratios, which is why the
    logarithmic form is preferred to the linear one for combining opinions.
    """
    _, held_out, level, hybrid, _ = parts
    dc, gbm = level.predict_proba(held_out), hybrid.predict_proba(held_out)
    weight = 0.5
    pooled = log_pool([gbm, dc], [weight, 1.0 - weight])
    assert np.abs(pooled.sum(axis=1) - 1.0).max() < 1e-12
    for i, j in ((0, 1), (0, 2), (1, 2)):
        expected = (
            weight * np.log(gbm[:, i] / gbm[:, j])
            + (1.0 - weight) * np.log(dc[:, i] / dc[:, j])
        )
        assert np.abs(np.log(pooled[:, i] / pooled[:, j]) - expected).max() < 1e-12


# --- reproducibility -------------------------------------------------------------------------

def test_two_fits_of_the_same_data_predict_identically(parts) -> None:
    """Pinned to one thread and LightGBM's deterministic mode, and asserted rather than trusted."""
    train, held_out, level, hybrid, spec = parts
    again = fit_hybrid(train, level, spec=spec)
    assert np.array_equal(hybrid.predict_proba(held_out), again.predict_proba(held_out))


# --- the design matrix -----------------------------------------------------------------------

def test_the_design_matrix_matches_its_declared_columns(parts) -> None:
    train, held_out, level, hybrid, _ = parts
    home, away = design_matrix(held_out, level, hybrid.effective)
    assert home.shape == (len(held_out), len(FEATURE_NAMES))
    assert away.shape == (len(held_out), len(FEATURE_NAMES))
    # The home flag is the one column whose contents are known a priori, so it pins the ordering.
    assert np.all(home[:, FEATURE_NAMES.index("is_home")] == 1.0)
    assert np.all(away[:, FEATURE_NAMES.index("is_home")] == 0.0)


def test_an_unknown_team_carries_nan_rather_than_a_substituted_value(parts) -> None:
    """Missing data is a value. A tree sends NaN down a learned direction; a zero would be a lie."""
    _, held_out, level, hybrid, _ = parts
    invented = held_out.head(1).copy()
    invented["home_team"] = "Not A Real Club"
    home, _ = design_matrix(invented, level, hybrid.effective)
    for column in ("own_attack", "own_defence", "own_effective_n"):
        assert np.isnan(home[0, FEATURE_NAMES.index(column)]), column
    # The opponent is a real club, so its columns are still populated.
    assert not np.isnan(home[0, FEATURE_NAMES.index("opp_attack")])


def test_effective_history_counts_both_sides_under_decay(parts) -> None:
    train, _, level, _, _ = parts
    history = effective_history(train, level)
    assert set(history) >= set(level.teams)
    # Every club played matches, and decay weights are positive, so every total is.
    assert min(history.values()) > 0.0
    # A club that played twice as recently carries more weight than the corpus average.
    assert max(history.values()) > np.median(list(history.values()))


# --- the arm does something, and only through the axis it claims -------------------------------

def test_the_tree_model_produces_different_rates_from_the_linear_predictor(parts) -> None:
    _, held_out, level, hybrid, _ = parts
    lam_dc, mu_dc = level.rates(held_out["home_team"], held_out["away_team"], held_out["date"])
    lam_gb, mu_gb = hybrid.rates(held_out)
    assert np.abs(lam_gb - lam_dc).max() > 1e-3
    assert np.abs(mu_gb - mu_dc).max() > 1e-3
    assert np.all(lam_gb > 0) and np.all(mu_gb > 0)


def test_the_tree_model_borrows_the_scoreline_family_it_was_given(parts) -> None:
    """The rates change; the dependence device does not. That is what keeps this to one axis."""
    _, _, level, hybrid, _ = parts
    assert hybrid.rho == level.rho
    assert hybrid.max_goals == level.max_goals
    assert hybrid.half_life_days == level.half_life_days


def test_an_empty_history_raises_rather_than_fitting_nothing(parts) -> None:
    train, _, level, _, spec = parts
    with pytest.raises(HybridError):
        fit_hybrid(train.head(0), level, spec=spec)


def test_the_summary_reports_the_settings_and_the_importances(parts) -> None:
    _, _, _, hybrid, _ = parts
    block = hybrid.as_dict()["gbm"]
    assert set(block["feature_importance"]) == set(FEATURE_NAMES)
    assert block["n_training_rows"] == hybrid.n_rows
    assert block["num_leaves"] == hybrid.spec.num_leaves


# --- through the harness -----------------------------------------------------------------------

@pytest.mark.integration
def test_the_hybrid_arms_run_the_walk_and_each_one_moves(cfg, corpus) -> None:
    import hashlib

    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )

    def digest(probs: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()

    baseline, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    for name in ("gbm", "ens-gbm-half"):
        probs, _ = run_arm(ArmSpec.parse(name), corpus, splits, cfg)
        assert probs.shape == baseline.shape
        assert np.abs(probs - baseline).max() > 1e-3, f"{name} is indistinguishable from baseline"
        assert np.abs(probs.sum(axis=1) - 1.0).max() < 1e-12

    # A single season never accumulates the resolved history the fitted weight needs, so this arm
    # IS the baseline here -- and that is the behaviour to assert, not a reason to skip it.
    #
    # "Is the baseline" means to within a floating-point round trip, not byte for byte: a
    # logarithmic pool at weight zero still takes a log and an exponential of the baseline and
    # renormalises, which moves the last bit. Measured at 3e-16 on the full test decade.
    fitted, state = run_arm(ArmSpec.parse("ens-gbm"), corpus, splits, cfg)
    assert np.abs(fitted - baseline).max() < 1e-12
    assert set(state["weights"]) == {0.0}

    again, _ = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    assert digest(again) == digest(baseline), "the baseline moved while the hybrid arms ran"


@pytest.mark.integration
def test_the_report_carries_the_blend_weight(cfg, corpus) -> None:
    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, _fit_summary, run_arm

    splits = walk_forward(
        corpus, first_season="2024-25", last_season="2024-25",
        min_train_matches=cfg.backtest.min_train_matches,
    )
    _, state = run_arm(ArmSpec.parse("ens-gbm"), corpus, splits, cfg)
    block = _fit_summary(state)["blend"]
    assert block["n_barriers"] == len(splits)
    assert block["weight_on_gbm"]["share_zero"] == 1.0

    _, plain = run_arm(ArmSpec.parse("dixon-coles"), corpus, splits, cfg)
    assert "blend" not in _fit_summary(plain)


def test_rps_of_a_pool_is_bounded_by_its_parents_at_the_ends(parts) -> None:
    """A sanity check on the pooling direction: the ends are the parents, so their scores are too."""
    _, held_out, level, hybrid, _ = parts
    outcomes = metrics.outcome_from_scores(
        held_out["home_goals"].to_numpy(), held_out["away_goals"].to_numpy()
    )
    dc, gbm = level.predict_proba(held_out), hybrid.predict_proba(held_out)
    assert metrics.mean_rps(log_pool([gbm, dc], [0.0, 1.0]), outcomes) == pytest.approx(
        metrics.mean_rps(dc, outcomes), abs=1e-12
    )
    assert metrics.mean_rps(log_pool([gbm, dc], [1.0, 0.0]), outcomes) == pytest.approx(
        metrics.mean_rps(gbm, outcomes), abs=1e-12
    )
