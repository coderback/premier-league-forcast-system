"""config.yaml loads, and the acceptance rule survives the trip intact.

The acceptance rule is embedded verbatim in every harness report. If it can drift between the
config and the code, a report can claim a rule the project does not hold — so the rule's exact
text is asserted here, character for character.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from plmodel.config import DEFAULT_CONFIG_PATH, Config, ConfigError, load_config

# The rule as the build brief states it. Any edit to config.yaml's wording must be a deliberate
# change to this constant too — that is the point.
EXPECTED_ACCEPTANCE_RULE = """\
Accept a candidate change iff BOTH:
  (1) its paired-bootstrap RPS delta vs baseline on the test pool is favourable
      (95% CI excludes 0, OR P(better) >= 0.95), AND
  (2) its delta vs the Shin de-vigged market on the odds-covered subset does not degrade.
An accepted variant earns a hyperparameter retune BEFORE production wiring."""


@pytest.fixture(scope="module")
def cfg() -> Config:
    return load_config()


def test_default_config_exists() -> None:
    assert DEFAULT_CONFIG_PATH.exists(), f"config.yaml not found at {DEFAULT_CONFIG_PATH}"


def test_acceptance_rule_is_verbatim(cfg: Config) -> None:
    assert cfg.acceptance_rule == EXPECTED_ACCEPTANCE_RULE


def test_acceptance_rule_states_both_gates(cfg: Config) -> None:
    """A rule that lost a gate would still load; these assert it still says what it must."""
    rule = cfg.acceptance_rule
    assert "paired-bootstrap RPS delta vs baseline" in rule
    assert "Shin de-vigged market" in rule
    assert "does not degrade" in rule
    assert "retune BEFORE production wiring" in rule


def test_seed_is_set(cfg: Config) -> None:
    assert isinstance(cfg.seed, int)


def test_data_section(cfg: Config) -> None:
    assert cfg.data.base_url.startswith("https://")
    assert cfg.data.divisions[0] == "E0"
    assert set(cfg.data.divisions) <= {"E0", "E1", "E2", "E3"}
    assert cfg.data.first_season == "9394"
    # Row-count floors back the ingest smoke test; without them a truncated download reads as a
    # real dip in a division's fixture count.
    assert set(cfg.data.min_expected_rows) == set(cfg.data.divisions)
    assert all(v > 0 for v in cfg.data.min_expected_rows.values())


def test_paths_are_absolute(cfg: Config) -> None:
    for path in (cfg.cache_dir, cfg.static_dir, cfg.output_dir):
        assert path.is_absolute()


def test_odds_section(cfg: Config) -> None:
    from plmodel.data.odds import DEVIG_METHODS, FAMILIES

    assert cfg.odds.devig_primary in DEVIG_METHODS
    assert cfg.odds.devig_sensitivity in DEVIG_METHODS
    assert cfg.odds.devig_primary != cfg.odds.devig_sensitivity
    assert cfg.odds.gate_benchmark in FAMILIES
    assert all(name in FAMILIES for name in cfg.odds.diagnostic_benchmarks)
    assert cfg.odds.sum_tolerance > 0


def test_gate_benchmark_is_a_closing_line(cfg: Config) -> None:
    """A pre-close benchmark would measure the model against a stale market."""
    from plmodel.data.odds import FAMILIES

    assert FAMILIES[cfg.odds.gate_benchmark].is_closing


def test_backtest_section(cfg: Config) -> None:
    bt = cfg.backtest
    assert bt.prediction_division in cfg.data.divisions
    assert bt.test_span.first_season <= bt.test_span.last_season
    assert bt.refit_every >= 1
    assert bt.n_boot > 0 and 0 < bt.fdr_alpha < 1


def test_sensitivity_span_does_not_overlap_the_test_span(cfg: Config) -> None:
    """An earlier decade is only a check on era-dependence if it is a different decade."""
    assert cfg.backtest.sensitivity_span.last_season < cfg.backtest.test_span.first_season


def test_inverted_span_raises(tmp_path: Path) -> None:
    import yaml

    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["backtest"]["test_span"] = {"first_season": "2020-21", "last_season": "2016-17"}
    bad = tmp_path / "config.yaml"
    bad.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="is after last_season"):
        load_config(bad)


def test_model_section(cfg: Config) -> None:
    m = cfg.model
    assert m.max_goals > 0 and m.decay_half_life_days > 0
    assert set(m.param_bounds) >= {"intercept", "home_advantage", "rho", "strength"}
    lo, hi = m.param_bounds["rho"]
    # rho must stay inside the region where every tau cell is positive; 1 - lam*mu*rho fails
    # around 0.4 at football rates.
    assert -0.4 < lo < 0 < hi < 0.4


def test_every_seam_ships_off(cfg: Config) -> None:
    """The production configuration is the one the byte-identity tests pin."""
    assert cfg.model.seams_are_inert()


def test_the_season_block_is_typed_and_complete(cfg: Config) -> None:
    """Every section that has landed is typed; nothing about it is defaulted in code."""
    assert cfg.season.n_replicates > 0 and cfg.season.validation_replicates > 0
    assert set(cfg.season.questions) == {"title", "top_four", "relegation"}
    assert set(cfg.season.drift) == {"attack_sd", "defence_sd", "correlation", "horizon_exponent"}


def test_a_season_block_missing_a_setting_raises(tmp_path: Path) -> None:
    """A half-populated section must fail at load, not run with a silent default."""
    source = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    trimmed = "\n".join(
        line for line in source.splitlines() if "validation_replicates" not in line
    )
    path = tmp_path / "config.yaml"
    path.write_text(trimmed, encoding="utf-8")
    with pytest.raises(ConfigError, match="season section missing"):
        load_config(path)


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config not found"):
        load_config(tmp_path / "nope.yaml")


def test_missing_acceptance_rule_raises(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text("seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="acceptance_rule"):
        load_config(bad)


def test_empty_acceptance_rule_raises(tmp_path: Path) -> None:
    bad = tmp_path / "config.yaml"
    bad.write_text('acceptance_rule: "   "\nseed: 1\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="acceptance_rule is empty"):
        load_config(bad)
