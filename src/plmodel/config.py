"""Load and validate the central config.yaml into a typed structure.

No magic numbers live in code: every hyperparameter is read from here. See NOTES.md for the
justification of each value, and tests/test_no_magic_numbers.py for the check that enforces it.

Sections are populated phase by phase; a section still empty in config.yaml loads as an empty
mapping rather than a default-filled object, so "not yet tuned" can never be mistaken for "tuned".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file (src/plmodel/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


class ConfigError(ValueError):
    """Raised when config.yaml is missing a required key or carries an impossible value."""


@dataclass(frozen=True)
class DataConfig:
    """football-data.co.uk ingest settings."""

    base_url: str
    divisions: tuple[str, ...]
    first_season: str
    min_expected_rows: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class OddsConfig:
    """Which market benchmark the acceptance rule's second gate is scored against, and how the
    bookmaker's margin is removed. The column names live in data/odds.py; only decisions here."""

    devig_primary: str
    devig_sensitivity: str
    gate_benchmark: str
    diagnostic_benchmarks: tuple[str, ...]
    sum_tolerance: float


@dataclass(frozen=True)
class SeasonSpan:
    """An inclusive span of seasons, labelled as the corpus labels them ("2016-17")."""

    first_season: str
    last_season: str


@dataclass(frozen=True)
class BacktestConfig:
    """The walk-forward protocol and the statistics applied to its output."""

    prediction_division: str
    test_span: SeasonSpan
    sensitivity_span: SeasonSpan
    tuning_span: SeasonSpan
    half_life_grid_days: tuple[float, ...]
    refit_every: int
    min_train_matches: int
    n_boot: int
    fdr_alpha: float


@dataclass(frozen=True)
class ModelConfig:
    """The production Dixon-Coles specification and its extension seams."""

    max_goals: int
    decay_half_life_days: float
    min_effective_share: float
    max_iter: int
    param_bounds: dict[str, tuple[float, float]]
    seams: dict[str, Any]

    def seams_are_inert(self) -> bool:
        """True when every seam is off — the configuration the byte-identity tests pin."""
        s = self.seams
        return (
            not s.get("covariates")
            and not (s.get("dynamics") or {}).get("enabled", False)
            and list((s.get("observation") or {}).get("channels", ["goals"])) == ["goals"]
            and not (s.get("ensemble") or {}).get("enabled", False)
            and (s.get("home_advantage") or {}).get("mode", "global") == "global"
            and list(s.get("tiers", ["E0"])) == ["E0"]
        )


@dataclass(frozen=True)
class AuditConfig:
    """The calibration slices. Diagnostics, never gates."""

    calibration_bins: int
    big_six: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    """The whole config.yaml, typed where a phase has populated it."""

    acceptance_rule: str
    seed: int
    cache_dir: Path
    static_dir: Path
    output_dir: Path
    data: DataConfig
    odds: OddsConfig
    backtest: BacktestConfig
    audit: AuditConfig
    model: ModelConfig
    # Sections not yet populated are carried as raw mappings so nothing invents a default.
    season: dict[str, Any] = field(default_factory=dict)


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ConfigError(f"config.yaml missing required key: {key!r}")
    return raw[key]


def _section(raw: dict[str, Any], key: str) -> dict[str, Any]:
    """A config section, tolerating the `key: {}` and `key:` (null) spellings of 'empty'."""
    value = raw.get(key) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"config.yaml section {key!r} must be a mapping, got {type(value).__name__}")
    return value


def _span(raw: Any, key: str) -> SeasonSpan:
    if not isinstance(raw, dict) or {"first_season", "last_season"} - set(raw):
        raise ConfigError(f"config.yaml {key} needs first_season and last_season")
    first, last = str(raw["first_season"]), str(raw["last_season"])
    if first > last:
        raise ConfigError(f"config.yaml {key}: first_season {first} is after last_season {last}")
    return SeasonSpan(first_season=first, last_season=last)


def load_config(path: Path | str | None = None) -> Config:
    """Read config.yaml into a :class:`Config`.

    The acceptance rule is validated to be present and non-empty here rather than at its use
    sites: every harness report embeds it, so a missing rule must fail at load, not at write.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config not found: {cfg_path}")
    with cfg_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config.yaml must parse to a mapping, got {type(raw).__name__}")

    rule = str(_require(raw, "acceptance_rule")).strip()
    if not rule:
        raise ConfigError("acceptance_rule is empty; every harness report must embed it")

    paths = _section(raw, "paths")
    missing_paths = [k for k in ("cache_dir", "static_dir", "output_dir") if k not in paths]
    if missing_paths:
        raise ConfigError(f"config.yaml paths section missing: {missing_paths}")

    d = _section(raw, "data")
    missing_data = [
        k for k in ("base_url", "divisions", "first_season") if k not in d
    ]
    if missing_data:
        raise ConfigError(f"config.yaml data section missing: {missing_data}")

    o = _section(raw, "odds")
    odds_keys = ("devig_primary", "devig_sensitivity", "gate_benchmark", "sum_tolerance")
    missing_odds = [k for k in odds_keys if k not in o]
    if missing_odds:
        raise ConfigError(f"config.yaml odds section missing: {missing_odds}")

    b = _section(raw, "backtest")
    backtest_keys = (
        "prediction_division", "test_span", "sensitivity_span", "tuning_span",
        "half_life_grid_days", "refit_every", "min_train_matches", "n_boot", "fdr_alpha",
    )
    missing_backtest = [k for k in backtest_keys if k not in b]
    if missing_backtest:
        raise ConfigError(f"config.yaml backtest section missing: {missing_backtest}")

    a = _section(raw, "audit")
    missing_audit = [k for k in ("calibration_bins", "big_six") if k not in a]
    if missing_audit:
        raise ConfigError(f"config.yaml audit section missing: {missing_audit}")

    m = _section(raw, "model")
    model_keys = ("max_goals", "decay_half_life_days", "min_effective_share", "max_iter",
                  "param_bounds", "seams")
    missing_model = [k for k in model_keys if k not in m]
    if missing_model:
        raise ConfigError(f"config.yaml model section missing: {missing_model}")
    required_bounds = ("intercept", "home_advantage", "rho", "strength")
    missing_bounds = [k for k in required_bounds if k not in m["param_bounds"]]
    if missing_bounds:
        raise ConfigError(f"config.yaml model.param_bounds missing: {missing_bounds}")

    return Config(
        acceptance_rule=rule,
        seed=int(_require(raw, "seed")),
        cache_dir=REPO_ROOT / str(paths["cache_dir"]),
        static_dir=REPO_ROOT / str(paths["static_dir"]),
        output_dir=REPO_ROOT / str(paths["output_dir"]),
        data=DataConfig(
            base_url=str(d["base_url"]).rstrip("/"),
            divisions=tuple(str(x) for x in d["divisions"]),
            first_season=str(d["first_season"]),
            min_expected_rows={str(k): int(v) for k, v in (d.get("min_expected_rows") or {}).items()},
        ),
        odds=OddsConfig(
            devig_primary=str(o["devig_primary"]),
            devig_sensitivity=str(o["devig_sensitivity"]),
            gate_benchmark=str(o["gate_benchmark"]),
            diagnostic_benchmarks=tuple(str(x) for x in (o.get("diagnostic_benchmarks") or ())),
            sum_tolerance=float(o["sum_tolerance"]),
        ),
        backtest=BacktestConfig(
            prediction_division=str(b["prediction_division"]),
            test_span=_span(b["test_span"], "test_span"),
            sensitivity_span=_span(b["sensitivity_span"], "sensitivity_span"),
            tuning_span=_span(b["tuning_span"], "tuning_span"),
            half_life_grid_days=tuple(float(x) for x in b["half_life_grid_days"]),
            refit_every=int(b["refit_every"]),
            min_train_matches=int(b["min_train_matches"]),
            n_boot=int(b["n_boot"]),
            fdr_alpha=float(b["fdr_alpha"]),
        ),
        audit=AuditConfig(
            calibration_bins=int(a["calibration_bins"]),
            big_six=tuple(str(x) for x in a["big_six"]),
        ),
        model=ModelConfig(
            max_goals=int(m["max_goals"]),
            decay_half_life_days=float(m["decay_half_life_days"]),
            min_effective_share=float(m["min_effective_share"]),
            max_iter=int(m["max_iter"]),
            param_bounds={
                k: (float(v[0]), float(v[1])) for k, v in m["param_bounds"].items()
            },
            seams=dict(m["seams"] or {}),
        ),
        season=_section(raw, "season"),
    )
