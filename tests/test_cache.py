"""The walk cache, and the fingerprint that is the only reason it is allowed to exist.

Reading forecasts off disk is the one thing this harness otherwise forbids: probabilities that did
not come from the arm the report names. Every test here is about the key being strict enough that a
cached comparison is worth exactly as much as a fresh one.

The load-bearing test is :func:`test_an_in_memory_override_changes_the_fingerprint`. It is written
because the first version of this module keyed on the hash of ``config.yaml`` alone, which is
stable under ``dataclasses.replace`` — so a sweep asking for a 2555-day walk would have been handed
the 730-day one, silently, with the right filename and a passing fingerprint check.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from plmodel.config import load_config
from plmodel.eval import cache
from plmodel.eval.backtest import Split


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _pool(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01"] * n),
        "division": ["E0"] * n,
        "home_team": [f"H{i}" for i in range(n)],
        "away_team": [f"A{i}" for i in range(n)],
    })


def _splits(n: int = 2) -> list[Split]:
    return [
        Split(index=i, barrier=pd.Timestamp("2020-01-01"), fit_barrier=pd.Timestamp("2020-01-01"),
              train_end=100 + i, test_start=100 + i, test_stop=101 + i, is_refit=True)
        for i in range(n)
    ]


def _probs(n: int = 3) -> np.ndarray:
    return np.tile(np.array([0.5, 0.3, 0.2]), (n, 1))


# --- the fingerprint ------------------------------------------------------------------------------

def test_the_same_walk_gives_the_same_fingerprint(cfg) -> None:
    a = cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
    b = cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
    assert a == b


def test_a_different_arm_changes_the_fingerprint(cfg) -> None:
    assert (cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
            != cache.fingerprint("dc-gas", _pool(), _splits(), cfg))


def test_an_in_memory_override_changes_the_fingerprint(cfg) -> None:
    """The bug this module was rewritten for.

    A sweep builds its variants with ``dataclasses.replace``, which never touches config.yaml. Key
    on the file and every variant of a walk collides with every other, which is not a stale cache —
    it is the harness scoring one model's forecasts under another model's name.
    """
    other = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, decay_half_life_days=cfg.model.decay_half_life_days * 2
    ))
    assert (cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
            != cache.fingerprint("dixon-coles", _pool(), _splits(), other))


def test_a_seam_override_changes_the_fingerprint(cfg) -> None:
    """Seams are nested dicts, so this checks the serialisation really walks the whole tree."""
    seam = {**cfg.model.seams["dynamics"], "score_loading": 0.0}
    other = dataclasses.replace(cfg, model=dataclasses.replace(
        cfg.model, seams={**cfg.model.seams, "dynamics": seam}
    ))
    assert (cache.fingerprint("dc-gas", _pool(), _splits(), cfg)
            != cache.fingerprint("dc-gas", _pool(), _splits(), other))


def test_different_barriers_change_the_fingerprint(cfg) -> None:
    assert (cache.fingerprint("dixon-coles", _pool(), _splits(2), cfg)
            != cache.fingerprint("dixon-coles", _pool(), _splits(3), cfg))


def test_different_matches_change_the_fingerprint(cfg) -> None:
    other = _pool().assign(home_team=["X", "Y", "Z"])
    assert (cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
            != cache.fingerprint("dixon-coles", other, _splits(), cfg))


def test_reordered_matches_change_the_fingerprint(cfg) -> None:
    """Order is part of the identity: the pool is scored positionally against the arm's output."""
    reversed_pool = _pool().iloc[::-1].reset_index(drop=True)
    assert (cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
            != cache.fingerprint("dixon-coles", reversed_pool, _splits(), cfg))


def test_the_config_file_digest_is_part_of_the_identity(cfg) -> None:
    """Covers edits to parts of the file the typed config does not carry — comments included."""
    assert (cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
            != cache.fingerprint("dixon-coles", _pool(), _splits(),
                                 dataclasses.replace(cfg, digest="something else")))


# --- storing and reading --------------------------------------------------------------------------

def test_a_miss_is_none_not_an_error(tmp_path, cfg) -> None:
    assert cache.load(tmp_path, "dixon-coles", "nothing-here") is None


def test_a_round_trip_returns_the_same_numbers(tmp_path, cfg) -> None:
    key = cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
    cache.save(tmp_path, "dixon-coles", key, _probs(), {"n_fits": 7})
    probs, summary = cache.load(tmp_path, "dixon-coles", key)
    assert np.array_equal(probs, _probs())
    assert summary == {"n_fits": 7}


def test_a_tampered_entry_is_refused_rather_than_used(tmp_path, cfg) -> None:
    """The filename is a convenience; the fingerprint inside the entry is the authority."""
    key = cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
    cache.save(tmp_path, "dixon-coles", key, _probs(), None)
    meta = next(tmp_path.glob("*.json"))
    meta.write_text(json.dumps({"arm": "dixon-coles", "fingerprint": "forged"}), encoding="utf-8")
    with pytest.raises(cache.CacheError, match="different walk"):
        cache.load(tmp_path, "dixon-coles", key)


def test_an_entry_of_the_wrong_shape_is_refused(tmp_path, cfg) -> None:
    key = cache.fingerprint("dixon-coles", _pool(), _splits(), cfg)
    cache.save(tmp_path, "dixon-coles", key, np.zeros((3, 5)), None)
    with pytest.raises(cache.CacheError, match="expected"):
        cache.load(tmp_path, "dixon-coles", key)


# --- through the harness --------------------------------------------------------------------------

@pytest.mark.integration
def test_a_cached_walk_reproduces_an_uncached_one_exactly(cfg, tmp_path) -> None:
    """Not "close to": identical. A cached comparison must be worth what a fresh one is worth."""
    import hashlib

    from plmodel.eval.backtest import walk_forward
    from plmodel.eval.compare import ArmSpec, run_arm

    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    corpus = pd.read_parquet(path, columns=["date", "division", "season", "played", "result",
                                            "home_team", "away_team", "home_goals", "away_goals"])
    corpus = corpus[(corpus["division"] == cfg.backtest.prediction_division) & corpus["played"]]
    corpus = corpus.sort_values("date", kind="stable").reset_index(drop=True)
    splits = walk_forward(corpus, first_season="2024-25", last_season="2024-25",
                          min_train_matches=cfg.backtest.min_train_matches)
    spec = ArmSpec.parse("dixon-coles")

    fresh, fresh_state = run_arm(spec, corpus, splits, cfg, cache_dir=tmp_path)
    cached, cached_state = run_arm(spec, corpus, splits, cfg, cache_dir=tmp_path)

    def digest(probs):
        return hashlib.sha256(np.ascontiguousarray(probs, dtype=np.float64).tobytes()).hexdigest()

    assert not fresh_state.get("from_cache")
    assert cached_state.get("from_cache")
    assert digest(fresh) == digest(cached)
    # The fit summary has to survive too: a cached run must be able to report on itself.
    assert cached_state["fit_summary"]["n_fits"] == len(fresh_state["fits"])
