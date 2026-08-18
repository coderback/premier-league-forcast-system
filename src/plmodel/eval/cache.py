"""A content-addressed cache for a completed arm walk.

A full walk of the production model over the test decade is 1,153 fits and takes minutes. That is
affordable to run once and expensive to run again because the statistics afterwards ran out of
memory, or because a slice needs recomputing, or because a sub-analysis occurred to someone the day
after. This module lets the forecasts be stored and the analysis re-run against them.

**The fingerprint is the whole point.** Reading probabilities from disk introduces the one thing
this harness otherwise does not allow — forecasts that did not come from the arm the report names.
So an entry is keyed by a digest of everything that determines those numbers:

* the arm's name;
* the exact split boundaries, barrier by barrier;
* the identity of every match in the pool, in order;
* the **effective** configuration — every field of the :class:`~plmodel.config.Config` actually in
  force, not merely the file it was loaded from.

That last point is the one that bit. Sweeps and sensitivity runs build variants with
``dataclasses.replace(cfg, model=...)``, which changes the half-life the arm fits at while leaving
the source file untouched. Keying on the file's digest alone would have handed a 730-day walk back
to a caller asking for a 2555-day one, silently and with the right filename. The file digest is
still folded in — it catches edits to comments and to sections not yet typed — but the serialised
configuration is what makes the key correct.

A cache entry whose fingerprint does not match is not stale, it is *wrong*, and it is ignored
rather than repaired. Change a half-life, re-order the corpus, add a barrier, and the entry stops
being found — which is the behaviour that makes a cached run as trustworthy as a fresh one.

What it is not for: a cache is not a substitute for re-running an arm whose code changed. The
fingerprint covers configuration, not source, because hashing the tree would invalidate every
entry on a docstring edit. The discipline that goes with this module is therefore: **delete the
cache directory whenever model or eval code changes.** It lives under ``output/`` and is
gitignored, so throwing it away costs nothing but time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from plmodel.config import Config

# Columns that identify a match, mirroring the harness's alignment guard. Imported lazily at use
# to avoid a circular import with compare.py.
_IDENTITY: tuple[str, ...] = ("date", "division", "home_team", "away_team")


class CacheError(ValueError):
    """Raised when a cache entry exists but cannot be trusted."""


def config_identity(cfg: Config) -> str:
    """Canonical serialisation of the configuration actually in force.

    ``dataclasses.asdict`` walks the whole tree, so an in-memory override made with
    ``dataclasses.replace`` shows up here even though ``cfg.digest`` — the hash of the file on disk
    — does not move. Both go into the fingerprint: this one catches overrides, that one catches
    edits to parts of the file the typed config does not carry.
    """
    return json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str)


def fingerprint(arm: str, pool: pd.DataFrame, splits, cfg: Config) -> str:
    """Identity of one arm's walk: the arm, the barriers, the matches, the configuration."""
    digest = hashlib.sha256()
    digest.update(arm.encode("utf-8"))
    digest.update(cfg.digest.encode("utf-8"))
    digest.update(config_identity(cfg).encode("utf-8"))
    for split in splits:
        digest.update(
            f"{split.index}|{split.train_end}|{split.test_start}|{split.test_stop}|"
            f"{split.fit_barrier}|{split.is_refit}".encode("utf-8")
        )
    identity = pool[list(_IDENTITY)].astype(str).agg("|".join, axis=1)
    digest.update(pd.util.hash_pandas_object(identity, index=False).to_numpy().tobytes())
    return digest.hexdigest()


# How much of the fingerprint goes in the filename. A prefix keeps names readable; the FULL
# fingerprint is stored inside the entry and verified on load, so this length affects only how
# often two entries would want the same filename, never whether a wrong entry can be used.
_FILENAME_DIGEST_CHARS = 16


def _paths(cache_dir: Path, arm: str, key: str) -> tuple[Path, Path]:
    stem = f"{arm}-{key[:_FILENAME_DIGEST_CHARS]}"
    return cache_dir / f"{stem}.npy", cache_dir / f"{stem}.json"


def load(cache_dir: Path, arm: str, key: str) -> tuple[np.ndarray, dict | None] | None:
    """The cached forecasts for this exact walk, or None if there is no matching entry."""
    probs_path, meta_path = _paths(Path(cache_dir), arm, key)
    if not probs_path.exists() or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("fingerprint") != key:
        # A digest collision in the filename is not a licence to use the file.
        raise CacheError(
            f"cache entry {probs_path.name} carries fingerprint {meta.get('fingerprint')!r}, "
            f"not {key!r}; refusing to score forecasts that were made for a different walk"
        )
    probs = np.load(probs_path)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise CacheError(f"cache entry {probs_path.name} has shape {probs.shape}, expected (n, 3)")
    return probs, meta.get("fit_summary")


def save(
    cache_dir: Path, arm: str, key: str, probs: np.ndarray, fit_summary: dict | None
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    probs_path, meta_path = _paths(cache_dir, arm, key)
    np.save(probs_path, np.ascontiguousarray(probs, dtype=np.float64))
    meta_path.write_text(
        json.dumps({"arm": arm, "fingerprint": key, "fit_summary": fit_summary},
                   indent=1, default=str),
        encoding="utf-8",
    )
