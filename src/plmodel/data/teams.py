"""Team-name canonicalisation, failing loudly on anything unrecognised.

Two files, both committed, both curated by hand:

* ``team_aliases.yaml`` — source spelling -> canonical name. Only entries where the two differ.
* ``team_roster.yaml``  — the full set of canonical names the corpus is allowed to contain.

Ingest maps through the aliases and then asserts every resulting name is on the roster. An
unrecognised name **raises**, listing the offenders so the map can be extended deliberately.

Why the roster exists as well as the aliases: the real hazard is not a name we have never seen,
it is a name that changes spelling between seasons and silently becomes a *second team* with a
fresh, empty history. That failure is invisible without a closed roster — the frame still looks
well-formed, the model just quietly forgets a club's past. Fuzzy matching would paper over exactly
this, so it is never used.

The WC2026 project's two worst bugs were both silent key misses (a substring config key matching
the wrong tier; an accented tournament name falling through to a fallback). Both were invisible
until audited. This module is the guard against the same class of bug here.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd
import yaml

ALIAS_FILENAME = "team_aliases.yaml"
ROSTER_FILENAME = "team_roster.yaml"


class TeamNameError(ValueError):
    """Raised when a team name does not resolve to a canonical roster entry."""


def _read_yaml(path: Path) -> object:
    if not path.exists():
        raise TeamNameError(f"required team file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_aliases(static_dir: Path) -> dict[str, str]:
    """Source spelling -> canonical name."""
    raw = _read_yaml(static_dir / ALIAS_FILENAME) or {}
    if not isinstance(raw, dict):
        raise TeamNameError(f"{ALIAS_FILENAME} must be a mapping of source name -> canonical name")
    return {str(k).strip(): str(v).strip() for k, v in raw.items()}


def load_roster(static_dir: Path) -> set[str]:
    """The closed set of canonical team names."""
    raw = _read_yaml(static_dir / ROSTER_FILENAME) or []
    if not isinstance(raw, list):
        raise TeamNameError(f"{ROSTER_FILENAME} must be a list of canonical team names")
    roster = {str(x).strip() for x in raw}
    if not roster:
        raise TeamNameError(f"{ROSTER_FILENAME} is empty; the roster guard would be vacuous")
    return roster


def canonicalise(
    names: pd.Series, aliases: dict[str, str], roster: set[str], *, source: str = ""
) -> pd.Series:
    """Map a column of source team names to canonical names, or raise listing the unknowns."""
    cleaned = names.astype(str).str.strip()
    mapped = cleaned.map(lambda n: aliases.get(n, n))
    unknown = sorted(set(mapped) - roster)
    if unknown:
        where = f"{source}: " if source else ""
        raise TeamNameError(
            f"{where}{len(unknown)} team name(s) not on the roster: {unknown}\n"
            f"Add each to {ALIAS_FILENAME} (if it is a spelling of an existing club) or to "
            f"{ROSTER_FILENAME} (if it is a club the corpus has not seen before). "
            f"Never fuzzy-match: a near-miss is how one club silently becomes two."
        )
    return mapped


# --- curation helpers -------------------------------------------------------------------------
# Used when extending the roster, never on the ingest path.

def _fold(name: str) -> str:
    """Aggressively normalise a name for near-duplicate detection only."""
    decomposed = unicodedata.normalize("NFKD", name.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in stripped if c.isalnum())


def find_near_duplicates(names: set[str]) -> list[tuple[str, str]]:
    """Pairs of roster names that fold to the same key — the silent-duplicate smell.

    Catches 'Nott'm Forest' vs 'Nottm Forest' and 'Sheffield Weds' vs 'Sheffield Weds.'. Reported
    for human review during curation; it is not a substitute for reading the roster.
    """
    by_key: dict[str, list[str]] = {}
    for name in sorted(names):
        by_key.setdefault(_fold(name), []).append(name)
    pairs: list[tuple[str, str]] = []
    for group in by_key.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pairs.append((group[i], group[j]))
    return pairs
