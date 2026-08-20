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
this, so **it is never used on the ingest path**.

It is used in exactly one place, at the other end of the program: :func:`resolve_team`, which maps
what a person typed at a command line onto a canonical name. The two are opposite problems. An
unrecognised name arriving from the *source* is a data error and must stop the ingest; an
unrecognised name arriving from a *person* is a typo, and refusing to forecast Man United because
someone wrote "Man Utd" helps nobody. The resolver never touches the corpus, reports every
substitution it makes, and refuses to guess between two plausible clubs.

The WC2026 project's two worst bugs were both silent key misses (a substring config key matching
the wrong tier; an accented tournament name falling through to a fallback). Both were invisible
until audited. This module is the guard against the same class of bug here.
"""
from __future__ import annotations

import difflib
import unicodedata
from collections.abc import Sequence
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


# How much better the best fuzzy match must be than the runner-up before it is accepted
# without asking. Below this the two names are close enough that guessing is a coin flip.
_RESOLVE_MARGIN = 0.08


# --- resolving a name a human typed ------------------------------------------------------------

class AmbiguousTeamError(TeamNameError):
    """Raised when typed input matches no club, or several equally well."""


def resolve_team(typed: str, known: Sequence[str], *, aliases: dict[str, str] | None = None,
                 cutoff: float = 0.6) -> str:
    """Best canonical name for something a person typed at a command line.

    Deliberately separate from :func:`load_aliases`, which maps *source* spellings and is a closed
    set guarding corpus integrity: a typo there is a data error and must fail. Here a typo is a
    person in a hurry, and "Man Utd" should reach Man United rather than a stack trace.

    Resolution is reported rather than silent, and an ambiguous input raises with the candidates
    instead of guessing — picking the alphabetically-first of two plausible clubs is how a forecast
    ends up quietly about the wrong team.
    """
    names = list(known)
    folded = {_fold(n): n for n in names}
    key = _fold(typed)
    if typed in names:
        return typed
    if aliases and typed in aliases and aliases[typed] in names:
        return aliases[typed]
    if key in folded:
        return folded[key]

    close = difflib.get_close_matches(typed.lower(), [n.lower() for n in names], n=4, cutoff=cutoff)
    lower = {n.lower(): n for n in names}
    if len(close) == 1:
        return lower[close[0]]
    if len(close) > 1:
        best, runner = difflib.SequenceMatcher(None, typed.lower(), close[0]).ratio(), \
            difflib.SequenceMatcher(None, typed.lower(), close[1]).ratio()
        # A clear winner is accepted; a near-tie is the caller's to settle.
        if best - runner >= _RESOLVE_MARGIN:
            return lower[close[0]]
        raise AmbiguousTeamError(
            f"{typed!r} could be any of {[lower[c] for c in close]}; write the name in full"
        )
    raise AmbiguousTeamError(
        f"no club matches {typed!r}. Closest: "
        f"{difflib.get_close_matches(typed.lower(), [n.lower() for n in names], n=5, cutoff=0.3)}"
    )
