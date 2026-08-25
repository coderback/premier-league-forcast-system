"""Committing a frozen forecast, which is what makes the ledger's claim checkable.

A frozen file's whole assertion is *this forecast existed before the result did*, and an untracked
file in a gitignored directory cannot support it: the filename is chosen by whoever writes it, so
any file could be created at any moment with any date in its name. Committing puts it in a hash
chain instead.

The two properties that matter here are both about not doing damage. A freeze happens with a match
about to kick off, so the commit must never swallow unrelated work and must never be the reason the
forecast is lost.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from plmodel.eval.live import commit_frozen


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repository with one commit, an unrelated dirty file, and a ledger directory."""
    root = tmp_path / "repo"
    (root / "output" / "live").mkdir(parents=True)
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "config", "user.name", "Test")
    (root / "tracked.py").write_text("original\n", encoding="utf-8")
    git(root, "add", "tracked.py")
    git(root, "commit", "-qm", "initial")
    # Work in progress that must survive a freeze untouched.
    (root / "tracked.py").write_text("half-finished refactor\n", encoding="utf-8")
    return root


def test_a_frozen_file_is_committed(repo: Path) -> None:
    path = repo / "output" / "live" / "2026-08-29.json"
    path.write_text('{"barrier": "2026-08-29"}', encoding="utf-8")

    receipt = commit_frozen(path)

    assert receipt["committed"] is True
    assert receipt["commit"]
    assert "2026-08-29" in receipt["message"]
    assert git(repo, "log", "-1", "--format=%s") == "chore: freeze live forecasts for 2026-08-29"


def test_the_commit_does_not_sweep_up_unrelated_work(repo: Path) -> None:
    """The load-bearing one.

    Somebody mid-refactor who freezes a matchday must not find their work-in-progress committed
    under a message about forecasts. ``git commit -- <path>`` commits that path regardless of the
    index, which is why it is used instead of a bare commit.
    """
    path = repo / "output" / "live" / "2026-08-29.json"
    path.write_text('{"barrier": "2026-08-29"}', encoding="utf-8")

    assert commit_frozen(path)["committed"] is True

    committed = git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["output/live/2026-08-29.json"]
    # The refactor is still uncommitted, and still says what it said.
    assert (repo / "tracked.py").read_text(encoding="utf-8") == "half-finished refactor\n"
    assert "tracked.py" in git(repo, "status", "--porcelain")


def test_a_staged_but_unrelated_change_is_left_staged(repo: Path) -> None:
    """Even when the index is dirty, only the ledger file goes in."""
    git(repo, "add", "tracked.py")
    path = repo / "output" / "live" / "2026-08-29.json"
    path.write_text('{"barrier": "2026-08-29"}', encoding="utf-8")

    assert commit_frozen(path)["committed"] is True

    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == [
        "output/live/2026-08-29.json"
    ]
    assert git(repo, "diff", "--cached", "--name-only") == "tracked.py"


def test_no_upstream_is_reported_rather_than_assumed(repo: Path) -> None:
    """A branch nobody has pushed has no external witness, and the caller has to be able to say so.

    This is the honest half of the feature: a local commit date is settable, so a commit alone does
    not establish when a file was written. Pushing is what records that a third party saw it.
    """
    path = repo / "output" / "live" / "2026-08-29.json"
    path.write_text('{"barrier": "2026-08-29"}', encoding="utf-8")

    assert commit_frozen(path)["unpushed_commits"] is None


def test_a_failed_commit_never_raises_and_says_why(tmp_path: Path) -> None:
    """The freeze is already on disk. Losing it to a git problem would be the worse outcome."""
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    path = outside / "2026-08-29.json"
    path.write_text('{"barrier": "2026-08-29"}', encoding="utf-8")

    receipt = commit_frozen(path)

    assert receipt["committed"] is False
    assert "git" in receipt["reason"] or "repository" in receipt["reason"]
    assert path.exists(), "the frozen forecast must survive a failed commit"


def test_committing_the_same_file_twice_is_not_an_error(repo: Path) -> None:
    """The second call has nothing to commit, and must report that rather than blow up.

    Reachable whenever someone re-runs `pl live` after a freeze; the freeze itself already refuses
    to overwrite, so the only thing left to go wrong is here.
    """
    path = repo / "output" / "live" / "2026-08-29.json"
    path.write_text('{"barrier": "2026-08-29"}', encoding="utf-8")
    assert commit_frozen(path)["committed"] is True

    receipt = commit_frozen(path)

    assert receipt["committed"] is False
    assert receipt["reason"]


# --- the CLI path, end to end ---------------------------------------------------------------------

def test_pl_live_freezes_and_commits_in_one_go(repo: Path, monkeypatch) -> None:
    """Drives `cmd_live` itself, because the freeze and the commit are wired together there.

    Worth a test rather than an inspection: this project has already been bitten once by a CLI
    branch that was never exercised, where `pl backtest` would have skipped its whole readout in
    silence if the production arm were ever renamed.
    """
    import argparse
    import dataclasses

    import pandas as pd

    from plmodel import cli
    from plmodel.config import load_config

    cfg = dataclasses.replace(load_config(), output_dir=repo / "output")
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01", "2026-08-08"]),
            "season": "2026-27", "division": "E0",
            "home_team": ["Arsenal", "Chelsea"], "away_team": ["Chelsea", "Everton"],
            "home_goals": [2.0, 1.0], "away_goals": [1.0, 1.0], "result": ["H", "D"],
            "played": [True, True],
        }
    )
    fixtures = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-29", "2026-08-29"]),
            "season": "2026-27", "division": "E0",
            "home_team": ["Everton", "Fulham"], "away_team": ["Arsenal", "Chelsea"],
            "home_goals": [pd.NA, pd.NA], "away_goals": [pd.NA, pd.NA],
            "result": [None, None], "played": [False, False],
        }
    )
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(cli, "_load_corpus", lambda c: (history, history))
    monkeypatch.setattr(
        "plmodel.data.football_data.upcoming_fixtures", lambda *a, **k: fixtures
    )

    args = argparse.Namespace(
        config=None, arms="uniform,home-rate", score=False, dry_run=False, no_feed=True
    )
    assert cli.cmd_live(args) == 0

    frozen = repo / "output" / "live" / "2026-08-29.json"
    assert frozen.exists(), "the matchday was not frozen"
    assert git(repo, "log", "-1", "--format=%s") == "chore: freeze live forecasts for 2026-08-29"
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == [
        "output/live/2026-08-29.json"
    ]
    # And the unrelated refactor from the fixture is still sitting there, uncommitted.
    assert "tracked.py" in git(repo, "status", "--porcelain")
