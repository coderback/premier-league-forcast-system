"""The ``pl`` command runs single-threaded BLAS, like the suite.

Each check runs in a fresh interpreter with the pin variables removed from its environment. The
suite's own conftest has already pinned THIS process, so an in-process check would pass whether or
not ``plmodel`` pins anything.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import plmodel


def _fresh(code: str, **env: str) -> str:
    clean = {k: v for k, v in os.environ.items() if k not in plmodel.BLAS_THREAD_VARIABLES}
    clean.update(env)
    done = subprocess.run([sys.executable, "-c", code], env=clean, capture_output=True,
                          text=True, check=True)
    return done.stdout.strip()


def test_importing_the_cli_pins_every_blas_to_one_thread() -> None:
    pytest.importorskip("threadpoolctl")
    out = _fresh(
        "import plmodel.cli, numpy, scipy.optimize\n"
        "from threadpoolctl import threadpool_info\n"
        "print(sorted({i['num_threads'] for i in threadpool_info()}))"
    )
    assert out == "[1]"


def test_an_explicit_setting_is_respected() -> None:
    """setdefault: a user who exports a thread count gets it."""
    out = _fresh("import plmodel, os; print(os.environ['OPENBLAS_NUM_THREADS'])",
                 OPENBLAS_NUM_THREADS="4")
    assert out == "4"


def test_the_suite_and_the_package_pin_the_same_variables() -> None:
    """The two pins must not drift: conftest runs before plmodel is importable in some paths."""
    from pathlib import Path

    conftest = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    for variable in plmodel.BLAS_THREAD_VARIABLES:
        assert f'"{variable}"' in conftest
