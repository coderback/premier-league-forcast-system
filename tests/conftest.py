"""Suite-wide setup. Must run before numpy is imported anywhere, which is why it lives here.

**Why pin the BLAS thread count.** OpenBLAS allocates per-thread scratch buffers when it
initialises, sized for the machine rather than for the work. On a 16-core host that costs about a
gigabyte of *commit* before a single test runs — measured here at 1,106 MB to import
numpy/pandas/scipy/plmodel against 84 MB with the threads pinned. The paired bootstrap needs two
contiguous ~300 MB arrays, so on a machine whose page file is already near its limit that gigabyte
is the difference between the acceptance harness running and raising ``MemoryError``.

Nothing in this project benefits from threaded BLAS. The largest matrix the model ever touches is
the ~50x50 of a league's attack and defence parameters, far below the size where threading pays,
and the walk is a long sequence of tiny problems rather than a few large ones. Pinning also removes
a source of run-to-run variation in floating-point summation order, which is a small gain for a
project that asserts byte-identical output in six places.

Set as environment variables rather than through ``threadpoolctl`` because the allocation happens
at import time: by the time Python can call a library function, the memory is already committed.
"""
from __future__ import annotations

import os

# Read by OpenBLAS, MKL, OpenMP and Accelerate respectively. Setting all four means the pin holds
# whichever BLAS the installed numpy/scipy wheels happen to carry.
for _variable in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
):
    os.environ.setdefault(_variable, "1")

# Below the pin on purpose, not by accident: importing pandas pulls in numpy, which is when the
# BLAS commits its scratch buffers. Anything that imports numpy must come after the loop above.
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from plmodel.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def corpus(cfg) -> pd.DataFrame:
    """The played top-flight matches, sorted by date. Skips when nothing has been ingested."""
    path = cfg.cache_dir / "matches.parquet"
    if not path.exists():
        pytest.skip("run `pl ingest` first")
    frame = pd.read_parquet(path)
    frame = frame[(frame["division"] == cfg.backtest.prediction_division) & frame["played"]]
    return frame.sort_values("date", kind="stable").reset_index(drop=True)
