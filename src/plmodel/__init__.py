"""Premier League forecasting model.

**The BLAS thread pin lives here, and has to.** Python runs a package's ``__init__`` before any of
its submodules, so this is the one place guaranteed to execute before ``plmodel`` imports numpy. It
is the same pin ``tests/conftest.py`` has applied to the suite since 2026-08, for the same two
reasons (NOTES.md, the Arm 3 memory entry):

* OpenBLAS commits per-thread scratch buffers at import, sized for the machine. On a 16-core host
  that is about a gigabyte before any work, against 84 MB pinned, and this project's largest matrix
  is a league's ~50x50 parameters, far below where threading pays.
* Threaded BLAS varies floating-point summation order from run to run. The ``pl`` command ran
  unpinned until 2026-11-17, and identical ``pl compare`` runs moved P(better) in the third decimal
  (0.835 against 0.838).

``setdefault``, so anyone who wants threads back can export the variables before running ``pl``. A
caller that imports numpy BEFORE importing ``plmodel`` gets whatever numpy already started with;
nothing set here can reach back into a library that is already initialised.
"""
import os

# Read by OpenBLAS, MKL, OpenMP and Accelerate respectively, so the pin holds whichever BLAS the
# installed numpy/scipy wheels carry. Kept identical to tests/conftest.py's list.
BLAS_THREAD_VARIABLES: tuple[str, ...] = (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
for _variable in BLAS_THREAD_VARIABLES:
    os.environ.setdefault(_variable, "1")

__version__ = "0.1.0"
