"""Enforce "no magic numbers in code" across the model and eval paths.

Non-negotiable #3: every tunable lives in config.yaml with an inline justification. This test is
what makes that a checkable claim rather than a habit. It AST-walks the model and eval packages
and fails on any numeric literal that is not one of:

  1. a structural value in ``ALLOWED`` — array indices, ``ndim`` checks, arithmetic identities;
  2. a literal on a line carrying a ``# MATH:`` marker — a mathematical constant that belongs in
     the code because it defines the formula (RPS's 1/2), not a choice that could have gone
     another way;
  3. a module-level ``UPPER_CASE`` constant that carries an explanatory comment;
  4. a file in ``VERBATIM_PORTS`` — see that mapping for why each is exempt.

Anything else is a decision, and decisions live in config.yaml.

The checker is exercised against synthetic clean/dirty sources below, so it has teeth from commit
one rather than passing vacuously while the packages it scans are still empty.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCANNED_PACKAGES = ("model", "eval")

# Structural values: array indices, shape/ndim comparisons, and arithmetic identities. These
# cannot meaningfully be configured — `probs.shape[1] != 3` is the three-class contract itself.
ALLOWED: frozenset[float] = frozenset({0, 1, 2, 3, -1})

MARKER = "# MATH:"

# Verbatim ports: file -> (reason, the specific literals that file is licensed to contain).
#
# The licence is per-literal, not per-file, so a NEW magic number in a ported file still fails.
# Exempting the whole file would put the project's most load-bearing module permanently outside
# the check. The only admissible reason is a requirement that overrides this one; "it was
# inconvenient" is not a reason.
VERBATIM_PORTS: dict[str, tuple[str, frozenset[float]]] = {
    "eval/metrics.py": (
        "Ported verbatim from the WC2026 project and byte-identity tested against it, so the "
        "ported functions' code lines must not be edited — not even to add markers, which would "
        "make a future diff against the source noisier. A companion test asserts every "
        "paired_delta call site passes n_boot and seed explicitly from config.",
        # EPS clip; the 1/2 in the RPS formula; the bootstrap default; the 95% CI percentiles.
        frozenset({1e-15, 0.5, 10000, 2.5, 97.5}),
    ),
}

# Compound statements whose span must not be treated as a single markable statement — a marker on
# a `for` line must not license every literal in the loop body.
_COMPOUND = (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.For, ast.AsyncFor,
    ast.While, ast.If, ast.With, ast.AsyncWith, ast.Try,
)


def _marked_lines(source: str, tree: ast.AST) -> set[int]:
    """Line numbers licensed by a ``# MATH:`` marker.

    A marker licenses its own physical line, plus the full span of a *simple* statement whose
    first line carries it (so a wrapped call keeps its literals licensed).
    """
    lines = source.splitlines()
    marked = {i for i, text in enumerate(lines, start=1) if MARKER in text}
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or isinstance(node, _COMPOUND):
            continue
        end = getattr(node, "end_lineno", None) or node.lineno
        if node.lineno in marked:
            marked.update(range(node.lineno, end + 1))
    return marked


def _documented_constant_lines(source: str, tree: ast.AST) -> set[int]:
    """Line numbers inside module-level UPPER_CASE assignments that carry a comment."""
    lines = source.splitlines()
    allowed: set[int] = set()
    for node in tree.body if isinstance(tree, ast.Module) else []:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        names = [t for t in targets if isinstance(t, ast.Name)]
        # Tuple unpacking (`_LO, _HI = -6.0, 4.0`) counts when every name is UPPER_CASE.
        for t in targets:
            if isinstance(t, ast.Tuple):
                names.extend(e for e in t.elts if isinstance(e, ast.Name))
        if not names or not all(n.id.isupper() for n in names):
            continue
        own = lines[node.lineno - 1] if node.lineno <= len(lines) else ""
        # Walk back past blank lines and sibling constant assignments: one comment above a block
        # of related constants documents all of them, which is how they are naturally written.
        sibling_lines = {n.lineno for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))}
        idx = node.lineno - 2
        while idx >= 0 and (not lines[idx].strip() or (idx + 1) in sibling_lines):
            idx -= 1
        prev = lines[idx].strip() if idx >= 0 else ""
        if "#" not in own and not prev.startswith("#"):
            continue  # an undocumented constant is still a magic number, just relocated
        for child in ast.walk(value):
            if isinstance(child, ast.Constant):
                allowed.add(child.lineno)
    return allowed


def find_violations(
    source: str, filename: str = "<source>", *, extra_allowed: frozenset[float] = frozenset()
) -> list[str]:
    """Numeric literals in ``source`` that no rule licenses, as human-readable messages."""
    tree = ast.parse(source, filename=filename)
    licensed = _marked_lines(source, tree) | _documented_constant_lines(source, tree)
    allowed = ALLOWED | extra_allowed
    lines = source.splitlines()

    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        # bool is a subclass of int; True/False are not magic numbers.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value in allowed or node.lineno in licensed:
            continue
        text = lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
        violations.append(f"{filename}:{node.lineno}: bare literal {value!r} in `{text}`")
    return violations


def scanned_files() -> list[Path]:
    files: list[Path] = []
    for package in SCANNED_PACKAGES:
        files.extend(sorted((REPO_ROOT / "src" / "plmodel" / package).rglob("*.py")))
    return files


def test_no_magic_numbers_in_model_and_eval() -> None:
    violations: list[str] = []
    for path in scanned_files():
        rel = path.relative_to(REPO_ROOT / "src" / "plmodel").as_posix()
        _, extra = VERBATIM_PORTS.get(rel, ("", frozenset()))
        violations.extend(
            find_violations(path.read_text(encoding="utf-8"), rel, extra_allowed=extra)
        )
    assert not violations, "magic numbers must live in config.yaml:\n  " + "\n  ".join(violations)


def test_every_exemption_carries_a_reason() -> None:
    """An exemption without a stated reason is a hole, not an exemption."""
    for name, (reason, literals) in VERBATIM_PORTS.items():
        assert reason.strip(), f"{name} is exempt with no reason given"
        assert literals, f"{name} is exempt but licenses no literals; drop the entry"
        assert (REPO_ROOT / "src" / "plmodel" / name).exists(), f"stale exemption: {name}"


def test_exemptions_do_not_license_unused_literals() -> None:
    """A licensed literal that no longer appears is a stale hole; remove it."""
    for name, (_, literals) in VERBATIM_PORTS.items():
        source = (REPO_ROOT / "src" / "plmodel" / name).read_text(encoding="utf-8")
        present = {
            n.value for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Constant)
            and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool)
        }
        unused = {lit for lit in literals if lit not in present}
        assert not unused, f"{name} licenses literals it no longer contains: {sorted(unused)}"


# --- the checker's own teeth -------------------------------------------------------------------
# Without these the test above passes vacuously while model/ and eval/ are empty.

def test_checker_flags_a_bare_literal() -> None:
    assert find_violations("def f(x):\n    return x * 0.7\n")


def test_checker_allows_structural_values() -> None:
    assert not find_violations("def f(p):\n    if p.ndim != 2 or p.shape[1] != 3:\n        raise ValueError\n")


def test_checker_allows_marked_math() -> None:
    src = "def rps(a, b):\n    return 0.5 * (a + b)  # MATH: RPS is the mean of squared cumulative errors\n"
    assert not find_violations(src)


def test_checker_allows_documented_module_constant() -> None:
    src = "# Clip the linear predictor so the optimiser cannot overflow exp().\n_LOG_RATE_MIN = -6.0\n"
    assert not find_violations(src)


def test_one_comment_documents_a_block_of_constants() -> None:
    """Related constants are naturally written under a single comment."""
    src = "# Newey-West automatic bandwidth parameters.\n_A = 4.0\n_B = 100.0\n_C = 9.0\n"
    assert not find_violations(src)


def test_undocumented_constant_after_a_documented_one_is_still_caught() -> None:
    src = "# Documented.\n_A = 4.0\n\n\ndef f():\n    return 7.7\n"
    assert find_violations(src)


def test_extra_allowed_licenses_only_what_it_names() -> None:
    src = "def f(n_boot=10000, other=1234):\n    return n_boot, other\n"
    assert len(find_violations(src)) == 2
    remaining = find_violations(src, extra_allowed=frozenset({10000}))
    assert len(remaining) == 1 and "1234" in remaining[0]


def test_checker_rejects_undocumented_module_constant() -> None:
    assert find_violations("_LOG_RATE_MIN = -6.0\n")


def test_checker_rejects_function_default_tunable() -> None:
    """The shape that hides most tunables: a default argument nobody re-reads."""
    assert find_violations("def paired_delta(a, b, n_boot=10000):\n    return n_boot\n")


def test_checker_ignores_booleans_and_strings() -> None:
    assert not find_violations('def f(flag=True, name="x"):\n    """Doc with 4 words."""\n    return flag, name\n')


def test_marker_on_compound_statement_does_not_license_its_body() -> None:
    """A marker on a `for` line must not license every literal in the loop."""
    src = "def f(xs):\n    for x in xs:  # MATH: iterate\n        y = x * 0.7\n    return y\n"
    assert find_violations(src)


@pytest.mark.parametrize("package", SCANNED_PACKAGES)
def test_scanned_packages_exist(package: str) -> None:
    """Guard against the scan silently covering nothing because a path was renamed."""
    assert (REPO_ROOT / "src" / "plmodel" / package).is_dir()
