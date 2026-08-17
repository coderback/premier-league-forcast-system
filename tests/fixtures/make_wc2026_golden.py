"""Generate the metrics byte-identity fixture by running the WC2026 implementation.

Run with the WC2026 venv so the numbers come from that project's code, not a reimplementation.
The fixture is deterministic: a fixed seed, a fixed construction, full float repr in the output.
"""
import json
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\tobio\PycharmProjects\fifa-wc2026-prediction-model\src")
from wc2026.eval import metrics  # noqa: E402

SEED = 20260822
N = 500


def build_case():
    """A fixed (probs_a, probs_b, outcomes) triple spanning sharp, flat and skewed forecasts."""
    rng = np.random.default_rng(SEED)
    # Dirichlet gives well-formed probability triples with a realistic spread of sharpness.
    probs_a = rng.dirichlet([4.0, 3.0, 3.0], size=N)
    probs_b = rng.dirichlet([2.0, 2.0, 2.0], size=N)
    # Outcomes drawn from a third distribution so neither forecaster is trivially right.
    truth = rng.dirichlet([3.5, 3.0, 3.0], size=N)
    outcomes = np.array([rng.choice(3, p=row) for row in truth])
    return probs_a, probs_b, outcomes


def main() -> int:
    probs_a, probs_b, outcomes = build_case()
    uniform = metrics.uniform_baseline(len(outcomes))

    out = {
        "_provenance": {
            "source": "wc2026.eval.metrics",
            "repo": "fifa-wc2026-prediction-model",
            "numpy": np.__version__,
            "seed": SEED,
            "n": N,
        },
        "inputs": {
            "probs_a": probs_a.tolist(),
            "probs_b": probs_b.tolist(),
            "outcomes": outcomes.tolist(),
        },
        "expected": {
            "rps_a": metrics.rps(probs_a, outcomes).tolist(),
            "rps_b": metrics.rps(probs_b, outcomes).tolist(),
            "log_loss_a": metrics.log_loss(probs_a, outcomes).tolist(),
            "brier_a": metrics.brier(probs_a, outcomes).tolist(),
            "mean_rps_a": metrics.mean_rps(probs_a, outcomes),
            "mean_rps_b": metrics.mean_rps(probs_b, outcomes),
            "mean_log_loss_a": metrics.mean_log_loss(probs_a, outcomes),
            "mean_brier_a": metrics.mean_brier(probs_a, outcomes),
            "mean_rps_uniform": metrics.mean_rps(uniform, outcomes),
            "summary_a": metrics.summary(probs_a, outcomes),
            "skill_a": metrics.skill(
                metrics.mean_rps(probs_a, outcomes), metrics.mean_rps(uniform, outcomes)
            ),
            "paired_delta_ab": metrics.paired_delta(probs_a, probs_b, outcomes),
            "paired_delta_ab_seed7_boot2000": metrics.paired_delta(
                probs_a, probs_b, outcomes, n_boot=2000, seed=7
            ),
            "outcome_from_scores": metrics.outcome_from_scores(
                np.array([3, 1, 0, 2]), np.array([1, 1, 2, 5])
            ).tolist(),
        },
    }
    path = "wc2026_metrics_golden.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {path}")
    print("mean_rps_a       =", repr(out["expected"]["mean_rps_a"]))
    print("mean_rps_b       =", repr(out["expected"]["mean_rps_b"]))
    print("mean_rps_uniform =", repr(out["expected"]["mean_rps_uniform"]))
    print("paired_delta     =", out["expected"]["paired_delta_ab"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
