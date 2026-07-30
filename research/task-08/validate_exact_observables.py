#!/usr/bin/env python3
"""Validate Task 08 with exact TensorCircuit light-cone contractions.

This is independent of the expert's sequential sampling path: it computes
each published evaluator observable directly from the circuit tensor network.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import tensorcircuit as tc


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-abs-error", type=float, default=2e-6)
    args = parser.parse_args()

    expert_path = ROOT / "references/task-08/solution_8.py"
    evaluator_path = ROOT / "tasks/task-08/evaluator/evaluate_8.py"
    expert = load_module("task08_exact_expert", expert_path)
    evaluator = load_module("task08_exact_evaluator", evaluator_path)
    circuit = expert.build_circuit(dict(evaluator.DEFAULT_CONFIG))

    started = time.perf_counter()
    values = []
    timings = []
    for support in evaluator.HIDDEN_Z_STRINGS:
        item_started = time.perf_counter()
        value = circuit.expectation_ps(z=list(support), enable_lightcone=True)
        values.append(float(tc.backend.numpy(tc.backend.real(value))))
        timings.append(time.perf_counter() - item_started)
    elapsed = time.perf_counter() - started

    expected = np.asarray(evaluator.EXACT_HIDDEN_Z_STRING_VALUES)
    errors = np.abs(np.asarray(values) - expected)
    passed = bool(np.max(errors) <= args.max_abs_error)
    report = {
        "schema_version": 1,
        "task_id": "08",
        "method": "TensorCircuit expectation_ps exact contraction with enable_lightcone=True",
        "sampling_path_used": False,
        "observable_count": len(values),
        "max_abs_error_threshold": args.max_abs_error,
        "max_abs_error": float(np.max(errors)),
        "mean_abs_error": float(np.mean(errors)),
        "elapsed_sec": elapsed,
        "per_observable_elapsed_sec": timings,
        "computed_values": values,
        "expected_values": expected.tolist(),
        "passed": passed,
        "provenance": {
            "expert_sha256": sha256(expert_path),
            "evaluator_sha256": sha256(evaluator_path),
            "tensorcircuit_version": tc.__version__,
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
