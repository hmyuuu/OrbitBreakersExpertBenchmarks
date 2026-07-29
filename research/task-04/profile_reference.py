#!/usr/bin/env python3
"""Profile the immutable Task 04 expert without changing benchmark semantics."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import optax


ROOT = Path(__file__).resolve().parents[2]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronized_seconds(function):
    start = time.perf_counter()
    value = function()
    if isinstance(value, tuple):
        leaves = value
    else:
        leaves = (value,)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value, time.perf_counter() - start


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steady-repeats", type=int, default=8)
    args = parser.parse_args()

    solution = load_module(
        ROOT / "references" / "task-04" / "solution_4.py",
        "task04_reference_profile",
    )
    evaluator = load_module(
        ROOT / "tasks" / "task-04" / "evaluator" / "evaluate_4.py",
        "task04_evaluator_profile",
    )
    config = dict(evaluator.DEFAULT_CONFIG)
    K = solution.K

    true_p01 = K.convert_to_tensor(config["true_p01"])
    true_p10 = K.convert_to_tensor(config["true_p10"])
    target, target_seconds = synchronized_seconds(
        lambda: solution.observable_table(true_p01, true_p10, config)
    )

    params = solution.initial_parameters(config)
    optimizer = optax.adam(config["learning_rate"])
    opt_state = optimizer.init(params)

    def loss_fn(p):
        return solution.loss_and_observables(p, target, config)

    value_and_grad = K.value_and_grad(loss_fn, has_aux=True)

    def train_step(p, state):
        (loss, aux), grads = value_and_grad(p)
        updates, state = optimizer.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss, aux

    compiled_step = K.jit(train_step)
    lower_start = time.perf_counter()
    lowered = compiled_step.lower(params, opt_state)
    lower_seconds = time.perf_counter() - lower_start
    compile_start = time.perf_counter()
    executable = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start

    first, first_seconds = synchronized_seconds(
        lambda: executable(params, opt_state)
    )
    params, opt_state = first[0], first[1]
    steady_seconds = []
    for _ in range(args.steady_repeats):
        result, elapsed = synchronized_seconds(
            lambda: executable(params, opt_state)
        )
        params, opt_state = result[0], result[1]
        steady_seconds.append(elapsed)

    cost_analysis = executable.cost_analysis()
    cost_analysis = {
        key: cost_analysis[key]
        for key in ("flops", "bytes accessed", "transcendentals")
        if key in cost_analysis
    }
    payload = {
        "schema_version": 1,
        "task_id": "04",
        "reference_path": "references/task-04/solution_4.py",
        "reference_sha256": (
            "04e37b73e7246599ed3eb8f65e38bb7e084db7aab511d7af3b20baa0867b21ae"
        ),
        "configuration": config,
        "structural_counts": {
            "probes_per_table": 4,
            "channels_per_probe": 22,
            "channels_per_table": 88,
            "kraus_branches_per_channel": 3,
            "observables_per_probe": 13,
            "observables_per_table": 52,
            "training_steps": 120,
        },
        "target_observable_table_seconds": target_seconds,
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "first_compiled_step_seconds": first_seconds,
        "steady_step_seconds": steady_seconds,
        "steady_step_mean_seconds": statistics.mean(steady_seconds),
        "steady_step_median_seconds": statistics.median(steady_seconds),
        "projected_120_step_seconds": (
            120 * statistics.mean(steady_seconds)
        ),
        "stablehlo_line_count": len(lowered.as_text().splitlines()),
        "cost_analysis": jsonable(cost_analysis),
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
    else:
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
