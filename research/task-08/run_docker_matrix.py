#!/usr/bin/env python3
"""Run Task 08 reduced pairs or canonical candidate repeats in one container."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import statistics
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_RE = re.compile(r"End-to-end solution time:\s*([0-9.]+)s")
T_CRITICAL_95 = {5: 2.5705818366}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stats(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    stdev = statistics.stdev(values) if n > 1 else None
    return {
        "n": n,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "sample_stdev": stdev,
        "stderr": stdev / math.sqrt(n) if stdev is not None else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def run_command(
    command: list[str], *, timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def host_record() -> dict[str, object]:
    commands = {
        "uname": ["uname", "-a"],
        "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "physical_memory": ["sysctl", "-n", "hw.memsize"],
    }
    record: dict[str, object] = {}
    for name, command in commands.items():
        try:
            result = run_command(command)
            record[name] = result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            record[name] = None
    fingerprint_bytes = json.dumps(record, sort_keys=True).encode()
    record["fingerprint_sha256"] = hashlib.sha256(fingerprint_bytes).hexdigest()
    return record


def image_record(image: str) -> dict[str, object]:
    result = run_command(["docker", "image", "inspect", image])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"cannot inspect {image}")
    raw = json.loads(result.stdout)[0]
    return {
        "reference": image,
        "id": raw.get("Id"),
        "repo_digests": raw.get("RepoDigests") or [],
        "created": raw.get("Created"),
        "architecture": raw.get("Architecture"),
        "os": raw.get("Os"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("paired", "candidate-only"), required=True)
    parser.add_argument("--n-samples", type=int, required=True)
    parser.add_argument("--repeat", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cpus", type=float, default=6.0)
    parser.add_argument("--memory", default="7g")
    parser.add_argument(
        "--image",
        default="orbitbreakers-expert-benchmarks:tensorcircuit-py311",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples <= 0 or args.repeat <= 0:
        raise SystemExit("n-samples and repeat must be positive")
    timeout = min(args.timeout, 300.0)
    output = args.output.expanduser().resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    sources = {
        "candidate": ROOT / "src/solutions/task-08/solution_8.py",
        "reference": ROOT / "references/task-08/solution_8.py",
    }
    evaluator = ROOT / "tasks/task-08/evaluator/evaluate_8.py"
    sitecustomize = ROOT / "envs/tensorcircuit-py311/sitecustomize.py"
    for path in [*sources.values(), evaluator, sitecustomize]:
        if not path.is_file():
            raise SystemExit(f"missing required file: {path}")

    image = image_record(args.image)
    container_name = f"orbit-task08-matrix-{uuid.uuid4().hex[:10]}"
    rows: list[dict[str, object]] = []
    started_at = utc_now()
    session_started = time.perf_counter()
    container_id: str | None = None

    staging_root = ROOT / ".tmp"
    staging_root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="task08-matrix-", dir=staging_root) as tmp:
        staging = Path(tmp)
        shutil.copy2(evaluator, staging / "evaluate_8.py")
        environment = staging / "environment"
        environment.mkdir()
        shutil.copy2(sitecustomize, environment / "sitecustomize.py")
        modules: dict[str, str] = {}
        for role, source in sources.items():
            module = f"solution_8_{role}"
            modules[role] = module
            shutil.copy2(source, staging / f"{module}.py")

        snapshot = {
            role: sha256(staging / f"{module}.py")
            for role, module in modules.items()
        }
        snapshot["evaluator"] = sha256(staging / "evaluate_8.py")
        snapshot["sitecustomize"] = sha256(environment / "sitecustomize.py")
        snapshot_sha256 = hashlib.sha256(
            json.dumps(snapshot, sort_keys=True).encode()
        ).hexdigest()

        start_command = [
            "docker",
            "run",
            "--detach",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,size=1g",
            "--mount",
            f"type=bind,src={staging.resolve()},dst=/session,readonly",
            "--workdir",
            "/session",
            "--env",
            "NUMBA_DISABLE_JIT=1",
            "--env",
            "PYTHONPATH=/session:/session/environment",
            "--cpus",
            str(args.cpus),
            "--memory",
            args.memory,
            args.image,
            "tail",
            "-f",
            "/dev/null",
        ]
        started = run_command(start_command, timeout=60)
        if started.returncode != 0:
            raise SystemExit(started.stderr.strip() or "container start failed")
        container_id = started.stdout.strip()

        plan: list[tuple[int, int, str, str]] = []
        if args.mode == "paired":
            for pair in range(1, args.repeat + 1):
                roles = (
                    ("reference", "candidate")
                    if pair % 2
                    else ("candidate", "reference")
                )
                order = "->".join(roles)
                for position, role in enumerate(roles, start=1):
                    plan.append((pair, position, role, order))
        else:
            plan = [
                (repeat, 1, "candidate", "candidate-only")
                for repeat in range(1, args.repeat + 1)
            ]

        try:
            for cell_index, (pair, position, role, order) in enumerate(plan, start=1):
                module = modules[role]
                command = [
                    "docker",
                    "exec",
                    "--workdir",
                    "/session",
                    "--env",
                    "NUMBA_DISABLE_JIT=1",
                    "--env",
                    "PYTHONPATH=/session:/session/environment",
                    container_name,
                    "python",
                    "/session/evaluate_8.py",
                    "--solution",
                    module,
                    "--n-samples",
                    str(args.n_samples),
                ]
                cell_started_at = utc_now()
                wall_started = time.perf_counter()
                timed_out = False
                try:
                    result = run_command(command, timeout=timeout)
                    stdout = result.stdout
                    stderr = result.stderr
                    returncode = result.returncode
                except subprocess.TimeoutExpired as exc:
                    timed_out = True
                    stdout = exc.stdout or ""
                    stderr = exc.stderr or ""
                    returncode = None
                wall_sec = time.perf_counter() - wall_started
                stdout_path = logs / f"cell-{cell_index:02d}-{role}.stdout.txt"
                stderr_path = logs / f"cell-{cell_index:02d}-{role}.stderr.txt"
                stdout_path.write_text(stdout, encoding="utf-8")
                stderr_path.write_text(stderr, encoding="utf-8")
                match = RUNTIME_RE.search(stdout)
                runtime = float(match.group(1)) if match else None
                passed = (
                    not timed_out
                    and returncode == 0
                    and runtime is not None
                    and "Overall: PASS" in stdout
                )
                row = {
                    "cell_id": f"task08-{cell_index:02d}",
                    "pair": pair,
                    "position": position,
                    "order": order,
                    "role": role,
                    "n_samples": args.n_samples,
                    "runtime_sec": runtime,
                    "controller_wall_sec": wall_sec,
                    "passed": passed,
                    "timed_out": timed_out,
                    "returncode": returncode,
                    "started_at": cell_started_at,
                    "source_sha256": snapshot[role],
                    "evaluator_sha256": snapshot["evaluator"],
                    "stdout_sha256": sha256(stdout_path),
                    "stderr_sha256": sha256(stderr_path),
                }
                rows.append(row)
                print(
                    f"{row['cell_id']} {role} passed={passed} "
                    f"runtime={runtime} wall={wall_sec:.3f}",
                    flush=True,
                )
                if timed_out:
                    break
        finally:
            run_command(["docker", "rm", "-f", container_name], timeout=30)

    summary: dict[str, object] = {}
    by_role = {
        role: [
            float(row["runtime_sec"])
            for row in rows
            if row["role"] == role and row["passed"]
        ]
        for role in ("reference", "candidate")
    }
    summary["reference"] = stats(by_role["reference"])
    summary["candidate"] = stats(by_role["candidate"])
    if args.mode == "paired":
        pair_speeds = []
        pair_rows = []
        for pair in range(1, args.repeat + 1):
            pair_cells = [row for row in rows if row["pair"] == pair]
            values = {
                str(row["role"]): float(row["runtime_sec"])
                for row in pair_cells
                if row["passed"] and row["runtime_sec"] is not None
            }
            if set(values) == {"reference", "candidate"}:
                speedup = values["reference"] / values["candidate"]
                pair_speeds.append(speedup)
                pair_rows.append({"pair": pair, **values, "speedup": speedup})
        speed_stats = stats(pair_speeds)
        df = len(pair_speeds) - 1
        critical = T_CRITICAL_95.get(df)
        if critical is not None and speed_stats["stderr"] is not None:
            margin = critical * float(speed_stats["stderr"])
            speed_stats["ci_95_low"] = float(speed_stats["mean"]) - margin
            speed_stats["ci_95_high"] = float(speed_stats["mean"]) + margin
        ref_mean = summary["reference"]["mean"]  # type: ignore[index]
        cand_mean = summary["candidate"]["mean"]  # type: ignore[index]
        summary["eligible_pairs"] = pair_rows
        summary["pairwise_speedup"] = speed_stats
        summary["ratio_of_means_speedup"] = (
            float(ref_mean) / float(cand_mean)
            if ref_mean is not None and cand_mean is not None
            else None
        )
        summary["ratio_of_means_improvement_pct"] = (
            100.0 * (float(ref_mean) - float(cand_mean)) / float(ref_mean)
            if ref_mean is not None and cand_mean is not None
            else None
        )

    report = {
        "schema_version": 1,
        "task_id": "08",
        "mode": args.mode,
        "n_samples": args.n_samples,
        "planned_repeats": args.repeat,
        "timeout_sec": timeout,
        "started_at": started_at,
        "finished_at": utc_now(),
        "controller_elapsed_sec": time.perf_counter() - session_started,
        "container": {
            "name": container_name,
            "id": container_id,
            "cpus": args.cpus,
            "memory": args.memory,
            "network": "none",
            "process_state": "fresh evaluator process per cell",
        },
        "image": image,
        "host": host_record(),
        "provenance": {
            "snapshot_sha256": snapshot_sha256,
            "files": snapshot,
        },
        "results": rows,
        "summary": summary,
    }
    report_path = output / "results.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"report={report_path}")
    if not rows or any(not row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
