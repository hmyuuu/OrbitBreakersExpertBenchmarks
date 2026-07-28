# Task 10 Autoresearch Campaign

Task: `task-10`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and gate setup

Date: 2026-07-29

Branch: `codex/orbitbreakers/task-10/extreme-cmz`

Latest accepted parent commit:
`7be259106c9dae899e5ea8f82c2251ca285640a4`

The live open-PR search on `sxzgroup/ORBIT-Q` found whole-suite benchmark PRs
4, 5, and 6, but no active Task 10 human-expert optimization PR. This campaign
selects Task 10 and will not edit or benchmark another task.

The immutable reference and editable source were byte-identical at campaign
start:
`sha256:0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172`.

The first build attempt for the pinned
`orbitbreakers-expert-benchmarks:tensorcircuit-py311` image failed during
`apt-get update` after a network interruption. No candidate code had been
changed.

A workload-semantic validation was then run with the immutable human expert
in the already available ORBIT-Q TensorCircuit-NG image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
at 6 CPUs / 7 GiB, network disabled. The evaluator reported:

```text
End-to-end solution time: 18.117244s
Initial energy density: 0.9718865752
Final energy density: -1.1781760454
Exact ground energy density: -1.2925285569
Overall: PASS
```

This validates the canonical public configuration and output contract. It is
not a paired performance claim and is not pooled with future pinned-image
measurements.

## Experiment `e01-exact-mps-scan`

Branch: `codex/orbitbreakers/task-10/extreme-cmz`

## Hypothesis

An exact TensorCircuit-NG `MPSCircuit` formulation will retain bond dimension
at most four across the two bond-2 CMZ applications, and a whole-training scan
will eliminate host dispatch. Together they will reduce valid end-to-end
runtime relative to the immutable general-network/OMECo expert.

## Parent commit and diff digest

Latest accepted parent commit:
`7be259106c9dae899e5ea8f82c2251ca285640a4`

Hypothesis commit: pending gate completion

Candidate file: `src/solutions/task-10/solution_10.py`

Candidate SHA-256: pending

Diff SHA-256: pending

## Data used

Public dataset version: pending Task 10 gate artifact

Private evaluation used: `no`

## Command, seed, and environment

Benchmark command: pending

Public seed: `2040`

Reference SHA-256:
`0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172`

Evaluator SHA-256:
`0ab012597cfa79ec32ebc55bb28307c7b15309315f8057604760df5ad4be71db`

Pair order: odd reference-first, even candidate-first

Timeout: `300 seconds per evaluator process`

Measured region: evaluator timer around `run_solution(config)`

## Result

Not measured. Candidate editing is blocked until the survey and public
workload gates pass.

Decision: `baseline`

## Failure signal and interpretation

The first pinned-image build failed at Debian package index download after the
reported network interruption. This is an environment acquisition failure,
not evidence about the hypothesis.

## Next pivot

Complete the public Task 10 workload record, run reference profiling, retry
the pinned image build when network service returns, then commit e01 before
its first benchmark.

## Append-only corrections

None.
