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

## Experiment `e01-exact-mps-scan`: measured result

Candidate commit: `05ba40c`

Candidate SHA-256:
`c1705d11d5c9e984ec7628b2dfcc08272c8ae3ff2e568c901de52f9a5ed1fc56`

Diff SHA-256:
`79d54f78de99c6b4242b43ef5a1809791bf9c686b836d29c4fe9de13fd991197`

Environment image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
(exploratory newer ORBIT-Q image, 6 CPUs / 7 GiB, network disabled).

The exact bond-2 CMZ MPO, bond-3 TFIM contraction, fused rotations, and
whole-training scan completed in 10.260367 seconds, but failed the functional
energy-gap gate:

```text
initial_energy_density: 0.9718867540
final_energy_density: -0.7872041464
exact_ground_energy_density: -1.2925285569
vqe_gap: 0.5053244105
Overall: FAIL
```

The first five reference and candidate histories agreed to low absolute error:

```text
reference: 0.9718865752, 0.8913648129, 0.7948441505, 0.6822248101, 0.5538985729
candidate: 0.9718867540, 0.8913646340, 0.7948451042, 0.6822254658, 0.5539006591
max parameter delta after five updates: 5.95897e-05
```

This establishes semantic equivalence at the start but numerical trajectory
divergence over 200 complex64 updates. The result is invalid and has no
performance standing.

Decision: `invalid`

### e01 numerical follow-ups

Commit `d9a5336` removed rotation fusion to reproduce the expert's explicit
`RX -> RZ -> RY` gate sequence. Its five-step maximum parameter delta grew to
`1.01790e-04`, so it was discarded as a regression.

Commit `3d48da0` restored fused rotations and changed only the MPS/CMZ/TFIM
intermediate dtype to complex128 while retaining float32 parameters and Adam
state. It completed `run_solution` in 10.790999 seconds but ended at
`-0.2665115864`, also invalid. Higher precision does not cure the optimization
basin sensitivity.

Failure interpretation: differentiating through exact QR/RQ canonicalization
is mathematically valid, but its gauge/rounding path perturbs this sensitive
200-step Adam trajectory. The next experiment will remove canonicalization
entirely: the known bond dimensions grow only `1 -> 2 -> 4`, so the raw
framework MPO-times-MPS tensors are already exact and bounded.
