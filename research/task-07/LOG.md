# Task 07 Autoresearch Campaign

Destination: `research/task-07/LOG.md`

Task: `task-07`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and setup

Selected task: `task-07` (16-qubit measurement-feedback VQE).

Base commit: `5af98f27b9404c513df8eee0f4568b1512edee19`.

Branch: `codex/orbitbreakers/task-07/extreme-native`.

Worktree:
`/Users/qqy/.codex/visualizations/2026/07/28/019fa982-7244-7e20-99f5-f609bdd0cf27/task07-extreme`.

The branch and worktree were created before the Task 07 survey and candidate
files. No candidate source was edited before the survey and public workload
gates were completed.

Open pull requests in `hmyuuu/OrbitBreakersExpertBenchmarks` were inspected
before selection. The open optimization PRs covered Tasks 08, 09, and 10;
none covered Task 07.

## Immutable expert bootstrap: canonical

Date: 2026-07-29

Reference SHA-256:
`ac483319363f3c386a7646eaa867670ae3d3cd687f8517e6d4201e69240ff0a3`.

Evaluator SHA-256:
`69717d98a90a7e53c31686128b3ef3e7cea3c96685ec538662a12163fe324b31`.

Docker image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
(TensorCircuit-NG `1.8.0.dev20260726`, JAX/JAXLIB `0.10.0`).

Allocation: six CPUs, 7 GiB memory, no network; timeout 300 seconds.

```text
workload: canonical max_steps=100
terminal_status: SUCCESS
valid: true
runtime_sec: 135.815605
initial_energy: -6.8462643623
final_history_energy: -10.0279636383
improvement: 3.1816992760
final_trajectory_mean: -10.0333871841
final_trajectory_std: 0.0000000000
history_length: 100
```

Decision: `bootstrap baseline`. A performance claim still requires six
matched reference/candidate pairs in one container.

## Immutable expert update-count profile

Date: 2026-07-29

All runs used the same source, image, six-CPU/7-GiB limits, seed, layers,
trajectory batch, output schema, thresholds, and evaluator. Only the
evaluator-supported `--max-steps` argument changed.

| Updates | Runtime (s) | Final history energy | Overall |
| ---: | ---: | ---: | --- |
| 1 | 47.853128 | -6.8462719917 | FAIL (expected thresholds) |
| 10 | 51.638506 | -7.4798407555 | FAIL (target) |
| 20 | 60.074862 | -7.8647251129 | FAIL (target) |
| 32 | 69.596274 | -8.1460399628 | FAIL (target) |
| 50 | 91.540316 | -8.7927856445 | PASS |
| 100 | 135.815605 | -10.0279636383 | PASS |

The one-step run establishes about 48 seconds of fixed trace, compilation,
contraction-path, and final-evaluation work. The remaining 99 canonical
updates add approximately 0.89 seconds each. Decision: prioritize the
per-step energy/gradient contractions, retain 50 updates as a passing screen,
and make claims only on the canonical 100-step case.

## Frozen hypotheses

The complete pre-edit hypothesis definitions and falsification rules are in
`SURVEY.md`.

Primary experiment: `e01`, single native TensorCircuit state contraction plus
one native sparse eight-data-qubit Hamiltonian expectation per trajectory.

Secondary experiments, each isolated from the latest accepted commit:

- `e02`: exact measured-ancilla feedback `RZZ` to data `RZ` reduction;
- `e03`: whole-training `K.jaxy_scan`;
- `e04`: OMECo contractor-budget sweep;
- `e05`: native-state measurement-round reuse, only if lower-risk ideas leave
  substantial headroom.

## Experiment `e01`

Branch and fresh hypothesis worktree: pending.

### Hypothesis

Replacing 15 separate per-trajectory Pauli expectation contractions with one
TensorCircuit `state` contraction and one TensorCircuit-native sparse TFIM
operator expectation materially reduces trace/compile and repeated gradient
cost while preserving energies, gradients, one Adam update, all 100
pre-update history values, and the final trajectory vector within declared
complex64 tolerances.

### Pre-run frozen environment

Public dataset version: `orbitq-workloads-v20260729.5`.

Private evaluation used: `no`.

Reference/evaluator/image hashes: as recorded above.

Pair order for final promotion: odd `reference -> candidate`, even
`candidate -> reference`; six pairs; 300-second cap.

### Result

Pending. Append results; never overwrite prior failure evidence.

## Append-only corrections

Append corrections below this heading. Never rewrite a result after it has
informed another experiment.
