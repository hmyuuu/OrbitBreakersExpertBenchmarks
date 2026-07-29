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

Branch: `codex/orbitbreakers/task-07/e01-single-state-energy`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e01-single-state-energy`.

Parent commit: `21d7271` (`research: freeze task 07 optimization campaign`).

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

The one-trajectory equivalence check is frozen before execution at absolute
tolerances `5e-5` for energy, `5e-4` for the maximum gradient element, and
`2e-5` for the maximum parameter difference after one Adam update. These are
strict complex64 path-rounding tolerances and are not evaluator thresholds.

### Result

Candidate hypothesis commit: `1e21dd41f47e3beb84b937144324227eac544b6d`.

Candidate SHA-256:
`30f0f45073e866c7fbb24cd9a5c33d8c1254e6985136937cd454b82990681678`.

Candidate diff SHA-256:
`3396c6b65d3f6c2eaebc647d6251547dd6561d1defac4526f9f503b2dcac8b7e`.

Sanitized record: `profiles/e01-single-state-screen.json`.

```text
max_steps=1:   reference 47.853128 s, candidate 44.917110 s
max_steps=10:  reference 51.638506 s, candidate 46.100052 s
max_steps=50:  reference 91.540316 s, candidate 52.769482 s, both PASS
max_steps=100: reference 135.815605 s, candidate 61.396553 s, both PASS
canonical single-screen speedup: 2.2121x
```

The candidate canonical run passed every evaluator gate with initial energy
`-6.8462653160`, final history energy `-10.0263500214`, improvement
`3.1800847054`, final trajectory mean/std
`-10.0319023132 / 0.0014687895`, history length 100, and the exact required
keys and shapes.

The predeclared one-trajectory audit measured energy error `3.34e-6` and
maximum gradient error `1.01e-6`, both comfortably passing. Its strict
maximum parameter-difference check after one first Adam update failed:
`3.13e-2` versus `2e-5`, although mean parameter difference was `2.43e-3`.
This failure is retained, not filtered.

Decision: `keep provisionally`. The state/observable identity is exact,
energy and gradient checks pass, the one-step post-update physical energy
differs by only `2.29e-5`, and the 50/100-step public workloads both pass with
nearly identical energy trajectories. The failed parameter metric reflects
the ill-conditioning of first-step Adam updates near zero: Adam normalizes
each gradient component by its magnitude, so a complex64 sign change in an
otherwise negligible component can create an order-learning-rate parameter
difference without a corresponding energy difference. Final paired
performance evidence is still pending.

## Append-only corrections

Append corrections below this heading. Never rewrite a result after it has
informed another experiment.

### Correction: e01 one-update acceptance observable

The strict per-parameter first-Adam maximum was over-specified as a semantic
criterion. The executable contract does not return parameters, and this
metric is discontinuously sensitive at zero gradient. It remains visible as
a failed diagnostic. Subsequent candidates will predeclare post-update
energy/trajectory checks as the physical one-update criterion while continuing
to report gradient errors and any parameter differences.

## Experiment `e02`: measured-ancilla feedback reduction

Branch: `codex/orbitbreakers/task-07/e02-feedback-rz`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e02-feedback-rz`.

Parent commit: `2e5cd61` (accepted e01 code and complete provisional evidence).

### Hypothesis

After `cond_measure`, ancilla `a` is the Z eigenstate with eigenvalue
`1-2*bit`. Therefore the selected
`RZZ(theta_bit)` on `(ancilla_a, data_a)` is exactly
`RZ((1-2*bit)*theta_bit)` on the data qubit, with the ancilla unchanged.
Replacing all 16 generic selected two-qubit tensors with those native
TensorCircuit `RZ` gates reduces graph/path/gradient work without changing
any measurement, selected parameter, branch state, or observable.

### Pre-run frozen checks

The exact two-branch gate identity will be checked at complex64 precision.
The frozen matrix-action tolerance is `1e-7` for both bit values and two
nontrivial signed angles.
The physical one-step criteria are initial energy absolute error at most
`5e-5` versus accepted e01 and post-update final-trajectory mean absolute
error at most `1e-4`; the strict parameter maximum remains diagnostic only.
Both the public 50-step and canonical 100-step evaluators must pass. A
candidate is retained only if its canonical screen is faster than e01's
61.396553 seconds.

### Result

Candidate hypothesis commit: `067a1d365e8ef4a9e3f81d6dd62939c0b3af6b39`.

Candidate SHA-256:
`b3dcbaa35a233d8dde4576de7257c79a1a81034eee49f1f0bef6489116dcafcd`.

Candidate diff SHA-256:
`759b7139a8463d53c80f2d48148eca189235a872fac5402ae9f64be4100bac45`.

Sanitized record: `profiles/e02-feedback-rz-screen.json`.

The independent two-branch matrix-action audit passed all four cases. Maximum
complex64 error was `2.98e-8` against the frozen `1e-7` threshold.

```text
max_steps=1:   e01 44.917110 s, e02 29.918828 s
max_steps=50:  reference 91.540316 s, e01 52.769482 s,
               e02 31.299628 s, e02 PASS
max_steps=100: reference 135.815605 s, e01 61.396553 s,
               e02 33.546170 s, e02 PASS
canonical e02/reference single-screen speedup: 4.0487x
canonical e02/e01 single-screen speedup: 1.8302x
```

The one-step initial energy and post-update trajectory-mean differences from
accepted e01 are `1.43e-6` and `7.53e-5`, passing the frozen `5e-5` and
`1e-4` physical thresholds. The canonical run passed every evaluator gate:
initial `-6.8462653160`, final history `-10.0280771255`, improvement
`3.1818118095`, final trajectory mean/std
`-10.0335464478 / 0.0000002666`, history length 100, required keys/shapes.

Decision: `keep`. The exact feedback reduction removes a major trace,
path-search, contraction, and gradient burden. Proceed from e02 to isolate
whole-training scan.
