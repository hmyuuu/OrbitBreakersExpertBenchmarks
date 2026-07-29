# Task 02 Autoresearch Campaign

Task: `task-02`

Campaign branch: `codex/orbitbreakers/task-02/extreme-native`

Campaign task: `task-02`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign objective

Reduce evaluator-reported runtime for the immutable Task 02 human expert while
preserving all 243 trainable parameters, three block-checkpoint Renyi-2
entropies, the 45-term XXZ energy, and exactly 500 sequential Adam updates.

## Selection and gates

- `2026-07-29T13:30Z`: inspected all open PRs on `sxzgroup/ORBIT-Q` and the
  benchmark repository. No active Task 02 optimization was present. This
  campaign and every hypothesis worktree are bound to Task 02.
- Base commit:
  `d13e2591574cc1480507b00bcb33b0c6a48e6b99`.
- Public data only; no hidden or private evaluation is used.
- Reference and initial optimized sources were byte-identical at selection,
  SHA-256
  `cd5776dfb223924edd83795bd8222032351008a584b55d747110521c709dcdd3`.

## Baseline `reference-six-20260729`

Hypothesis: the latest TensorCircuit-NG image and fixed 6-CPU/7-GiB envelope
produce a valid repeated reference suitable for paired promotion decisions.

Command:

`./bench run 02 --solution reference --repeat 6 --engine docker --cpus 6
--memory 7g --timeout 300 --no-build --output
results/task-02-reference-20260729`

Environment:

- image ID:
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`;
- TensorCircuit-NG `1.8.0.dev20260726`;
- JAX/JAXLIB `0.10.0` / `0.10.0`;
- Optax `0.2.8`;
- TensorNetwork `0.5.1`; Quimb `1.11.1`;
- host fingerprint:
  `c627504db97dc65b8d998afb4b9cdf73cfc3eff2ce2ac589b4a7a4aa0c7fdc48`;
- one shared container, six CPUs, 7 GiB, no network, 300-second per-cell cap;
- evaluator SHA-256:
  `0d9360d4812d033d364253dd478444f68918f1b05f04146eddcb846a3dca3a1e`;
- dependency-lock SHA-256:
  `cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`.

Result:

```text
terminal_status: SUCCESS
valid: true
timed_out: false
passing_runs: 6/6
runtimes_sec: 4.559075, 4.986945, 4.990731, 5.188701, 4.813289, 6.035628
mean_runtime_sec: 5.0957281667
median_runtime_sec: 4.988838
sample_stdev_sec: 0.5064942904
runtime_stderr_sec: 0.2067754282
min_runtime_sec: 4.559075
max_runtime_sec: 6.035628
```

Immutable report SHA-256:
`7d450e074cc5a4cabca72e9f6cd5787e2f5226c42c69cf6c3016ec53465b9912`.

Decision: `baseline`. The repeated-reference promotion gate passes. No
improvement is claimed without matched candidate pairs.

## Append-only campaign events

- `2026-07-29T13:31Z`: inspected installed TensorCircuit sources. The expert's
  Pauli MVP traces one path per Hamiltonian term; Renyi-2 forms `rho @ rho`;
  and the backend's `K.jaxy_scan` delegates to `jax.lax.scan`.
- `2026-07-29T13:31Z`: completed the cited survey and canonical public workload
  record. Candidate editing remains blocked until the manifest is rebound and
  the fail-closed gate passes.
- `2026-07-29T13:33Z`: survey, public dataset, and repeated-reference gates all
  pass. Added an immutable-reference profiler to separate lowering,
  compilation, trajectory, entropy, Hamiltonian action, and steady optimizer
  execution before selecting candidate factors.
- `2026-07-29T13:35Z`: immutable-reference profile completed. Report
  `research/task-02/profiles/reference-profile.json`, SHA-256
  `b7d190ea225ca5586d813759b89fea0cba05c045ca41e72e8f687a8769241d40`;
  profiler SHA-256
  `538a86104c32925586606394f50be069800c18b054bae45a6ebb786ac0e76a8b`.
  One update lowers in `0.803158 s`, compiles in `1.929769 s`, contains
  15,479 StableHLO lines, and executes in `0.002643 s` mean over 12 steady
  calls. The projected 500-update kernel time is `1.321653 s`. The forward
  trajectory accounts for about 93.5% of the separately measured loss
  execution (`0.000889/0.000951 s`); a Hamiltonian action is only
  `0.000176 s`, and one generic entropy is `0.000132 s`. Decision: prioritize
  graph reduction and host-loop removal, while treating sparse Hamiltonian and
  purity rewrites as secondary measured factors.

## Append-only corrections

Append corrections below this heading. Never rewrite an earlier result after
it has informed another experiment.

## Experiment `e07-batched-purity`

Branch: `codex/orbitbreakers/task-02/e07-batched-purity`

Parent commit: `edf1e91` (accepted whole-training scan).

Execution note: this hypothesis uses a fresh local shared Git checkout and
independent branch at the accepted commit after worktree permission review
timed out.

Hypothesis: the predeclared shared-kernel combination of checkpoint
`K.vmap` and exact Frobenius purity will batch the three reduced-density
matrices while removing all three dense `rho @ rho` products. The prior
single-factor reports remain the subfactor ablations; neither subfactor alone
passed the confidence rule.

Permitted data: public Task 02 artifacts and the two completed single-factor
reports only. No hidden/private evaluation.

Pre-evaluation correctness rule: initial loss/auxiliary values, gradient, and
all four 12-step histories must satisfy the frozen survey tolerances.

Candidate source SHA-256:
`181c0459c8194aa259be813e23896d3c70691adf5999a26353fe57cab14d3f3d`.

Candidate source-diff SHA-256:
`ad7f082c18273b9f8c67cab5aef8477983791011261bb604ea62f6768d3a3fcf`.

The 12-step audit passed: initial loss error `0`, auxiliary maximum error
`2.981e-7`, gradient maximum error `1.211e-8`, and maximum history error
`1.073e-6`. Audit report SHA-256:
`107c59ce1785ee9ada91e061d01970f1522ea73f3d8fe3efea37c03138d06b53`.

Six direct parent/factor pairs all passed. Parent/factor means were
`4.531128/4.218862 s`; factor wins `6/6`; mean paired speedup `1.074457x`,
95% Student-t interval `[1.044174x, 1.104740x]`. The runner roles were
intentionally reversed so its registered `optimized` source was the factor;
the tracked report normalizes every ratio as parent/factor. Raw report
SHA-256:
`6dd10f3c24054668734022000058391c192c87f0554d404a6d3b7d61593ee07e`.

Decision: `keep`. The shared-kernel combination passes the confidence rule
although each predeclared subfactor was inconclusive alone.

## Experiment `e02-training-scan`

Branch: `codex/orbitbreakers/task-02/e02-training-scan`

Parent commit: `77939b1`

Hypothesis: carrying parameters and Optax state through one TensorCircuit
`K.jaxy_scan`, compiled by `K.jit`, will preserve all 500 sequential updates
and pre-update histories while removing 500 Python-to-device dispatches.

Permitted data: public Task 02 artifacts and tracked reference profile only.
No hidden/private evaluation.

Candidate SHA-256:
`6e44df512170e071655eee7697f7a0c084704dd34c266f441b2755cb1f29bc1a`.

Candidate diff SHA-256:
`9b590f4cb193a39327b38165bcd255f3e051f5eb76d495b141194ebefbcbe1d8`.

Pre-evaluation correctness rule: all four 12-step pre-update histories must
match the immutable expert within `2e-4` and retain identical shapes.

The 12-step audit passed with bit-identical values and shapes for all four
returned histories. Audit report SHA-256:
`8864dd25031e00e1df188f705824738925e3679e4e46e06ae7078c5ccb6a1493`.

Six alternating canonical pairs all passed. Reference/candidate means were
`4.724217/4.563051 s`; candidate wins `6/6`; mean paired speedup
`1.035580x`, 95% Student-t interval `[1.010063x, 1.061098x]`. Raw report
SHA-256:
`549e267378bc1eca2ea2b839810af4c46780ac8bdc7800eacdfc15385bc1e33b`.

Decision: `keep`. Whole-training `K.jaxy_scan` is the first accepted parent
and removes about 3.41% of end-to-end runtime in this session.
