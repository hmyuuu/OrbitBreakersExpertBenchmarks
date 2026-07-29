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
