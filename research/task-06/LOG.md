# Task 06 Autoresearch Campaign

Task: `task-06`

Campaign branch: `codex/orbitbreakers/task-06/extreme-native`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign scope

Optimize only the immutable Task 06 human-expert TensorCircuit-NG solution.
Preserve the four true continuous-time ODE blocks, all parameters, all 100 Adam
updates, and the complete output contract. Use the latest installed framework;
do not attribute host hardware differences to solution performance.

At `2026-07-29T01:26Z`, the upstream repository returned zero open pull
requests matching `Task 06`.

## Frozen provenance

- Parent: `5af98f27b9404c513df8eee0f4568b1512edee19`.
- Reference:
  `0e7fec8d11135241eb3f3501f3651f3f337e08c636407b3da8a2858c2b3d85d1`.
- Evaluator:
  `0d2dfc7f30087896fb599925f9110190a3a61358263688dbb09cc36115a23998`.
- Image:
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`.
- TensorCircuit-NG `1.8.0.dev20260726`; JAX/JAXLIB `0.10.0`; Diffrax
  `0.7.2`; Optax `0.2.8`.
- Resource profile: six CPUs, 7 GiB, network disabled, 300-second cell cap.
- Pair order: odd `reference -> candidate`, even
  `candidate -> reference`.
- Private or hidden evaluation used: no.

## Append-only campaign events

- `2026-07-29T01:27:26Z`: created the dedicated branch from current
  `origin/main`. Reference and editable source were byte-identical.
- `2026-07-29T01:28Z`: latest-image one-update compatibility run completed in
  `7.889008 s`. The ODE API is functional; the one-update evaluator
  intentionally fails convergence thresholds and is diagnostic only.
- `2026-07-29T01:29:10Z` to `2026-07-29T01:34:01Z`: completed six immutable
  canonical reference runs in one no-network container with a fresh evaluator
  process per run. All six passed. Runtimes were `43.976600`, `45.097434`,
  `44.415753`, `44.864424`, `45.481100`, and `46.387673` seconds. Mean
  `45.037164 s`, median `44.980929 s`, standard error `0.344740 s`.
  Report: `profiles/reference-baseline-six.json`;
  SHA-256
  `bdb7b985d6a0fa6bd6e7044816f9764e90b285fa9be0e6f07b79aa2587e9ff35`.
- `2026-07-29T01:35Z`: the first runner version omitted the gate checker's
  `repeat` alias and exact `docker exec` command from serialized rows. Both
  metadata fields were reconstructed deterministically from the unmodified
  checkpoint; no measurement, output, status, order, or hash changed. The
  runner now records both fields directly.
- `2026-07-29T01:36:02Z`: froze the public canonical Task 06 workload and
  source-backed survey. Candidate edits are allowed only after
  `research/check_gates.py` reports `research_ready: true`.

## Baseline interpretation

The canonical expert passes with initial energy density `-0.5182266235`, final
history energy density `-1.5754342079`, and independently evaluated sparse
ground energy density about `-1.6025561094`. Its learned analog parameters are
well inside all bounds. The small `1.9%` max/min runtime spread makes the
baseline suitable for paired screening.

## Append-only corrections

Append later corrections here; do not rewrite any result after it informs a
candidate.

