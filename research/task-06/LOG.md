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

## Profiling event: immutable compiled update

Recorded at `2026-07-29T01:41Z`.

Profiler: `profile_reference.py`
(`sha256:ba1a9072b8b76b439429ef7cbfb9325c4f525c531dedcc9153813d17f4f183fd`).
Report: `profiles/reference-profile.json`
(`sha256:b96624021fde8f5a5c374c1e60d1e0aa5c24c1e15d624a520f35d7c0dc243a4e`).

Lowering and compilation took `2.851043 s` and `3.194721 s`. Eight early
post-compile optimizer updates averaged `0.280501 s` and projected
`28.050100 s` for 100 executions. XLA reported about `23.52 million` FLOPs,
`101.16 MB` bytes accessed, and `18.78 MB` temporary storage per update.
The canonical evaluator mean is `45.037164 s`, so both compilation and
steady differentiated ODE execution are material; Python-only cleanup cannot
produce a large gain.

## Profiling event: Hamiltonian action

Recorded at `2026-07-29T01:42Z`.

Profiler: `profile_hamiltonian_actions.py`
(`sha256:fcc5914a4bb1feb50bd3b1ca2bbc89c3b706a17ce701e8dbafba29c1058b0e46`).
Report: `profiles/hamiltonian-action-profile.json`
(`sha256:6d864f1d3ff26ec1625edacb39347ed8ba3575d1a081d4f67aedd9c314584472`).

The TensorCircuit COO operators agreed with the expert analog action to
`8.72e-9` maximum absolute error and with the target action to `1.91e-6`.
However, native BCOO multiplication was slower on this CPU/JAX stack:

```text
analog termwise MVP: 0.249930 ms
analog sparse BCOO:  0.851897 ms
termwise/sparse:     0.293381x

target termwise MVP: 0.344365 ms
target sparse BCOO:  1.209447 ms
termwise/sparse:     0.284729x
```

Decision: `discard before candidate integration`. The installed 1.8
`PauliStringSum2MVP` implementation is already reshape/slice/broadcast based
and XLA fuses these short local Pauli sums effectively. A full sparse rewrite
has no source-independent reason to reverse a 3.4-3.5x isolated steady
regression. Do not repeat BCOO unchanged.

## Profiling event: exact digital Euler fusion

Recorded at `2026-07-29T01:44Z`.

Profiler: `profile_digital_fusion.py`
(`sha256:681c4c55ba4550db5ca2dd0729ea7c4f0495252d375102aca48cd6c2369e1d39`).
Report: `profiles/digital-fusion-profile.json`
(`sha256:48185620664abf2da8857ed17d12a148b0ed7c8f685b5ba124c2a3b3df52c113`).

Replacing each `RZ -> RY -> RZ` triple by the exactly phased TensorCircuit
`U` gate produced maximum state, energy, and gradient errors
`8.94e-8`, `9.54e-7`, and `9.65e-7`. The isolated energy-gradient steady
speedup was only `1.0042x`, but compile-plus-first-execution fell from
`2.5819 s` to `2.2421 s`. Decision: `test end to end`; this is a compile-cost
hypothesis, not a steady-execution claim.

## Frozen follow-up ODE hypotheses

Source inspection after the initial survey exposed two additional
TensorCircuit-native controls that preserve a true adaptive ODE:

1. pass `dt0=None` so Diffrax chooses its initial step instead of forcing
   `0.01` for every smooth time-independent block;
2. compare TensorCircuit's `ode_backend="jaxode"` with the current Diffrax
   path, preserving `rtol`, `atol`, and `max_steps`.

Each is isolated after the digital-fusion screen. A candidate must pass a
canonical 100-update evaluator; lower-step runs are diagnostics only.
