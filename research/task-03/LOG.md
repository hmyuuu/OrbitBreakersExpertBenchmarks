# Task 03 Autoresearch Campaign

Task: `task-03`

Campaign branch: `codex/orbitbreakers/task-03/extreme-native`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign scope and frozen provenance

Optimize only the immutable Task 03 human-expert TensorCircuit-NG solution.
At `2026-07-29T12:28Z`, the benchmark repository had no open Task 03
optimization PR.

- Parent: `d13e2591574cc1480507b00bcb33b0c6a48e6b99`.
- Reference:
  `a451480b08610e24072098171ac57364efc764ddc5e77e3d95028f009b6d5c89`.
- Evaluator:
  `b0a173f181857735b94fea3a9d03e11b595d4584b2a4bc33da80973e4b5c6d36`.
- Image:
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`.
- TensorCircuit-NG `1.8.0.dev20260726`; JAX/JAXLIB `0.10.0`; Optax `0.2.8`.
- Resource profile: 6 CPUs, 7 GiB, network disabled, 300-second cell cap.
- Pair order: odd `reference -> candidate`, even
  `candidate -> reference`.
- Private or hidden evaluation used: no.

## Append-only campaign events

- `2026-07-29T12:27Z`: created the dedicated branch from current
  `origin/main`. Reference and editable Task 03 sources were byte-identical.
- `2026-07-29T12:30Z`: reused the already-reviewed per-invocation resource
  override and factor-ablation workflow so the 6-CPU Docker backend does not
  require a tracked `bench.toml` mutation.
- `2026-07-29T12:34Z` to `2026-07-29T12:35Z`: six immutable canonical
  reference runs all passed in one no-network container. Runtimes were
  `4.262369`, `4.044382`, `4.064228`, `4.297515`, `4.053138`, and `4.200180`
  seconds; mean `4.153635 s`, median `4.132204 s`, standard error
  `0.046447 s`. Report:
  `profiles/reference-baseline-six.json`; original report SHA-256
  `da63de7d86bd158adfd720b1cd2c3ed653fac05f97ea75c505297df65eb8ed22`.
- `2026-07-29T12:38Z`: froze the public canonical Task 03 record and cited
  survey. The first candidate hypothesis is the exact product-state
  conditional-map reduction described in `SURVEY.md`.

## Baseline interpretation

The reference passes with initial/final energy density
`-0.44931078 / -1.02593327`, final success probability `1.55883934e-2`,
and final mean log event probability `-0.0693538114`. The runtime spread is
small enough for paired screening.

## Append-only corrections

Append later corrections here; do not rewrite an earlier event after it informs
another experiment.
