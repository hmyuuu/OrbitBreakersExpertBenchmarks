# Task 01 Autoresearch Campaign

Task: `task-01`

Campaign branch: `codex/orbitbreakers/task-01/extreme-native`

Reference parent: `0819ed34dd3f1eaa8f77587c2a40d95a420ea829`

## Objective

Optimize the immutable Task 01 human expert to the limit while preserving the
full DMRG-MPS input, 570-parameter circuit, 500 Adam updates, history semantics,
TensorCircuit-NG framework fidelity, and exact original docstring. The older
MPO candidate on `main` is a historical factor, not the campaign parent.

## Initial gate status

At `2026-07-29T14:59:00Z`,
`python3 research/check_gates.py --task 01 --json` reported:

- Task 01 `SURVEY.md`: missing;
- public manifest selection/coverage for Task 01: missing;
- current repeated immutable-reference report: missing;
- `research_ready`: false.

No candidate source was changed while these gates were closed.

## Public overlap check

Open upstream PRs inspected at `2026-07-29T14:59:00Z`: #3, #4, #5, #6, and
#7. No active PR targets Task 01 expert-runtime optimization.

## Append-only campaign events

- `2026-07-29T14:59:00Z`: created the Task 01 campaign branch from current
  `origin/main`; recorded immutable source hashes, environment versions, and
  the gate failure; began a six-run immutable-reference baseline.
- `2026-07-29T15:04:00Z`: froze the public survey and predeclared twelve
  single-factor hypotheses.
- `2026-07-29T15:04:30Z`: the canonical public workload and six-run
  immutable-reference validation completed. All six runs passed; runtimes
  were `54.896941`, `56.031343`, `52.621569`, `55.264405`, `58.445563`,
  and `56.618919` seconds (mean `55.6464567 s`, median `55.647874 s`).
  Report SHA-256:
  `61aca162f2fd80634f253464e885d83ca16bbb7ffefaa99ade2e83b23412e8c2`.
  The Task 01 public dataset is version
  `orbitq-workloads-v20260729.8`; the survey is now `READY`.

## Corrections

Append corrections here; never rewrite an experiment after it informs a later
choice.
