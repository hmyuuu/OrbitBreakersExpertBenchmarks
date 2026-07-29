# Human-expert optimization workflow

This workflow turns one ORBIT-Q human-expert optimization campaign into a
repeatable sequence with explicit gates, single-factor attribution, matched
timing, and PR-ready evidence. `GOAL.md`, `program.md`, `AGENTS.md`, and
[`README.md`](../README.md) remain authoritative.

## 1. Select and bind one task

1. Fetch the latest accepted branch.
2. Inspect the live open pull requests on `sxzgroup/ORBIT-Q`.
3. Create a dedicated campaign branch and worktree from the latest accepted
   commit.
4. Confirm `research/task-XX/SURVEY.md` is `READY`.
5. Confirm the public manifest contains a validated case for Task `XX`, then
   bind `selected_task_id` to `XX` before any candidate edit.
6. Record the inspection time, PR conclusion, base commit, workload version,
   image ID, TensorCircuit/JAX versions, CPU/memory limits, and timeout in the
   append-only task ledger.

Run the fail-closed gate:

```bash
python3 research/check_gates.py --task XX --json
```

Do not edit a candidate until `research_ready` is true. Promotion additionally
requires a current repeated reference report.

## 2. Freeze the same-machine reference

Use the image already selected by the repository. Resource limits are
per-invocation overrides, so a host with fewer CPUs does not require a tracked
`bench.toml` edit:

```bash
./bench run XX \
  --solution reference \
  --repeat 6 \
  --engine docker \
  --cpus 6 \
  --memory 7g \
  --timeout 300 \
  --no-build \
  --output results/task-XX-reference-YYYYMMDD
```

The repository promotion protocol uses six runs, which also satisfies a
five-run request. Preserve all runs, including failures and timeouts. Record
the report hash and use only same-session or same-environment relative timing;
absolute runtime across different hosts is not a claim.

## 3. Profile before choosing factors

Profile end-to-end execution first, then split compilation, steady-state
execution, and the largest scientific components. Rank hypotheses by expected
impact, correctness risk, and framework fidelity. Prefer TensorCircuit-native
operations for the central quantum computation.

Each hypothesis gets:

- one mechanism;
- one branch and worktree;
- one committed candidate before timed evaluation;
- an equivalence harness appropriate to its numerical risk;
- one immutable evidence record and one `keep`/`discard` decision.

## 4. Validate semantics before full timing

For an exact execution rewrite, compare at least:

1. the relevant state or observable on deterministic reduced inputs;
2. initial loss or energy;
3. gradients;
4. one optimizer update;
5. a short history and final parameters when the contract permits it;
6. the full evaluator and static/framework policy.

Use tolerances tighter than the evaluator threshold. A reduced smoke case is
diagnostic only; it never replaces the canonical workload.

## 5. Attribute performance with factor ablation

Maintain a factor table whose rows are:

- immutable expert;
- accepted parent;
- parent plus exactly one factor;
- final combination;
- final combination minus each retained factor.

Time every important retained factor with matched alternating pairs. Prefer six
pairs; never claim a factor from an unpaired one-off profile. If an old
eligible report already isolates the same factor under the same environment,
reuse it and cite its hash. Otherwise rerun.

For each measured factor, plot:

- paired runtime or paired speedup for every pair;
- the factor mean;
- a 95% Student-t interval when at least two pairs exist;
- pass/fail counts and exact report hash in the caption or adjacent table.

Rejected factors remain in the report so the final speedup is not incorrectly
attributed to every attempted change.

## 6. Promote and remeasure

After choosing the fastest valid candidate, run the canonical comparison:

```bash
./bench run XX \
  --solution optimized \
  --compare-to reference \
  --repeat 6 \
  --engine docker \
  --cpus 6 \
  --memory 7g \
  --timeout 300 \
  --no-build \
  --output results/task-XX-final-YYYYMMDD
```

Odd pairs run reference then candidate; even pairs reverse the order. Report
the six-pair result as authoritative and optionally show the first-five mean
for compatibility with a five-run request. Never remove an unfavorable pair.

## 7. Package the PR

The PR should contain:

- the optimized human-expert source;
- semantic-equivalence checks and profiling helpers;
- sanitized immutable result JSON;
- one machine-readable ablation summary;
- one chart per measured factor;
- `IMPLEMENTATION_COMPARISON.md` with mechanism, results, limitations, and
  exact reproduction commands;
- append-only `LOG.md` evidence and consolidated `INSIGHTS.md`.

Before push:

```bash
python3 -m pytest -q
./bench verify
python3 research/check_gates.py \
  --task XX \
  --baseline-report results/task-XX-reference-YYYYMMDD/results.json \
  --json
git diff --check
```

Open the PR against `hmyuuu/OrbitBreakersExpertBenchmarks:main`. The title and
body should state the task, central mechanism, six-pair same-machine result,
confidence interval, pass count, and links to the report and ablation figures.
