# Task 06 Research Insights

Task: `task-06`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

No optimized candidate has been accepted yet. The immutable expert's six-run
mean is `45.037164 s` under the frozen six-CPU Docker profile.

## Preserved semantics

- 14-qubit Neel state and four hybrid blocks.
- Four true adaptive ODE evolutions per loss, with the original tolerances and
  maximum-step bound.
- All analog time/coupling/detuning variables and all 168 independent digital
  Euler angles.
- Seed 2026, complex64 TensorCircuit/JAX computation, target Hamiltonian,
  exactly 100 Adam updates, and every pre-update energy.
- Original NumPy output keys, shapes, and physical meanings.

## Confirmed bottlenecks

Both compilation and differentiated ODE execution are material. Immutable
profiling measured `2.851 s` lowering, `3.195 s` compilation, and
`0.2805 s` per early optimizer update, projecting `28.05 s` for 100 compiled
executions against the `45.04 s` evaluator mean.

The expected sparse-Hamiltonian shortcut is not viable on this CPU/JAX stack.
TensorCircuit BCOO multiplication was 3.4-3.5x slower than the installed
termwise MVP in isolated analog and target actions. XLA already fuses the
short reshape/slice/broadcast Pauli sums effectively.

Exact digital Euler fusion primarily affects compilation: the isolated
energy-gradient first call improved by about `0.34 s`, while steady execution
changed by only `0.4%`.

## What worked

None yet.

## What did not work

The old bootstrap's raw-mode ODE API failure is obsolete in the current image
and should not be treated as a present blocker.

Do not replace the current Hamiltonian actions with TensorCircuit/JAX BCOO
unchanged. Numerical equivalence passed, but isolated steady runtime regressed
by more than 3x.

## Open hypotheses

1. End-to-end exact fusion of each `RZ -> RY -> RZ` triple into one
   differentiable TensorCircuit gate.
2. Diffrax automatic initial-step selection with `dt0=None`.
3. TensorCircuit's `jaxode` backend under unchanged tolerances and step bound.
4. Whole-training `K.jaxy_scan`.
5. Diffrax solver sweep under unchanged tolerances.
6. Parameter-tree simplification only if later profiling supports it.

## Evidence limits

The baseline covers one fixed public workload, one image, and one host
resource profile. It establishes neither cross-hardware performance nor global
SOTA. No candidate result exists yet.
