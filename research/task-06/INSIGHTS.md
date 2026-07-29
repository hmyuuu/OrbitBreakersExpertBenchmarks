# Task 06 Research Insights

Task: `task-06`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment e01 is the provisional current best. Its canonical single screen
is `42.412637 s` versus the immutable expert's six-run mean `45.037164 s`.
The result is not yet eligible for a formal speedup claim.

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

Exact fusion of every digital `RZ -> RY -> RZ` sequence into one
phase-corrected TensorCircuit `U` gate passed state/energy/gradient audits and
reduced the canonical single screen by about 5.8%.

## What did not work

The old bootstrap's raw-mode ODE API failure is obsolete in the current image
and should not be treated as a present blocker.

Do not replace the current Hamiltonian actions with TensorCircuit/JAX BCOO
unchanged. Numerical equivalence passed, but isolated steady runtime regressed
by more than 3x.

## Open hypotheses

1. Diffrax automatic initial-step selection with `dt0=None`.
2. TensorCircuit's `jaxode` backend under unchanged tolerances and step bound.
3. Whole-training `K.jaxy_scan`.
4. Diffrax solver sweep under unchanged tolerances.
5. Parameter-tree simplification only if later profiling supports it.

## Evidence limits

The baseline covers one fixed public workload, one image, and one host
resource profile. It establishes neither cross-hardware performance nor global
SOTA. No candidate result exists yet.
