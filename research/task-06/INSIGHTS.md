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

Source inspection identifies the strongest likely bottleneck: every Diffrax
stage evaluates 40 termwise full-state Pauli transforms through two
`PauliStringSum2MVP` closures, and reverse mode differentiates through this
work. A measured component profile is still required before calling its
runtime share confirmed.

## What worked

None yet.

## What did not work

The old bootstrap's raw-mode ODE API failure is obsolete in the current image
and should not be treated as a present blocker.

## Open hypotheses

1. TensorCircuit-native sparse analog and target Hamiltonian actions.
2. Exact fusion of each `RZ -> RY -> RZ` triple into one differentiable gate.
3. Whole-training `K.jaxy_scan`.
4. TensorCircuit Diffrax solver sweep under unchanged tolerances.
5. Parameter-tree simplification if profiling supports it.

## Evidence limits

The baseline covers one fixed public workload, one image, and one host
resource profile. It establishes neither cross-hardware performance nor global
SOTA. No candidate result exists yet.

