# Task 01 Optimization Insights

This file records hypotheses and measured conclusions separately from the final
comparison report.

## Starting observations

- The immutable expert is already tensor-network based and never materializes
  a dense `2^32` state, but it asks a generic circuit contractor to process the
  same shallow one-dimensional structure on every objective evaluation.
- The loss creates 63 Pauli-pattern contractions. A direct TFIM MPO has bond
  dimension three and is an exact algebraic replacement.
- Every physical bond is touched by only two two-qubit layers. Starting from
  DMRG bond dimension eight and allowing rank four per fused two-qubit gate
  gives a worst exact bond dimension of 128, with no truncation.
- `RXX`, `RYY`, and `RZZ` commute. Their product lies exactly in the four-term
  operator basis `I⊗I`, `X⊗X`, `Y⊗Y`, `Z⊗Z`; this avoids a dynamic SVD when
  applying the fused gate as a local MPO.
- The expert dispatches a compiled update 500 times from Python. A backend
  scan is semantically straightforward but may be secondary if each gradient
  contraction dominates.
- Initial parameters are near identity. Exploiting that fact to truncate small
  singular values would change the algorithm and is prohibited; exact ranks
  must be retained.

## Historical evidence, not accepted parent

`research/TASK_01_MPO_ENERGY_COMPARISON.md` reports a three-pair, different-image
experiment in which only the Hamiltonian evaluation changed and median runtime
fell from about 102.55 seconds to 67.02 seconds (1.53x). This motivates the MPO
factor but is not eligible promotion evidence and does not define the final
implementation.

## Pending measurements

- immutable-reference six-run baseline;
- component/XLA profile;
- exact operator and gradient equivalence;
- single-factor runtimes;
- leave-one-factor-out ablations;
- six alternating matched promotion pairs.
