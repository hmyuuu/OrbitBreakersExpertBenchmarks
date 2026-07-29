# Task 02 Research Insights

Task: `task-02`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Whole-training TensorCircuit `K.jaxy_scan`, source SHA-256
`6e44df512170e071655eee7697f7a0c084704dd34c266f441b2755cb1f29bc1a`,
is the accepted parent. Its six-pair mean speedup is `1.035580x`, 95%
Student-t interval `[1.010063x, 1.061098x]`, with `6/6` wins.

## Preserved semantics

- 12-qubit Neel input and complex64 TensorCircuit-NG evolution;
- six brickwork sublayers with all 243 independent parameters;
- three half-chain Renyi-2 checkpoints and the fixed target profile;
- the 45-term open XXZ plus staggered-field Hamiltonian;
- fixed seed 2026, Adam learning rate `0.015`, and 500 sequential updates;
- every pre-update history and the original four-key NumPy result contract.

## Confirmed bottlenecks

- One update lowers in `0.803158 s`, compiles in `1.929769 s`, and contains
  15,479 StableHLO lines. The 500 steady updates project to `1.321653 s`, so
  both cold graph staging and steady execution are material.
- The separately compiled trajectory is about 93.5% of the forward loss
  runtime (`0.000889/0.000951 s`), making exact gate-graph reduction the
  highest-priority factor.
- The host dispatches one compiled optimizer update 500 times. A whole-training
  scan can remove that boundary, although it cannot eliminate the steady
  quantum kernel itself.
- The termwise Hamiltonian MVP traces 45 independent paths, but its separately
  measured forward action is only `0.000176 s`. Sparse conversion is therefore
  a secondary compile-size hypothesis, not the primary steady bottleneck.
- One generic Renyi-2 call is `0.000132 s`; its Frobenius-purity rewrite is
  algebraically exact but expected to be a smaller factor than gate fusion.

Profile: [`profiles/reference-profile.json`](profiles/reference-profile.json).

## What worked

- Whole-training `K.jaxy_scan` preserved the four 12-step histories
  bit-for-bit and reduced canonical mean runtime from `4.724217` to
  `4.563051 s` in its six-pair session.

## What did not work

No candidate result yet.

## Open hypotheses

1. exact Frobenius-purity Renyi-2 evaluation on the scan parent;
2. TensorCircuit-native sparse XXZ action on the accepted parent;
3. final leave-one-factor-out checks for every retained mechanism.

## Evidence limits

Evidence covers only the canonical public Task 02 configuration, one latest
TensorCircuit-NG image, and one 6-CPU/7-GiB same-host profile. It establishes
a small paired improvement over the bundled expert, not global SOTA or
scaling.
