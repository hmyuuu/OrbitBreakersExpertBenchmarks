# Task 04 Research Insights

Task: `task-04`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment `e02-probe-vmap`, source SHA-256
`6247d16ca78364925ada690b32b350eadf3ed4c87760922e7704885514f82ec1`.
Six paired Docker runs: reference `15.029802 ± 0.215846 s`, candidate
`6.962292 ± 0.203499 s`, paired speedup `2.165207 ± 0.050131x`, 95% CI
`[2.036340x, 2.294074x]`, 6/6 wins and 12/12 cells PASS.

## Preserved semantics

- Four fixed 12-qubit probe preparations and the exact even/odd `RXX(0.31)`
  brickwork order.
- User-defined asymmetric bit-flip channel represented by the specified three
  differentiable Kraus tensors and applied independently after each
  entangler.
- Target observables generated inside `run_solution` from true probabilities.
- All 12 single-Z values plus full-chain Z parity for every probe.
- Sigmoid probability parameterization, fixed initialization, exactly 120
  Adam updates at learning rate `0.04`, and pre-update loss history.
- Complex64 TensorCircuit-NG central computation and unchanged NumPy output
  schema.

## Confirmed bottlenecks

- `DMCircuit.apply_general_kraus` eagerly contracts the circuit before every
  channel, materializes three full-density Kraus branches, and sums them.
- A 12-qubit complex64 density matrix occupies 128 MiB before gradient
  temporaries; one observable table invokes 88 one-qubit channels.
- The exact canonical reference takes `14.671115 ± 0.123236 s` end to end.
- The tracked profile measures `2.629 s` target construction, `3.068 s`
  lowering, `10.712 s` XLA compilation, but only `1.257 ms` per compiled
  update (`0.151 s` projected for all 120). Compilation of the 21,720-line
  StableHLO graph, not Python optimizer dispatch, dominates this workload.

## What worked

Building the four exact probe states once with `tc.Circuit` and batching the
identical noisy density-matrix circuit through TensorCircuit `K.vmap` reduced
StableHLO from 21,720 to 4,855 lines, lowering from `3.068` to `0.616 s`, and
XLA compilation from `10.712` to `2.014 s`. This dominates the increase in
steady per-update execution from `1.257` to `8.035 ms`.

## What did not work

The first baseline command session was interrupted after two passing cells and
did not produce a complete report. Those timings are excluded rather than
spliced into the complete baseline.

Changing only `tc.DMCircuit` to `tc.DMCircuit2` is semantically valid but not
promotable: `1.0180x ± 0.0159x`, 95% CI `[0.9772x, 1.0588x]`, with only 4/6
pair wins. The newer class must be combined with an explicit scalar-TN
contraction/reuse strategy to change the compiled graph materially.

## Open hypotheses

1. TensorCircuit `K.jaxy_scan` over all 120 optimizer steps as a secondary
   factor; the profile limits standalone dispatch headroom.
2. Scalar `reuse=False` versus reusable full-density `reuse=True`
   expectations on the accepted batched parent.
3. Exact composite two-qubit noisy superoperators and reusable contraction
   paths.
4. Exact MPO/purification formulations through TensorCircuit-native classes if
   simpler delayed contraction is insufficient.

## Evidence limits

- Claims are limited to the canonical fixed Task 04 workload and current host.
- No external matched runtime establishes global state of the art.
- Source inspection identifies eager density materialization, but factor
  magnitude awaits single-factor paired benchmarks.
