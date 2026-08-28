# Stage 0 gate

Stage 0 validates implementation correctness on CPU or the 4 GB RTX 3050. It
does not establish robotics performance.

## Exit criteria

- [x] Specification-aligned repository and packaging structure exists.
- [x] Typed bus messages validate shapes, serialize, reject stale packets, and
  retain correlation IDs.
- [x] Reusable RMSNorm, RoPE, GQA, SwiGLU, masking, and Transformer blocks exist.
- [x] Tiny implementations of all eight neural modules exist.
- [x] A synthetic full-agent loop executes candidate generation, prediction,
  deterministic scoring, safety filtering, short-prefix execution, and memory
  update.
- [x] Tests cover finite forward/backward passes, shapes, determinism,
  serialization, checkpoint round trips, and mixed-precision fallback.
- [x] Each tiny module passes its recorded tiny-set overfit experiment.
- [x] Stage 0 gate report is generated from a clean test run.

## Verified baseline

On 28 August 2026, `scripts/run_stage0_gate.py` passed all 23 tests under Python
3.12.3 and CPU PyTorch 2.13.0. The machine-readable local record is generated at
`artifacts/reports/stage0_gate.json` and is intentionally excluded from Git as a
run artifact.

## Explicit boundaries

The mock robot, token vocabulary, object labels, and synthetic supervision are
test fixtures. Isaac Sim assets, public dataset converters, real hardware I/O,
full parameter budgets, and production training are later stages.
