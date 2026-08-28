# Stage 0 gate — v1.3

Stage 0 validates implementation correctness on CPU or the 4 GB RTX 3050. It
does not establish robotics performance.

## Exit criteria

- [x] Specification-aligned repository and packaging structure exists.
- [x] Typed bus messages validate shapes, serialize, reject stale/duplicate
  packets, and preserve correlation IDs.
- [x] Readable RMSNorm, RoPE, GQA, SwiGLU, masking and Transformer blocks exist.
- [x] Tiny implementations of all eight neural modules exist.
- [x] JEPA masking/EMA, camera-validity masks, flow sampling, body uncertainty,
  structured Executive output, text/audio semantics, memory and routing work at
  correctness scale.
- [x] Each of the eight modules passes a recorded tiny-set overfit experiment.
- [x] Checkpoint round trips, deterministic seeds and CPU precision fallback are
  tested.
- [x] A 512D-bus smoke test and complete synthetic candidate/predict/score/safe
  execute/memory loop pass.
- [x] The Stage 0 gate report is generated from a clean active-baseline test run.

## Explicit boundaries

The mock robot, byte vocabulary, object labels and synthetic supervision are
test fixtures. Stage 0 does not prove dataset quality, learned robotic skill,
real hardware readiness, or full 1.024B training feasibility.

Under v1.3, public dataset registry/adapters, bulk source validation, real
SO-101 I/O, specialist pretraining and production evaluation are later stages.
The historical custom-Isaac generator is excluded from this gate.
