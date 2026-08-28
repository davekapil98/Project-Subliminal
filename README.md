# Project Subliminal

Project Subliminal is a modular, fully local robot brain for the LeRobot SO-101
arm. The authoritative architecture and development plan is
[`Modular_Robot_Brain_Functional_Master_Spec_v1.2.docx`](./Modular_Robot_Brain_Functional_Master_Spec_v1.2.docx).

The repository is currently at **Stage 0**: small, readable models and synthetic
integration tests that validate tensor contracts, losses, checkpoints, and the
closed-loop control design before expensive simulation or training begins.

## Stage 0 quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python scripts/run_stage0_demo.py
```

The demo runs this local synthetic path:

```text
text -> Language -> TASK_GOAL -> Executive -> MOTOR_GOAL
     -> Motor Cortex candidates -> Body/World predictions
     -> deterministic MPC score -> safety clamp -> mock robot
     -> EXECUTION_RESULT -> memory
```

It does not control physical hardware and it does not claim task performance.
Its purpose is code correctness and architectural learning.

## Repository rules

- Raw datasets under `data/raw/` are immutable and append-only.
- Derived data is regenerated through versioned code and manifests.
- Physical values remain typed numeric tensors; they are never encoded as prose.
- Learned output is always wrapped by deterministic safety checks.
- Model architecture, trainers, data conversion, control, and evaluation stay
  separate.
- Every important neural concept receives a small readable implementation and
  test before an optimized production implementation is introduced.

See [`docs/stage0.md`](docs/stage0.md) for the current gate and
[`docs/tensor_shapes.md`](docs/tensor_shapes.md) for interface shapes.
