# Terminology

- **Body state**: SO-101 joint position and velocity in radians and radians per
  second, plus optional actuator telemetry.
- **Motor goal**: typed desired joint position/velocity, duration, and physical
  constraints. It contains no language or camera pixels.
- **Action chunk**: `H` relative joint-position commands for six joints.
- **World token**: a compact latent representation of scene and sensor context.
- **Candidate**: one possible action chunk sampled by the Motor Cortex.
- **Prediction residual**: difference between predicted and observed state.
- **Short-prefix execution**: applying only the first few commands before
  observing and replanning.
- **Stage 0**: correctness-scale models and synthetic data, not a performance
  baseline.
