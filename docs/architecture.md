# Stage 0 architecture

The eight neural modules retain the separation required by the master spec:

1. `jepa_encoder` converts images and proprioception to compact world tokens.
2. `jepa_world` predicts action-conditioned future world tokens and events.
3. `motor_cortex` uses flow matching to generate joint-position chunks from
   body state and a physical motor goal only.
4. `body_dynamics` predicts future joint state and calibrated uncertainty.
5. `executive` converts task/world/prediction context into structured subgoals.
6. `language_speech` emits structured task semantics from local inputs.
7. `memory` controls reads and writes to external stores.
8. `orchestrator` validates and routes messages without planning tasks.

`control/` owns deterministic inverse-kinematics placeholders, MPC scoring,
safety limits, and the mock servo plant. Neural predictions are advisory.
