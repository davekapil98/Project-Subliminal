# v1.3 architecture baseline

The system uses eight independently trainable neural modules connected through
typed adapters to a 512-dimensional bus. Stage 0 constructors permit smaller
dimensions for CPU correctness tests, and the gate also exercises a 512D bus.

| Module | Full v1.3 budget | Stage 0 responsibility |
|---|---:|---|
| Multimodal JEPA Encoder | 280M | Encode camera and proprioceptive inputs into world tokens; exercise masks and EMA-target prediction. |
| JEPA World Predictor | 160M | Predict action-conditioned future latents, events and uncertainty. |
| Motor Cortex | 128M | Produce flow-matched SO-101 action chunks from body state and physical motor goals only. |
| Body Dynamics | 32M | Predict future SO-101 q/qdot distributions and uncertainty. |
| Executive Brain | 330M | Convert grounded task/world/prediction context into structured physical subgoals. |
| Language + Speech | 80M | Convert local text or audio features into structured task semantics. |
| Memory Controller | 12M | Control reads, writes, retrieval and compression around external stores. |
| Orchestrator | 2M | Route validated messages without taking over planning. |

The mocked execution path preserves the final control ordering:

```text
TASK_GOAL -> WORLD_STATE -> MOTOR_GOAL -> K ACTION_CANDIDATES
          -> BODY_PREDICTION + WORLD_PREDICTION
          -> deterministic score -> hard safety -> short prefix
          -> EXECUTION_RESULT -> memory -> replan
```

`control/` owns deterministic inverse-kinematics placeholders, MPC scoring,
safety limits and the mock servo plant. Neural predictions remain advisory.
Full parameter counts are targets for later specialist training, not claims
about the Stage 0 tiny models.
