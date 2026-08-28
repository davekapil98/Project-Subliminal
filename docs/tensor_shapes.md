# Tensor shapes

Symbols: `B` batch, `V` camera views, `N` world tokens, `K` candidates, `H`
action horizon, `J=6` joints, `D` bus width, and `T` sequence length.

| Interface | Shape | Units / meaning |
|---|---:|---|
| RGB camera input | `[B,V,3,height,width]` | normalized image values |
| Proprioception | `[B,18]` | q, qdot, previous command |
| WORLD_STATE tokens | `[B,N,D]` | JEPA latent bus tokens |
| q_goal / qdot_goal | `[B,6]` | radians / radians per second |
| Action candidates | `[B,K,H,6]` | relative joint-position commands, radians |
| Body mean/log variance | `[B,K,H,12]` | future q and qdot |
| World event logits | `[B,K,3]` | contact, grasp, collision |
| Memory entries | `[B,M,D]` | external memory embeddings |

Production adapters use `D=512`. Unit tests may explicitly use a smaller bus
width to keep Stage 0 CPU checks fast; all model constructors accept `bus_dim`.
