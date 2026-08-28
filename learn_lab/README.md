# Learn lab

These files rebuild core ideas in small, readable forms. They intentionally do
not import production classes from `src/`. Work through them in this order:

1. `rmsnorm_from_scratch.py`
2. `attention_from_scratch.py`
3. `transformer_from_scratch.py`
4. `jepa_toy.py`
5. `flow_matching_toy.py`
6. `gaussian_dynamics_toy.py`

Each file exposes tensor shapes directly and relies on PyTorch only for tensor
operations, automatic differentiation, and standard linear layers.
