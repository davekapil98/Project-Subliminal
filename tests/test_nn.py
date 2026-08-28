import torch

from nn import GroupedQueryAttention, RMSNorm, RotaryEmbedding, TransformerBlock


def test_rmsnorm_matches_equation_and_has_gradients() -> None:
    layer = RMSNorm(8)
    inputs = torch.randn(2, 3, 8, requires_grad=True)
    actual = layer(inputs)
    expected = inputs * torch.rsqrt(inputs.float().square().mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_rope_preserves_shape_and_vector_magnitude() -> None:
    rope = RotaryEmbedding(4)
    query = torch.randn(2, 3, 5, 4)
    key = torch.randn(2, 1, 5, 4)
    rotated_query, rotated_key = rope(query, key)
    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    torch.testing.assert_close(rotated_query.square().sum(-1), query.square().sum(-1))


def test_causal_attention_cannot_read_future_tokens() -> None:
    torch.manual_seed(2)
    attention = GroupedQueryAttention(16, num_heads=4, num_kv_heads=2, dropout=0.0)
    attention.eval()
    original = torch.randn(1, 4, 16)
    changed = original.clone()
    changed[:, 3] = 100.0
    original_output = attention(original, causal=True)
    changed_output = attention(changed, causal=True)
    torch.testing.assert_close(original_output[:, :3], changed_output[:, :3])


def test_transformer_forward_backward_is_finite() -> None:
    block = TransformerBlock(16, 4, num_kv_heads=2)
    inputs = torch.randn(2, 5, 16, requires_grad=True)
    output = block(inputs)
    assert output.shape == inputs.shape
    output.mean().backward()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in block.parameters()
    )
