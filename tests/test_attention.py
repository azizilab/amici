import pytest
import torch
from einops import repeat

from amici._components import AttentionBlock


@pytest.fixture
def attention_params():
    return {
        "query_dim": 32,
        "kv_dim": 32,
        "head_size": 16,
        "num_heads": 4,
        "dummy_attn_score": 0.0,
        "dropout": 0.1,
        "add_res_connection": False,
    }


@pytest.fixture
def attention_block(attention_params):
    return AttentionBlock(**attention_params)


@pytest.fixture
def sample_inputs():
    batch_size = 8
    query_len = 5
    key_len = 10
    num_heads = 4
    query_dim = 32
    kv_dim = 32

    return {
        "query": torch.randn(batch_size, query_len, num_heads, query_dim),
        "key": torch.randn(batch_size, key_len, num_heads, kv_dim),
        "value": torch.randn(batch_size, key_len, num_heads, kv_dim),
        "batch_size": batch_size,
        "query_len": query_len,
        "key_len": key_len,
    }


def test_initialization(attention_block, attention_params):
    """Test that the attention block initializes correctly."""
    assert attention_block.query_dim == attention_params["query_dim"]
    assert attention_block.kv_dim == attention_params["kv_dim"]
    assert attention_block.head_size == attention_params["head_size"]
    assert attention_block.num_heads == attention_params["num_heads"]
    assert attention_block.embed_dim == attention_params["num_heads"] * attention_params["head_size"]

    # Check parameter shapes
    embed_dim = attention_params["num_heads"] * attention_params["head_size"]
    assert attention_block.W_Q.shape == (
        attention_params["head_size"],
        attention_params["num_heads"],
        attention_params["query_dim"],
    )
    assert attention_block.b_Q.shape == (
        attention_params["num_heads"],
        attention_params["head_size"],
    )
    assert attention_block.W_K.shape == (
        attention_params["head_size"],
        attention_params["num_heads"],
        attention_params["kv_dim"],
    )
    assert attention_block.b_K.shape == (
        attention_params["num_heads"],
        attention_params["head_size"],
    )
    assert attention_block.W_V.shape == (
        attention_params["head_size"],
        attention_params["num_heads"],
        attention_params["kv_dim"],
    )
    assert attention_block.b_V.shape == (
        attention_params["num_heads"],
        attention_params["head_size"],
    )
    assert attention_block.W_O.shape == (
        embed_dim,
        attention_params["num_heads"],
        attention_params["head_size"],
    )


def test_forward_output_shape(attention_block, sample_inputs):
    """Test that the forward pass produces output of the correct shape."""
    output = attention_block(sample_inputs["query"], sample_inputs["key"], sample_inputs["value"])
    assert isinstance(output, dict)
    assert "x" in output

    # Check output shape
    assert output["x"].shape == (
        sample_inputs["batch_size"],
        sample_inputs["query_len"],
        attention_block.num_heads * attention_block.head_size,
    )


def test_compute_qkv_matrices(attention_block, sample_inputs):
    """Test the computation of Q, K, V matrices."""
    q, k, v = attention_block.compute_qkv_matrices(
        sample_inputs["query"],
        sample_inputs["key"],
        sample_inputs["value"],
    )

    # Check shapes
    assert q.shape == (
        sample_inputs["batch_size"],
        sample_inputs["query_len"],
        attention_block.num_heads,
        attention_block.head_size,
    )
    assert k.shape == (
        sample_inputs["batch_size"],
        sample_inputs["key_len"],
        attention_block.num_heads,
        attention_block.head_size,
    )
    assert v.shape == (
        sample_inputs["batch_size"],
        sample_inputs["key_len"],
        attention_block.num_heads,
        attention_block.head_size,
    )


def test_compute_base_attention_scores(attention_block, sample_inputs):
    """Test the computation of base attention scores."""
    batch_size = sample_inputs["batch_size"]
    query_len = sample_inputs["query_len"]
    key_len = sample_inputs["key_len"]

    q = torch.randn(batch_size, query_len, attention_block.num_heads, attention_block.head_size)
    k = torch.randn(batch_size, key_len, attention_block.num_heads, attention_block.head_size)

    attn_scores = attention_block.compute_base_attention_scores(q, k)

    # Check shape (including dummy column)
    assert attn_scores.shape == (
        batch_size,
        attention_block.num_heads,
        query_len,
        key_len + 1,
    )


def test_attention_mask(attention_block, sample_inputs):
    """Test that attention masking works correctly."""
    # Create a mask that masks out the first key position
    mask = torch.ones(sample_inputs["batch_size"], sample_inputs["key_len"])
    mask[:, 0] = 0  # Mask out first position

    output = attention_block(
        sample_inputs["query"],
        sample_inputs["key"],
        sample_inputs["value"],
        attention_mask=mask,
        return_attn_patterns=True,
    )

    # Check that the attention pattern for the first position is zero
    attn_patterns = output["attn_patterns"]
    assert torch.allclose(attn_patterns[:, :, :, 0], torch.zeros_like(attn_patterns[:, :, :, 0]))


def test_positional_attention_score(attention_block, sample_inputs):
    """Test that positional attention scores are added correctly."""
    batch_size = sample_inputs["batch_size"]
    key_len = sample_inputs["key_len"]
    num_heads = attention_block.num_heads

    # Create positional attention scores
    pos_attn_score = torch.randn(batch_size, key_len, num_heads)

    output = attention_block(
        sample_inputs["query"],
        sample_inputs["key"],
        sample_inputs["value"],
        pos_attn_score=pos_attn_score,
        return_base_attn_scores=True,
    )

    # The difference between attn_scores and base_attn_scores should match the positional scores
    diff = output["attn_scores"] - output["base_attn_scores"]

    # Check only the non-dummy positions
    expanded_pos_scores = repeat(pos_attn_score, "b n h -> b h 1 n")
    assert torch.allclose(diff[:, :, :, :-1], expanded_pos_scores, rtol=1e-5, atol=1e-5)


def test_residual_connection(sample_inputs):
    """Test that residual connections work correctly."""
    # Create attention block with residual connection
    attn_block_with_res = AttentionBlock(query_dim=32, kv_dim=32, head_size=16, num_heads=4, add_res_connection=True)

    # Create attention block without residual connection
    attn_block_no_res = AttentionBlock(query_dim=32, kv_dim=32, head_size=16, num_heads=4, add_res_connection=False)

    # Use the same weights for fair comparison
    attn_block_with_res.W_Q.data = attn_block_no_res.W_Q.data.clone()
    attn_block_with_res.b_Q.data = attn_block_no_res.b_Q.data.clone()
    attn_block_with_res.W_K.data = attn_block_no_res.W_K.data.clone()
    attn_block_with_res.b_K.data = attn_block_no_res.b_K.data.clone()
    attn_block_with_res.W_V.data = attn_block_no_res.W_V.data.clone()
    attn_block_with_res.b_V.data = attn_block_no_res.b_V.data.clone()
    attn_block_with_res.W_O.data = attn_block_no_res.W_O.data.clone()

    # Get outputs
    output_with_res = attn_block_with_res(sample_inputs["query"], sample_inputs["key"], sample_inputs["value"])

    output_no_res = attn_block_no_res(sample_inputs["query"], sample_inputs["key"], sample_inputs["value"])

    # Outputs should be different
    assert not torch.allclose(output_with_res["x"], output_no_res["x"])
