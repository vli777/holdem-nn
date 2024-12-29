import pytest
import torch
from src.PokerLinformerModel import PokerLinformerModel

def test_poker_linformer_model_forward():
    batch_size = 8
    seq_len = 1
    input_dim = 106
    hidden_dim = 128
    output_dim = 3

    model = PokerLinformerModel(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=4,
        num_layers=2,
    )

    states = torch.randn(batch_size, input_dim)
    positions = torch.randint(0, 6, (batch_size,))
    player_ids = torch.randint(0, 6, (batch_size,))
    recent_actions = torch.randint(0, 3, (batch_size,))  # Ensure valid range

    print("Generated recent_actions:", recent_actions)

    try:
        policy_logits, value = model(states, positions, player_ids, recent_actions)
        print("Forward pass succeeded")
        assert policy_logits.shape == (batch_size, output_dim)
        assert value.shape == (batch_size, 1)
    except Exception as e:
        print(f"Error in forward pass: {e}")
        pytest.fail(f"PokerLinformerModel forward pass failed with error: {e}")
