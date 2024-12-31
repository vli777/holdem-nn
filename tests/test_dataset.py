import pytest
import torch
from torch.utils.data import Subset
from pathlib import Path
from config import config  
from PokerDataset import PokerDataset
import random  

# Convert DATA_PATH to string to avoid AttributeError
DATA_PATH = str(config.data_path)


@pytest.fixture(scope="module")
def dataset():
    if not Path(DATA_PATH).exists():
        pytest.skip("Data file missing, skipping dataset fixture.")
    ds = PokerDataset(DATA_PATH)
    subset_size = 10000
    if len(ds) > subset_size:
        indices = list(range(len(ds)))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        ds = Subset(ds, subset_indices)
    return ds


def test_dataset_item_structure():
    dataset = PokerDataset(DATA_PATH)
    state, action_label, position, player_id, recent_action = dataset[0]
    
    # Test state
    assert isinstance(state, torch.Tensor), "State should be a torch.Tensor"
    assert state.shape == (config.input_dim,), f"State tensor shape mismatch: expected {config.input_dim}, got {state.shape}"
    
    # Test action_label
    assert isinstance(action_label, torch.Tensor), "Action label should be a torch.Tensor"
    assert action_label.dim() == 0, "Action label tensor should be scalar"
    assert action_label.item() in range(config.output_dim), f"Invalid action label: {action_label.item()}"
    
    # Test position
    assert isinstance(position, torch.Tensor), "Position should be a torch.Tensor"
    assert position.dim() == 0, "Position tensor should be scalar"
    
    # Test player_id
    assert isinstance(player_id, torch.Tensor), "Player ID should be a torch.Tensor"
    assert player_id.dim() == 0, "Player ID tensor should be scalar"
    
    # Test recent_action
    assert isinstance(recent_action, torch.Tensor), "Recent action should be a torch.Tensor"
    assert recent_action.dim() == 0, "Recent action tensor should be scalar"


def test_data_validity(dataset):
    assert len(dataset) > 0, "Dataset should not be empty"


def test_invalid_actions(dataset):
    invalid_actions = [
        action_label.item()
        for _, action_label, _, _, _ in dataset
        if action_label.item() not in range(config.output_dim)
    ]
    assert not invalid_actions, f"Invalid actions found: {invalid_actions}"


def test_invalid_player_ids(dataset):
    num_players = 6  # Assuming player IDs range from 0 to 5
    invalid_player_ids = [
        player_id.item()
        for _, _, player_id, _, _ in dataset
        if player_id.item() < 0 or player_id.item() >= num_players
    ]
    assert not invalid_player_ids, f"Invalid player IDs found: {invalid_player_ids}"


def test_invalid_positions(dataset):
    max_positions = 6  # Assuming positions range from 0 to 5
    invalid_positions = [
        position.item()
        for _, _, position, _, _ in dataset
        if position.item() < 0 or position.item() >= max_positions
    ]
    assert not invalid_positions, f"Invalid positions found: {invalid_positions}"


def test_invalid_state_dimensions(dataset):
    expected_state_dim = config.input_dim  # Should be 4 as per config
    invalid_states = [
        state
        for state, _, _, _, _ in dataset
        if len(state) != expected_state_dim
    ]
    if invalid_states:
        for idx, state in enumerate(invalid_states[:10]):  # Limit to first 10 for brevity
            print(f"Invalid state at index {idx}: {state}, Length: {len(state)}")
    assert not invalid_states, f"Invalid state dimensions found: {len(invalid_states)} entries"


def test_invalid_recent_actions(dataset):
    valid_recent_actions = [0, 1, 2]  # Define valid recent actions
    invalid_recent_actions = [
        recent_action.item()
        for _, _, _, _, recent_action in dataset
        if recent_action.item() not in valid_recent_actions
    ]
    assert not invalid_recent_actions, f"Invalid recent actions found: {invalid_recent_actions}"


def test_invalid_hand_strength_and_pot_odds(dataset):
    invalid_hand_strengths_or_pot_odds = [
        (state[-2].item(), state[-1].item())
        for state, _, _, _, _ in dataset
        if not (0 <= state[-2].item() <= 1) or not (0 <= state[-1].item() <= 1)
    ]
    assert not invalid_hand_strengths_or_pot_odds, f"Invalid hand strengths or pot odds found: {invalid_hand_strengths_or_pot_odds}"
