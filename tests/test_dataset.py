import pytest
from src.PokerDataset import PokerDataset

DATA_PATH = "data/texas_holdem_data.npz"

def test_data_validity():
    try:
        dataset = PokerDataset(DATA_PATH)
        assert len(dataset) > 0, "Dataset should not be empty"
    except Exception as e:
        pytest.fail(f"Failed to load dataset: {e}")

def test_invalid_actions():
    dataset = PokerDataset(DATA_PATH)
    invalid_actions = [
        action_label for _, action_label, _, _, _ in dataset
        if action_label not in [0, 1, 2]
    ]
    assert not invalid_actions, f"Invalid actions found: {invalid_actions}"

def test_invalid_player_ids():
    dataset = PokerDataset(DATA_PATH)
    num_players = 6  # Replace with your actual value
    invalid_player_ids = [
        player_id for _, _, _, player_id, _ in dataset
        if player_id < 0 or player_id >= num_players
    ]
    assert not invalid_player_ids, f"Invalid player IDs found: {invalid_player_ids}"

def test_invalid_positions():
    dataset = PokerDataset(DATA_PATH)
    max_positions = 10  # Replace with your actual value
    invalid_positions = [
        position for _, _, position, _, _ in dataset
        if position < 0 or position >= max_positions
    ]
    assert not invalid_positions, f"Invalid positions found: {invalid_positions}"

def test_invalid_state_dimensions():
    dataset = PokerDataset(DATA_PATH)
    expected_state_dim = 54  # Replace with the correct dimension for your state vector
    invalid_states = [
        state for state, _, _, _, _ in dataset
        if len(state) != expected_state_dim
    ]
    assert not invalid_states, f"Invalid state dimensions found: {invalid_states}"

def test_invalid_recent_actions():
    dataset = PokerDataset(DATA_PATH)
    invalid_recent_actions = [
        recent_action for _, _, _, _, recent_action in dataset
        if recent_action not in [0, 1, 2]
    ]
    assert not invalid_recent_actions, f"Invalid recent actions found: {invalid_recent_actions}"

def test_invalid_hand_strength_and_pot_odds():
    dataset = PokerDataset(DATA_PATH)
    invalid_hand_strengths_or_pot_odds = [
        (state[-2], state[-1]) for state, _, _, _, _ in dataset
        if not (0 <= state[-2] <= 1) or not (0 <= state[-1] <= 1)
    ]
    assert not invalid_hand_strengths_or_pot_odds, (
        f"Invalid hand strengths or pot odds found: {invalid_hand_strengths_or_pot_odds}"
    )
