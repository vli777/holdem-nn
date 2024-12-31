import os
from pathlib import Path
import pytest
from PokerDataset import PokerDataset
import torch
from config import config

# Ensure DATA_PATH is a string to avoid AttributeError
DATA_PATH = str(config.data_path)


@pytest.fixture(scope="module")
def dataset():
    if not os.path.exists(DATA_PATH):
        pytest.skip("Data file missing, skipping dataset fixture.")
    ds = PokerDataset(DATA_PATH)
    subset_size = 10000
    if len(ds) > subset_size:
        from torch.utils.data import Subset
        import random

        indices = list(range(len(ds)))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        ds = Subset(ds, subset_indices)
    return ds


def test_dataset_item_structure():
    data_path = Path("data/texas_holdem_data.npz")
    dataset = PokerDataset(str(data_path))
    state, action_label, position, player_id, recent_action = dataset[0]
    assert isinstance(state, torch.Tensor) and state.shape == (
        4,
    ), "State tensor shape mismatch"
    assert action_label in [0, 1, 2], "Invalid action label"
    assert isinstance(position, torch.Tensor), "Position should be a tensor"
    assert isinstance(player_id, torch.Tensor), "Player ID should be a tensor"
    assert isinstance(recent_action, torch.Tensor), "Recent action should be a tensor"


def test_data_validity(dataset):
    try:
        assert len(dataset) > 0, "Dataset should not be empty"
    except Exception as e:
        pytest.fail(f"Failed to load dataset: {e}")


def test_invalid_actions(dataset):
    invalid_actions = [
        action_label
        for _, action_label, _, _, _ in dataset
        if action_label not in [0, 1, 2]
    ]
    assert not invalid_actions, f"Invalid actions found: {invalid_actions}"


def test_invalid_player_ids(dataset):
    num_players = 6
    invalid_player_ids = [
        player_id
        for _, _, _, player_id, _ in dataset
        if player_id < 0 or player_id >= num_players
    ]
    assert not invalid_player_ids, f"Invalid player IDs found: {invalid_player_ids}"


def test_invalid_positions(dataset):
    max_positions = 6
    invalid_positions = [
        position
        for _, _, position, _, _ in dataset
        if position < 0 or position >= max_positions
    ]
    assert not invalid_positions, f"Invalid positions found: {invalid_positions}"


def test_invalid_state_dimensions(dataset):
    expected_state_dim = 106
    invalid_states = [
        state for state, _, _, _, _ in dataset if len(state) != expected_state_dim
    ]
    if invalid_states:
        for idx, state in enumerate(invalid_states):
            print(f"Invalid state at index {idx}: {state}, Length: {len(state)}")
    assert not invalid_states, f"Invalid state dimensions found: {invalid_states}"


def test_invalid_recent_actions(dataset):
    invalid_recent_actions = [
        recent_action
        for _, _, _, _, recent_action in dataset
        if recent_action not in [0, 1, 2]
    ]
    assert (
        not invalid_recent_actions
    ), f"Invalid recent actions found: {invalid_recent_actions}"


def test_invalid_hand_strength_and_pot_odds(dataset):
    invalid_hand_strengths_or_pot_odds = [
        (state[-2], state[-1])
        for state, _, _, _, _ in dataset
        if not (0 <= state[-2] <= 1) or not (0 <= state[-1] <= 1)
    ]
    assert (
        not invalid_hand_strengths_or_pot_odds
    ), f"Invalid hand strengths or pot odds found: {invalid_hand_strengths_or_pot_odds}"
