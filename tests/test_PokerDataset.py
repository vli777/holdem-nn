import pytest
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Subset
from config import config
from PokerDataset import PokerDataset
from utils import encode_action
import random


@pytest.fixture
def dataset(tmp_path):
    data_path = config.data_path
    if not Path(data_path).exists():
        pytest.skip(f"Data file missing at {data_path}, skipping dataset fixture.")
    ds = PokerDataset(str(data_path))
    subset_size = 100
    if len(ds) > subset_size:
        indices = list(range(len(ds)))
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        ds = Subset(ds, subset_indices)
    return ds


def test_dataset_loading(tmp_path):
    """
    Test that the PokerDataset loads correctly and is not empty.
    """
    dataset = PokerDataset(str(config.data_path))
    assert len(dataset) > 0, "Dataset should not be empty"


@pytest.mark.parametrize(
    "action,expected",
    [
        ("fold", 0),
        ("call", 1),
        ("raise", 2),
    ],
)
def test_encode_action_valid(action, expected):
    """
    Test that encode_action correctly encodes valid actions.
    """
    assert (
        encode_action(action) == expected
    ), f"encode_action('{action}') should return {expected}"


@pytest.mark.parametrize("action", ["invalid_action", "check", "bet"])
def test_encode_action_invalid(action):
    """
    Test that encode_action raises ValueError for invalid actions.
    """
    with pytest.raises(ValueError, match=r"Unexpected action: .*"):
        encode_action(action)


def test_valid_dataset(tmp_path):
    """
    Test that a valid dataset is loaded correctly with all assertions passing.
    """
    # Create valid data with the correct state dimension
    valid_data = [
        {
            "state": np.random.rand(config.input_dim),
            "action": 1,  # Valid action (0, 1, or 2)
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
            "bet_to_call": 10,
            "pot_odds": 0.5,
        }
        for _ in range(10)
    ]

    # Save the valid dataset to a temporary file
    valid_dataset_path = tmp_path / "valid_dataset.npz"
    np.savez_compressed(valid_dataset_path, updated_data=valid_data)

    # Load the dataset
    dataset = PokerDataset(str(valid_dataset_path))
    assert len(dataset) > 0, "Dataset should not be empty"

    # Iterate through the dataset and perform assertions
    for idx, data in enumerate(dataset):
        state, action_label, position, player_id, recent_action = data

        # Assert state
        assert isinstance(state, torch.Tensor), "State should be a torch.Tensor"
        assert state.shape == (config.input_dim,), (
            f"State tensor shape mismatch at index {idx}: "
            f"expected {config.input_dim}, got {state.shape}"
        )

        # Assert action_label
        assert isinstance(
            action_label, torch.Tensor
        ), "Action label should be a torch.Tensor"
        assert action_label.dim() == 0, "Action label tensor should be scalar"
        assert action_label.item() in range(
            config.output_dim
        ), f"Invalid action label at index {idx}: {action_label.item()}"

        # Assert position
        assert isinstance(position, torch.Tensor), "Position should be a torch.Tensor"
        assert position.dim() == 0, "Position tensor should be scalar"

        # Assert player_id
        assert isinstance(player_id, torch.Tensor), "Player ID should be a torch.Tensor"
        assert player_id.dim() == 0, "Player ID tensor should be scalar"

        # Assert recent_action
        assert isinstance(
            recent_action, torch.Tensor
        ), "Recent action should be a torch.Tensor"
        assert recent_action.dim() == 0, "Recent action tensor should be scalar"


def test_invalid_dataset_all_invalid(tmp_path):
    """
    Test that the PokerDataset raises a ValueError when all data points are invalid.
    """
    # Create invalid data with an invalid action label
    invalid_data = [
        {
            "state": np.random.rand(config.input_dim),
            "action": 5,  # Invalid action
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
            "bet_to_call": 10,
            "pot_odds": 0.5,
        }
    ]
    invalid_dataset_path = tmp_path / "invalid_dataset_all_invalid.npz"
    np.savez_compressed(invalid_dataset_path, updated_data=invalid_data)

    # Attempt to load the invalid dataset and expect a ValueError
    with pytest.raises(
        ValueError, match="No valid data points were parsed from the dataset."
    ):
        PokerDataset(str(invalid_dataset_path))


def test_invalid_dataset_partial(tmp_path):
    """
    Test that the PokerDataset skips invalid data points but loads valid ones.
    """
    # Create a mix of valid and invalid data
    mixed_data = [
        {
            "state": np.random.rand(config.input_dim),
            "action": 1,  # Valid action
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
            "bet_to_call": 10,
            "pot_odds": 0.5,
        },
        {
            "state": np.random.rand(config.input_dim),
            "action": 5,  # Invalid action
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
            "bet_to_call": 10,
            "pot_odds": 0.5,
        },
    ]
    mixed_dataset_path = tmp_path / "invalid_dataset_partial.npz"
    np.savez_compressed(mixed_dataset_path, updated_data=mixed_data)

    # Load the dataset
    dataset = PokerDataset(str(mixed_dataset_path))
    assert len(dataset) == 1, "Dataset should contain only valid data points"

    # Check the valid data point
    state, action_label, position, player_id, recent_action = dataset[0]
    assert action_label.item() == 1, "Action label should be 1 (call)"


def test_missing_field_in_action(tmp_path):
    """
    Test that the PokerDataset raises a ValueError when no valid data points are present due to missing fields.
    """
    # Create data missing the "action" field
    missing_field_data = [
        {
            "state": np.random.rand(config.input_dim),
            # Missing "action" field
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
            "bet_to_call": 10,
            "pot_odds": 0.5,
        }
    ]
    missing_field_dataset_path = tmp_path / "missing_field_dataset.npz"
    np.savez_compressed(missing_field_dataset_path, updated_data=missing_field_data)

    # Attempt to load the dataset and expect a ValueError since no valid data points are parsed
    with pytest.raises(
        ValueError, match="No valid data points were parsed from the dataset."
    ):
        PokerDataset(str(missing_field_dataset_path))


def test_empty_dataset(tmp_path):
    """
    Test that the PokerDataset raises a ValueError when the dataset is empty.
    """
    # Create an empty dataset
    empty_data = []
    empty_dataset_path = tmp_path / "empty_dataset.npz"
    np.savez_compressed(empty_dataset_path, updated_data=empty_data)

    # Attempt to load the empty dataset and expect a ValueError
    with pytest.raises(ValueError, match="Loaded dataset is empty!"):
        PokerDataset(str(empty_dataset_path))


def test_invalid_player_ids(dataset):
    """
    Test that player IDs are within the valid range.
    """
    num_players = 6  # Assuming player IDs range from 0 to 5
    invalid_player_ids = [
        player_id.item()
        for _, _, player_id, _, _ in dataset
        if player_id.item() < 0 or player_id.item() >= num_players
    ]
    assert not invalid_player_ids, f"Invalid player IDs found: {invalid_player_ids}"


def test_invalid_positions(dataset):
    """
    Test that positions are within the valid range.
    """
    max_positions = 6  # Assuming positions range from 0 to 5
    invalid_positions = [
        position.item()
        for _, _, position, _, _ in dataset
        if position.item() < 0 or position.item() >= max_positions
    ]
    assert not invalid_positions, f"Invalid positions found: {invalid_positions}"


def test_invalid_recent_actions(dataset):
    """
    Test that recent actions are within the valid range.
    """
    valid_recent_actions = [0, 1, 2]  # Define valid recent actions
    invalid_recent_actions = [
        recent_action.item()
        for _, _, _, _, recent_action in dataset
        if recent_action.item() not in valid_recent_actions
    ]
    assert (
        not invalid_recent_actions
    ), f"Invalid recent actions found: {invalid_recent_actions}"


def test_invalid_hand_strength_and_pot_odds(dataset):
    """
    Test that hand strengths and pot odds are within the range [0, 1].
    Assumes that the last two elements of the state tensor represent hand_strength and pot_odds.
    """
    invalid_hand_strengths_or_pot_odds = [
        (state[-2].item(), state[-1].item())
        for state, _, _, _, _ in dataset
        if not (0 <= state[-2].item() <= 1) or not (0 <= state[-1].item() <= 1)
    ]
    assert (
        not invalid_hand_strengths_or_pot_odds
    ), f"Invalid hand strengths or pot odds found: {invalid_hand_strengths_or_pot_odds}"
