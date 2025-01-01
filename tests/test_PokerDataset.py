import h5py
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
    valid_data = [
        {
            "state": np.random.rand(config.input_dim),
            "action": 1,
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
            "bet_to_call": 10,
            "pot_odds": 0.5,
        }
        for _ in range(10)
    ]

    valid_dataset_path = tmp_path / "valid_dataset.h5"
    with h5py.File(valid_dataset_path, "w") as hdf5_file:
        for key in ["state", "action", "position", "player_id", "recent_action"]:
            data_to_append = [entry[key] for entry in valid_data]
            data_to_append = np.array(data_to_append)

            # Adjust the maxshape and shape based on rank
            if key == "state":
                hdf5_file.create_dataset(
                    key,
                    data=data_to_append,
                    maxshape=(None, config.input_dim),  # Match rank of `data_to_append`
                    chunks=(1, config.input_dim),  # Chunk size must also match rank
                    compression="gzip",
                )
            else:
                hdf5_file.create_dataset(
                    key,
                    data=data_to_append,
                    maxshape=(None,),  # 1D for scalar values
                    chunks=(1,),
                    compression="gzip",
                )

    dataset = PokerDataset(str(valid_dataset_path))
    assert len(dataset) > 0, "Dataset should not be empty"

    for idx, data in enumerate(dataset):
        state, action_label, position, player_id, recent_action = data

        assert isinstance(state, torch.Tensor), "State should be a torch.Tensor"
        assert state.shape == (
            config.input_dim,
        ), f"State tensor shape mismatch at index {idx}: expected {config.input_dim}, got {state.shape}"

        assert isinstance(
            action_label, torch.Tensor
        ), "Action label should be a torch.Tensor"
        assert action_label.dim() == 0, "Action label tensor should be scalar"
        assert action_label.item() in range(
            config.output_dim
        ), f"Invalid action label at index {idx}: {action_label.item()}"


def test_invalid_dataset_partial(tmp_path):
    """
    Test that the PokerDataset skips invalid data points but loads valid ones.
    """
    mixed_data = [
        {
            "state": np.random.rand(config.input_dim).astype("float32"),
            "action": 1,  # Valid action
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
        },
        {
            "state": np.random.rand(config.input_dim).astype("float32"),
            "action": 5,  # Invalid action
            "position": 3,
            "player_id": 2,
            "recent_action": 1,
        },
    ]

    mixed_dataset_path = tmp_path / "invalid_dataset_partial.h5"
    with h5py.File(mixed_dataset_path, "w") as hdf5_file:
        for key in ["state", "action", "position", "player_id", "recent_action"]:
            data_to_append = np.array([entry[key] for entry in mixed_data])
            hdf5_file.create_dataset(
                key,
                data=data_to_append,
                maxshape=(
                    (None,) + data_to_append.shape[1:]
                    if len(data_to_append.shape) > 1
                    else (None,)
                ),
                chunks=True,
                compression="gzip",
            )

    dataset = PokerDataset(str(mixed_dataset_path))
    assert len(dataset) == 1, "Dataset should contain only valid data points"
    state, action_label, position, player_id, recent_action = dataset[0]
    assert action_label.item() == 1, "Action label should be 1 (call)"


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
