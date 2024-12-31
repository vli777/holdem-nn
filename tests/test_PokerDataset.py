import pytest
import numpy as np
from PokerDataset import PokerDataset
from utils import encode_action


def test_dataset_loading():
    dataset = PokerDataset("data/texas_holdem_data.npz")
    assert len(dataset) > 0, "Dataset should not be empty"


def test_encode_action():
    assert encode_action("fold") == 0
    assert encode_action("call") == 1
    assert encode_action("raise") == 2
    with pytest.raises(ValueError):
        encode_action("invalid_action")


def test_valid_dataset():
    valid_data = [
        [
            {
                "state": np.random.rand(2),
                "action": 1,  # Valid action (0, 1, or 2)
                "position": 3,
                "player_id": 2,
                "recent_action": 1,
                "bet_to_call": 10,
                "pot_odds": 0.5,
            }
            for _ in range(10)
        ]
    ]

    np.savez_compressed("data/valid_dataset.npz", updated_data=valid_data)

    dataset = PokerDataset("data/valid_dataset.npz")
    assert len(dataset) > 0, "Dataset should not be empty"
    for idx, data in enumerate(dataset):
        state, action_label, position, player_id, recent_action = data
        assert action_label in [
            0,
            1,
            2,
        ], f"Invalid action label at index {idx}: {action_label}"
        assert 0 <= state[-2] <= 1, f"Invalid hand strength at index {idx}: {state[-2]}"
        assert 0 <= state[-1] <= 1, f"Invalid pot odds at index {idx}: {state[-1]}"
        assert recent_action in [
            0,
            1,
            2,
        ], f"Invalid recent action at index {idx}: {recent_action}"


def test_invalid_dataset():
    invalid_data = [
        [
            {
                "state": np.random.rand(2),
                "action": 5,  # Invalid action
                "position": 3,
                "player_id": 2,
                "recent_action": 1,
                "bet_to_call": 10,
                "pot_odds": 0.5,
            }
        ]
    ]
    np.savez_compressed("data/invalid_dataset.npz", updated_data=invalid_data)

    with pytest.raises(ValueError, match="Invalid encoded action"):
        PokerDataset("data/invalid_dataset.npz")


def test_missing_field_in_action():
    missing_field_data = [
        [
            {
                "state": np.random.rand(2),
                # Missing "action" field
                "position": 3,
                "player_id": 2,
                "recent_action": 1,
                "bet_to_call": 10,
                "pot_odds": 0.5,
            }
        ]
    ]
    np.savez_compressed(
        "data/missing_field_dataset.npz", updated_data=missing_field_data
    )

    with pytest.raises(KeyError, match="Missing key: action"):
        PokerDataset("data/missing_field_dataset.npz")


def test_empty_dataset():
    empty_data = []
    np.savez_compressed("data/empty_dataset.npz", updated_data=empty_data)

    with pytest.raises(ValueError, match="Loaded dataset is empty"):
        PokerDataset("data/empty_dataset.npz")
