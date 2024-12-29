import pytest
import numpy as np
from src.PokerDataset import PokerDataset

def test_dataset_loading():
    dataset = PokerDataset("data/texas_holdem_data.npz")
    assert len(dataset) > 0, "Dataset should not be empty"

def test_encode_action():
    assert PokerDataset.encode_action("fold") == 0
    assert PokerDataset.encode_action("call") == 1
    assert PokerDataset.encode_action("raise") == 2
    with pytest.raises(ValueError):
        PokerDataset.encode_action("invalid_action")

def test_valid_dataset():
    valid_data = [
        [
            {
                "state": np.random.rand(106),
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
    for data in dataset:
        assert data[1] in [0, 1, 2], f"Invalid action label: {data[1]}"

def test_invalid_dataset():
    invalid_data = [
        [
            {
                "state": np.random.rand(106),
                "action": 5,  # Invalid action (should be 0, 1, or 2)
                "position": 3,
                "player_id": 2,
                "recent_action": 1,
                "bet_to_call": 10,
                "pot_odds": 0.5,
            }
            for _ in range(10)
        ]
    ]
    np.savez_compressed("data/invalid_dataset.npz", updated_data=invalid_data)

    with pytest.raises(ValueError, match="Invalid encoded action: 5"):
        PokerDataset("data/invalid_dataset.npz")

def test_empty_dataset():
    empty_data = []
    np.savez_compressed("data/empty_dataset.npz", updated_data=empty_data)
    
    with pytest.raises(ValueError, match="Loaded dataset is empty"):
        PokerDataset("data/empty_dataset.npz")
