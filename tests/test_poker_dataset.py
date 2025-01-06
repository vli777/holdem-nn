import pytest
import h5py
import numpy as np
from torch.utils.data import DataLoader

from PokerSequenceDataset import PokerSequenceDataset, poker_collate_fn

@pytest.fixture
def hdf5_test_file(tmp_path):
    file_path = tmp_path / "test_data.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        for i in range(3):  # Create three sequences
            group = hdf5_file.create_group(f"game_{i}")
            group.create_dataset("states", data=np.random.rand(i + 1, 5))  # 5 features per state
            group.create_dataset("actions", data=np.random.randint(0, 3, size=(i + 1,)))
            group.create_dataset("player_ids", data=np.random.randint(0, 6, size=(i + 1,)))
            group.create_dataset("positions", data=np.random.randint(0, 6, size=(i + 1,)))
            group.create_dataset("recent_actions", data=np.random.randint(0, 3, size=(i + 1,)))
            group.create_dataset("strategies", data=np.random.randint(0, 3, size=(i + 1,)))
            group.create_dataset("bluffing_probs", data=np.random.rand(i + 1,))
    return file_path


def test_dataset_length(hdf5_test_file):
    dataset = PokerSequenceDataset(hdf5_path=hdf5_test_file)
    assert len(dataset) == 3  # Dataset should have 3 sequences


def test_sequence_retrieval(hdf5_test_file):
    dataset = PokerSequenceDataset(hdf5_path=hdf5_test_file, max_seq_len=5)
    sample = dataset[0]
    assert sample["states"].shape == (5, 5)  # max_seq_len=5, 5 features
    assert sample["actions"].shape == (5,)
    assert sample["mask"].shape == (5,)
    assert sample["mask"].sum() <= 5  # Mask should have at most 5 valid entries


def test_padding_and_truncation(hdf5_test_file):
    dataset = PokerSequenceDataset(hdf5_path=hdf5_test_file, max_seq_len=3)
    sample = dataset[2]
    assert sample["states"].shape == (3, 5)  # Truncated or padded to max_seq_len=3
    assert sample["mask"].shape == (3,)
    assert sample["mask"].sum() == 3  # All sequences are valid for this test case


def test_dataloader(hdf5_test_file):
    dataset = PokerSequenceDataset(hdf5_path=hdf5_test_file, max_seq_len=5)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=poker_collate_fn)
    batch = next(iter(dataloader))
    assert batch["states"].shape == (2, 5, 5)  # Batch size=2, seq_len=5, input_dim=5
    assert batch["actions"].shape == (2, 5)


def test_invalid_data_handling(tmp_path):
    file_path = tmp_path / "invalid_data.hdf5"
    with h5py.File(file_path, "w") as hdf5_file:
        group = hdf5_file.create_group("game_0")
        group.create_dataset("states", data=np.random.rand(3, 5))
        group.create_dataset("actions", data=np.array([1, -1, 3]))  # Invalid action (-1)
        group.create_dataset("player_ids", data=np.random.randint(0, 6, size=(3,)))
        group.create_dataset("positions", data=np.random.randint(0, 6, size=(3,)))
        group.create_dataset("recent_actions", data=np.random.randint(0, 3, size=(3,)))
        group.create_dataset("strategies", data=np.random.randint(0, 3, size=(3,)))
        group.create_dataset("bluffing_probs", data=np.random.rand(3))

    dataset = PokerSequenceDataset(hdf5_path=file_path, max_seq_len=5)
    sample = dataset[0]
    assert (sample["actions"] >= 0).all(), "Invalid actions should be filtered or set to a valid default"
