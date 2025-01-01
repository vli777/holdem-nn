import numpy as np
import pytest
import torch
from PokerDataset import PokerDataset
from models.PokerLinformerModel import PokerLinformerModel
from training.cross_validation import k_fold_cross_validation
import logging
from training.hdf5 import append_to_hdf5, initialize_hdf5


@pytest.fixture
def config(tmp_path):
    """
    Configuration fixture for tests.
    """
    return {
        "data_path": tmp_path / "poker_dataset.h5",
        "state_dim": 4,  # Adjust based on your 'state' feature dimensionality
        "learning_rate": 0.001,
        "batch_size": 32,
        "hidden_dim": 128,
        "output_dim": 3,  # Example: 3 possible actions
        "seq_len": 5,
        "num_heads": 4,
        "num_layers": 2,
        "num_epochs": 20,
        "early_stop_limit": 5,
        "model_path": tmp_path / "models" / "poker_model.pt",
    }


@pytest.fixture
def setup_hdf5_mock_data(config):
    """
    Fixture to set up a mock HDF5 dataset for testing.
    """
    # Initialize HDF5 file
    initialize_hdf5(
        file_path=str(config["data_path"]),
        state_dim=config["state_dim"],
        initial_size=0,
        chunk_size=1000,
        compression="gzip",
    )

    # Create sample data to append
    sample_data = [
        {
            "state": np.random.rand(config["state_dim"]).astype("float32"),
            "action": np.random.randint(0, config["output_dim"]),
            "position": np.random.randint(0, 6),  # Assuming 6 possible positions
            "player_id": np.random.randint(1000, 2000),
            "recent_action": np.random.randint(0, 3),
        }
        for _ in range(100)
    ]

    # Append sample data to HDF5
    append_to_hdf5(str(config["data_path"]), sample_data, config["state_dim"])

    # Return the path and config for use in tests
    return config["data_path"], config


@pytest.fixture
def setup_mock_data(setup_hdf5_mock_data):
    """
    Fixture to provide dataset and model parameters using HDF5-based data.
    """
    data_path, config = setup_hdf5_mock_data
    dataset = PokerDataset(data_path)

    model_params = {
        "input_dim": config["state_dim"] + 6,
        "hidden_dim": config["hidden_dim"],
        "output_dim": config["output_dim"],
        "seq_len": config["seq_len"],
        "num_heads": config["num_heads"],
        "num_layers": config["num_layers"],
        "num_players": 6,  # As per actual model
    }

    return dataset, model_params


def test_k_fold_cross_validation_valid_data(setup_mock_data, tmp_path, caplog, config):
    """
    Test k-fold cross-validation with valid HDF5 data.
    """
    dataset, model_params = setup_mock_data
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.WARNING):
        results = k_fold_cross_validation(
            dataset=dataset,
            device=torch.device("cpu"),
            model_class=PokerLinformerModel,
            model_params=model_params,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": config["learning_rate"]},
            k=5,
            epochs=3,
            batch_size=config["batch_size"],
            model_save_dir=str(model_save_dir),
        )

    # Check results
    assert len(results) == 5, "Expected results for all 5 folds"
    for fold_result in results:
        assert "fold" in fold_result, "Missing 'fold' key in results"
        assert "train_loss" in fold_result, "Missing 'train_loss' key in results"
        assert "val_loss" in fold_result, "Missing 'val_loss' key in results"
        assert "val_accuracy" in fold_result, "Missing 'val_accuracy' key in results"


def test_k_fold_cross_validation_empty_dataset(tmp_path, config):
    """
    Test k-fold cross-validation with an empty HDF5 dataset.
    """
    # Initialize an empty HDF5 file
    initialize_hdf5(
        file_path=str(config["data_path"]),
        state_dim=config["state_dim"],
        initial_size=0,
        chunk_size=1000,
        compression="gzip",
    )

    # Attempt to initialize an empty dataset and ensure it raises ValueError
    with pytest.raises(ValueError, match=r"Loaded dataset is empty!"):
        dataset = PokerDataset(config["data_path"])

        # Ensure dataset length is 0 if it didn't raise (unlikely but good to check)
        assert len(dataset) == 0, "Dataset should be empty for this test"

        model_params = {
            "input_dim": config["state_dim"] + 6,
            "hidden_dim": config["hidden_dim"],
            "output_dim": config["output_dim"],
            "seq_len": config["seq_len"],
            "num_heads": config["num_heads"],
            "num_layers": config["num_layers"],
            "num_players": 6,
        }

        model_save_dir = tmp_path / "models"
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # Expect ValueError to be raised when using an empty dataset in cross-validation
        with pytest.raises(ValueError, match=r"Loaded dataset is empty!"):
            k_fold_cross_validation(
                dataset=dataset,
                device=torch.device("cpu"),
                model_class=PokerLinformerModel,
                model_params=model_params,
                criterion=torch.nn.CrossEntropyLoss(),
                optimizer_class=torch.optim.Adam,
                optimizer_params={"lr": config["learning_rate"]},
                k=5,
                epochs=3,
                batch_size=config["batch_size"],
                model_save_dir=str(model_save_dir),
            )



def test_k_fold_cross_validation_early_stopping(
    setup_mock_data, tmp_path, caplog, config
):
    """
    Test k-fold cross-validation with early stopping triggered.
    """
    dataset, model_params = setup_mock_data
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    with caplog.at_level(logging.INFO):
        results = k_fold_cross_validation(
            dataset=dataset,
            device=torch.device("cpu"),
            model_class=PokerLinformerModel,
            model_params=model_params,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": config["learning_rate"]},
            k=5,
            epochs=20,
            batch_size=config["batch_size"],
            model_save_dir=str(model_save_dir),
        )

    assert len(results) == 5, "Expected results for all 5 folds"
    for fold_result in results:
        assert "fold" in fold_result, "Missing 'fold' key in results"
        assert "train_loss" in fold_result, "Missing 'train_loss' key in results"
        assert "val_loss" in fold_result, "Missing 'val_loss' key in results"
        assert "val_accuracy" in fold_result, "Missing 'val_accuracy' key in results"

    # Check if early stopping logs are present
    early_stop_logs = [
        record for record in caplog.records if "Early stopping" in record.message
    ]
    assert len(early_stop_logs) >= 1, "Expected early stopping logs"
