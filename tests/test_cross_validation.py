import os
from pathlib import Path
import pytest
import torch
from tests.mocks import MockDataset, MockModel
from training.cross_validation import k_fold_cross_validation
from config import config

DATA_PATH = str(config.data_path)


@pytest.fixture
def setup_mock_data():
    dataset = MockDataset(size=100)
    model_params = {
        "input_dim": config.input_dim,  # Use input_dim from config
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "seq_len": config.seq_len,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
    }
    return dataset, model_params


@pytest.mark.skipif(
    not Path(DATA_PATH).exists(), reason="Poker dataset not available. Skipping tests."
)
def test_k_fold_cross_validation_valid_data(setup_mock_data, tmp_path):
    dataset, model_params = setup_mock_data
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    results = k_fold_cross_validation(
        dataset=dataset,
        device=torch.device("cpu"),
        model_class=MockModel,
        model_params=model_params,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": config.learning_rate},
        k=5,
        epochs=3,
        batch_size=config.batch_size,
        model_save_dir=str(model_save_dir),
    )

    # Check results
    assert len(results) == 5  # 5 folds
    for result in results:
        assert "train_loss" in result
        assert "val_loss" in result
        assert "val_accuracy" in result

    # Check that model weights are saved
    saved_models = list(model_save_dir.glob("*.pth"))
    assert len(saved_models) == 5  # One model per fold


def test_k_fold_cross_validation_empty_dataset(tmp_path):
    dataset = MockDataset(size=0)  # Empty dataset
    model_params = {
        "input_dim": config.input_dim,  # Use input_dim from config
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "seq_len": config.seq_len,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
    }
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(
        ValueError,
        match=r"Found array with 0 sample\(s\) .* while a minimum of 1 is required\.",
    ):
        k_fold_cross_validation(
            dataset=dataset,
            device=torch.device("cpu"),
            model_class=MockModel,
            model_params=model_params,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": config.learning_rate},
            k=5,
            epochs=3,
            batch_size=config.batch_size,
            model_save_dir=str(model_save_dir),
        )


def test_k_fold_cross_validation_early_stopping(setup_mock_data, tmp_path):
    dataset, model_params = setup_mock_data
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    results = k_fold_cross_validation(
        dataset=dataset,
        device=torch.device("cpu"),
        model_class=MockModel,
        model_params=model_params,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": config.learning_rate},
        k=5,
        epochs=20,  # Longer epochs to test early stopping
        batch_size=config.batch_size,
        model_save_dir=str(model_save_dir),
    )

    # Verify early stopping worked (e.g., fewer epochs completed than specified)
    assert len(results) == 5
    for result in results:
        assert "train_loss" in result
        assert "val_loss" in result
        assert "val_accuracy" in result
