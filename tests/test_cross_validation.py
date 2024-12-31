import pytest
import torch
from torch.utils.data import Dataset
from models.PokerLinformerModel import PokerLinformerModel
from training.cross_validation import k_fold_cross_validation
from tests.mocks.MockDataset import MockDataset
from config import config
import logging


@pytest.fixture
def setup_mock_data():
    dataset = MockDataset(size=100)
    model_params = {
        "input_dim": config.input_dim
        + 10
        + 5
        + 3,  # Combined input dimensions: states + positions + player_ids + recent_actions
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "seq_len": config.seq_len,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "num_players": 6,  # As per actual model
    }
    assert model_params["input_dim"] == config.input_dim + 10 + 5 + 3
    assert model_params["hidden_dim"] == config.hidden_dim
    return dataset, model_params


def test_k_fold_cross_validation_valid_data(setup_mock_data, tmp_path, caplog):
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
            optimizer_params={"lr": config.learning_rate},
            k=5,
            epochs=3,
            batch_size=config.batch_size,
            model_save_dir=str(model_save_dir),
        )

    # Check results
    assert len(results) == 5, "Expected results for all 5 folds"
    for fold_result in results:
        assert "fold" in fold_result, "Missing 'fold' key in results"
        assert "train_loss" in fold_result, "Missing 'train_loss' key in results"
        assert "val_loss" in fold_result, "Missing 'val_loss' key in results"
        assert "val_accuracy" in fold_result, "Missing 'val_accuracy' key in results"

    # Check that model weights are saved
    saved_models = list(model_save_dir.glob("best_model_fold*.pth"))
    assert len(saved_models) == 5, "Expected one saved model per fold"

    # Ensure no warnings about parameters not requiring grad
    for record in caplog.records:
        assert (
            "does not require gradients" not in record.message
        ), "Model parameters should require gradients"


def test_k_fold_cross_validation_empty_dataset(tmp_path):
    class EmptyDataset(Dataset):
        def __init__(self):
            self.data = []
            self.labels = []

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError("Dataset is empty")

    dataset = EmptyDataset()
    model_params = {
        "input_dim": config.input_dim + 10 + 5 + 3,  # Combined input dimensions
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "seq_len": config.seq_len,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "num_players": 6,
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
            model_class=PokerLinformerModel,
            model_params=model_params,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": config.learning_rate},
            k=5,
            epochs=3,
            batch_size=config.batch_size,
            model_save_dir=str(model_save_dir),
        )


def test_k_fold_cross_validation_early_stopping(setup_mock_data, tmp_path, caplog):
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
            optimizer_params={"lr": config.learning_rate},
            k=5,
            epochs=20,  # Longer epochs to test early stopping
            batch_size=config.batch_size,
            model_save_dir=str(model_save_dir),
        )

    # Check results
    assert len(results) == 5, "Expected results for all 5 folds"
    for fold_result in results:
        assert "fold" in fold_result, "Missing 'fold' key in results"
        assert "train_loss" in fold_result, "Missing 'train_loss' key in results"
        assert "val_loss" in fold_result, "Missing 'val_loss' key in results"
        assert "val_accuracy" in fold_result, "Missing 'val_accuracy' key in results"

    # Check that model weights are saved
    saved_models = list(model_save_dir.glob("best_model_fold*.pth"))
    assert len(saved_models) == 5, "Expected one saved model per fold"

    # Check if early stopping logs are present
    early_stop_logs = [
        record for record in caplog.records if "Early stopping" in record.message
    ]
    assert len(early_stop_logs) >= 1, "Expected early stopping logs"
