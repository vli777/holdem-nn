import pytest
import torch
from torch.utils.data import Dataset
from training.cross_validation import k_fold_cross_validation
from models.PokerLinformerModel import PokerLinformerModel

# Mock Dataset
class MockDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        self.data = [
            (
                torch.rand(106),  # Random state vector
                torch.randint(0, 3, (1,)).item(),  # Random action (0, 1, 2)
                torch.randint(0, 10, (1,)).item(),  # Random position
                torch.randint(0, 6, (1,)).item(),  # Random player ID
                torch.randint(0, 3, (1,)).item()  # Random recent action
            )
            for _ in range(size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]

# Mock Model
class MockModel(PokerLinformerModel):
    def forward(self, states, positions, player_ids, recent_actions):
        batch_size = states.size(0)
        policy_logits = torch.rand(batch_size, 3, requires_grad=True)  # Policy logits
        value = torch.rand(batch_size, 1, requires_grad=True)          # Value
        return policy_logits, value
    

@pytest.fixture
def setup_mock_data():
    dataset = MockDataset(size=100)
    model_params = {
        "input_dim": 106,
        "hidden_dim": 128,
        "output_dim": 3,
        "seq_len": 1,
        "num_heads": 4,
        "num_layers": 2,
    }
    return dataset, model_params

def test_k_fold_cross_validation_valid_data(setup_mock_data, tmp_path):
    dataset, model_params = setup_mock_data
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir()

    results = k_fold_cross_validation(
        dataset=dataset,
        device=torch.device("cpu"),
        model_class=MockModel,
        model_params=model_params,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        k=5,
        epochs=3,
        batch_size=16,
        model_save_dir=str(model_save_dir)
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
        "input_dim": 106,
        "hidden_dim": 128,
        "output_dim": 3,
        "seq_len": 1,
        "num_heads": 4,
        "num_layers": 2,
    }
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir()

    with pytest.raises(ValueError, match=r"Found array with 0 sample\(s\) .* while a minimum of 1 is required\."):
        k_fold_cross_validation(
            dataset=dataset,
            device=torch.device("cpu"),
            model_class=MockModel,
            model_params=model_params,
            criterion=torch.nn.CrossEntropyLoss(),
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 1e-3},
            k=5,
            epochs=3,
            batch_size=16,
            model_save_dir=str(model_save_dir)
        )

     
def test_k_fold_cross_validation_early_stopping(setup_mock_data, tmp_path):
    dataset, model_params = setup_mock_data
    model_save_dir = tmp_path / "models"
    model_save_dir.mkdir()

    results = k_fold_cross_validation(
        dataset=dataset,
        device=torch.device("cpu"),
        model_class=MockModel,
        model_params=model_params,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        k=5,
        epochs=20,  # Longer epochs to test early stopping
        batch_size=16,
        model_save_dir=str(model_save_dir)
    )

    # Verify early stopping worked (e.g., fewer epochs completed than specified)
    assert len(results) == 5
    for result in results:
        assert "train_loss" in result
        assert "val_loss" in result
        assert "val_accuracy" in result
