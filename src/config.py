from pathlib import Path
from pydantic_settings import BaseSettings
import logging


class Settings(BaseSettings):
    model_path: Path = Path("saved_models/poker_model_full.pt")
    data_path: Path = Path("data/poker_dataset.h5")
    learning_rate = 1e-4
    batch_size = 64
    hidden_dim = 256
    output_dim = 3  # "fold", "call", "raise"
    seq_len = 100  # Adjust based on EDA
    num_heads = 8
    num_layers = 4
    num_players = 6
    max_positions = 10
    num_actions = 3
    num_strategies = 3
    num_epochs = 50
    early_stop_limit = 5
    dropout = 0.1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        frozen = True  # Make settings immutable


# Initialize config
config = Settings()

# Ensure necessary directories exist
config.model_path.parent.mkdir(parents=True, exist_ok=True)
config.data_path.parent.mkdir(parents=True, exist_ok=True)

# Validate paths
assert config.data_path.suffix == ".h5", "data_path must be an HDF5 file"
assert config.model_path.suffix == ".pt", "model_path must be a PyTorch model file"


# Setup logging
def setup_logging(debug: bool):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


setup_logging(config.debug)
