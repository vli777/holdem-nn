from pydantic import BaseSettings
from pathlib import Path
import logging


class Settings(BaseSettings):
    model_path: Path = Path("saved_models/poker_model_full.pt")
    data_path: Path = Path("data/poker_dataset.h5")
    num_hands: int = 10000
    learning_rate: float = 1e-4
    batch_size: int = 64
    hidden_dim: int = 256
    output_dim: int = 3  # "fold", "call", "raise"
    seq_len: int = 100  # Adjust based on EDA
    num_heads: int = 8
    num_layers: int = 4
    num_players: int = 6
    max_positions: int = 10
    num_actions: int = 3
    num_strategies: int = 3
    num_epochs: int = 50
    early_stop_limit: int = 5
    dropout: float = 0.1
    debug: bool = False  # Added for logging purposes

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        frozen = True  # Keep settings immutable to prevent accidental changes


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
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )


setup_logging(config.debug)
