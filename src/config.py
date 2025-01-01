from pathlib import Path
from pydantic_settings import BaseSettings
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")


class Settings(BaseSettings):
    # General Configurations
    learning_rate: float = 1e-3
    batch_size: int = 32
    hidden_dim: int = 128
    output_dim: int = 3
    seq_len: int = 1
    num_heads: int = 4
    num_layers: int = 2
    num_epochs: int = 10
    early_stop_limit: int = 5
    input_dim: int = 4
    state_dim: int = 10
    model_path: Path = Path("saved_models/poker_model_full.pt")
    data_path: Path = Path("data/poker_dataset.h5")
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Settings()

# Ensure necessary directories exist
config.model_path.parent.mkdir(parents=True, exist_ok=True)
config.data_path.parent.mkdir(parents=True, exist_ok=True)

# Adjust logging based on debug mode
logging.basicConfig(
    level=logging.DEBUG if config.debug else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
