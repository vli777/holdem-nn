from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError


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

    # Configuration for Settings
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


try:
    config = Settings()
except ValidationError as e:
    print("Configuration validation error:", e)
    exit(1)
