# poker_model.py

import argparse
import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from torch.amp import GradScaler
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter

from PokerSequenceDataset import PokerSequenceDataset, poker_collate_fn
from models import PokerTransformerModel
from training.utils import initialize_hdf5
from config import Settings, config
from training.utils import get_class_weights


# Optional: Set a seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# Common Functions
# ===========================


def load_dataset(data_path, max_samples=None, shuffle_subset=False):
    """
    Load the Poker dataset from an HDF5 file, optionally limiting it to a smaller subset.

    Args:
        data_path (str or Path): Path to the HDF5 data file.
        max_samples (int, optional): If not None, the maximum number of samples to load.
        shuffle_subset (bool): Whether to shuffle before taking the subset.

    Returns:
        Dataset: A PyTorch Dataset object, potentially a Subset.
    """
    try:
        dataset = PokerSequenceDataset(data_path)
        logging.info(f"Dataset loaded with {len(dataset)} samples.")

        if max_samples is not None and max_samples < len(dataset):
            indices = list(range(len(dataset)))
            if shuffle_subset:
                random.shuffle(indices)
            subset_indices = indices[:max_samples]

            dataset = Subset(dataset, subset_indices)
            logging.info(f"Using partial dataset with {len(subset_indices)} samples.")

        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit(1)


def initialize_model(input_dim, device, config):
    """
    Initialize the PokerTransformerModel, optimizer, and scheduler.

    Args:
        input_dim (int): Dimensionality of the input features.
        device (torch.device): Device to load the model on.
        config (dict): Configuration dictionary with hyperparameters.

    Returns:
        tuple: (model, optimizer, scheduler)
    """
    model_dir = Path(config["model_path"]).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    model = PokerTransformerModel(
        input_dim=input_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        seq_len=config.seq_len,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_players=config.num_players,
        max_positions=config.max_positions,
        num_actions=config.num_actions,
        num_strategies=config.num_strategies,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )
    logging.info("Model, optimizer, and scheduler initialized.")
    return model, optimizer, scheduler


def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.
        scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision.
        device (torch.device): Device to perform training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(train_loader):
        states = batch["states"].to(device)  # [batch_size, seq_len, input_dim]
        actions = batch["actions"].to(device)  # [batch_size, seq_len]
        player_ids = batch["player_ids"].to(device)  # [batch_size, seq_len]
        positions = batch["positions"].to(device)  # [batch_size, seq_len]
        recent_actions = batch["recent_actions"].to(device)  # [batch_size, seq_len]
        strategies = batch["strategies"].to(device)  # [batch_size, seq_len]
        bluffing_probabilities = batch["bluffing_probabilities"].to(
            device
        )  # [batch_size, seq_len]
        mask = batch["mask"].to(device)  # [batch_size, seq_len]

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            policy_logits = model(
                states,
                player_ids,
                positions,
                recent_actions,
                strategies,
                bluffing_probabilities,
                mask=mask,
            )  # [batch_size, seq_len, output_dim]
            # Reshape for loss computation
            policy_logits = policy_logits.view(
                -1, policy_logits.size(-1)
            )  # [(batch_size * seq_len), output_dim]
            actions = actions.view(-1)  # [(batch_size * seq_len)]

            loss = criterion(policy_logits, actions)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            logging.info(
                f"Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}"
            )

    average_loss = running_loss / max(len(train_loader), 1)
    logging.info(f"Epoch Training Loss: {average_loss:.4f}")
    return average_loss


def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to perform validation on.

    Returns:
        tuple: (average validation loss, accuracy, F1 score)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for batch in val_loader:
            states = batch["states"].to(device)  # [batch_size, seq_len, input_dim]
            actions = batch["actions"].to(device)  # [batch_size, seq_len]
            player_ids = batch["player_ids"].to(device)  # [batch_size, seq_len]
            positions = batch["positions"].to(device)  # [batch_size, seq_len]
            recent_actions = batch["recent_actions"].to(device)  # [batch_size, seq_len]
            strategies = batch["strategies"].to(device)  # [batch_size, seq_len]
            bluffing_probabilities = batch["bluffing_probabilities"].to(
                device
            )  # [batch_size, seq_len]
            mask = batch["mask"].to(device)  # [batch_size, seq_len]

            policy_logits = model(
                states,
                player_ids,
                positions,
                recent_actions,
                strategies,
                bluffing_probabilities,
                mask=mask,
            )  # [batch_size, seq_len, output_dim]
            # Reshape for loss computation
            policy_logits = policy_logits.view(
                -1, policy_logits.size(-1)
            )  # [(batch_size * seq_len), output_dim]
            actions = actions.view(-1)

            loss = criterion(policy_logits, actions)
            val_loss += loss.item()

            # Predictions
            _, predicted = torch.max(policy_logits, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(actions.cpu().numpy())

    average_val_loss = val_loss / max(len(val_loader), 1)
    accuracy = (correct / total) * 100 if total > 0 else 0
    f1 = (
        f1_score(true_labels, predicted_labels, average="weighted")
        if total > 0
        else 0.0
    )

    logging.info(
        f"Validation Loss: {average_val_loss:.4f}, Accuracy: {accuracy:.2f}%, F1 Score: {f1:.4f}"
    )
    return average_val_loss, accuracy, f1


def train_model(dataset, device, config, writer, early_stop=True):
    """
    Perform full training of the model.

    Args:
        dataset (Dataset): The dataset to train on.
        device (torch.device): Device to perform training on.
        config (dict): Configuration dictionary with hyperparameters.
        writer (SummaryWriter): TensorBoard writer.
        early_stop (bool): Whether to use early stopping.

    Returns:
        float: Best validation F1 score achieved during training.
    """
    input_dim = dataset[0]["states"].shape[
        1
    ]  # Assuming states are [seq_len, input_dim]
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=4,
        collate_fn=poker_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=4,
        collate_fn=poker_collate_fn,
    )

    # Initialize model, optimizer, scheduler
    model, optimizer, scheduler = initialize_model(input_dim, device, config)
    all_actions = []
    for key in (
        dataset.dataset.game_keys if isinstance(dataset, Subset) else dataset.game_keys
    ):
        all_actions.extend(
            dataset.dataset.hdf5_file[key]["actions"][:]
            if isinstance(dataset, Subset)
            else dataset.hdf5_file[key]["actions"][:]
        )
    class_weights = get_class_weights(all_actions, config["output_dim"]).to(device)
    criterion = CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    scaler = GradScaler(enabled=device.type == "cuda")

    best_val_f1 = 0
    early_stop_counter = 0

    for epoch in range(config["num_epochs"]):
        logging.info(f"Starting Epoch {epoch + 1}/{config['num_epochs']}")

        # Training step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # Validation step
        val_loss, accuracy, f1 = validate(model, val_loader, criterion, device)

        # Logging to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.add_scalar("F1_Score", f1, epoch)

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Early stopping and model saving based on F1-score
        if early_stop:
            if f1 > best_val_f1:
                best_val_f1 = f1
                early_stop_counter = 0
                torch.save(model.state_dict(), config["model_path"])
                logging.info(f"New best model saved to '{config['model_path']}'")
            else:
                early_stop_counter += 1
                logging.info(
                    f"No improvement in F1. Early stopping counter: {early_stop_counter}/{config['early_stop_limit']}"
                )
                if early_stop_counter >= config["early_stop_limit"]:
                    logging.info("Early stopping triggered. Training stopped.")
                    break
        else:
            torch.save(model.state_dict(), config["model_path"])
            logging.info(
                f"Model saved to '{config['model_path']}' after epoch {epoch + 1}"
            )

    if isinstance(dataset, PokerSequenceDataset):
        dataset.hdf5_file.close()
    elif isinstance(dataset, Subset):
        dataset.dataset.hdf5_file.close()

    logging.info("Training completed.")
    return best_val_f1


# ===========================
# Training Mode
# ===========================


def train_main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )

    hyperparameters_config = {
        "data_path": config.data_path,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "seq_len": config.seq_len,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "num_players": config.num_players,
        "max_positions": config.max_positions,
        "num_actions": config.num_actions,
        "num_strategies": config.num_strategies,
        "num_epochs": config.num_epochs,
        "early_stop_limit": config.early_stop_limit,
        "model_path": config.model_path,
        "dropout": config.dropout,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not Path(config.data_path).exists():
        logging.info(
            f"HDF5 file not found at {config.data_path}. Initializing a new dataset."
        )
        initialize_hdf5(
            file_path=str(config.data_path),
            state_dim=config.state_dim,
            initial_size=0,
            chunk_size=1000,
            compression="gzip",
        )

    dataset = load_dataset(
        data_path=config.data_path, max_samples=10000, shuffle_subset=True
    )

    writer = SummaryWriter(log_dir="logs")

    best_f1 = train_model(dataset, device, hyperparameters_config, writer)

    writer.close()

    logging.info(f"Best Validation F1: {best_f1:.4f}")
    logging.info("Training process completed.")


# ===========================
# Tuning Mode
# ===========================


def tuning_main():
    import sys  # Imported here to avoid confusion in imports

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("tuning.log")],
    )

    def objective(trial):
        # Suggest hyperparameters
        suggested_params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512),
            "num_heads": trial.suggest_int("num_heads", 2, 16),
            "num_layers": trial.suggest_int("num_layers", 2, 6),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "dropout": trial.suggest_uniform("dropout", 0.1, 0.5),
            "seq_len": trial.suggest_int("seq_len", 50, 200),
        }

        # Create a new Settings instance with suggested hyperparameters
        trial_config = Settings.parse_obj(
            {
                **config.dict(),
                **suggested_params,
                "model_path": f"saved_models/poker_model_trial_{trial.number}.pt",
            }
        )

        # Ensure the model directory exists
        Path(trial_config.model_path).parent.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Trial {trial.number}: Using device {device}")

        # Load dataset
        dataset = load_dataset(
            data_path=trial_config.data_path, max_samples=10000, shuffle_subset=True
        )

        # Split dataset
        best_val_f1 = train_model(
            dataset,
            device,
            trial_config,
            SummaryWriter(log_dir=f"logs/trial_{trial.number}"),
            early_stop=True,
        )

        # Clean up
        if isinstance(dataset, Subset):
            dataset.dataset.hdf5_file.close()
        else:
            dataset.hdf5_file.close()

        return best_val_f1

    # Create an Optuna study
    sampler = TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, study_name="PokerModelOptimization"
    )

    # Optimize the study
    study.optimize(
        objective, n_trials=50, timeout=7 * 24 * 60 * 60
    )  # Adjust n_trials and timeout as needed

    # Print the best hyperparameters
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (Best Validation F1): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Convert study results to DataFrame
    df = study.trials_dataframe()

    # Visualization with Seaborn
    sns.set(style="whitegrid")

    # 1. Hyperparameter Importance
    try:
        # Prepare the data
        df_importance = study.trials_dataframe().dropna().reset_index()
        param_columns = [
            col for col in df_importance.columns if col.startswith("params_")
        ]
        X = df_importance[param_columns]
        y = df_importance["value"]

        # Encode categorical variables
        for col in X.columns:
            if X[col].dtype == "object" or isinstance(X[col].iloc[0], str):
                le = LabelEncoder()
                X[col] = le.fit_transform(
                    X[col].astype(str)
                )  # Ensure all data is string type

        # Fit a Random Forest to estimate feature importances
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        feature_importances = pd.Series(
            rf.feature_importances_, index=param_columns
        ).sort_values(ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=feature_importances.values, y=feature_importances.index, palette="viridis"
        )
        plt.title("Hyperparameter Importances")
        plt.xlabel("Importance")
        plt.ylabel("Hyperparameters")
        plt.tight_layout()
        plt.savefig("hyperparameter_importance.png")
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting feature importances: {e}")

    # 2. Parallel Coordinates Plot
    try:
        plt.figure(figsize=(12, 8))
        # Select a subset of trials for clarity if too many
        sample_df = df.sample(n=min(100, len(df)), random_state=42)
        sns.lineplot(
            data=sample_df,
            x="params_num_layers",
            y="value",
            hue="params_hidden_dim",
            palette="viridis",
            legend=False,
        )
        plt.title("Parallel Coordinates Plot")
        plt.xlabel("Number of Layers")
        plt.ylabel("Validation F1 Score")
        plt.tight_layout()
        plt.savefig("parallel_coordinates.png")
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting parallel coordinates: {e}")

    # 3. Pair Plot
    try:
        param_columns = [col for col in df.columns if col.startswith("params_")]
        pairplot_df = df[param_columns + ["value"]].dropna()
        pairplot_df.columns = [
            col.replace("params_", "") for col in pairplot_df.columns
        ]
        # To prevent overcrowding, you might limit the number of points
        sns.pairplot(
            pairplot_df, hue="value", palette="viridis", plot_kws={"alpha": 0.5}
        )
        plt.suptitle("Pair Plot of Hyperparameters vs. Validation F1 Score", y=1.02)
        plt.tight_layout()
        plt.savefig("pair_plot.png")
        plt.show()
    except Exception as e:
        logging.error(f"Error in plotting pair plot: {e}")

    logging.info("Hyperparameter tuning completed.")


# ===========================
# Main Execution
# ===========================


def main():
    parser = argparse.ArgumentParser(
        description="Poker Transformer Model: Training and Hyperparameter Tuning"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")

    # Training mode
    train_parser = subparsers.add_parser(
        "train", help="Train the Poker Transformer model"
    )

    # Tuning mode
    tune_parser = subparsers.add_parser(
        "tune", help="Tune hyperparameters using Optuna"
    )

    args = parser.parse_args()

    if args.mode == "train":
        set_seed()
        train_main()
    elif args.mode == "tune":
        set_seed()
        tuning_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
