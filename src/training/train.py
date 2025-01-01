import random
import torch
import logging
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from models.PokerLinformerModel import PokerLinformerModel
from PokerDataset import PokerDataset
from config import config
from training.hdf5 import initialize_hdf5


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
        dataset = PokerDataset(data_path)
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
    Initialize the PokerLinformerModel, optimizer, and scheduler.

    Args:
        input_dim (int): Dimensionality of the input features.
        device (torch.device): Device to load the model on.
        config (dict): Configuration dictionary with hyperparameters.

    Returns:
        tuple: (model, optimizer, scheduler)
    """
    model_dir = config["model_path"].parent
    model_dir.mkdir(parents=True, exist_ok=True)

    model = PokerLinformerModel(
        input_dim=input_dim,
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        seq_len=config["seq_len"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
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

    for batch_idx, (
        states,
        actions,
        positions,
        player_ids,
        recent_actions,
    ) in enumerate(train_loader):
        states, actions, positions, player_ids, recent_actions = (
            states.to(device),
            actions.to(device),
            positions.to(device),
            player_ids.to(device),
            recent_actions.to(device),
        )

        with torch.amp.autocast(enabled=device.type == "cuda"):
            policy_logits, _ = model(states, positions, player_ids, recent_actions)
            loss = criterion(policy_logits, actions)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

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
        for states, actions, positions, player_ids, recent_actions in val_loader:
            states, actions, positions, player_ids, recent_actions = (
                states.to(device),
                actions.to(device),
                positions.to(device),
                player_ids.to(device),
                recent_actions.to(device),
            )

            policy_logits, _ = model(states, positions, player_ids, recent_actions)
            loss = criterion(policy_logits, actions)
            val_loss += loss.item()

            # Accuracy
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


def train_model(dataset, device, config):
    """
    Perform full training of the model.

    Args:
        dataset (Dataset): The dataset to train on.
        device (torch.device): Device to perform training on.
        config (dict): Configuration dictionary with hyperparameters.
    """
    input_dim = len(dataset[0][0])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=4,
    )

    for states, actions, positions, player_ids, recent_actions in train_loader:
        print("States:", states.mean().item(), states.std().item())
        print("Actions:", actions.unique())
        print("Positions:", positions.unique())
        print("Player IDs:", player_ids.unique())
        print("Recent Actions:", recent_actions.unique())
        break
    exit()

    model, optimizer, scheduler = initialize_model(input_dim, device, config)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    writer = SummaryWriter(log_dir="logs")

    best_val_loss = float("inf")
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

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), config["model_path"])
            logging.info(f"New best model saved to '{config['model_path']}'")
        else:
            early_stop_counter += 1
            logging.info(
                f"No improvement. Early stopping counter: {early_stop_counter}/{config['early_stop_limit']}"
            )
            if early_stop_counter >= config["early_stop_limit"]:
                logging.info("Early stopping triggered. Training stopped.")
                break

    writer.close()
    logging.info("Training completed.")


# Main function
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
    )

    hyperparameters_config = {
        "data_path": config.data_path,
        "state_dim": config.state_dim,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "seq_len": config.seq_len,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "num_epochs": config.num_epochs,
        "early_stop_limit": config.early_stop_limit,
        "model_path": config.model_path,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not config["data_path"].exists():
        logging.info(
            f"HDF5 file not found at {config['data_path']}. Initializing a new dataset."
        )
        initialize_hdf5(
            file_path=str(config["data_path"]),
            state_dim=config["state_dim"],
            initial_size=0,
            chunk_size=1000,
            compression="gzip",
        )

    dataset = load_dataset(
        data_path=config["data_path"], max_samples=10000, shuffle_subset=True
    )

    train_model(dataset, device, hyperparameters_config)

    if isinstance(dataset, PokerDataset):
        dataset.close()
    elif isinstance(dataset, Subset):
        dataset.dataset.close()

    logging.info("Training completed.")


if __name__ == "__main__":
    main()
