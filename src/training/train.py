import os
import torch
import logging
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
from models.PokerLinformerModel import PokerLinformerModel
from PokerDataset import PokerDataset


# Function to load and validate dataset
def load_dataset(data_path):
    try:
        dataset = PokerDataset(data_path)
        logging.info(f"Dataset loaded. Total samples: {len(dataset)}")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        exit(1)


# Function to initialize the model, optimizer, and scheduler
def initialize_model(input_dim, device, config):
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
    return model, optimizer, scheduler


# Training function
def train_one_epoch(model, train_loader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0

    for states, actions, positions, player_ids, recent_actions in train_loader:
        states, actions, positions, player_ids, recent_actions = (
            states.to(device),
            actions.to(device),
            positions.to(device),
            player_ids.to(device),
            recent_actions.to(device),
        )

        with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
            policy_logits, _ = model(states, positions, player_ids, recent_actions)
            loss = criterion(policy_logits, actions)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    return running_loss / max(len(train_loader), 1)


# Validation function
def validate(model, val_loader, criterion, device):
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

    val_loss /= max(len(val_loader), 1)
    accuracy = correct / total * 100 if total > 0 else 0
    f1 = f1_score(true_labels, predicted_labels, average="weighted")

    return val_loss, accuracy, f1


# Function to perform full training
def train_and_validate(dataset, device, config):
    input_dim = len(dataset[0][0])  # Dynamically fetch input dimension

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    model, optimizer, scheduler = initialize_model(input_dim, device, config)
    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device="cuda")

    writer = SummaryWriter(log_dir="logs")

    best_val_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(config["num_epochs"]):
        # Training step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )

        # Validation step
        val_loss, accuracy, f1 = validate(model, val_loader, criterion, device)

        # Logging
        logging.info(
            f"Epoch {epoch + 1}/{config['num_epochs']}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Accuracy: {accuracy:.2f}%, "
            f"F1 Score: {f1:.4f}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy", accuracy, epoch)
        writer.add_scalar("F1_Score", f1, epoch)

        # Learning rate adjustment
        scheduler.step(val_loss)

        # Early stopping
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


# Main function
def main():
    logging.basicConfig(level=logging.INFO)

    # Configuration
    config = {
        "learning_rate": 1e-3,
        "batch_size": 32,
        "hidden_dim": 128,
        "output_dim": 3,
        "seq_len": 1,
        "num_heads": 4,
        "num_layers": 2,
        "num_epochs": 10,
        "early_stop_limit": 5,
        "model_path": "models/poker_model_full.pth",
    }

    # Paths
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    data_path = os.path.join(base_dir, "data", "texas_holdem_data.npz")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset(data_path)

    # Train and validate
    train_and_validate(dataset, device, config)


if __name__ == "__main__":
    main()
