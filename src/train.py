import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import logging
from PokerDataset import PokerDataset
from PokerLinformerModel import PokerLinformerModel
from cross_validation import k_fold_cross_validation
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Paths
DATA_PATH = "data/texas_holdem_data.npy"
MODEL_SAVE_DIR = "models"
FULL_MODEL_PATH = f"{MODEL_SAVE_DIR}/poker_model_full.pth"

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 1e-3
num_heads = 4
num_layers = 2  # Number of transformer layers
hidden_dim = 128
output_dim = 3  # "fold", "call", "raise"
seq_len = 1
k_folds = 5  # Number of folds for cross-validation

# Load dataset
try:
    dataset = PokerDataset(DATA_PATH)
    logging.info(f"Dataset loaded. Total samples: {len(dataset)}")
    if len(dataset) == 0:
        logging.error("Dataset is empty! Exiting...")
        exit(1)
except Exception as e:
    logging.error(f"Error during dataset initialization: {e}")
    exit(1)

input_dim = len(dataset[0][0])  # Dynamically fetch input dimension

# Perform K-Fold Cross-Validation
logging.info(f"Starting K-Fold Cross-Validation with {k_folds} folds...")
results = k_fold_cross_validation(
    dataset=dataset,
    device=device,
    model_class=PokerLinformerModel,
    model_params={
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'seq_len': seq_len,
    },
    criterion=torch.nn.CrossEntropyLoss(),
    optimizer_class=torch.optim.Adam,
    optimizer_params={'lr': learning_rate},
    k=k_folds,
    epochs=num_epochs,
    batch_size=batch_size,
    model_save_dir=MODEL_SAVE_DIR
)

# Log average results
avg_train_loss = sum(r['train_loss'] for r in results) / k_folds
avg_val_loss = sum(r['val_loss'] for r in results) / k_folds
avg_accuracy = sum(r['val_accuracy'] for r in results) / k_folds

logging.info("Cross-Validation Complete!")
logging.info(f"Average Train Loss: {avg_train_loss:.4f}")
logging.info(f"Average Validation Loss: {avg_val_loss:.4f}")
logging.info(f"Average Accuracy: {avg_accuracy:.2f}%")

# Train on full dataset after cross-validation
logging.info("Training on the full dataset...")

# Reinitialize the model
model = PokerLinformerModel(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    seq_len=seq_len
).to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Split dataset into training and validation sets for final training
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=4)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5)
scaler = torch.cuda.amp.GradScaler()

writer = SummaryWriter(log_dir="logs")

# Final Training Loop
for epoch in range(num_epochs):
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

        with torch.amp.autocast(device_type='cuda', enabled=device.type == 'cuda'):
            policy_logits, _ = model(
                states, positions, player_ids, recent_actions)
            loss = criterion(policy_logits, actions)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for states, actions, positions, player_ids, recent_actions in val_loader:
            states, actions, positions, player_ids, recent_actions = (
                states.to(device),
                actions.to(device),
                positions.to(device),
                player_ids.to(device),
                recent_actions.to(device),
            )

            policy_logits, _ = model(
                states, positions, player_ids, recent_actions)
            loss = criterion(policy_logits, actions)
            val_loss += loss.item()

            # Accuracy
            _, predicted = torch.max(policy_logits, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

    # Log epoch metrics
    train_loss = running_loss / max(len(train_loader), 1)
    val_loss = val_loss / max(len(val_loader), 1)
    accuracy = correct / total * 100 if total > 0 else 0

    logging.info(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%"
    )

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy", accuracy, epoch)

    scheduler.step(val_loss)  # Adjust LR based on validation loss

writer.close()

# Save the model
torch.save(model.state_dict(), FULL_MODEL_PATH)
logging.info(
    f"Final model trained on full dataset saved as '{FULL_MODEL_PATH}'")
