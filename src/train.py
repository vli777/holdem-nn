import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from . import PokerDataset, PokerModel
import logging

logging.basicConfig(level=logging.DEBUG)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Load dataset and split into training/validation
try:
    dataset = PokerDataset("data/texas_holdem_data.npy")
    logging.info(f"Dataset loaded. Total samples: {len(dataset)}")
    if len(dataset) == 0:
        logging.error("Dataset is empty! Exiting...")
        exit(1)
except Exception as e:
    logging.error(f"Error during dataset initialization: {e}")
    exit(1)

# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
try:
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
except Exception as e:
    logging.error(f"Error during dataset splitting: {e}")
    exit(1)

# Data loaders
try:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
except Exception as e:
    logging.error(f"Error initializing data loaders: {e}")
    exit(1)

# Inspect a batch
try:
    for batch in train_loader:
        states, actions = batch
        logging.info(f"First batch - States shape: {states.shape}, Actions shape: {actions.shape}")
        break
except Exception as e:
    logging.error(f"Error during batch inspection: {e}")
    exit(1)

# Hyperparameters
input_dim = states.shape[1]
hidden_dim = 128
output_dim = 3  # "fold", "call", "raise"
learning_rate = 1e-3
num_epochs = 10

# Model, Loss, Optimizer
model = PokerModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
logging.info("Starting training...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for states, actions in train_loader:
        states, actions = states.to(device), actions.to(device)
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

        optimizer.step()
        running_loss += loss.item()

    # Validation Loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for states, actions in val_loader:
            states, actions = states.to(device), actions.to(device)
            outputs = model(states)
            loss = criterion(outputs, actions)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += actions.size(0)
            correct += (predicted == actions).sum().item()

    # Logging
    train_loss = running_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    accuracy = (correct / total * 100) if total > 0 else 0
    logging.info(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%"
    )

# Save 
torch.save(model.state_dict(), "models/poker_model.pth")
