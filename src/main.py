import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from poker_dataset import PokerDataset
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

# Load dataset and split into training/validation
dataset = PokerDataset("texas_holdem_data.npy")
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Inspect a batch
for batch in train_loader:
    states, actions = batch
    logging.info(f"States shape: {states.shape}, Actions shape: {actions.shape}")
    break


# Model Definition
class PokerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PokerModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# Hyperparameters
input_dim = states.shape[1]  # Automatically match dataset output
hidden_dim = 128
output_dim = 3  # "fold", "call", "raise"
learning_rate = 1e-3
num_epochs = 10

# Model, Loss, Optimizer
model = PokerModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for states, actions in train_loader:
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
    accuracy = correct / total * 100
    logging.info(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Train Loss: {train_loss:.4f}, "
        f"Val Loss: {val_loss:.4f}, "
        f"Accuracy: {accuracy:.2f}%"
    )
