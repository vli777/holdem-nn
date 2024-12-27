import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
from PokerDataset import PokerDataset
from PokerLinformerModel import PokerLinformerModel
from cross_validation import k_fold_cross_validation

logging.basicConfig(level=logging.INFO)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# Paths
DATA_PATH = "data/texas_holdem_data.npy"
MODEL_SAVE_DIR = "models"

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
    seq_len=1
).to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss()

# Train the model
model.train()
running_loss = 0.0
for epoch in range(num_epochs):
    for states, actions in DataLoader(dataset, batch_size=32, shuffle=True):
        states, actions = states.to(device), actions.to(device)
        optimizer.zero_grad()
        outputs = model(states)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataset)
    logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "models/poker_model.pth")
logging.info("Final model trained on full dataset saved as 'poker_model.pth'")
