import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import os

def k_fold_cross_validation(dataset, device, model_class, model_params, criterion, optimizer_class, optimizer_params, k=5, epochs=10, batch_size=32, model_save_dir="models"):
    """
    Perform K-Fold Cross-Validation and save model weights for each fold.
    Args:
        dataset (Dataset): The PyTorch Dataset.
        device: GPU or CPU.
        model_class (nn.Module): The model class.
        model_params (dict): Parameters to initialize the model.
        criterion: Loss function.
        optimizer_class: Optimizer class (e.g., Adam).
        optimizer_params (dict): Parameters for the optimizer.
        k (int): Number of folds.
        epochs (int): Number of epochs per fold.
        batch_size (int): Batch size for training and validation.
        model_save_dir (str): Directory to save model weights.
    Returns:
        list: Results for each fold (train loss, validation loss, accuracy).
    """
    os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
        print(f"Fold {fold + 1}/{k}")

        # Split dataset into training and validation sets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Initialize model, optimizer, and criterion
        model = model_class(**model_params).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for states, actions in train_loader:
                states, actions = states.to(device), actions.to(device)
                optimizer.zero_grad()
                outputs = model(states)
                loss = criterion(outputs, actions)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation loop
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

            val_accuracy = correct / total * 100
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Save the model weights for this fold
        fold_model_path = os.path.join(model_save_dir, f"poker_model_fold{fold + 1}.pth")
        torch.save(model.state_dict(), fold_model_path)
        print(f"Model for Fold {fold + 1} saved to {fold_model_path}")

        # Record fold results
        fold_results.append({
            'fold': fold + 1,
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_accuracy
        })

    return fold_results
