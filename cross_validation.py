import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np

def k_fold_cross_validation(dataset, model_class, model_params, criterion, optimizer_class, optimizer_params, k=5, epochs=10, batch_size=32):
    """
    Perform K-Fold Cross-Validation.
    Args:
        dataset (Dataset): The PyTorch Dataset.
        model_class (nn.Module): The model class.
        model_params (dict): Parameters to initialize the model.
        criterion: Loss function.
        optimizer_class: Optimizer class (e.g., Adam).
        optimizer_params (dict): Parameters for the optimizer.
        k (int): Number of folds.
        epochs (int): Number of epochs per fold.
        batch_size (int): Batch size for training and validation.
    Returns:
        list: Average training and validation losses for each fold.
    """
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
        model = model_class(**model_params)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for states, actions in train_loader:
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
                    outputs = model(states)
                    loss = criterion(outputs, actions)
                    val_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    total += actions.size(0)
                    correct += (predicted == actions).sum().item()

            val_accuracy = correct / total * 100
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Record fold results
        fold_results.append({
            'train_loss': train_loss / len(train_loader),
            'val_loss': val_loss / len(val_loader),
            'val_accuracy': val_accuracy
        })

    # Average results
    avg_train_loss = np.mean([result['train_loss'] for result in fold_results])
    avg_val_loss = np.mean([result['val_loss'] for result in fold_results])
    avg_val_accuracy = np.mean([result['val_accuracy'] for result in fold_results])

    print(f"Average Results - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.2f}%")
    return fold_results
