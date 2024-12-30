import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import os
import logging


def run_epoch(loader, model, criterion, device, optimizer=None):
    """
    Runs one epoch for either training or validation.
    Args:
        loader: DataLoader for training or validation data.
        model: The model to train/evaluate.
        criterion: Loss function.
        device: The device to use (CPU/GPU).
        optimizer: Optimizer (if None, runs validation instead of training).
    Returns:
        Tuple of (average_loss, total_correct, total_samples)
    """
    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for states, actions, positions, player_ids, recent_actions in loader:
        states, actions, positions, player_ids, recent_actions = (
            states.to(device),
            actions.to(device),
            positions.to(device),
            player_ids.to(device),
            recent_actions.to(device),
        )

        if states.ndim == 2:
            states = states.unsqueeze(1)

        # Forward pass
        with torch.set_grad_enabled(is_training):
            policy_logits, _ = model(states, positions, player_ids, recent_actions)
            loss = criterion(policy_logits, actions)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(policy_logits, 1)
            correct += (predicted == actions).sum().item()
            total += actions.size(0)

            # Backpropagation for training
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else float("inf")
    accuracy = correct / total * 100 if total > 0 else 0.0
    return avg_loss, accuracy


def k_fold_cross_validation(
    dataset,
    device,
    model_class,
    model_params,
    criterion,
    optimizer_class,
    optimizer_params,
    k=5,
    epochs=10,
    batch_size=32,
    model_save_dir="models",
):
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
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    fold_results = []
    patience_limit = 3  # Set a threshold for stopping

    for fold, (train_indices, val_indices) in enumerate(
        # (fold, call, raise)
        kfold.split(dataset.data, [d[1] for d in dataset.data])
    ):
        logging.info(f"Fold {fold + 1}/{k}")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_class(**model_params).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)

        # Reset early stopping vars
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            # Training loop
            train_loss, train_accuracy = run_epoch(
                train_loader, model, criterion, device, optimizer
            )

            # Validation loop
            val_loss, val_accuracy = run_epoch(val_loader, model, criterion, device)

            # Early stopping and logging
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(
                    model.state_dict(),
                    f"{model_save_dir}/best_model_fold{fold + 1}.pth",
                )
                logging.info(f"New best model saved for Fold {fold + 1}")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logging.info(f"Early stopping at epoch {epoch + 1}")
                    break

            logging.info(
                f"Fold {fold + 1} Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, "
                f"Patience Counter: {patience_counter}/{patience_limit}"
            )

        # Record fold results
        fold_results.append(
            {
                "fold": fold + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )

    return fold_results
