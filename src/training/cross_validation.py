import os
import torch
import logging
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter


def k_fold_cross_validation(
    dataset,
    device,
    model_class,
    model_params,
    optimizer_class,
    optimizer_params,
    criterion_class,
    k=5,
    epochs=10,
    batch_size=32,
    model_save_dir="models",
    patience_limit=3,
    collate_fn=None,
):
    """
    Perform a hybrid K-Fold Cross-Validation with enhanced features.

    Args:
        dataset (Dataset): The dataset to use.
        device (torch.device): Device for computation.
        model_class (nn.Module): Model class.
        model_params (dict): Model initialization parameters.
        optimizer_class: Optimizer class (e.g., Adam).
        optimizer_params (dict): Optimizer parameters.
        criterion_class: Loss function class.
        k (int): Number of folds.
        epochs (int): Number of epochs per fold.
        batch_size (int): Batch size for DataLoader.
        model_save_dir (str): Directory to save model weights.
        patience_limit (int): Early stopping patience.
        collate_fn (callable): Optional custom collate function.
    """
    os.makedirs(model_save_dir, exist_ok=True)

    # Extract labels for stratified splitting
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_indices, val_indices) in enumerate(skf.split(range(len(dataset)), labels)):
        logging.info(f"Fold {fold + 1}/{k}")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = model_class(**model_params).to(device)
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        criterion = criterion_class()
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

        writer = SummaryWriter(log_dir=f"logs/fold_{fold + 1}")

        best_val_f1 = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            logging.info(f"Fold {fold + 1} - Epoch {epoch + 1}/{epochs}")

            # Training loop
            train_loss, train_accuracy = run_epoch(
                train_loader, model, criterion, device, optimizer, scaler
            )

            # Validation loop
            val_loss, val_accuracy, val_f1 = validate(
                val_loader, model, criterion, device
            )

            # Log metrics
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)
            writer.add_scalar("F1_Score/Validation", val_f1, epoch)

            # Early stopping based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), f"{model_save_dir}/best_model_fold{fold + 1}.pt")
                logging.info(f"Fold {fold + 1} - New best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    logging.info(f"Fold {fold + 1} - Early stopping triggered.")
                    break

            logging.info(
                f"Fold {fold + 1} Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, "
                f"Validation F1 Score: {val_f1:.4f}, Patience Counter: {patience_counter}/{patience_limit}"
            )

        writer.close()

    logging.info("Cross-validation completed.")


def run_epoch(loader, model, criterion, device, optimizer=None, scaler=None):
    """
    Runs one epoch for training or validation.
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        inputs, targets = batch[0].to(device), batch[1].to(device)
        with torch.set_grad_enabled(is_training):
            with torch.amp.autocast(enabled=scaler is not None):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            if is_training:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

    avg_loss = total_loss / len(loader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def validate(loader, model, criterion, device):
    """
    Validate the model and calculate F1 score.
    """
    from sklearn.metrics import f1_score

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = sum([p == t for p, t in zip(all_preds, all_targets)]) / len(all_targets) * 100
    f1 = f1_score(all_targets, all_preds, average="weighted")

    return avg_loss, accuracy, f1
