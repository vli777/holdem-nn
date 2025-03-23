from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from config import config
from PokerSequenceDataset import PokerSequenceDataset, poker_collate_fn
from models.PokerTransformerModel import PokerTransformerModel


def plot_confusion_matrix(
    true_labels, predicted_labels, classes=["fold", "call", "raise"]
):
    """
    Plot the confusion matrix.

    Args:
        true_labels (list or np.array): True action labels.
        predicted_labels (list or np.array): Predicted action labels.
        classes (list): List of class names.

    Returns:
        None
    """
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def plot_multiclass_roc(
    true_labels, predicted_probs, classes=["fold", "call", "raise"]
):
    """
    Plot ROC curves for each class in a multi-class setting.

    Args:
        true_labels (list or np.array): True action labels.
        predicted_probs (np.array): Predicted probabilities for each class.
        classes (list): List of class names.

    Returns:
        None
    """
    # Binarize the output
    y_true = label_binarize(true_labels, classes=[0, 1, 2])
    n_classes = y_true.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predicted_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"ROC curve of class {classes[i]} (area = {roc_auc[i]:0.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--")  # Diagonal
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.show()


def plot_multiclass_precision_recall(
    true_labels, predicted_probs, classes=["fold", "call", "raise"]
):
    """
    Plot Precision-Recall curves for each class in a multi-class setting.

    Args:
        true_labels (list or np.array): True action labels.
        predicted_probs (np.array): Predicted probabilities for each class.
        classes (list): List of class names.

    Returns:
        None
    """
    # Binarize the output
    y_true = label_binarize(true_labels, classes=[0, 1, 2])
    n_classes = y_true.shape[1]

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_true[:, i], predicted_probs[:, i]
        )
        average_precision[i] = average_precision_score(
            y_true[:, i], predicted_probs[:, i]
        )

    # Plot all Precision-Recall curves
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(
            recall[i],
            precision[i],
            label=f"Precision-Recall curve of class {classes[i]} (AP = {average_precision[i]:0.2f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.show()


def load_trained_model(config, device):
    """
    Load the trained model from disk.

    Args:
        config (Config): Configuration object.
        device (torch.device): Device to load the model on.

    Returns:
        nn.Module: Loaded model.
    """
    model = PokerTransformerModel(
        input_dim=config.hidden_dim,  # Adjust based on your encode_state
        hidden_dim=config.hidden_dim,
        output_dim=config.output_dim,
        seq_len=config.seq_len,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_players=config.num_players,
        max_positions=config.max_positions,
        num_actions=config.num_actions,
        num_strategies=config.num_strategies,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(torch.load(config.model_path))
    model.eval()
    return model


def evaluate_model(model, test_loader, device, config):
    """
    Evaluate the model and plot confusion matrix.
    """
    predicted_labels = []
    true_labels = []
    predicted_probs = []

    with torch.no_grad():
        for batch in test_loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            player_ids = batch["player_ids"].to(device)
            positions = batch["positions"].to(device)
            recent_actions = batch["recent_actions"].to(device)
            strategies = batch["strategies"].to(device)
            bluffing_probabilities = batch["bluffing_probabilities"].to(device)
            mask = batch["mask"].to(device)

            policy_logits = model(
                states,
                player_ids,
                positions,
                recent_actions,
                strategies,
                bluffing_probabilities,
                mask=mask,
            )  # [batch_size, seq_len, output_dim]

            # Get predictions
            probs = torch.softmax(policy_logits, dim=-1)
            _, predicted = torch.max(policy_logits, dim=-1)
            predicted = predicted.view(-1).cpu().numpy()
            actions = actions.view(-1).cpu().numpy()
            probs = probs.view(-1, config.output_dim).cpu().numpy()

            # Mask out padding
            valid_indices = actions != -1
            predicted = predicted[valid_indices]
            actions = actions[valid_indices]
            probs = probs[valid_indices]

            predicted_labels.extend(predicted)
            true_labels.extend(actions)
            predicted_probs.extend(probs)

    # Calculate metrics
    accuracy = (np.array(predicted_labels) == np.array(true_labels)).mean() * 100
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    cm = confusion_matrix(true_labels, predicted_labels, labels=[0, 1, 2])

    # Print metrics
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Test F1 Score: {f1:.4f}")

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["fold", "call", "raise"]
    )
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix on Test Set")
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Replace this with your actual dataset
    dataset = PokerSequenceDataset(
        hdf5_path=config.data_path, max_seq_len=config.max_seq_len
    )
    test_dataset = torch.utils.data.Subset(
        dataset, indices=range(int(0.8 * len(dataset)), len(dataset))
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=4,
        collate_fn=poker_collate_fn,
    )

    # Load the model
    model = load_trained_model(config, device)

    # Evaluate the model
    evaluate_model(model, test_loader, device, config)

    # Close the HDF5 file
    dataset.close()
