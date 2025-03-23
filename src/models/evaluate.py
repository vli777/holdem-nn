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
from typing import List, Dict, Any, Tuple
import seaborn as sns
from tqdm import tqdm

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


def evaluate_model(
    model: torch.nn.Module,
    test_data: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, float]:
    """
    Evaluate the model on test data using multiple metrics.

    Args:
        model (nn.Module): Trained PokerTransformerModel.
        test_data (list): List of test game histories.
        device (torch.device): Device to run the model on.
        batch_size (int): Batch size for evaluation.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    model.eval()
    metrics = {}

    # Initialize lists to store predictions and targets
    all_predictions = []
    all_targets = []
    all_values = []
    all_value_targets = []
    all_confidences = []

    # Process data in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i:i + batch_size]
        
        # Prepare batch data
        states = torch.stack([torch.tensor(d["state"]) for d in batch]).to(device)
        actions = torch.stack([torch.tensor(d["action"]) for d in batch]).to(device)
        player_ids = torch.stack([torch.tensor(d["player_id"]) for d in batch]).to(device)
        positions = torch.stack([torch.tensor(d["position"]) for d in batch]).to(device)
        recent_actions = torch.stack([torch.tensor(d["recent_action"]) for d in batch]).to(device)
        strategies = torch.stack([torch.tensor(d.get("strategy", 2)) for d in batch]).to(device)
        bluffing_probs = torch.stack([torch.tensor(d.get("bluffing_probability", 0.0)) for d in batch]).to(device)
        value_targets = torch.stack([torch.tensor(d.get("value_target", 0.0)) for d in batch]).to(device)

        # Create attention mask
        mask = torch.ones_like(actions, dtype=torch.bool).to(device)

        with torch.no_grad():
            policy_logits, value_pred = model(
                x=states,
                player_ids=player_ids,
                positions=positions,
                recent_actions=recent_actions,
                strategies=strategies,
                bluffing_probabilities=bluffing_probs,
                mask=mask
            )

        # Get predictions
        predictions = torch.argmax(policy_logits, dim=-1)
        confidences = torch.softmax(policy_logits, dim=-1)

        # Store results
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(actions.cpu().numpy())
        all_values.extend(value_pred.cpu().numpy())
        all_value_targets.extend(value_targets.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_values = np.array(all_values)
    all_value_targets = np.array(all_value_targets)
    all_confidences = np.array(all_confidences)

    # Calculate metrics
    metrics["accuracy"] = np.mean(all_predictions == all_targets)
    metrics["value_mse"] = np.mean((all_values - all_value_targets) ** 2)
    metrics["value_mae"] = np.mean(np.abs(all_values - all_value_targets))
    
    # Calculate per-action metrics
    for action_idx, action_name in enumerate(["fold", "call", "raise"]):
        action_mask = all_targets == action_idx
        if np.any(action_mask):
            metrics[f"{action_name}_accuracy"] = np.mean(
                all_predictions[action_mask] == all_targets[action_mask]
            )
            metrics[f"{action_name}_confidence"] = np.mean(
                all_confidences[action_mask, action_idx]
            )

    # Calculate ROC curves for each action
    for action_idx, action_name in enumerate(["fold", "call", "raise"]):
        fpr, tpr, _ = roc_curve(
            (all_targets == action_idx).astype(int),
            all_confidences[:, action_idx]
        )
        metrics[f"{action_name}_auc"] = auc(fpr, tpr)

    return metrics


def plot_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_path: str = "confusion_matrix.png"
) -> None:
    """
    Plot and save confusion matrix.

    Args:
        predictions (np.ndarray): Model predictions.
        targets (np.ndarray): Ground truth targets.
        save_path (str): Path to save the plot.
    """
    cm = confusion_matrix(targets, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["fold", "call", "raise"],
        yticklabels=["fold", "call", "raise"]
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(
    confidences: np.ndarray,
    targets: np.ndarray,
    save_path: str = "roc_curves.png"
) -> None:
    """
    Plot and save ROC curves for each action.

    Args:
        confidences (np.ndarray): Model confidence scores.
        targets (np.ndarray): Ground truth targets.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(10, 8))
    for action_idx, action_name in enumerate(["fold", "call", "raise"]):
        fpr, tpr, _ = roc_curve(
            (targets == action_idx).astype(int),
            confidences[:, action_idx]
        )
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{action_name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Each Action")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()


def evaluate_on_specific_scenarios(
    model: torch.nn.Module,
    scenario_data: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on specific poker scenarios (e.g., pre-flop decisions,
    bluffing situations, etc.).

    Args:
        model (nn.Module): Trained PokerTransformerModel.
        scenario_data (list): List of scenario-specific game histories.
        device (torch.device): Device to run the model on.

    Returns:
        Dict[str, float]: Dictionary of scenario-specific metrics.
    """
    model.eval()
    scenario_metrics = {}

    # Group scenarios by type
    scenario_types = {}
    for scenario in scenario_data:
        scenario_type = scenario.get("scenario_type", "unknown")
        if scenario_type not in scenario_types:
            scenario_types[scenario_type] = []
        scenario_types[scenario_type].append(scenario)

    # Evaluate each scenario type
    for scenario_type, scenarios in scenario_types.items():
        predictions = []
        targets = []
        values = []
        value_targets = []

        for scenario in scenarios:
            # Prepare input data
            states = torch.tensor(scenario["state"]).unsqueeze(0).to(device)
            actions = torch.tensor(scenario["action"]).unsqueeze(0).to(device)
            player_ids = torch.tensor(scenario["player_id"]).unsqueeze(0).to(device)
            positions = torch.tensor(scenario["position"]).unsqueeze(0).to(device)
            recent_actions = torch.tensor(scenario["recent_action"]).unsqueeze(0).to(device)
            strategies = torch.tensor(scenario.get("strategy", 2)).unsqueeze(0).to(device)
            bluffing_probs = torch.tensor(scenario.get("bluffing_probability", 0.0)).unsqueeze(0).to(device)
            value_target = torch.tensor(scenario.get("value_target", 0.0)).unsqueeze(0).to(device)

            # Create attention mask
            mask = torch.ones_like(actions, dtype=torch.bool).to(device)

            with torch.no_grad():
                policy_logits, value_pred = model(
                    x=states,
                    player_ids=player_ids,
                    positions=positions,
                    recent_actions=recent_actions,
                    strategies=strategies,
                    bluffing_probabilities=bluffing_probs,
                    mask=mask
                )

            predictions.append(torch.argmax(policy_logits, dim=-1).item())
            targets.append(actions.item())
            values.append(value_pred.item())
            value_targets.append(value_target.item())

        # Calculate metrics for this scenario type
        scenario_metrics[f"{scenario_type}_accuracy"] = np.mean(
            np.array(predictions) == np.array(targets)
        )
        scenario_metrics[f"{scenario_type}_value_mse"] = np.mean(
            (np.array(values) - np.array(value_targets)) ** 2
        )
        scenario_metrics[f"{scenario_type}_value_mae"] = np.mean(
            np.abs(np.array(values) - np.array(value_targets))
        )

    return scenario_metrics


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
    metrics = evaluate_model(model, test_loader, device)

    # Close the HDF5 file
    dataset.close()

    # Plot confusion matrix
    plot_confusion_matrix(metrics["predictions"], metrics["targets"])

    # Plot ROC curves
    plot_roc_curves(metrics["confidences"], metrics["targets"])
