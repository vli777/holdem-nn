from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np


def get_class_weights(actions, num_classes):
    """
    Compute class weights to handle class imbalance.

    Args:
        actions (list or np.array): Array of action labels.
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: Class weights tensor.
    """
    classes = np.arange(num_classes)
    class_weights = compute_class_weight("balanced", classes=classes, y=actions)
    return torch.tensor(class_weights, dtype=torch.float32)
