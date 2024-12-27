import torch
from PokerDataset import PokerDataset


class PokerEnsemblePredictor:
    def __init__(self, model_class, model_paths, input_dim, hidden_dim, output_dim, **model_kwargs):
        """
        Initialize the ensemble predictor with multiple model paths.
        Args:
            model_class (nn.Module): The model class (e.g., PokerLinformerModel).
            model_paths (list): List of paths to the saved model weights.
            input_dim (int): Number of input features.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Number of output classes.
            **model_kwargs: Additional keyword arguments for the model (e.g., seq_len, num_heads, num_layers).
        """
        self.models = []
        for path in model_paths:
            model = model_class(
                input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **model_kwargs)
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()  # Set to evaluation mode
            self.models.append(model)

        self.action_map = {0: "fold", 1: "call", 2: "raise"}

    def predict_action(self, sample_action):
        """
        Predict the action using the ensemble of models.
        Args:
            sample_action (dict): A dictionary containing the hand state.
        Returns:
            str: Predicted action ("fold", "call", or "raise").
        """
        # Encode the state
        encoded_state = PokerDataset.encode_state(sample_action)
        input_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(
            0)  # Add batch dimension

        # Aggregate predictions from all models
        outputs = []
        for model in self.models:
            with torch.no_grad():
                outputs.append(model(input_tensor))

        # Average the outputs across the ensemble
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        predicted_action = torch.argmax(avg_output, dim=1).item()

        # Map prediction to action name
        return self.action_map[predicted_action]

    def predict_with_confidence(self, sample_action, threshold=0.7):
        """
        Predict with confidence and reject uncertain predictions.
        Args:
            sample_action (dict): A dictionary containing the hand state.
            threshold (float): Confidence threshold for making predictions.
        Returns:
            str: Predicted action ("fold", "call", "raise") or "uncertain".
        """
        # Encode the state
        encoded_state = PokerDataset.encode_state(sample_action)
        input_tensor = torch.tensor(encoded_state, dtype=torch.float32).unsqueeze(
            0)  # Add batch dimension

        # Aggregate predictions
        outputs = []
        for model in self.models:
            with torch.no_grad():
                outputs.append(model(input_tensor))
        avg_output = torch.mean(torch.stack(
            outputs), dim=0)  # Average predictions

        # Calculate confidence
        probabilities = torch.softmax(avg_output, dim=1)
        confidence, predicted_action = torch.max(probabilities, dim=1)
        if confidence.item() < threshold:
            return "uncertain"

        return self.action_map[predicted_action.item()]
