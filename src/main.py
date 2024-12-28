import logging
from PokerLinformerModel import PokerLinformerModel
from PokerPredictor import PokerPredictor
from PokerEnsemblePredictor import PokerEnsemblePredictor
from simulate import randomize_sample_action, play_out_game
import os
from glob import glob
from PokerDataset import PokerDataset

# Logging setup
logging.basicConfig(level=logging.INFO)

# Path to model directory
MODEL_DIR = "models"
FULL_MODEL_PATH = os.path.join(MODEL_DIR, "poker_model.pth")


def main():
    # Check for the full model
    if not os.path.exists(FULL_MODEL_PATH):
        logging.error(
            f"Full model not found at {FULL_MODEL_PATH}. Ensure the model is trained and saved.")
        exit(1)

    # Model parameters
    sample_encoded_state = PokerDataset.encode_state(randomize_sample_action())
    input_dim = len(sample_encoded_state)
    hidden_dim = 128
    output_dim = 3
    num_heads = 4
    num_layers = 2
    seq_len = 1

    # Initialize the single predictor
    predictor = PokerPredictor(
        model_class=PokerLinformerModel,
        model_path=FULL_MODEL_PATH,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # Randomize a sample action
    sample_action = randomize_sample_action()

    # Display the initial state
    logging.info("--- Single Model Prediction ---")
    predictor.display_hand(sample_action)
    logging.info(
        f"Predicted Action: {
            predictor.predict_action(sample_action)}")

    # Dynamically find fold models
    fold_model_paths = glob(os.path.join(MODEL_DIR, "*fold*.pth"))
    if not fold_model_paths:
        logging.warning(
            "No fold models found. Proceeding with the full model only.")
        play_out_game(predictor, sample_action, num_players=6)
        return

    # Include the full model in the ensemble
    model_paths = [FULL_MODEL_PATH] + fold_model_paths
    logging.info(
        f"Found {
            len(model_paths)} models for ensemble: {model_paths}")

    # Initialize the ensemble predictor
    ensemble_predictor = PokerEnsemblePredictor(
        model_class=PokerLinformerModel,
        model_paths=model_paths,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    # Perform ensemble predictions
    logging.info("\n--- Ensemble Model Prediction ---")
    action_with_confidence = ensemble_predictor.predict_with_confidence(
        sample_action, threshold=0.8)
    if action_with_confidence == "uncertain":
        logging.warning("Prediction confidence is too low. Action: uncertain.")
    else:
        logging.info(
            f"Predicted Action with Confidence: {action_with_confidence}")

    # Simulate and play out the game
    logging.info("\n--- Simulated Game ---")
    play_out_game(ensemble_predictor, sample_action, num_players=6)


if __name__ == "__main__":
    main()
