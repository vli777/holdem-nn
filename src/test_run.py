import logging
from mc import randomize_sample_action
from models.PokerLinformerModel import PokerLinformerModel
from predictors.PokerPredictor import PokerPredictor
from predictors.PokerEnsemblePredictor import PokerEnsemblePredictor
from simulate_play import play_out_game
from utils import encode_state
from pathlib import Path

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"
FULL_MODEL_PATH = MODEL_DIR / "poker_model_full.pth"

def initialize_models(
    full_model_path,
    fold_model_paths,
    input_dim,
    hidden_dim,
    output_dim,
    seq_len,
    num_heads,
    num_layers,
):
    """
    Initialize single and ensemble predictors.
    """
    predictor = PokerPredictor(
        model_class=PokerLinformerModel,
        model_path=full_model_path,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
    )
    logging.info("Single predictor initialized.")

    ensemble_predictor = None
    if fold_model_paths:
        ensemble_predictor = PokerEnsemblePredictor(
            model_class=PokerLinformerModel,
            model_paths=fold_model_paths,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        logging.info(f"Ensemble predictor initialized with {len(fold_model_paths)} models.")

    return predictor, ensemble_predictor

def main():
    """
    Main execution function for poker prediction and simulation.
    """
    if not FULL_MODEL_PATH.exists():
        logging.error(f"Full model not found at {FULL_MODEL_PATH}. Ensure the model is trained and saved.")
        exit(1)

    # Generate sample action
    original_sample_action = randomize_sample_action()
    logging.debug(f"Generated sample action: {original_sample_action}")

    # Encode state
    sample_encoded_state = encode_state(**original_sample_action)
    input_dim = len(sample_encoded_state)
    hidden_dim = 128
    output_dim = 3
    num_heads = 4
    num_layers = 2
    seq_len = 1

    # Initialize predictors
    fold_model_paths = list(MODEL_DIR.glob("*fold*.pth"))
    predictor, ensemble_predictor = initialize_models(
        FULL_MODEL_PATH, fold_model_paths, input_dim, hidden_dim, output_dim, seq_len, num_heads, num_layers
    )

    # Predict actions
    single_predicted_action = predictor.predict_action(original_sample_action)
    ensemble_predicted_action = (
        ensemble_predictor.predict_with_confidence(original_sample_action, threshold=0.8)
        if ensemble_predictor
        else "uncertain"
    )

    # Log results
    logging.info("--- Predictor Results ---")
    logging.info(f"Single Predictor Action: {single_predicted_action}")
    logging.info(f"Ensemble Predictor Action: {ensemble_predicted_action}")

    # Determine chosen predictor
    chosen_predictor = (
        "Single Predictor"
        if ensemble_predicted_action == "uncertain" or not ensemble_predictor
        else "Ensemble Predictor"
    )
    logging.info(f"Chosen Predictor for Game: {chosen_predictor}")

    # Play out game
    chosen_action = (
        original_sample_action if chosen_predictor == "Single Predictor" else original_sample_action
    )
    play_out_game(
        predictor if chosen_predictor == "Single Predictor" else ensemble_predictor,
        chosen_action,
        num_players=6,
    )

if __name__ == "__main__":
    main()
