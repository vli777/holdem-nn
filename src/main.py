from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from treys import Evaluator, Card
from models.PokerLinformerModel import PokerLinformerModel
from utils import calculate_pot_odds
from .predictors.PokerPredictor import PokerPredictor
from .predictors.PokerEnsemblePredictor import PokerEnsemblePredictor
from pathlib import Path
import logging

evaluator = Evaluator()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("uvicorn.error")  # Use Uvicorn's logger

# Initialize FastAPI app
app = FastAPI()
logger.info("Starting Poker Predictor API")

# Add CORS middleware if necessary
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"
FULL_MODEL_PATH = MODEL_DIR / "poker_model_full.pth"
ENSEMBLE_MODEL_PATHS = [MODEL_DIR / f"best_model_fold{i}.pth" for i in range(1, 6)]

# Model parameters
input_dim = 2
hidden_dim = 128
output_dim = 3
num_heads = 4
num_layers = 2
seq_len = 1

# Initialize predictors
predictors = {
    "standard": PokerPredictor(
        model_class=PokerLinformerModel,
        model_path=FULL_MODEL_PATH,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
    ),
    "ensemble": PokerEnsemblePredictor(
        model_class=PokerLinformerModel,
        model_paths=ENSEMBLE_MODEL_PATHS,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        seq_len=seq_len,
        num_heads=num_heads,
        num_layers=num_layers,
    ),
}


# Input data model
class HandState(BaseModel):
    hole_cards: List[str] = Field(
        ..., description="Two hole cards in standard poker notation, e.g., ['Ah', 'Kd']"
    )
    community_cards: List[str] = Field(
        ...,
        description="Community cards in standard poker notation, e.g., ['Qs', 'Jd']",
    )


# Output response model
class PredictionResponse(BaseModel):
    predicted_action: str = Field(
        ..., description="Predicted action: 'fold', 'call', or 'raise'"
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_action(
    hand_state: HandState,
    predictor_type: Optional[str] = Query(
        "standard",
        enum=["standard", "ensemble"],
        description="standard or ensemble (default standard)",
    ),
    current_pot: Optional[float] = Query(
        100.0, description="Current pot size (default 100)"
    ),
    bet_amount: Optional[float] = Query(
        10.0, description="Current bet amount (default 10)"
    ),
):
    """
    Predict the poker action for a given hand state using either PokerPredictor or PokerEnsemblePredictor.
    """
    try:
        # Convert input cards to Treys format
        logging.info("Converting input cards to Treys format...")
        hole_cards = [Card.new(card) for card in hand_state.hole_cards]
        community_cards = [Card.new(card) for card in hand_state.community_cards]

        # Calculate hand strength using Treys Evaluator
        logging.info("Calculating hand strength using Treys...")
        hand_strength = evaluator.evaluate(hole_cards, community_cards)
        logging.info(f"Hand strength calculated: {hand_strength}")

        # Calculate pot odds
        logging.info("Calculating pot odds...")
        pot_odds = calculate_pot_odds(current_pot, bet_amount)
        logging.info(f"Pot odds calculated: {pot_odds}")

        # Prepare sample action
        sample_action = {
            "hole_cards": hole_cards,
            "community_cards": community_cards,
            "hand_strength": hand_strength,
            "pot_odds": pot_odds,
        }
        logging.info(f"Sample action prepared: {sample_action}")

        # Select predictor
        predictor = predictors.get(predictor_type)
        if predictor is None:
            raise HTTPException(status_code=400, detail="Invalid predictor type")

        # Predict action
        logging.info(f"Using predictor: {predictor_type}")
        predicted_action = predictor.predict_action(sample_action)
        logging.info(f"Prediction successful: {predicted_action}")
        return PredictionResponse(predicted_action=predicted_action)

    except ValueError as ve:
        logging.error(f"ValueError during prediction: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logging.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred during prediction."
        )


@app.get("/")
def root():
    """
    Root endpoint for health check.
    """
    return {"message": "Poker Predictor API is running"}
