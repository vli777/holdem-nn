import logging
from treys import Deck, Evaluator, Card


def determine_optimal_action(player_strength, max_opponent_strength):
    """
    Determine the optimal action based on the player's and opponent's hand strengths.

    Args:
        player_strength (int): The strength of the player's hand (lower is better).
        max_opponent_strength (int): The maximum strength of the opponents' hands (lower is better).

    Returns:
        str: The optimal action ("fold", "call", or "raise").
    """
    if player_strength > 4000 and player_strength > max_opponent_strength:
        return "fold"
    elif player_strength < max_opponent_strength:
        return "raise"
    else:
        return "call"


def play_out_game(predictor, sample_action, num_players=6):
    deck = Deck()

    # Convert known cards to Treys format
    known_cards = sample_action["hole_cards"] + sample_action["community_cards"]
    known_cards = [Card.new(card) for card in known_cards]

    # Remove known cards from the deck
    for card in known_cards:
        deck.cards.remove(card)

    # Ensure enough cards remain in the deck
    if len(deck.cards) < num_players * 2 + (5 - len(sample_action["community_cards"])):
        logging.error("Not enough cards remaining to simulate the game.")
        return

    # Predictor's action and confidence
    predicted_action = predictor.predict_action(sample_action)

    # Check for confidence method
    if hasattr(predictor, "predict_with_confidence"):
        predicted_confidence = predictor.predict_with_confidence(
            sample_action, threshold=0.8
        )
        if predicted_confidence == "uncertain":
            if sample_action["pot_odds"] > 0.5 or sample_action["hand_strength"] > 3000:
                predicted_action = "call"
                logging.info(
                    "Confidence too low, but pot odds/hand strength favor calling."
                )
            else:
                predicted_action = "fold"
                logging.info("Confidence too low. Defaulting to fold.")
    else:
        predicted_confidence = "N/A"

    logging.info(
        f"Player's Hole Cards: {[Card.int_to_pretty_str(card) for card in known_cards[:2]]}"
    )
    logging.info(
        f"Community Cards: {[Card.int_to_pretty_str(card) for card in known_cards[2:]]}"
    )
    logging.info(
        f"Predicted Action: {predicted_action}, Confidence: {predicted_confidence}"
    )

    # Complete the community cards
    while len(sample_action["community_cards"]) < 5:
        new_card = deck.draw(1)[0]
        sample_action["community_cards"].append(Card.int_to_pretty_str(new_card))
        logging.info(f"New Community Card: {Card.int_to_pretty_str(new_card)}")

    # Deal opponent hands
    opponent_hands = [
        [deck.draw(1)[0], deck.draw(1)[0]] for _ in range(num_players - 1)
    ]
    logging.info(
        f"Opponent Hands: {[[Card.int_to_pretty_str(card) for card in hand] for hand in opponent_hands]}"
    )

    # Evaluate hand strengths
    evaluator = Evaluator()

    player_hand_strength = evaluator.evaluate(
        known_cards[:2], [Card.new(card) for card in sample_action["community_cards"]]
    )
    opponent_strengths = [
        evaluator.evaluate(
            hand, [Card.new(card) for card in sample_action["community_cards"]]
        )
        for hand in opponent_hands
    ]

    logging.info(f"Player's Final Hand Strength: {player_hand_strength}")
    logging.info(f"Opponent Hand Strengths: {opponent_strengths}")

    max_opponent_strength = min(opponent_strengths)  # Lower is better
    player_outcome = (
        "Win"
        if player_hand_strength < max_opponent_strength
        else ("Tie" if player_hand_strength == max_opponent_strength else "Lose")
    )

    logging.info(f"Player Outcome: {player_outcome}")

    # Determine if the action was optimal
    optimal_action = determine_optimal_action(
        player_hand_strength, max_opponent_strength
    )
    logging.info(f"Optimal Action: {optimal_action}")
    logging.info(
        f"Was the predicted action optimal? {'Yes' if predicted_action == optimal_action else 'No'}"
    )
