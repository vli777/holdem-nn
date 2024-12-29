import eval7
import random
from training.generate_data import monte_carlo_hand_strength


def randomize_sample_action():
    """
    Generate a random poker hand state.
    Returns:
        dict: Randomized sample action with hole cards, community cards, hand strength, and pot odds.
    """
    deck = eval7.Deck()
    deck.shuffle()

    # Deal random hole cards for the player
    hole_cards = deck.deal(2)

    # Deal a random number of community cards (0 to 5)
    community_cards = deck.deal(random.randint(0, 5))

    hand_strength = monte_carlo_hand_strength(hole_cards, community_cards)

    # Random pot odds (example range: 0.1 to 0.9)
    pot_odds = random.uniform(0.1, 0.9)

    return {
        "hole_cards": hole_cards,
        "community_cards": community_cards,
        "hand_strength": hand_strength,
        "pot_odds": pot_odds
    }


def play_out_game(predictor, sample_action, num_players=6):
    """
    Simulate the game and print each step after predicting the best action.
    Args:
        predictor: The predictor object to make predictions.
        sample_action (dict): The initial poker state.
        num_players (int): Total number of players in the game.
    """
    deck = eval7.Deck()
    deck.shuffle()

    # Remove known cards from the deck
    known_cards = sample_action["hole_cards"] + \
        sample_action["community_cards"]
    for card in known_cards:
        deck.cards.remove(card)

    # Predict the action
    predicted_action = predictor.predict_action(sample_action)
    predicted_confidence = predictor.predict_with_confidence(
        sample_action, threshold=0.8)

    print(
        f"Player's Hole Cards: {[str(card) for card in sample_action['hole_cards']]}")
    print(
        f"Community Cards: {[str(card) for card in sample_action['community_cards']]}")
    print(f"Predicted Action: {predicted_action}")
    print(f"Confidence Prediction: {predicted_confidence}")

    # Deal the rest of the community cards
    while len(sample_action["community_cards"]) < 5:
        new_card = deck.deal(1)[0]
        sample_action["community_cards"].append(new_card)
        print(f"New Community Card: {new_card}")

    # Simulate opponent hands
    opponent_hands = [deck.deal(2) for _ in range(num_players - 1)]

    print(
        f"Opponent Hands: {[[str(card) for card in hand] for hand in opponent_hands]}")

    # Evaluate final hands
    player_hand_strength = eval7.evaluate(
        sample_action["hole_cards"] + sample_action["community_cards"]
    )
    opponent_strengths = [
        eval7.evaluate(opponent_hand + sample_action["community_cards"])
        for opponent_hand in opponent_hands
    ]

    print(f"Player's Final Hand Strength: {player_hand_strength}")
    print(f"Opponent Hand Strengths: {opponent_strengths}")

    # Determine the winner
    max_opponent_strength = max(opponent_strengths)
    if player_hand_strength > max_opponent_strength:
        print("Player Wins!")
    elif player_hand_strength == max_opponent_strength:
        print("It's a Tie!")
    else:
        print("Opponent Wins!")
