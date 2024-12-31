class OpponentBehavior:
    def __init__(self, strategy="balanced"):
        self.strategy = strategy

    def decide_action(self, hand_strength, pot_odds, position):
        """
        Decide the action based on hand strength, pot odds, and player position.

        Args:
            hand_strength (float): Strength of the opponent's hand (normalized to [0, 1]).
            pot_odds (float): Pot odds (normalized to [0, 1]).
            position (int): Player position (0 = early, higher = later).

        Returns:
            str: The decided action ("fold", "call", or "raise").
        """
        # Modify behavior based on position
        if position < 2:  # Early position: tighter play
            if self.strategy == "tight-aggressive":
                return (
                    "raise"
                    if hand_strength > 0.7 and hand_strength > pot_odds
                    else "fold"
                )
            elif self.strategy == "loose-passive":
                return "call" if hand_strength > 0.3 else "fold"
            else:  # Balanced
                return "call" if hand_strength > pot_odds else "fold"

        elif position > 4:  # Late position: looser play
            if self.strategy == "tight-aggressive":
                return "raise" if hand_strength > pot_odds else "call"
            elif self.strategy == "loose-passive":
                return "call" if hand_strength > 0.1 else "fold"
            else:  # Balanced
                return "raise" if hand_strength > 0.5 else "call"

        else:  # Middle position: moderate play
            if self.strategy == "tight-aggressive":
                return "raise" if hand_strength > pot_odds else "fold"
            elif self.strategy == "loose-passive":
                return "call" if hand_strength > 0.2 else "fold"
            else:  # Balanced
                return "raise" if hand_strength > pot_odds else "call"
