import random
from typing import Literal

StrategyType = Literal["tight-aggressive", "loose-passive", "balanced"]

class OpponentBehavior:
    __slots__ = ['strategy', 'bluffing_probability']

    def __init__(
        self,
        strategy: StrategyType = "balanced",
        bluffing_probability: float = 0.2
    ):
        self.strategy = strategy
        self.bluffing_probability = bluffing_probability

    def decide_action(
        self,
        hand_strength: float,
        pot_odds: float,
        position: int
    ) -> Literal["fold", "call", "raise"]:
        """
        Decide the action based on hand strength, pot odds, and player position.

        Args:
            hand_strength (float): Strength of the opponent's hand (normalized to [0, 1]).
            pot_odds (float): Pot odds (normalized to [0, 1]).
            position (int): Player position (0 = early, higher = later).

        Returns:
            str: The decided action ("fold", "call", or "raise").
        """
        is_bluffing = hand_strength < 0.4 and random.random() < self.bluffing_probability

        # Early position (0-1)
        if position < 2:
            if self.strategy == "tight-aggressive":
                if is_bluffing:
                    return "raise"
                return "raise" if hand_strength > 0.7 and hand_strength > pot_odds else "fold"

            elif self.strategy == "loose-passive":
                if is_bluffing and hand_strength < 0.3:
                    return "raise"
                return "call" if hand_strength > 0.3 else "fold"

            else:  # balanced
                if is_bluffing and hand_strength < 0.6:
                    return "raise"
                return "call" if hand_strength > pot_odds else "fold"

        # Middle position (2-4)
        elif position <= 4:
            if self.strategy == "tight-aggressive":
                if is_bluffing:
                    return "raise"
                return "raise" if hand_strength > 0.6 and hand_strength > pot_odds else "fold"

            elif self.strategy == "loose-passive":
                if is_bluffing and hand_strength < 0.4:
                    return "raise"
                return "call" if hand_strength > 0.2 else "fold"

            else:  # balanced
                if is_bluffing and hand_strength < 0.5:
                    return "raise"
                if hand_strength > pot_odds:
                    return "raise"
                return "call" if hand_strength > 0.4 else "fold"

        # Late position (>4)
        else:
            if self.strategy == "tight-aggressive":
                if is_bluffing:
                    return "raise"
                return "raise" if hand_strength > pot_odds else "call"

            elif self.strategy == "loose-passive":
                if is_bluffing:
                    return "raise"
                return "call" if hand_strength > 0.1 else "fold"

            else:  # balanced
                if is_bluffing:
                    return "raise"
                return "raise" if hand_strength > 0.5 else "call"
