import random


class OpponentBehavior:
    def __init__(self, strategy="balanced", bluffing_probability=0.2):
        self.strategy = strategy
        self.bluffing_probability = bluffing_probability

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
        is_bluffing = False
        if hand_strength < 0.4 and random.random() < self.bluffing_probability:
            is_bluffing = True

        # 2) Position-based logic
        if position < 2:
            # --- EARLY POSITION: Tighter play ---
            if self.strategy == "tight-aggressive":
                # A typical tight-aggressive in early position might only raise with > 0.7 strength
                # or occasionally bluff.
                if is_bluffing:
                    return "raise"
                elif hand_strength > 0.7 and hand_strength > pot_odds:
                    return "raise"
                else:
                    return "fold"

            elif self.strategy == "loose-passive":
                # A looser, more passive player might call more often if they have at least moderate strength
                # or occasionally bluff.
                if is_bluffing and hand_strength < 0.3:
                    return "raise"
                elif hand_strength > 0.3:
                    return "call"
                else:
                    return "fold"

            else:  # "balanced"
                # Balanced in early position: mostly fold unless they have decent odds,
                # but might occasionally bluff.
                if is_bluffing and hand_strength < 0.6:
                    return "raise"
                elif hand_strength > pot_odds:
                    return "call"
                else:
                    return "fold"

        elif position <= 4:
            # --- MIDDLE POSITION: Moderate play ---
            if self.strategy == "tight-aggressive":
                # Slightly looser than early, but still not crazy.
                if is_bluffing:
                    return "raise"
                elif hand_strength > 0.6 and hand_strength > pot_odds:
                    return "raise"
                else:
                    return "fold"

            elif self.strategy == "loose-passive":
                # Might call quite frequently if strength is at least moderate,
                # and can occasionally bluff with weaker hands.
                if is_bluffing and hand_strength < 0.4:
                    # For variety, you could do "raise" half the time or "call" half the time, etc.
                    return "raise"
                elif hand_strength > 0.2:
                    return "call"
                else:
                    return "fold"

            else:  # "balanced"
                # Balanced in middle might raise with stronger hands,
                # call if moderate, fold if weak, and occasionally bluff.
                if is_bluffing and hand_strength < 0.5:
                    return "raise"
                elif hand_strength > pot_odds:
                    return "raise"
                elif hand_strength > 0.4:
                    return "call"
                else:
                    return "fold"

        else:
            # --- LATE POSITION (> 4): Looser play ---
            if self.strategy == "tight-aggressive":
                # Even a tight-aggressive player in late position can open up more.
                if is_bluffing:
                    return "raise"
                elif hand_strength > pot_odds:
                    return "raise"
                else:
                    return "call"

            elif self.strategy == "loose-passive":
                # A loose-passive late position player calls a lot, occasionally folds if super weak,
                # might bluff sometimes.
                if is_bluffing:
                    return "raise"
                elif hand_strength > 0.1:
                    return "call"
                else:
                    return "fold"

            else:  # "balanced"
                # Balanced in late position: more likely to raise or call,
                # plus the occasional bluff.
                if is_bluffing:
                    return "raise"
                elif hand_strength > 0.5:
                    return "raise"
                else:
                    return "call"
