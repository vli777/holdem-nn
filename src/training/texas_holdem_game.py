import logging
import random
import config
from training.player_state import PlayerState
from treys import Deck

from utils import (
    calculate_pot_odds,
    encode_action,
    encode_state,
    evaluate_hand,
)


class TexasHoldemGame:
    def __init__(self, num_players=6, starting_chips=1000):
        self.num_players = num_players
        self.starting_chips = starting_chips

        # Create players with random or assigned strategies
        strategies = ["tight-aggressive", "loose-passive", "balanced"]
        self.players = [
            PlayerState(
                player_id=i,
                strategy=random.choice(strategies),
                starting_chips=starting_chips,
                bluffing_probability=0.05 + random.random() * 0.15,
            )
            for i in range(num_players)
        ]

        # Deck, community cards, pot, minimal raise
        self.deck = Deck()
        self.community_cards = []
        self.current_pot = 0
        self.minimum_raise = 2

        # For storing states/actions, etc.
        self.game_data = []  # Will hold the final state-action records

    def reset_for_new_hand(self):
        """Reset or re-init fields for a new hand."""
        self.deck = Deck()
        self.deck.shuffle()
        self.community_cards = []
        self.current_pot = 0
        for p in self.players:
            p.reset_for_new_hand()

    def deal_preflop(self):
        """Deal 2 hole cards to each player."""
        for p in self.players:
            p.hole_cards = self.deck.draw(2)

    def post_blinds(self):
        """Simple approach: small blind = 1 chip, big blind = 2 chips."""
        if self.num_players < 2:
            return

        # Player 0 posts small blind
        sb = min(self.players[0].chips, 1)
        self.players[0].chips -= sb
        self.players[0].current_bet = sb

        # Player 1 posts big blind
        bb = min(self.players[1].chips, 2)
        self.players[1].chips -= bb
        self.players[1].current_bet = bb

        self.current_pot = sb + bb

    def single_pass_betting_round(self, round_name="pre-flop"):
        """
        Conduct a single pass of betting across all players.
        Returns True if a raise occurred, False otherwise.
        """
        logging.info(f"--- {round_name.upper()} BETTING ROUND (SINGLE PASS) ---")

        action_happened = False
        current_highest_bet = max(p.current_bet for p in self.players if p.in_hand)

        for p in self.players:
            if not p.in_hand:
                continue

            bet_to_call = current_highest_bet - p.current_bet
            if bet_to_call < 0:
                bet_to_call = 0

            # Evaluate the player's hand strength, pot odds, etc.
            normalized_strength = evaluate_hand(p.hole_cards, self.community_cards)
            pot_odds = calculate_pot_odds(self.current_pot, bet_to_call)

            action_str = p.opponent_behavior.decide_action(
                hand_strength=normalized_strength,
                pot_odds=pot_odds,
                position=p.player_id,
            )

            p.last_action = action_str

            if action_str == "fold":
                p.in_hand = False
                logging.info(f"Player {p.player_id} folds.")
            elif action_str == "call":
                call_amount = min(p.chips, bet_to_call)
                p.chips -= call_amount
                p.current_bet += call_amount
                self.current_pot += call_amount
                logging.info(f"Player {p.player_id} calls {call_amount}.")
            elif action_str == "raise":
                raise_amount = bet_to_call + self.minimum_raise
                raise_amount = min(p.chips, raise_amount)
                p.chips -= raise_amount
                p.current_bet += raise_amount
                self.current_pot += raise_amount
                current_highest_bet = p.current_bet
                action_happened = True
                logging.info(f"Player {p.player_id} raises to {p.current_bet}.")

            encoded_state = encode_state(
                p.hole_cards, self.community_cards, normalized_strength, pot_odds
            )
            encoded_act = encode_action(action_str)
            self.game_data.append(
                {
                    "state": encoded_state,
                    "action": encoded_act,
                    "player_id": p.player_id,
                    "position": p.player_id,
                    "recent_action": encoded_act,
                }
            )

        # Return True if a raise occurred, else False
        return action_happened

    def multi_betting_round(self, round_name="pre-flop"):
        """
        Conduct a multiple-pass betting round for a single street
        (pre-flop, flop, turn, or river).
        We repeat single_pass_betting_round until no new raises occur
        or until only one player remains.
        """
        logging.info(f"=== STARTING MULTI-PASS {round_name.upper()} BETTING ===")

        while True:
            # If only one player remains, betting is done
            if self.count_in_hand() < 2:
                logging.info("Betting ended because only one player remains.")
                break

            # Perform a single pass
            did_raise = self.single_pass_betting_round(round_name=round_name)

            # If no one raised, betting is complete
            if not did_raise:
                logging.info("No raise in this pass. Betting round ends.")
                break

            # If at least one raise occurred, we loop back
            # giving players a chance to respond in another pass
            # But note that in real poker, you'd track who still needs
            # to act. This simplified approach just goes around again.
        logging.info(f"=== MULTI-PASS {round_name.upper()} BETTING COMPLETE ===")

    def deal_flop(self):
        """Deal 3 community cards."""
        if len(self.deck.cards) < 3:
            logging.warning("Not enough cards to deal the flop.")
            return
        self.community_cards += self.deck.draw(3)

    def deal_turn(self):
        """Deal 1 community card (turn)."""
        if len(self.deck.cards) < 1:
            logging.warning("Not enough cards to deal the turn.")
            return
        self.community_cards += self.deck.draw(1)

    def deal_river(self):
        """Deal 1 community card (river)."""
        if len(self.deck.cards) < 1:
            logging.warning("Not enough cards to deal the river.")
            return
        self.community_cards += self.deck.draw(1)

    def play_hand(self):
        self.reset_for_new_hand()
        self.deal_preflop()
        self.post_blinds()
        # Pre-flop betting with multiple passes
        self.multi_betting_round(round_name="pre-flop")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        # Flop
        self.deal_flop()
        self.reset_bets_for_next_round()
        self.multi_betting_round(round_name="flop")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        # Turn
        self.deal_turn()
        self.reset_bets_for_next_round()
        self.multi_betting_round(round_name="turn")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        # River
        self.deal_river()
        self.reset_bets_for_next_round()
        self.multi_betting_round(round_name="river")
        if self.count_in_hand() < 2:
            self.showdown()
            return

        self.showdown()

    def showdown(self):
        # Collect players still in hand
        remaining_players = [p for p in self.players if p.in_hand]
        if not remaining_players:
            return  # Everyone folded earlier

        best_rank = None
        winners = []
        for p in remaining_players:
            # Evaluate final 7 cards: p.hole_cards + self.community_cards
            final_rank = evaluate_hand(self.community_cards, p.hole_cards)
            # Lower rank value = better in Treys, so invert if needed
            if best_rank is None or final_rank < best_rank:
                best_rank = final_rank
                winners = [p]
            elif final_rank == best_rank:
                winners.append(p)

        # If there's exactly one winner:
        if len(winners) == 1:
            winners[0].chips += self.current_pot
            logging.info(
                f"Player {winners[0].player_id} wins the pot of {self.current_pot}."
            )
        else:
            # Split pot among winners
            split_pot = self.current_pot // len(winners)
            for w in winners:
                w.chips += split_pot
            logging.info(
                f"Players {[w.player_id for w in winners]} split the pot of {self.current_pot}."
            )

        self.current_pot = 0

    def reset_bets_for_next_round(self):
        """Reset each player's bet to 0 after a street ends."""
        for p in self.players:
            p.current_bet = 0

    def count_in_hand(self):
        """Return how many players are still in the hand."""
        return sum(p.in_hand for p in self.players)

    def get_game_data(self):
        """Return or transform the recorded data for outside use."""
        return self.game_data


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG if config.debug else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    game = TexasHoldemGame(num_players=6)
    game.play_hand()

    final_data = game.get_game_data()
    logging.info(f"Collected {len(final_data)} state-action records from the hand.")
