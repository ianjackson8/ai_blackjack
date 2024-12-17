import random
import json
from play_blackjack import BlackjackGame, Player
from calc_stats import parse_log_and_plot_bot
from datetime import datetime

class BlackjackBot:
    def __init__(self, name, balance, log_file="bot_logs.json"):
        self.player = Player(name, balance)
        self.log_file = log_file
        self.round_logs = []

    def log_action(self, action, hand, dealer_visible_card):
        """
        Log the bot's action during the round.
        :param action: Action taken (e.g., 'hit', 'stand').
        :param hand: The bot's current hand.
        :param dealer_visible_card: The dealer's visible card.
        """
        log_entry = {
            "action": action,
            "player_hand": [str(card) for card in hand.cards],
            "hand_value": hand.calculate_value(),
            "dealer_visible_card": dealer_visible_card
        }
        self.round_logs.append(log_entry)

    def log_round(self, dealer_hand, dealer_value, result):
        """
        Log the full round's outcome for the bot.
        :param dealer_hand: The dealer's final hand.
        :param dealer_value: The dealer's final hand value.
        :param result: The bot's result for the round.
        """
        round_entry = {
            "dealer_hand": [str(card) for card in dealer_hand.cards],
            "dealer_value": dealer_value,
            "bot": {
                "name": self.player.name,
                "bet": self.player.current_bet,
                "actions": self.round_logs,
                "final_hand": [str(card) for card in self.player.hands[0].cards],
                "final_value": self.player.hands[0].calculate_value(),
                "result": result,
                "balance": self.player.balance
            }
        }
        # Append to log file
        try:
            with open(self.log_file, 'r') as file:
                logs = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            logs = []

        logs.append(round_entry)

        with open(self.log_file, 'w') as file:
            json.dump(logs, file, indent=4)

        # Clear the round logs for the next round
        self.round_logs = []

    def make_decision(self, hand, dealer_visible_card):
        """
        Decide the action based on a simple strategy:
        - Hit if hand value <= 16.
        - Stand if hand value > 16.
        :param hand: The current hand of the bot.
        :param dealer_visible_card: The dealer's visible card.
        :return: A string representing the chosen action.
        """
        hand_value = hand.calculate_value()
        strat = "lt16"

        if strat == "lt16":
            if hand_value <= 16:
                return "hit"
            else:
                return "stand"
        elif strat == "rand":
            choice = random.randint(0,1)
            if choice == 0: return "hit"
            elif choice == 1: return "stand"

    def play_round(self, game: BlackjackGame):
        """
        Simulate the bot playing a round of Blackjack.
        :param game: An instance of the Blackjack game.
        """
        # Place a random bet
        bet = 10
        # bet = random.randint(10, min(self.player.balance, 100))
        try:
            self.player.place_bet(bet)
        except ValueError:
            return
        print(f"{self.player.name} bets ${bet}")

        # Play the turn
        for hand in self.player.hands:
            while not hand.is_blackjack() and not hand.is_busted():
                dealer_visible_card = str(game.dealer.hands[0].cards[1])
                action = self.make_decision(hand, dealer_visible_card)
                print(f"{self.player.name} decides to {action}")

                if action == "hit":
                    card = game.shoe.draw_card()
                    hand.add_card(card)
                    print(f"{self.player.name} hits and receives {card}. Hand value: {hand.calculate_value()}")
                    self.log_action("hit", hand, dealer_visible_card)
                elif action == "stand":
                    print(f"{self.player.name} stands with hand value: {hand.calculate_value()}")
                    self.log_action("stand", hand, dealer_visible_card)
                    break

        # Log the round result
        dealer_value = game.dealer.hands[0].calculate_value()
        result = "Win" if hand.calculate_value() > dealer_value and not hand.is_busted() else "Lose"
        self.log_round(game.dealer.hands[0], dealer_value, result)


# Example usage:
if __name__ == "__main__":
    # Initialize game and bot
    player_names = []  # No human players
    game = BlackjackGame(num_decks=7, player_names=player_names, starting_balance=100, log_file=None)

    log_file = f'logs/session_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.json'
    bot = BlackjackBot("AI Bot", balance=100, log_file=log_file)
    game.players.append(bot.player)  # Add bot as a player

    # Play multiple rounds
    for _ in range(50):  # Play 5 rounds
        print("\nStarting a new round...")
        game.setup_round()
        game.deal_cards()

        # Bot plays
        bot.play_round(game)

        # Dealer's turn
        game.dealer_turn()

        # Determine results
        for player in game.players:
            game.determine_winner(player)

    parse_log_and_plot_bot(log_file)
