#!myenv/bin/python3.12
'''
Play blackjack in terminal

Author: Ian Jackson
Version v4.0
'''

import random
import os
import json
import argparse
import torch
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from typing import List, Optional, Union
from datetime import datetime

#== Global Variables ==#
SHOW_OUTPUT = True
LOG_GAME = True
SHOW_GRAPH = True
DEFAULT_BET = 10
TRAIN_MODE = False
GOD_MODE = False

#== Colors ==#
class C:
    RESET     = '\033[0m'
    BOLD      = '\033[1m'
    DIM       = '\033[2m'
    UNDERLINE = '\033[4m'
    RED       = '\033[91m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    BLUE      = '\033[94m'
    MAGENTA   = '\033[95m'
    CYAN      = '\033[96m'
    WHITE     = '\033[97m'

def clear() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')

#== Classes ==#

class Card:
    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.values = self._assign_value(rank)

    def _assign_value(self, rank: str) -> tuple:
        if rank in ["Jack", "King", "Queen"]:
            return (10,)
        elif rank == "Ace":
            return (1, 11)
        else:
            return (int(rank),)

    def __repr__(self) -> str:
        return f"{self.rank} of {self.suit}"


class Shoe:
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        self.cards = self._generate_shoe()
        self.shuffle()

    def _generate_shoe(self) -> List[Card]:
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        shoe = []
        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    shoe.append(Card(rank, suit))
        return shoe

    def shuffle(self) -> None:
        random.shuffle(self.cards)

    def draw_card(self) -> Card:
        if not self.cards:
            raise ValueError("The shoe is empty, cannot draw card")
        return self.cards.pop()

    def __len__(self) -> int:
        return len(self.cards)


class Hand:
    def __init__(self, hand: list[Card] = None):
        self.cards = [] if hand is None else hand

    def add_card(self, card: Card):
        self.cards.append(card)

    def calculate_value(self) -> int:
        values = [0]
        for card in self.cards:
            new_values = []
            for val in values:
                for card_val in card.values:
                    new_values.append(val + card_val)
            values = list(set(new_values))
        valid_values = [v for v in values if v <= 21]
        return max(valid_values) if valid_values else min(values)

    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.calculate_value() == 21

    def is_busted(self) -> bool:
        return self.calculate_value() > 21

    def num_cards(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        cards_str = ", ".join(str(card) for card in self.cards)
        return f"[{cards_str}] | Value: {self.calculate_value()}"


class Player:
    def __init__(self, name: str, balance: int):
        self.name = name
        self.balance = balance
        self.hands = [Hand()]
        self.current_bet = 0
        self.hand_bets = []
        self.actions = []
        self.results = []
        self.stats = {"win": 0, "lose": 0, "push": 0, "bust": 0, "blackjack": 0}

    def log_action(self, action: str, dealer_visible_card: str):
        self.actions.append({
            "action": action,
            "player_hand": [str(card) for card in self.hands[-1].cards],
            "hand_value": self.hands[-1].calculate_value(),
            "dealer_visible_card": dealer_visible_card
        })

    def place_bet(self, amount: int) -> None:
        if amount <= 0:
            raise ValueError(f"  {C.RED}Bet amount must be greater than zero.{C.RESET}")
        if amount > self.balance:
            raise ValueError(f"  {C.RED}Bet amount exceeds available balance.{C.RESET}")
        self.current_bet = amount
        self.hand_bets = [amount]
        self.balance -= amount

    def hit(self, card: Card) -> None:
        self.hands[-1].add_card(card)

    def stand(self) -> None:
        pass

    def split(self, hand_index: int = 0) -> None:
        current_hand = self.hands[hand_index]
        if len(current_hand.cards) != 2 or current_hand.cards[0].rank != current_hand.cards[1].rank:
            raise ValueError(f"  {C.RED}Cannot split: hand must have exactly two cards of the same rank.{C.RESET}")
        if self.current_bet > self.balance:
            raise ValueError(f"  {C.RED}Insufficient balance to split (need ${self.current_bet} for second hand).{C.RESET}")
        self.balance -= self.current_bet
        card1, card2 = current_hand.cards
        self.hands[hand_index] = Hand()
        self.hands[hand_index].add_card(card1)
        new_hand = Hand()
        new_hand.add_card(card2)
        self.hands.insert(hand_index + 1, new_hand)
        self.hand_bets.insert(hand_index + 1, self.current_bet)

    def double_down(self, card: Card, hand: Hand = None, hand_index: int = None):
        if hand is None:
            hand = self.hands[-1]
        if hand.num_cards() != 2:
            raise ValueError(f"  {C.RED}Cannot double down after a hit.{C.RESET}")
        if hand_index is None:
            hand_index = len(self.hands) - 1
        bet = self.hand_bets[hand_index] if self.hand_bets else self.current_bet
        if bet > self.balance:
            raise ValueError(f"  {C.RED}Insufficient balance to double down.{C.RESET}")
        self.balance -= bet
        self.hand_bets[hand_index] = bet * 2
        hand.add_card(card)

    def reset_for_new_round(self):
        self.hands = [Hand()]
        self.current_bet = 0
        self.hand_bets = []
        self.actions = []
        self.results = []
        self.insurance_bet = 0

    def __repr__(self) -> str:
        # BUG: always shows? set to nothing for now
        return f""


class Bot(Player):
    def __init__(self, name: str, balance: int, strategy: str):
        super().__init__(name, balance)
        self.strategy = strategy

    def play_turn(self, shoe: Shoe, dealer_visible_card: str, deal_delay: int):
        for hand in self.hands:
            print(f"\n  {C.BOLD}{self.name}{C.RESET}'s turn:")
            print(f"  Hand: {hand}")
            while not hand.is_blackjack() and not hand.is_busted():
                action = self.make_decision(dealer_visible_card, hand)
                print(f"  {C.DIM}{self.name} → {action}{C.RESET}")

                if action == "hit":
                    card = shoe.draw_card()
                    hand.add_card(card)
                    time.sleep(deal_delay)
                    print(f"  {self.name} hits: {hand}")
                elif action == "stand":
                    break
                elif action == "double":
                    if (float(self.balance) >= float(self.current_bet)) and (hand.num_cards() == 2):
                        bet = self.hand_bets[0] if self.hand_bets else self.current_bet
                        self.balance -= bet
                        self.hand_bets[0] = bet * 2
                        card = shoe.draw_card()
                        hand.add_card(card)
                        time.sleep(deal_delay)
                        print(f"  {self.name} doubles: {hand}")
                        break
                    else:
                        if hand.num_cards() > 2:
                            print(f"  {C.CYAN}Cannot double after hitting.{C.RESET}")
                        else:
                            print(f"  {C.CYAN}Cannot double — insufficient balance.{C.RESET}")
                        card = shoe.draw_card()
                        hand.add_card(card)
                        time.sleep(deal_delay)
                        print(f"  {self.name} hits: {hand}")
                else:
                    print(f"  {C.RED}Invalid action: {action}{C.RESET}")

    def make_decision(self, dealer_visible_card: str, hand: Hand) -> str:
        DECISION_TABLE_BTB = {
            2: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "hit",
                10: "double", 11: "double", 12: "hit", 13: "stand", 14: "stand",
                15: "stand", 16: "stand", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            3: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "double",
                10: "double", 11: "double", 12: "hit", 13: "stand", 14: "stand",
                15: "stand", 16: "stand", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            4: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "double",
                10: "double", 11: "double", 12: "stand", 13: "stand", 14: "stand",
                15: "stand", 16: "stand", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            5: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "double",
                10: "double", 11: "double", 12: "stand", 13: "stand", 14: "stand",
                15: "stand", 16: "stand", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            6: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "double",
                10: "double", 11: "double", 12: "stand", 13: "stand", 14: "stand",
                15: "stand", 16: "stand", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            7: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "hit",
                10: "double", 11: "double", 12: "hit", 13: "hit", 14: "hit",
                15: "hit", 16: "hit", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            8: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "hit",
                10: "double", 11: "double", 12: "hit", 13: "hit", 14: "hit",
                15: "hit", 16: "hit", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            9: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "hit",
                10: "double", 11: "double", 12: "hit", 13: "hit", 14: "hit",
                15: "hit", 16: "hit", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            10: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "hit",
                10: "hit", 11: "double", 12: "hit", 13: "hit", 14: "hit",
                15: "hit", 16: "hit", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
            11: {
                4: "hit", 5: "hit", 6: "hit", 7: "hit", 8: "hit", 9: "hit",
                10: "hit", 11: "hit", 12: "hit", 13: "hit", 14: "hit",
                15: "hit", 16: "hit", 17: "stand", 18: "stand", 19: "stand",
                20: "stand", 21: "stand"
            },
        }

        if self.strategy == "default":
            hand_value = hand.calculate_value()
            return "hit" if hand_value <= 16 else "stand"

        elif self.strategy == "by the books":
            dealer_value = get_card_value(dealer_visible_card)
            hand_value = hand.calculate_value()
            if dealer_value not in DECISION_TABLE_BTB or hand_value not in DECISION_TABLE_BTB[dealer_value]:
                return "stand"
            return DECISION_TABLE_BTB[dealer_value][hand_value]

        else:
            raise ValueError(f"  {C.RED}Invalid bot strategy: {self.strategy}.{C.RESET}")


class BlackjackGame:
    def __init__(self, num_decks: int, player_names: List[str], bot_info: List[dict], starting_balance: int, log_file: str,
                 deal_delay: int, allow_double_after_split: bool = True, split_limit: int = 1,
                 allow_insurance: bool = True, save_data: dict = None):
        self.num_decks = num_decks
        self.shoe = Shoe(num_decks)
        self.shoe.shuffle()
        self.dealer = Player("Dealer", balance=0)
        self.deal_delay = deal_delay
        self.allow_double_after_split = allow_double_after_split
        self.split_limit = split_limit if split_limit != 0 else 1
        self.allow_insurance = allow_insurance

        saved_player_balances = {}
        saved_bot_balances = {}
        saved_bot_epsilons = {}
        if save_data:
            for p in save_data.get("players", []):
                saved_player_balances[p["name"]] = p["balance"]
            for b in save_data.get("bots", []):
                saved_bot_balances[b["name"]] = b["balance"]
                if "epsilon" in b:
                    saved_bot_epsilons[b["name"]] = b["epsilon"]

        self.players = [Player(name, saved_player_balances.get(name, starting_balance)) for name in player_names]

        self.bots = []
        for bot in bot_info:
            bal = saved_bot_balances.get(bot['name'], starting_balance)
            if bot.get('strategy') == 'ai-nn':
                self.bots.append(TrainableBot(bot['name'], bal, 2, 3))
            else:
                self.bots.append(Bot(bot['name'], bal, bot['strategy']))

        self.log_file = log_file
        self.round_data = {}
        self.game_num = save_data["rounds_played"] if save_data else 0

        if save_data and "balance_history" in save_data:
            self.balance_history = save_data["balance_history"]
        else:
            self.balance_history = {p.name: [p.balance] for p in self.players}
            for bot in self.bots:
                self.balance_history[bot.name] = [bot.balance]

        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("saves"):
            os.makedirs("saves")

        for bot in self.bots:
            if isinstance(bot, TrainableBot):
                try:
                    bot_name = bot.name.replace(" ", "_")
                    additional_data = load_model(bot.q_network, bot.optimizer, f"models/{bot_name}_model.pth")
                    bot.epsilon = saved_bot_epsilons.get(bot.name, additional_data.get("epsilon", 1.0))
                except FileNotFoundError:
                    print(f"  {C.CYAN}AI save not found — creating new model.{C.RESET}")
                    bot.epsilon = saved_bot_epsilons.get(bot.name, 1.0)

    def setup_round(self) -> None:
        for player in self.players + self.bots:
            player.reset_for_new_round()
        self.dealer.reset_for_new_round()
        self.round_data = {}
        self.game_num += 1

    def deal_cards(self) -> None:
        global GOD_MODE
        for player in self.players + self.bots:
            if player.current_bet > 0:
                if GOD_MODE and not (isinstance(player, Bot) or isinstance(player, TrainableBot)):
                    player.hands = [Hand([Card("Ace", "Hearts"), Card("King", "Hearts")])]
                else:
                    for _ in range(2):
                        player.hit(self.shoe.draw_card())
        for _ in range(2):
            self.dealer.hit(self.shoe.draw_card())

    def display_table(self) -> None:
        clear()
        print()
        print(f'  {C.BOLD}{C.YELLOW}╔══════════════════════════════════════╗{C.RESET}')
        print(f'  {C.BOLD}{C.YELLOW}║          ♠  B L A C K J A C K  ♠    ║{C.RESET}')
        print(f'  {C.BOLD}{C.YELLOW}╚══════════════════════════════════════╝{C.RESET}')
        print()
        print(f'  {C.BOLD}Dealer:{C.RESET}  [Hidden] | {self.dealer.hands[0].cards[1]}')
        print()
        for player in self.players + self.bots:
            if player.current_bet > 0:
                print(f'  {C.BOLD}{player.name}{C.RESET}  {C.DIM}(${player.balance}){C.RESET}  {player.hands[0]}')
        print()

    def display_players(self) -> None:
        print(f'  {C.BOLD}Players at the table:{C.RESET}')
        for player in self.players + self.bots:
            print(f'    {player.name}: {C.GREEN}${player.balance}{C.RESET}')
        print()

    def player_turn(self, player: Player) -> None:
        dealer_visible_card = str(self.dealer.hands[0].cards[1])
        print(f"\n  {C.BOLD}{player.name}{C.RESET}, your turn:")

        hand_index = 0
        while hand_index < len(player.hands):
            hand = player.hands[hand_index]
            if len(player.hands) > 1:
                print(f"\n  {C.DIM}── Hand {hand_index + 1} of {len(player.hands)} ──{C.RESET}")

            while True:
                print(f"\n  Hand: {hand}")
                if hand.is_blackjack():
                    print(f"  {C.BOLD}{C.GREEN}Blackjack!{C.RESET}")
                    break
                elif hand.is_busted():
                    print(f"  {C.BOLD}{C.RED}Busted!{C.RESET}")
                    break

                print(f"  {C.DIM}[1] Hit  [2] Stand  [3] Double  [4] Split{C.RESET}")
                action = input("  > ").strip().lower()

                if action == "1":
                    action = "hit"
                elif action == "2":
                    action = "stand"
                elif action == "3":
                    action = "double"
                elif action == "4":
                    action = "split"

                if action == "hit":
                    hand.add_card(self.shoe.draw_card())
                    time.sleep(self.deal_delay)
                    player.log_action("hit", dealer_visible_card)
                elif action == "stand":
                    player.log_action("stand", dealer_visible_card)
                    break
                elif action == "double":
                    if len(player.hands) > 1 and not self.allow_double_after_split:
                        print(f"  {C.RED}Cannot double down after splitting (disabled in settings).{C.RESET}")
                    else:
                        try:
                            player.double_down(self.shoe.draw_card(), hand, hand_index)
                            player.log_action("double", dealer_visible_card)
                        except ValueError as e:
                            print(e)
                            continue
                        break
                elif action == "split":
                    can_split = (self.split_limit == -1) or (len(player.hands) <= self.split_limit)
                    if can_split:
                        try:
                            player.split(hand_index)
                            player.log_action("split", dealer_visible_card)
                        except ValueError as e:
                            print(e)
                            continue
                        player.hands[hand_index].add_card(self.shoe.draw_card())
                        time.sleep(self.deal_delay)
                        player.hands[hand_index + 1].add_card(self.shoe.draw_card())
                        time.sleep(self.deal_delay)
                        hand = player.hands[hand_index]
                        print(f"\n  {C.DIM}── Hand {hand_index + 1} of {len(player.hands)} ──{C.RESET}")
                    else:
                        print(f"  {C.RED}Cannot split: split limit of {self.split_limit} reached.{C.RESET}")
                else:
                    print(f"  {C.RED}Invalid action.{C.RESET}")

            hand_index += 1

    def dealer_turn(self) -> None:
        print(f"\n  {C.BOLD}Dealer's turn:{C.RESET}")
        print(f"  Hand: {self.dealer.hands[0]}")
        while self.dealer.hands[0].calculate_value() < 17:
            self.dealer.hands[0].add_card(self.shoe.draw_card())
            time.sleep(self.deal_delay)
            print(f"  Dealer hits: {self.dealer.hands[0]}")
        if self.dealer.hands[0].is_busted():
            print(f"  {C.BOLD}{C.RED}Dealer busted!{C.RESET}")

    def offer_insurance(self) -> None:
        for player in self.players:
            if player.current_bet == 0:
                continue
            max_insurance = player.current_bet / 2
            print(f"\n  {C.BOLD}{player.name}{C.RESET}  Dealer shows Ace — insurance? (max ${max_insurance:.0f}, 0 to skip)")
            while True:
                raw = input("  > ").strip()
                try:
                    amount = float(raw)
                except ValueError:
                    print(f"  {C.RED}Enter a number.{C.RESET}")
                    continue
                if amount < 0 or amount > max_insurance:
                    print(f"  {C.RED}Insurance must be between $0 and ${max_insurance:.0f}.{C.RESET}")
                    continue
                if amount > player.balance:
                    print(f"  {C.RED}Insufficient balance.{C.RESET}")
                    continue
                player.insurance_bet = amount
                player.balance -= amount
                if amount > 0:
                    print(f"  {C.DIM}Insurance bet: ${amount:.0f}{C.RESET}")
                break

    def determine_winner(self, player: Player) -> None:
        # Resolve insurance before main hand
        if player.insurance_bet > 0:
            if self.dealer.hands[0].is_blackjack():
                player.balance += player.insurance_bet * 3  # stake back + 2:1
                print(f"  {C.BOLD}{player.name}{C.RESET}  Insurance  →  {C.GREEN}Won ${player.insurance_bet * 2:.0f}.{C.RESET}")
            else:
                print(f"  {C.BOLD}{player.name}{C.RESET}  Insurance  →  {C.RED}Lost ${player.insurance_bet:.0f}.{C.RESET}")

        dealer_value = self.dealer.hands[0].calculate_value()

        for i, hand in enumerate(player.hands):
            bet = player.hand_bets[i] if player.hand_bets and i < len(player.hand_bets) else player.current_bet
            player_value = hand.calculate_value()
            if hand.is_busted():
                print(f"  {C.BOLD}{player.name}{C.RESET}  {player_value}  →  {C.RED}Busted! Lost ${bet}.{C.RESET}")
                player.results.append("busted")
                player.stats["bust"] += 1
            elif hand.is_blackjack():
                winnings = bet * 2.5
                player.balance += winnings
                print(f"  {C.BOLD}{player.name}{C.RESET}  {player_value}  →  {C.GREEN}Blackjack! Won ${winnings}.{C.RESET}")
                player.results.append("blackjack")
                player.stats["blackjack"] += 1
            elif dealer_value > 21 or player_value > dealer_value:
                winnings = bet * 2
                player.balance += winnings
                print(f"  {C.BOLD}{player.name}{C.RESET}  {player_value}  →  {C.GREEN}Win! Won ${winnings}.{C.RESET}")
                player.results.append("win")
                player.stats["win"] += 1
            elif player_value == dealer_value:
                player.balance += bet
                print(f"  {C.BOLD}{player.name}{C.RESET}  {player_value}  →  {C.CYAN}Push. ${bet} returned.{C.RESET}")
                player.results.append("push")
                player.stats["push"] += 1
            else:
                print(f"  {C.BOLD}{player.name}{C.RESET}  {player_value}  →  {C.RED}Lose. Lost ${bet}.{C.RESET}")
                player.results.append("lose")
                player.stats["lose"] += 1

    def play_round(self) -> None:
        self.check_shoe()
        self.setup_round()

        for player in self.players + self.bots:
            while True:
                try:
                    if isinstance(player, Bot) or isinstance(player, TrainableBot):
                        global DEFAULT_BET
                        bet = min(DEFAULT_BET, player.balance)
                        if float(bet) == float(0): break
                        player.place_bet(bet)
                        break
                    else:
                        is_bet = False
                        while not is_bet:
                            print(f"\n  {C.BOLD}{player.name}{C.RESET}   Balance: {C.GREEN}${player.balance}{C.RESET}")
                            bet = input("  Bet (or /command): ").strip()
                            if bet.startswith('/'):
                                self.parse_command(bet[1:])
                            else:
                                is_bet = True

                        try:
                            bet = int(bet)
                        except:
                            raise ValueError(f"  {C.RED}Bet must be a number.{C.RESET}")

                        if bet > 0:
                            player.place_bet(bet)
                        break
                except ValueError as e:
                    print(e)

        self.deal_cards()
        self.display_table()

        if self.allow_insurance and self.dealer.hands[0].cards[1].rank == "Ace":
            self.offer_insurance()

        for player in self.players + self.bots:
            if player.current_bet > 0:
                if isinstance(player, Bot) or isinstance(player, TrainableBot):
                    player.play_turn(self.shoe, str(self.dealer.hands[0].cards[1]), self.deal_delay)
                else:
                    self.player_turn(player)

        self.dealer_turn()

        print(f"\n  {C.BOLD}{C.UNDERLINE}Results{C.RESET}")
        dealer_value = self.dealer.hands[0].calculate_value()
        print(f"  Dealer: {dealer_value}")
        for player in self.players + self.bots:
            if player.current_bet > 0:
                self.determine_winner(player)
                if isinstance(player, TrainableBot):
                    player.update_final_reward(player.results[0] if player.results else "")

        for player in self.players + self.bots:
            self.balance_history[player.name].append(player.balance)

        global LOG_GAME
        if LOG_GAME: self.log_game()

    def play_game(self):
        ai_done = False

        while True:
            clear()
            print()
            print(f'  {C.BOLD}{C.YELLOW}╔══════════════════════════════╗{C.RESET}')
            print(f'  {C.BOLD}{C.YELLOW}║   ♠  BLACKJACK               ║{C.RESET}')
            print(f'  {C.BOLD}{C.YELLOW}╚══════════════════════════════╝{C.RESET}')
            print()
            self.display_players()

            global DEFAULT_BET, TRAIN_MODE
            if TRAIN_MODE and (not self.bots or self.bots[0].balance <= DEFAULT_BET):
                ai_done = True
            else:
                self.play_round()

            if TRAIN_MODE and not ai_done:
                continue

            while True:
                print()
                play_again = input("  Play another round? [Y/N]: ").strip().lower()
                if play_again in ["y", "yes", "n", "no"]:
                    break
                print(f"  {C.RED}Enter Y or N.{C.RESET}")

            if play_again in ["n", "no"] or ai_done:
                self.end_game()
                break

    def log_game(self):
        log_entry = {
            "game_number": self.game_num,
            "dealer": {
                "initial_hand": [str(card) for card in self.dealer.hands[0].cards[:1]] + ["Hidden"],
                "final_hand": [str(card) for card in self.dealer.hands[0].cards],
                "final_value": self.dealer.hands[0].calculate_value()
            },
            "players": []
        }

        for player in self.players + self.bots:
            player_log = {
                "name": player.name,
                "bet": player.current_bet,
                "insurance_bet": getattr(player, 'insurance_bet', 0),
                "actions": player.actions,
                "hands": [
                    {
                        "final_hand": [str(card) for card in hand.cards],
                        "final_value": hand.calculate_value(),
                    }
                    for hand in player.hands
                ],
                "results": player.results,
                "balance": player.balance
            }
            log_entry["players"].append(player_log)

        with open(self.log_file, 'r') as file:
            logs = json.load(file)
        logs.append(log_entry)
        with open(self.log_file, "w") as file:
            json.dump(logs, file, indent=4)

    def check_shoe(self, force: bool = False):
        if (len(self.shoe) < (len(self.players) + len(self.bots) + 1) * 3) or force:
            print(f"  {C.MAGENTA}Reshuffling deck...{C.RESET}")
            self.shoe = Shoe(self.num_decks)
            self.shoe.shuffle()

    def print_stats(self) -> None:
        header = f"Session Stats — Round {self.game_num}"
        print()
        print(f'  {C.BOLD}{C.YELLOW}╔══════════════════════════════════════════════════════════╗{C.RESET}')
        print(f'  {C.BOLD}{C.YELLOW}║  {header:<56}║{C.RESET}')
        print(f'  {C.BOLD}{C.YELLOW}╚══════════════════════════════════════════════════════════╝{C.RESET}')
        print()
        print(f"  {C.BOLD}{'Player':<16} {'Net':>7}   {'W':>4} {'L':>4} {'P':>4} {'BJ':>4}   {'Win%':>6}{C.RESET}")
        print(f"  {'─' * 56}")
        for player in self.players + self.bots:
            s = player.stats
            history = self.balance_history.get(player.name, [])
            init_bal = history[0] if history else player.balance
            net = player.balance - init_bal
            net_str = f"{C.GREEN}+${net:.0f}{C.RESET}" if net >= 0 else f"{C.RED}-${abs(net):.0f}{C.RESET}"
            total_hands = s["win"] + s["lose"] + s["push"] + s["bust"] + s["blackjack"]
            win_rate = (s["win"] + s["blackjack"]) / total_hands * 100 if total_hands > 0 else 0.0

            # largest single-round gain/loss from balance_history deltas
            if len(history) > 1:
                deltas = [history[i+1] - history[i] for i in range(len(history) - 1)]
                best = max(deltas)
                worst = min(deltas)
                swing_str = f"  {C.DIM}best {C.GREEN}+${best:.0f}{C.RESET}{C.DIM}  worst {C.RED}-${abs(worst):.0f}{C.RESET}" if worst < 0 else f"  {C.DIM}best {C.GREEN}+${best:.0f}{C.RESET}"
            else:
                swing_str = ""

            name_col = player.name[:15]
            print(f"  {C.BOLD}{name_col:<16}{C.RESET} {net_str:>7}   {s['win']:>4} {s['lose']:>4} {s['push']:>4} {s['blackjack']:>4}   {win_rate:>5.1f}%{swing_str}")
        print()

    def end_game(self):
        print()
        print(f'  {C.BOLD}{C.YELLOW}╔══════════════════════════════╗{C.RESET}')
        print(f'  {C.BOLD}{C.YELLOW}║        Final Balances        ║{C.RESET}')
        print(f'  {C.BOLD}{C.YELLOW}╚══════════════════════════════╝{C.RESET}')
        print()
        for player in self.players + self.bots:
            print(f"  {C.BOLD}{player.name}{C.RESET}: {C.GREEN}${player.balance}{C.RESET}")
        print()

        self.print_stats()

        for bot in self.bots:
            if isinstance(bot, TrainableBot):
                bot_name = bot.name.replace(" ", "_")
                save_model(bot.q_network, bot.optimizer, f"models/{bot_name}_model.pth", additional_data={"epsilon": bot.epsilon})

        global TRAIN_MODE
        if not TRAIN_MODE:
            while True:
                save_choice = input("  Save game? [Y/N]: ").strip().lower()
                if save_choice in ["y", "yes", "n", "no"]:
                    break
                print(f"  {C.RED}Enter Y or N.{C.RESET}")

            if save_choice in ["y", "yes"]:
                default_name = f"save_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
                save_name = input(f"  Save name (Enter for '{default_name}'): ").strip()
                if not save_name:
                    save_name = default_name
                self.save_game(save_name)

        global SHOW_GRAPH
        if SHOW_GRAPH: parse_log_and_plot(self.log_file)

    def save_game(self, save_name: str):
        save_name = save_name.strip().replace(".json", "")
        save_path = f"saves/{save_name}.json"

        save_data = {
            "saved_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "rounds_played": self.game_num,
            "players": [{"name": p.name, "balance": p.balance} for p in self.players],
            "bots": [],
            "balance_history": self.balance_history
        }

        for bot in self.bots:
            if isinstance(bot, TrainableBot):
                bot_entry = {"name": bot.name, "strategy": "ai-nn", "balance": bot.balance, "epsilon": bot.epsilon}
            else:
                bot_entry = {"name": bot.name, "strategy": bot.strategy, "balance": bot.balance}
            save_data["bots"].append(bot_entry)

        with open(save_path, 'w') as f:
            json.dump(save_data, f, indent=4)

        print(f"  {C.GREEN}Game saved to {save_path}{C.RESET}")
        print(f"  Load with: python3 play_blackjack.py --load {save_name}")

    def parse_command(self, command: str):
        if command == 'help':
            print()
            print(f"  {C.BOLD}{C.UNDERLINE}Commands{C.RESET}")
            print(f"  {C.DIM}/help{C.RESET}                                  Show this message")
            print(f"  {C.DIM}/exit{C.RESET}                                  Quit the game")
            print(f"  {C.DIM}/stats{C.RESET}                                 Show win/loss statistics")
            print(f"  {C.DIM}/graph{C.RESET}                                 Show balance graph")
            print(f"  {C.DIM}/editbalance [player] [balance]{C.RESET}        Edit a player's balance")
            print(f"  {C.DIM}/showbalance{C.RESET}                           Show all balances")
            print(f"  {C.DIM}/shuffle{C.RESET}                               Reshuffle the deck")
            print(f"  {C.DIM}/save [name]{C.RESET}                           Save the game")
            print(f"  {C.DIM}/load [name]{C.RESET}                           Info on loading a save")
            print()

        elif command == 'exit':
            self.end_game()
            quit()

        elif command == 'stats':
            self.print_stats()

        elif command == 'graph':
            parse_log_and_plot(self.log_file, block=False)

        elif command.split()[0] == 'editbalance':
            parts = command.split()
            if len(parts) < 3:
                print(f"  {C.RED}Usage: /editbalance [player name] [new balance]{C.RESET}")
                return
            new_balance_str = parts[-1]
            player_name = ' '.join(parts[1:-1])
            target_player = None
            for player in self.players + self.bots:
                if player.name.lower() == player_name.lower():
                    target_player = player
                    break
            if target_player is None:
                print(f"  {C.RED}Player '{player_name}' not found.{C.RESET}")
                return
            try:
                new_balance = float(new_balance_str)
                if new_balance < 0:
                    print(f"  {C.RED}Balance cannot be negative.{C.RESET}")
                    return
                old_balance = target_player.balance
                target_player.balance = new_balance
                print(f"  {C.GREEN}{target_player.name}: ${old_balance} → ${new_balance}{C.RESET}")
            except ValueError:
                print(f"  {C.RED}Invalid balance — must be a number.{C.RESET}")

        elif command == 'showbalance':
            print()
            print(f"  {C.BOLD}Balances{C.RESET}")
            for player in self.players + self.bots:
                print(f"  {player.name}: {C.GREEN}${player.balance}{C.RESET}")
            print()

        elif command == 'shuffle':
            self.check_shoe(force=True)

        elif command.split()[0] == 'godmode':
            global GOD_MODE
            parts = command.split()
            if parts[-1] == "on":
                GOD_MODE = True
                print(f"  {C.MAGENTA}GOD MODE ON{C.RESET}")
            elif parts[-1] == "off":
                GOD_MODE = False
                print(f"  {C.MAGENTA}GOD MODE OFF{C.RESET}")

        elif command.split()[0] == 'save':
            parts = command.split(maxsplit=1)
            save_name = parts[1].strip() if len(parts) > 1 else f"save_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            self.save_game(save_name)

        elif command.split()[0] == 'load':
            print(f"  {C.CYAN}To load a save: python3 play_blackjack.py --load <save name>{C.RESET}")

        else:
            print(f"  {C.RED}Unknown command — run /help for a list of commands.{C.RESET}")


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class TrainableBot(Player):
    def __init__(self, name, balance, state_size, action_size, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        super().__init__(name, balance)
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = 0.95
        self.memory = []

    def get_state(self, hand, dealer_visible_card):
        hand_value = hand.calculate_value() / 21.0
        dealer_value = get_card_value(dealer_visible_card) / 11.0
        return np.array([hand_value, dealer_value])

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(["hit", "stand", "double"])
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            return ["hit", "stand", "double"][torch.argmax(q_values).item()]

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, dtype=torch.float32)
            done_tensor = torch.tensor(done, dtype=torch.float32)
            q_values = self.q_network(state_tensor)
            target = q_values.clone().detach()
            action_index = ["hit", "stand", "double"].index(action)
            if done:
                target[action_index] = reward
            else:
                next_q_values = self.q_network(next_state_tensor)
                target[action_index] = reward + self.gamma * torch.max(next_q_values)
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target)
            loss.backward()
            self.optimizer.step()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def play_turn(self, shoe, dealer_visible_card, deal_delay: int):
        self.last_state = None
        self.last_action = None

        for hand in self.hands:
            print(f"\n  {C.BOLD}{self.name}{C.RESET}'s turn:")
            print(f"  Hand: {hand}")
            while not hand.is_blackjack() and not hand.is_busted():
                state = self.get_state(hand, dealer_visible_card)
                action = self.choose_action(state)
                print(f"  {C.DIM}{self.name} → {action}  (ε: {self.epsilon:.3f}){C.RESET}")
                if action == "hit":
                    card = shoe.draw_card()
                    hand.add_card(card)
                    next_state = self.get_state(hand, dealer_visible_card)
                    if hand.is_busted():
                        reward = -1
                        done = True
                    else:
                        reward = 0
                        done = False
                    self.store_experience(state, "hit", reward, next_state, done)
                    self.last_state = next_state
                    self.last_action = "hit"
                    time.sleep(deal_delay)
                    print(f"  {self.name} hits: {hand}")
                elif action == "stand":
                    self.last_state = state
                    self.last_action = "stand"
                    break
                elif action == "double" and self.balance >= self.current_bet:
                    bet = self.hand_bets[0] if self.hand_bets else self.current_bet
                    self.balance -= bet
                    self.hand_bets[0] = bet * 2
                    card = shoe.draw_card()
                    hand.add_card(card)
                    next_state = self.get_state(hand, dealer_visible_card)
                    reward = -1 if hand.is_busted() else 0
                    self.store_experience(state, "double", reward, next_state, True)
                    self.last_state = next_state
                    self.last_action = "double"
                    break

    def update_final_reward(self, result):
        if self.last_state is None:
            return
        if result == "blackjack":
            reward = 1.5
        elif result == "win":
            reward = 1.0
        elif result == "push":
            reward = 0.0
        elif result in ("lose", "busted"):
            reward = -1.0
        else:
            reward = 0.0
        if self.last_action and result != "busted":
            self.store_experience(self.last_state, self.last_action, reward, self.last_state, True)
        self.replay()

    def save_model(self, file_path):
        save_model(self.q_network, self.optimizer, file_path, additional_data={"epsilon": self.epsilon})

    def load_model(self, file_path):
        additional_data = load_model(self.q_network, self.optimizer, file_path)
        self.epsilon = additional_data.get("epsilon", 1.0)


#== Functions ==#

def get_card_value(card_str: str) -> int:
    rank = card_str.split(" ")[0]
    if rank in ["Jack", "Queen", "King"]:
        return 10
    elif rank == "Ace":
        return 11
    else:
        return int(rank)


def parse_log_and_plot(log_file, block=True):
    try:
        with open(log_file, 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        print(f"  {C.RED}Log file {log_file} not found.{C.RESET}")
        return
    except json.JSONDecodeError:
        print(f"  {C.RED}Failed to parse log file {log_file}.{C.RESET}")
        return

    with open('game_settings.json', 'r') as file:
        settings = json.load(file)

    try:
        init_bal = settings["init_balance"]
        player_balances = {}
        players = logs[0].get("players", [])
        for player in players:
            name = player.get("name")
            player_balances[name] = [(0, init_bal)]

        for round_number, entry in enumerate(logs, start=1):
            players = entry.get("players", [])
            for player in players:
                name = player.get("name")
                balance = player.get("balance")
                if name and balance is not None:
                    if name not in player_balances:
                        player_balances[name] = []
                    if round_number == 0:
                        player_balances[name].append((0, init_bal))
                    else:
                        player_balances[name].append((round_number, balance))

        plt.figure(figsize=(10, 6))
        for player, rounds_and_balances in player_balances.items():
            rounds, balances = zip(*rounds_and_balances)
            plt.plot(rounds, balances, marker='o', linestyle='-', label=player)

        plt.title("Player Balances Over Rounds")
        plt.xlabel("Round Number")
        plt.ylabel("Balance")
        plt.grid(True)
        plt.legend()
        plt.show(block=block)
    except IndexError:
        print(f'  {C.YELLOW}Cannot create graph — no data yet.{C.RESET}')


def save_model(model, optimizer, file_path, additional_data=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "additional_data": additional_data
    }
    torch.save(checkpoint, file_path)
    print(f"  {C.DIM}Model saved to {file_path}{C.RESET}")


def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"  {C.DIM}Model loaded from {file_path}{C.RESET}")
    return checkpoint.get("additional_data", {})


#== Main ==#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Train AI mode — no player interaction")
    parser.add_argument("--load", type=str, default=None, help="Load a saved game by save name")
    args = parser.parse_args()

    with open('game_settings.json', 'r') as file:
        settings = json.load(file)

    global SHOW_OUTPUT, LOG_GAME, SHOW_GRAPH, DEFAULT_BET, TRAIN_MODE
    SHOW_OUTPUT = bool(settings["show_output"])
    LOG_GAME = bool(settings["log_game"])
    SHOW_GRAPH = bool(settings["show_graph"])
    DEFAULT_BET = settings["default_bet"]
    TRAIN_MODE = args.train

    if SHOW_GRAPH and not LOG_GAME:
        print(f"  {C.YELLOW}show_graph is enabled but log_game is false — graph will be skipped.{C.RESET}")
        SHOW_GRAPH = False

    if TRAIN_MODE and not settings.get('bots'):
        print(f"  {C.RED}Train mode requires at least one bot configured in game_settings.json.{C.RESET}")
        quit()

    save_data = None
    if args.load:
        save_name = args.load.replace(".json", "")
        save_path = f"saves/{save_name}.json"
        try:
            with open(save_path, 'r') as f:
                save_data = json.load(f)
            print(f"  {C.GREEN}Loaded save: {save_name}{C.RESET}")
        except FileNotFoundError:
            print(f"  {C.RED}Save file '{save_path}' not found.{C.RESET}")
            quit()

    clear()
    print()
    print(f'  {C.BOLD}{C.YELLOW}╔══════════════════════════════╗{C.RESET}')
    print(f'  {C.BOLD}{C.YELLOW}║   ♠  BLACKJACK               ║{C.RESET}')
    print(f'  {C.BOLD}{C.YELLOW}╚══════════════════════════════╝{C.RESET}')
    print()

    players = []
    i = 1

    if save_data:
        players = [p["name"] for p in save_data["players"]]
    else:
        while True:
            if TRAIN_MODE:
                break
            player_i = input(f"  Player {i} name ('done' to start): ").strip()
            if player_i.lower() != "done":
                players.append(player_i)
                i += 1
            else:
                if players:
                    break
                print(f"  {C.RED}Must have at least one player.{C.RESET}")

    if LOG_GAME:
        if not os.path.exists("logs"):
            os.mkdir("logs")
        log_file = f'logs/session_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.json'
        with open(log_file, 'w') as file:
            file.write("[]")
    else:
        log_file = 'logs/session_NULL.json'

    bj_game = BlackjackGame(
        num_decks=settings['num_decks'],
        player_names=players,
        bot_info=settings['bots'],
        starting_balance=settings['init_balance'],
        log_file=log_file,
        deal_delay=settings['deal_delay'],
        allow_double_after_split=settings.get('allow_double_after_split', True),
        split_limit=settings.get('split_limit', 1),
        allow_insurance=settings.get('allow_insurance', True),
        save_data=save_data
    )

    bj_game.play_game()


if __name__ == "__main__":
    main()
