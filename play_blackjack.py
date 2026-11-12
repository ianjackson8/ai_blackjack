#!myenv/bin/python3.12
'''
Play blackjack in terminal

Author: Ian Jackson
Version v5.0
'''

import random
import os
import sys
import json
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from collections import deque
from itertools import count as _counter
from typing import List, Optional, Union
from datetime import datetime

from counter import BASE_EDGE, bet_units, load_counter, make_tracker, save_counter, train_counter, evaluate_counter
from strategy import basic_strategy
from tui import (HeadlessUI, RichUI, card_markup, edit_settings, esc, hand_label, hand_total, key_value_table, money,
                 signed_money, stats_table)

#== Global Variables ==#
SHOW_OUTPUT = True
LOG_GAME = True
SHOW_GRAPH = True
DEFAULT_BET = 10
TRAIN_MODE = False
GOD_MODE = False

NULL_UI = HeadlessUI(verbose=False)

SETTINGS_FILE = 'game_settings.json'
BOT_STRATEGIES = ["by the books", "default", "ai-nn", "counter", "counter-hilo"]

# Fields shown in the in-game settings editor (bots are edited separately)
SETTINGS_FIELDS = [
    {"key": "num_decks", "label": "Decks in shoe", "kind": "int", "min": 1, "desc": "Number of 52-card decks"},
    {"key": "init_balance", "label": "Starting balance", "kind": "money", "min": 1, "desc": "New players and bots start with this"},
    {"key": "default_bet", "label": "Default bet", "kind": "money", "min": 1, "desc": "Bots always bet this; your first suggested bet"},
    {"key": "deal_delay", "label": "Deal delay", "kind": "seconds", "min": 0, "desc": "Pause between cards (0 = instant)"},
    {"key": "allow_insurance", "label": "Insurance", "kind": "bool", "desc": "Offer insurance when the dealer shows an Ace"},
    {"key": "allow_double_after_split", "label": "Double after split", "kind": "bool", "desc": "Allow doubling down on split hands"},
    {"key": "split_limit", "label": "Split limit", "kind": "split_limit", "min": -1, "desc": "Max splits per hand (-1 = unlimited)"},
    {"key": "log_game", "label": "Log games", "kind": "bool", "desc": "Write every hand to logs/"},
    {"key": "show_graph", "label": "Balance graph", "kind": "bool", "desc": "Plot balances on exit (needs Log games)"},
    {"key": "show_output", "label": "Show table in --train", "kind": "bool", "desc": "Off = fast one-line progress"},
    {"key": "counter.max_spread", "label": "Counter bet spread", "kind": "int", "min": 1, "default": 8,
     "desc": "Most a counter bot bets, in units of default bet"},
    {"key": "train_param.rounds", "label": "Training rounds", "kind": "int", "min": 1, "default": 60000,
     "desc": "How many rounds --train plays"},
]

COMMANDS = [
    ("/help", "Show this message"),
    ("/exit", "Quit the game"),
    ("/stats", "Show win/loss statistics"),
    ("/graph", "Show balance graph"),
    ("/editbalance [player] [balance]", "Edit a player's balance"),
    ("/showbalance", "Show all balances"),
    ("/shuffle", "Reshuffle the deck"),
    ("/save [name]", "Save the game"),
    ("/load [name]", "Info on loading a save"),
]

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


HILO = {"2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "10": -1, "Jack": -1, "Queen": -1, "King": -1, "Ace": -1}


_SHOE_IDS = _counter(1)


class Shoe:
    def __init__(self, num_decks: int = 1):
        self.num_decks = num_decks
        self.shuffle_id = next(_SHOE_IDS)   # changes on every fresh shoe, so counters know to reset
        self.cards = self._generate_shoe()
        self.running_count = 0   # Hi-Lo count of every card drawn since the shuffle
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
            # ran dry mid-round (small shoe, lots of splits/hits): refill rather than crash
            self.cards = self._generate_shoe()
            self.running_count = 0
            self.shuffle_id = next(_SHOE_IDS)
            self.shuffle()
        card = self.cards.pop()
        self.running_count += HILO.get(card.rank, 0)
        return card

    def true_count(self) -> float:
        '''Running count per remaining deck — positive means the shoe is rich in tens and aces.'''
        return self.running_count / max(len(self.cards) / 52, 0.5)

    def penetration(self) -> float:
        '''Fraction of the shoe dealt so far.'''
        return 1 - len(self.cards) / (self.num_decks * 52)

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
        self.insurance_bet = 0
        self.last_bet = 0
        self.round_start_balance = balance
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
            raise ValueError("Bet amount must be greater than zero.")
        if amount > self.balance:
            raise ValueError("Bet amount exceeds available balance.")
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
            raise ValueError("Cannot split: hand must have exactly two cards of the same rank.")
        if self.current_bet > self.balance:
            raise ValueError(f"Insufficient balance to split (need ${self.current_bet} for second hand).")
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
            raise ValueError("Cannot double down after a hit.")
        if hand_index is None:
            hand_index = len(self.hands) - 1
        bet = self.hand_bets[hand_index] if self.hand_bets else self.current_bet
        if bet > self.balance:
            raise ValueError("Insufficient balance to double down.")
        self.balance -= bet
        self.hand_bets[hand_index] = bet * 2
        hand.add_card(card)

    def reset_for_new_round(self):
        self.round_start_balance = self.balance
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

    def play_turn(self, shoe: Shoe, dealer_visible_card: str, deal_delay: int, ui=None):
        ui = ui or NULL_UI
        name = f"[cyan]{esc(self.name)}[/]"
        for i, hand in enumerate(self.hands):
            ui.set_active(self, i)
            ui.pause(deal_delay / 2)
            while not hand.is_blackjack() and not hand.is_busted():
                action = self.make_decision(dealer_visible_card, hand)

                if action == "hit":
                    card = shoe.draw_card()
                    hand.add_card(card)
                    ui.log(f"{name} hits → {card_markup(card)}  ({hand_label(hand)})")
                    ui.pause(deal_delay)
                elif action == "stand":
                    ui.log(f"{name} stands on {hand_label(hand)}")
                    ui.pause(deal_delay / 2)
                    break
                elif action == "double":
                    if (float(self.balance) >= float(self.current_bet)) and (hand.num_cards() == 2):
                        bet = self.hand_bets[0] if self.hand_bets else self.current_bet
                        self.balance -= bet
                        self.hand_bets[0] = bet * 2
                        card = shoe.draw_card()
                        hand.add_card(card)
                        ui.log(f"{name} [gold1]doubles[/] → {card_markup(card)}  ({hand_label(hand)})")
                        ui.pause(deal_delay)
                        break
                    else:
                        card = shoe.draw_card()
                        hand.add_card(card)
                        reason = "after hitting" if hand.num_cards() > 3 else "— low balance"
                        ui.log(f"{name} can't double {reason}, hits → {card_markup(card)}  ({hand_label(hand)})")
                        ui.pause(deal_delay)
                else:
                    ui.log(f"[red]{name}: invalid action {esc(action)}[/]")
                    break
            if hand.is_busted():
                ui.log(f"{name} [red]busts[/] with {hand.calculate_value()}")

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
            raise ValueError(f"Invalid bot strategy: {self.strategy}.")


COUNTER_MODEL_PATH = "models/card_counter_lstm.pth"
# strategy name -> how the bot tracks the shoe ("counter" is the LSTM; the others are for comparison)
COUNTER_TRACKERS = {"counter": "lstm", "counter-hilo": "hi-lo", "counter-perfect": "perfect", "counter-flat": "flat"}


class CounterBot(Bot):
    '''
    Plays perfect basic strategy (including splits) and sizes bets from a card-count tracker.
    With strategy "counter" the tracker is the LSTM in counter.py; it watches every card that
    becomes visible and its predicted shoe composition is turned into an edge estimate.
    '''

    def __init__(self, name: str, balance: int, strategy: str = "counter", num_decks: int = 6,
                 model=None, das: bool = True, split_limit: int = -1, max_spread: int = 8):
        super().__init__(name, balance, strategy)
        kind = COUNTER_TRACKERS.get(strategy, "lstm")
        self.missing_model = kind == "lstm" and model is None
        if self.missing_model:
            kind = "flat"
        self.tracker_kind = kind
        self.tracker = make_tracker(kind, num_decks, model)
        self.das = das
        self.split_limit = split_limit
        self.max_spread = max_spread
        self.last_edge = BASE_EDGE
        self._shoe_id = None

    def _sync(self, shoe: Shoe) -> None:
        if shoe.shuffle_id != self._shoe_id:
            self.tracker.reset(shoe.num_decks)
            self._shoe_id = shoe.shuffle_id

    def choose_bet(self, shoe: Shoe, base_bet) -> float:
        self._sync(shoe)
        self.last_edge = self.tracker.edge()
        units = 1 if self.tracker_kind == "flat" else bet_units(self.last_edge, self.max_spread)
        return min(units * base_bet, self.balance)

    def observe(self, ranks: List[str], shoe: Shoe) -> None:
        '''Called after each round with every card that was shown.'''
        self._sync(shoe)
        self.tracker.observe(ranks)

    def count_summary(self) -> str:
        '''Short markup line for the seat panel.'''
        if self.tracker_kind == "flat":
            return "[grey50]no model — run --train-counter[/]" if self.missing_model else "[grey50]flat betting[/]"
        return f"TC {self.tracker.true_count():+.1f} · edge {self.tracker.edge() * 100:+.1f}%"

    def bet_note(self) -> str:
        '''Extra detail for the log when the bot raises its bet.'''
        note = f"edge {self.last_edge * 100:+.2f}%"
        if self.tracker_kind == "lstm":
            note += f" · LSTM count {self.tracker.true_count():+.1f} / real {self.tracker.exact_true_count():+.1f}"
        return note

    def play_turn(self, shoe: Shoe, dealer_visible_card: str, deal_delay: int, ui=None):
        ui = ui or NULL_UI
        name = f"[cyan]{esc(self.name)}[/]"
        up = dealer_visible_card.split(" ")[0]
        i = 0
        while i < len(self.hands):
            hand = self.hands[i]
            ui.set_active(self, i)
            ui.pause(deal_delay / 2)
            while not hand.is_busted() and hand.calculate_value() < 21:
                stake = self.hand_bets[i] if i < len(self.hand_bets) else self.current_bet
                two = hand.num_cards() == 2
                can_double = two and self.balance >= stake and (len(self.hands) == 1 or self.das)
                can_split = (two and hand.cards[0].rank == hand.cards[1].rank and self.balance >= self.current_bet
                             and (self.split_limit == -1 or len(self.hands) <= self.split_limit))
                action = basic_strategy([c.rank for c in hand.cards], up, can_double, can_split, self.das)

                if action == "split":
                    rank = hand.cards[0].rank
                    self.split(i)
                    ui.log(f"{name} [gold1]splits[/] a pair of {esc(rank)}s")
                    for j in (i, i + 1):
                        self.hands[j].add_card(shoe.draw_card())
                        ui.pause(deal_delay)
                    hand = self.hands[i]
                elif action == "hit":
                    card = shoe.draw_card()
                    hand.add_card(card)
                    ui.log(f"{name} hits → {card_markup(card)}  ({hand_label(hand)})")
                    ui.pause(deal_delay)
                elif action == "double":
                    self.double_down(shoe.draw_card(), hand, i)
                    ui.log(f"{name} [gold1]doubles[/] → {card_markup(hand.cards[-1])}  ({hand_label(hand)})")
                    ui.pause(deal_delay)
                    break
                else:
                    ui.log(f"{name} stands on {hand_label(hand)}")
                    ui.pause(deal_delay / 2)
                    break
            if hand.is_busted():
                ui.log(f"{name} [red]busts[/] with {hand.calculate_value()}")
            i += 1


class BlackjackGame:
    def __init__(self, num_decks: int, player_names: List[str], bot_info: List[dict], starting_balance: int, log_file: str,
                 deal_delay: int, allow_double_after_split: bool = True, split_limit: int = 1,
                 allow_insurance: bool = True, save_data: dict = None, ui=None, counter_max_spread: int = 8):
        self.ui = ui or NULL_UI
        self.num_decks = num_decks
        self.shoe = Shoe(num_decks)
        self.shoe.shuffle()
        self.dealer = Player("Dealer", balance=0)
        self.deal_delay = deal_delay
        self.allow_double_after_split = allow_double_after_split
        self.split_limit = split_limit if split_limit != 0 else 1
        self.allow_insurance = allow_insurance
        self.phase = "betting"

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

        self.counter_model = None
        if any(b.get('strategy') == 'counter' for b in bot_info):
            try:
                self.counter_model = load_counter(COUNTER_MODEL_PATH)
                self.ui.log(f"[grey50]Card-counter LSTM loaded from {COUNTER_MODEL_PATH}[/]")
            except (FileNotFoundError, ValueError):
                self.ui.log("[yellow]No card-counter model yet — counter bots bet flat. "
                            "Train one with: python3 play_blackjack.py --train-counter[/]")

        self.bots = []
        for bot in bot_info:
            bal = saved_bot_balances.get(bot['name'], starting_balance)
            if bot.get('strategy') == 'ai-nn':
                self.bots.append(TrainableBot(bot['name'], bal))
            elif bot.get('strategy', '') in COUNTER_TRACKERS:
                self.bots.append(CounterBot(bot['name'], bal, bot['strategy'], num_decks=num_decks,
                                            model=self.counter_model, das=allow_double_after_split,
                                            split_limit=self.split_limit, max_spread=counter_max_spread))
            else:
                self.bots.append(Bot(bot['name'], bal, bot['strategy']))

        self.log_file = log_file
        self.log_entries = []
        self.starting_balance = starting_balance
        self.train_until = None
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
                    additional_data = bot.load(f"models/{bot_name}_model.pth", log=self.ui.log)
                    bot.epsilon = saved_bot_epsilons.get(bot.name, additional_data.get("epsilon", 1.0))
                except FileNotFoundError:
                    self.ui.log(f"[cyan]No saved model for {esc(bot.name)} — starting fresh.[/]")
                    bot.epsilon = saved_bot_epsilons.get(bot.name, 1.0)
                except ValueError:
                    self.ui.log(f"[yellow]{esc(bot.name)}'s saved model is an older version — starting fresh "
                                f"(it will be overwritten on exit).[/]")
                    bot.epsilon = 1.0

    @property
    def seats(self) -> List[Player]:
        return self.players + self.bots

    def setup_round(self) -> None:
        for player in self.seats:
            player.reset_for_new_round()
        self.dealer.reset_for_new_round()
        self.round_data = {}
        self.game_num += 1
        self.phase = "betting"

    def deal_cards(self) -> None:
        global GOD_MODE
        self.phase = "playing"
        in_play = [p for p in self.seats if p.current_bet > 0]
        delay = min(self.deal_delay, 0.25)
        for player in in_play:
            if GOD_MODE and not isinstance(player, (Bot, TrainableBot)):
                player.hands = [Hand([Card("Ace", "Hearts"), Card("King", "Hearts")])]
        # deal like a real table: one card each, round twice, dealer last
        for _ in range(2):
            for player in in_play:
                if GOD_MODE and not isinstance(player, (Bot, TrainableBot)):
                    continue
                player.hit(self.shoe.draw_card())
                self.ui.pause(delay)
            self.dealer.hit(self.shoe.draw_card())
            self.ui.pause(delay)

    def display_table(self) -> None:
        self.ui.set_active(None)

    def _action_options(self, player: Player, hand: Hand, hand_index: int) -> list:
        options = [("h1", "hit", "Hit"), ("s2", "stand", "Stand")]
        bet = player.hand_bets[hand_index] if hand_index < len(player.hand_bets) else player.current_bet
        two_cards = hand.num_cards() == 2
        if two_cards and bet <= player.balance and (len(player.hands) == 1 or self.allow_double_after_split):
            options.append(("d3", "double", "Double"))
        can_split = (self.split_limit == -1) or (len(player.hands) <= self.split_limit)
        if two_cards and hand.cards[0].rank == hand.cards[1].rank and player.current_bet <= player.balance and can_split:
            options.append(("p4", "split", "Split"))
        return options

    def player_turn(self, player: Player) -> None:
        dealer_visible_card = str(self.dealer.hands[0].cards[1])
        name = f"[green3]{esc(player.name)}[/]"

        hand_index = 0
        while hand_index < len(player.hands):
            hand = player.hands[hand_index]

            while True:
                self.ui.set_active(player, hand_index)
                if hand.is_blackjack():
                    self.ui.log(f"{name} has [bold gold1]Blackjack![/]")
                    self.ui.pause(self.deal_delay / 2)
                    break
                elif hand.is_busted():
                    self.ui.log(f"{name} [red]busts[/] with {hand.calculate_value()}")
                    self.ui.pause(self.deal_delay / 2)
                    break
                elif hand.calculate_value() == 21:
                    self.ui.log(f"{name} stands on 21")
                    player.log_action("stand", dealer_visible_card)
                    break

                where = f" · hand {hand_index + 1}/{len(player.hands)}" if len(player.hands) > 1 else ""
                prompt = f"[bold]{esc(player.name)}[/]{where} — [bold]{hand_label(hand)}[/] vs dealer {card_markup(self.dealer.hands[0].cards[1])}"
                action = self.ui.choose(prompt, self._action_options(player, hand, hand_index))

                if action == "hit":
                    card = self.shoe.draw_card()
                    hand.add_card(card)
                    self.ui.log(f"{name} hits → {card_markup(card)}  ({hand_label(hand)})")
                    self.ui.pause(self.deal_delay)
                    player.log_action("hit", dealer_visible_card)
                elif action == "stand":
                    self.ui.log(f"{name} stands on {hand_label(hand)}")
                    player.log_action("stand", dealer_visible_card)
                    break
                elif action == "double":
                    if len(player.hands) > 1 and not self.allow_double_after_split:
                        self.ui.error("Cannot double down after splitting (disabled in settings).")
                    else:
                        try:
                            player.double_down(self.shoe.draw_card(), hand, hand_index)
                            player.log_action("double", dealer_visible_card)
                        except ValueError as e:
                            self.ui.error(str(e))
                            continue
                        self.ui.log(f"{name} [gold1]doubles[/] → {card_markup(hand.cards[-1])}  ({hand_label(hand)})")
                        self.ui.pause(self.deal_delay)
                        if hand.is_busted():
                            self.ui.log(f"{name} [red]busts[/] with {hand.calculate_value()}")
                        break
                elif action == "split":
                    can_split = (self.split_limit == -1) or (len(player.hands) <= self.split_limit)
                    if can_split:
                        try:
                            player.split(hand_index)
                            player.log_action("split", dealer_visible_card)
                        except ValueError as e:
                            self.ui.error(str(e))
                            continue
                        self.ui.log(f"{name} [gold1]splits[/] a pair of {esc(player.hands[hand_index].cards[0].rank)}s")
                        self.ui.pause(self.deal_delay / 2)
                        player.hands[hand_index].add_card(self.shoe.draw_card())
                        self.ui.pause(self.deal_delay)
                        player.hands[hand_index + 1].add_card(self.shoe.draw_card())
                        self.ui.pause(self.deal_delay)
                        hand = player.hands[hand_index]
                    else:
                        self.ui.error(f"Cannot split: split limit of {self.split_limit} reached.")
                else:
                    self.ui.error("Invalid action.")

            hand_index += 1

    def dealer_turn(self) -> None:
        self.phase = "dealer"
        hand = self.dealer.hands[0]
        self.ui.set_active(self.dealer)
        self.ui.log(f"Dealer reveals {card_markup(hand.cards[0])}  ({hand_label(hand)})")
        self.ui.pause(self.deal_delay)
        while hand.calculate_value() < 17:
            card = self.shoe.draw_card()
            hand.add_card(card)
            self.ui.log(f"Dealer hits → {card_markup(card)}  ({hand_label(hand)})")
            self.ui.pause(self.deal_delay)
        if hand.is_busted():
            self.ui.log("[bold red]Dealer busts![/]")
        else:
            self.ui.log(f"Dealer stands on {hand.calculate_value()}")

    def offer_insurance(self) -> None:
        for player in self.players:
            if player.current_bet == 0:
                continue
            max_insurance = player.current_bet / 2
            self.ui.set_active(player)
            while True:
                raw = self.ui.ask(
                    f"[bold]{esc(player.name)}[/] — dealer shows an Ace. Insurance? (up to {money(max_insurance)}, 0 to skip)",
                    default="0")
                try:
                    amount = float(raw)
                except ValueError:
                    self.ui.error("Enter a number.")
                    continue
                if amount < 0 or amount > max_insurance:
                    self.ui.error(f"Insurance must be between $0 and {money(max_insurance)}.")
                    continue
                if amount > player.balance:
                    self.ui.error("Insufficient balance.")
                    continue
                player.insurance_bet = amount
                player.balance -= amount
                if amount > 0:
                    self.ui.log(f"[green3]{esc(player.name)}[/] takes insurance for {money(amount)}")
                break

    def determine_winner(self, player: Player) -> None:
        name = f"[bold]{esc(player.name)}[/]"
        # Resolve insurance before main hand
        if player.insurance_bet > 0:
            if self.dealer.hands[0].is_blackjack():
                player.balance += player.insurance_bet * 3  # stake back + 2:1
                self.ui.log(f"{name} insurance pays [green]+{money(player.insurance_bet * 2)}[/]")
            else:
                self.ui.log(f"{name} insurance loses [red]-{money(player.insurance_bet)}[/]")

        dealer_value = self.dealer.hands[0].calculate_value()

        for i, hand in enumerate(player.hands):
            bet = player.hand_bets[i] if player.hand_bets and i < len(player.hand_bets) else player.current_bet
            player_value = hand.calculate_value()
            if hand.is_busted():
                self.ui.log(f"{name} {player_value} · [red]bust -{money(bet)}[/]")
                player.results.append("busted")
                player.stats["bust"] += 1
            elif hand.is_blackjack():
                winnings = bet * 2.5
                player.balance += winnings
                self.ui.log(f"{name} {player_value} · [bold gold1]blackjack +{money(bet * 1.5)}[/]")
                player.results.append("blackjack")
                player.stats["blackjack"] += 1
            elif dealer_value > 21 or player_value > dealer_value:
                winnings = bet * 2
                player.balance += winnings
                self.ui.log(f"{name} {player_value} · [green]win +{money(bet)}[/]")
                player.results.append("win")
                player.stats["win"] += 1
            elif player_value == dealer_value:
                player.balance += bet
                self.ui.log(f"{name} {player_value} · [grey70]push[/]")
                player.results.append("push")
                player.stats["push"] += 1
            else:
                self.ui.log(f"{name} {player_value} · [red]lose -{money(bet)}[/]")
                player.results.append("lose")
                player.stats["lose"] += 1

    def collect_bet(self, player: Player) -> None:
        if isinstance(player, TrainableBot):
            bet = player.choose_bet(self.shoe, DEFAULT_BET)
            if float(bet) > 0:
                player.place_bet(bet)
                if bet != DEFAULT_BET:
                    self.ui.log(f"[cyan]{esc(player.name)}[/] bets {money(bet)}  "
                                f"[grey50]true count {self.shoe.true_count():+.1f}[/]")
            return
        if isinstance(player, CounterBot):
            bet = player.choose_bet(self.shoe, DEFAULT_BET)
            if float(bet) > 0:
                player.place_bet(bet)
                if bet != DEFAULT_BET:
                    self.ui.log(f"[cyan]{esc(player.name)}[/] bets {money(bet)}  "
                                f"[grey50]{player.bet_note()}[/]")
            return
        if isinstance(player, Bot):
            bet = min(DEFAULT_BET, player.balance)
            if float(bet) > 0:
                player.place_bet(bet)
            return

        self.ui.set_active(player)
        while True:
            suggested = player.last_bet if 0 < player.last_bet <= player.balance else min(DEFAULT_BET, player.balance)
            raw = self.ui.ask(
                f"[bold]{esc(player.name)}[/] ([green]{money(player.balance)}[/]) — bet amount, 0 to sit out, /help for commands",
                default=f"{suggested:g}")
            if raw.startswith('/'):
                self.parse_command(raw[1:])
                continue
            try:
                bet = float(raw)
                if bet.is_integer():
                    bet = int(bet)
            except ValueError:
                self.ui.error("Bet must be a number.")
                continue
            if bet == 0:
                self.ui.log(f"[grey62]{esc(player.name)} sits out this round[/]")
                return
            try:
                player.place_bet(bet)
            except ValueError as e:
                self.ui.error(str(e))
                continue
            player.last_bet = bet
            self.ui.log(f"[green3]{esc(player.name)}[/] bets {money(bet)}")
            return

    def play_round(self) -> None:
        self.check_shoe()
        self.setup_round()
        self.ui.log(f"[bold gold1]── Round {self.game_num} ──[/]")

        for player in self.seats:
            self.collect_bet(player)
        self.ui.set_active(None)

        self.deal_cards()
        self.display_table()

        if self.allow_insurance and self.dealer.hands[0].cards[1].rank == "Ace":
            self.offer_insurance()

        for player in self.seats:
            if player.current_bet > 0:
                if isinstance(player, (Bot, TrainableBot)):
                    player.play_turn(self.shoe, str(self.dealer.hands[0].cards[1]), self.deal_delay, ui=self.ui)
                else:
                    self.player_turn(player)

        self.dealer_turn()

        for player in self.seats:
            if player.current_bet > 0:
                self.determine_winner(player)
                if isinstance(player, TrainableBot):
                    player.learn_from_round(player.results[0] if player.results else "",
                                            player.balance - player.round_start_balance, DEFAULT_BET)
        self.phase = "results"
        self.ui.set_active(None)

        # counters see every card that was shown this round
        shown = [c.rank for c in self.dealer.hands[0].cards]
        shown += [c.rank for p in self.seats for h in p.hands for c in h.cards]
        for bot in self.bots:
            if isinstance(bot, CounterBot):
                bot.observe(shown, self.shoe)

        for player in self.seats:
            self.balance_history[player.name].append(player.balance)

        global LOG_GAME
        if LOG_GAME: self.log_game()
        self.ui.on_round_end(self)

    def prepare_training(self, rounds: int) -> None:
        '''Train mode: run a fixed number of rounds, and pace exploration to fit them.'''
        self.train_rounds = rounds
        self.train_until = self.game_num + rounds
        self.deal_delay = 0
        for bot in self.bots:
            if isinstance(bot, TrainableBot) and bot.epsilon > bot.epsilon_min:
                # decay so exploration bottoms out ~60% of the way through this run
                steps = max(1, int(rounds * 0.6))
                bot.epsilon_decay = (bot.epsilon_min / bot.epsilon) ** (1 / steps)

    def rebuy_broke_bots(self) -> None:
        '''Train mode: a bot that can't cover a bet gets a fresh stack so it keeps learning.'''
        for bot in self.bots:
            if bot.balance < DEFAULT_BET:
                bot.balance = self.starting_balance
                bot.rebuys = getattr(bot, "rebuys", 0) + 1
                # rebase history so the top-up doesn't count as a winning round in the stats
                self.balance_history[bot.name][-1] = bot.balance
                self.ui.log(f"[magenta]{esc(bot.name)} rebuys for {money(self.starting_balance)} "
                            f"(rebuy #{bot.rebuys})[/]")

    def play_game(self):
        self.ui.attach(self)
        while True:
            if TRAIN_MODE:
                if not self.bots or self.game_num >= self.train_until:
                    break
                self.rebuy_broke_bots()
            self.play_round()

            if TRAIN_MODE:
                continue

            choice = None
            while choice not in ("y", "n"):
                choice = self.ui.choose("Round over.", [("y ", "y", "Next round"), ("nq", "n", "Quit")], default="y")
                if choice not in ("y", "n"):
                    self.ui.error("Enter Y or N.")
            if choice == "n":
                break
        self.end_game()

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

        for player in self.seats:
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

        self.log_entries.append(log_entry)
        # rewriting the whole file each round gets slow over long training runs, so batch it there
        if not TRAIN_MODE or len(self.log_entries) % 1000 == 0:
            self.flush_log()

    def flush_log(self) -> None:
        if not LOG_GAME or not self.log_entries:
            return
        with open(self.log_file, "w") as file:
            json.dump(self.log_entries, file, indent=4)

    def check_shoe(self, force: bool = False):
        if (len(self.shoe) < (len(self.players) + len(self.bots) + 1) * 3) or force:
            self.ui.log("[magenta]Shuffling a fresh shoe…[/]")
            self.shoe = Shoe(self.num_decks)
            self.shoe.shuffle()

    def print_stats(self) -> None:
        self.ui.show_info(f"Session Stats — Round {self.game_num}", stats_table(self))

    def save_models(self) -> None:
        for bot in self.bots:
            if isinstance(bot, TrainableBot):
                bot.save(f"models/{bot.name.replace(' ', '_')}_model.pth", log=self.ui.log)

    def end_game(self):
        self.phase = "over"
        self.ui.set_active(None)
        self.flush_log()

        self.save_models()

        saved_msg = None
        if not TRAIN_MODE:
            self.ui.set_overlay(f"Session Summary — {self.game_num} rounds", stats_table(self))
            choice = None
            while choice not in ("y", "n"):
                choice = self.ui.choose("Save this game?", [("y", "y", "Save"), ("n", "n", "Don't save")])
                if choice not in ("y", "n"):
                    self.ui.error("Enter Y or N.")

            if choice == "y":
                default_name = f"save_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
                save_name = self.ui.ask("Save name", default=default_name)
                saved_msg = self.save_game(save_name)
                self.ui.wait_key(f"[green]{saved_msg}[/]  —  press any key to exit")

        self.ui.clear_overlay()
        self.ui.stop()

        self.ui.print_after(stats_table(self))
        if saved_msg:
            self.ui.print_after(f"[green]{saved_msg}[/]")

        global SHOW_GRAPH
        if SHOW_GRAPH: parse_log_and_plot(self.log_file, log=self.ui.print_after)

    def save_game(self, save_name: str) -> str:
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

        msg = f"Saved to {save_path} — load with: python3 play_blackjack.py --load {save_name}"
        self.ui.log(f"[green]{esc(msg)}[/]")
        return msg

    def parse_command(self, command: str):
        parts = command.split()
        if not parts:
            self.ui.error("Empty command — run /help for a list of commands.")
            return
        name = parts[0].lower()

        if name == 'help':
            self.ui.show_info("Commands", key_value_table(COMMANDS))

        elif name == 'exit':
            self.end_game()
            sys.exit(0)

        elif name == 'stats':
            self.print_stats()

        elif name == 'graph':
            parse_log_and_plot(self.log_file, block=False, log=self.ui.error)

        elif name == 'editbalance':
            if len(parts) < 3:
                self.ui.error("Usage: /editbalance [player name] [new balance]")
                return
            new_balance_str = parts[-1]
            player_name = ' '.join(parts[1:-1])
            target_player = next((p for p in self.seats if p.name.lower() == player_name.lower()), None)
            if target_player is None:
                self.ui.error(f"Player '{player_name}' not found.")
                return
            try:
                new_balance = float(new_balance_str)
            except ValueError:
                self.ui.error("Invalid balance — must be a number.")
                return
            if new_balance < 0:
                self.ui.error("Balance cannot be negative.")
                return
            old_balance = target_player.balance
            target_player.balance = new_balance
            self.ui.log(f"[magenta]{esc(target_player.name)}: {money(old_balance)} → {money(new_balance)}[/]")

        elif name == 'showbalance':
            rows = [(esc(p.name), f"[green]{money(p.balance)}[/]") for p in self.seats]
            self.ui.show_info("Balances", key_value_table(rows, key_style="bold"))

        elif name == 'shuffle':
            self.check_shoe(force=True)

        elif name == 'godmode':
            global GOD_MODE
            if parts[-1] == "on":
                GOD_MODE = True
                self.ui.log("[magenta]GOD MODE ON[/]")
            elif parts[-1] == "off":
                GOD_MODE = False
                self.ui.log("[magenta]GOD MODE OFF[/]")

        elif name == 'save':
            split = command.split(maxsplit=1)
            save_name = split[1].strip() if len(split) > 1 else f"save_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}"
            self.save_game(save_name)

        elif name == 'load':
            self.ui.error("To load a save: python3 play_blackjack.py --load <save name>")

        else:
            self.ui.error("Unknown command — run /help for a list of commands.")


PLAY_ACTIONS = ["hit", "stand", "double"]
BET_MULTIPLIERS = [1, 2, 4, 6, 8]   # bet sizes the NN bot can pick, in units of default_bet
MODEL_ARCH = "v5"                    # bump when the network shapes or state features change


class QNetwork(nn.Module):
    '''Small MLP: state -> one Q-value per action.'''

    def __init__(self, state_size: int, action_size: int, hidden=(64, 32)):
        super().__init__()
        layers, prev = [], state_size
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU()]
            prev = width
        layers.append(nn.Linear(prev, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _masked_argmax(q_values: torch.Tensor, valid: List[bool]) -> int:
    q = q_values.clone()
    q[~torch.tensor(valid)] = -float("inf")
    return int(torch.argmax(q).item())


class TrainableBot(Player):
    '''
    Q-learning bot with two networks:

      play network  [hand total/21, soft hand, can double, dealer upcard/11] -> Q(hit, stand, double)
      bet network   [true count/10, shoe penetration]                        -> Q(bet 1x, 2x, 4x, 6x, 8x)

    Play rewards are in units of the hand's stake (doubling counts double). Bet rewards are the
    round's net result in units of default_bet, so the bet network learns expected profit per bet size.
    '''

    PLAY_FEATURES = 4
    BET_FEATURES = 2

    def __init__(self, name, balance, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 batch_size=64, target_sync=250, updates_per_round=4):
        super().__init__(name, balance)
        self.q_network = QNetwork(self.PLAY_FEATURES, len(PLAY_ACTIONS), hidden=(64, 32))
        self.target_network = QNetwork(self.PLAY_FEATURES, len(PLAY_ACTIONS), hidden=(64, 32))
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)

        self.bet_network = QNetwork(self.BET_FEATURES, len(BET_MULTIPLIERS), hidden=(32, 16))
        self.bet_optimizer = optim.Adam(self.bet_network.parameters(), lr=1e-3)

        self.criterion = nn.SmoothL1Loss()
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = 1.0          # episodes are a few steps long and only the end pays out: no discounting
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.updates_per_round = updates_per_round
        self.train_steps = 0

        self.memory = deque(maxlen=50_000)       # (state, action_idx, reward, next_state, done)
        self.bet_memory = deque(maxlen=20_000)   # (state, multiplier_idx, reward)
        self.trajectory = []                     # this round's (state, action_idx) decisions
        self.doubled = False
        self.bet_state = None
        self.bet_action = None

    # ── state ────────────────────────────────────────────────────────────────
    def get_state(self, hand: Hand, dealer_visible_card: str, can_double: bool) -> np.ndarray:
        total, soft = hand_total(hand)
        return np.array([total / 21.0, float(soft), float(can_double),
                         get_card_value(dealer_visible_card) / 11.0], dtype=np.float32)

    @staticmethod
    def get_bet_state(shoe: Shoe) -> np.ndarray:
        return np.array([max(-1.0, min(1.0, shoe.true_count() / 10.0)), shoe.penetration()], dtype=np.float32)

    # ── decisions ────────────────────────────────────────────────────────────
    def choose_action(self, state: np.ndarray) -> int:
        valid = [True, True, bool(state[2])]
        if random.random() < self.epsilon:
            return random.choice([i for i, ok in enumerate(valid) if ok])
        with torch.no_grad():
            return _masked_argmax(self.q_network(torch.from_numpy(state)), valid)

    def choose_bet(self, shoe: Shoe, base_bet) -> float:
        '''Pick a bet size from the count. Falls back to whatever is left if below one unit.'''
        self.bet_state = self.bet_action = None
        if self.balance < base_bet:
            return self.balance
        valid = [m * base_bet <= self.balance for m in BET_MULTIPLIERS]
        state = self.get_bet_state(shoe)
        if random.random() < self.epsilon:
            idx = random.choice([i for i, ok in enumerate(valid) if ok])
        else:
            with torch.no_grad():
                idx = _masked_argmax(self.bet_network(torch.from_numpy(state)), valid)
        self.bet_state, self.bet_action = state, idx
        return BET_MULTIPLIERS[idx] * base_bet

    # ── play ─────────────────────────────────────────────────────────────────
    def play_turn(self, shoe, dealer_visible_card, deal_delay: int, ui=None):
        ui = ui or NULL_UI
        name = f"[cyan]{esc(self.name)}[/]"
        self.trajectory = []
        self.doubled = False

        hand = self.hands[0]  # the NN bot never splits
        ui.set_active(self, 0)
        ui.pause(deal_delay / 2)
        while not hand.is_blackjack() and not hand.is_busted() and hand.calculate_value() < 21:
            stake = self.hand_bets[0] if self.hand_bets else self.current_bet
            can_double = hand.num_cards() == 2 and self.balance >= stake
            state = self.get_state(hand, dealer_visible_card, can_double)
            idx = self.choose_action(state)
            self.trajectory.append((state, idx))
            action = PLAY_ACTIONS[idx]
            tag = f"  [grey50]ε {self.epsilon:.3f}[/]"

            if action == "hit":
                card = shoe.draw_card()
                hand.add_card(card)
                ui.log(f"{name} hits → {card_markup(card)}  ({hand_label(hand)}){tag}")
                ui.pause(deal_delay)
            elif action == "stand":
                ui.log(f"{name} stands on {hand_label(hand)}{tag}")
                ui.pause(deal_delay / 2)
                break
            else:  # double
                self.balance -= stake
                self.hand_bets[0] = stake * 2
                self.doubled = True
                card = shoe.draw_card()
                hand.add_card(card)
                ui.log(f"{name} [gold1]doubles[/] → {card_markup(card)}  ({hand_label(hand)}){tag}")
                ui.pause(deal_delay)
                break
        if hand.is_busted():
            ui.log(f"{name} [red]busts[/] with {hand.calculate_value()}")

    # ── learning ─────────────────────────────────────────────────────────────
    def learn_from_round(self, result: str, net_change: float, base_bet: float) -> None:
        '''Credit the round's outcome to every decision that led to it, then train both networks.'''
        outcome = {"blackjack": 1.5, "win": 1.0, "push": 0.0, "lose": -1.0, "busted": -1.0}.get(result, 0.0)
        if self.doubled:
            outcome *= 2
        for t, (state, idx) in enumerate(self.trajectory):
            if t == len(self.trajectory) - 1:
                self.memory.append((state, idx, outcome, state, True))    # final decision gets the payout
            else:
                self.memory.append((state, idx, 0.0, self.trajectory[t + 1][0], False))
        self.trajectory = []

        if self.bet_state is not None and base_bet:
            self.bet_memory.append((self.bet_state, self.bet_action, net_change / base_bet))
            self.bet_state = self.bet_action = None

        for _ in range(self.updates_per_round):
            self.replay()
            self.replay_bets()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def replay(self) -> None:
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states = torch.from_numpy(np.stack([b[0] for b in batch]))
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([b[3] for b in batch]))
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

        q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states)
            next_q[next_states[:, 2] < 0.5, 2] = -float("inf")   # can't double after a hit
            target = rewards + self.gamma * next_q.max(dim=1).values * (1 - dones)

        loss = self.criterion(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_sync == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def replay_bets(self) -> None:
        if len(self.bet_memory) < self.batch_size:
            return
        batch = random.sample(self.bet_memory, self.batch_size)
        states = torch.from_numpy(np.stack([b[0] for b in batch]))
        actions = torch.tensor([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        q = self.bet_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.criterion(q, rewards)   # one-step: each bet's value is just its round result
        self.bet_optimizer.zero_grad()
        loss.backward()
        self.bet_optimizer.step()

    # ── persistence ──────────────────────────────────────────────────────────
    def save(self, file_path: str, log=None) -> None:
        torch.save({
            "arch": MODEL_ARCH,
            "play_state_dict": self.q_network.state_dict(),
            "play_optimizer": self.optimizer.state_dict(),
            "bet_state_dict": self.bet_network.state_dict(),
            "bet_optimizer": self.bet_optimizer.state_dict(),
            "additional_data": {"epsilon": self.epsilon, "train_steps": self.train_steps},
        }, file_path)
        if log:
            log(f"[grey50]Model saved to {esc(file_path)}[/]")

    def load(self, file_path: str, log=None) -> dict:
        '''Load a checkpoint. Raises FileNotFoundError, or ValueError for an older architecture.'''
        checkpoint = torch.load(file_path)
        if checkpoint.get("arch") != MODEL_ARCH:
            raise ValueError(f"{file_path} is from an older model version")
        self.q_network.load_state_dict(checkpoint["play_state_dict"])
        self.target_network.load_state_dict(checkpoint["play_state_dict"])
        self.optimizer.load_state_dict(checkpoint["play_optimizer"])
        self.bet_network.load_state_dict(checkpoint["bet_state_dict"])
        self.bet_optimizer.load_state_dict(checkpoint["bet_optimizer"])
        extra = checkpoint.get("additional_data", {})
        self.train_steps = extra.get("train_steps", 0)
        if log:
            log(f"[grey50]Model loaded from {esc(file_path)}[/]")
        return extra


#== Functions ==#

def get_card_value(card_str: str) -> int:
    rank = card_str.split(" ")[0]
    if rank in ["Jack", "Queen", "King"]:
        return 10
    elif rank == "Ace":
        return 11
    else:
        return int(rank)


def parse_log_and_plot(log_file, block=True, log=print):
    try:
        with open(log_file, 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        log(f"Log file {log_file} not found.")
        return
    except json.JSONDecodeError:
        log(f"Failed to parse log file {log_file}.")
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
        if not block:
            plt.pause(0.001)  # let the window draw without blocking the game
    except IndexError:
        log("Cannot create graph — no data yet.")


def settings_summary(settings: dict) -> str:
    bots = ", ".join(f"{esc(b['name'])} [grey50]({esc(b['strategy'])})[/]" for b in settings.get("bots", [])) or "none"
    decks = settings["num_decks"]
    return (f"{decks} deck{'s' if decks != 1 else ''} · start {money(settings['init_balance'])} · "
            f"bot bet {money(settings['default_bet'])} · deal {settings['deal_delay']:g}s\nBots: {bots}")


def apply_settings(settings: dict) -> None:
    global SHOW_OUTPUT, LOG_GAME, SHOW_GRAPH, DEFAULT_BET
    SHOW_OUTPUT = bool(settings["show_output"])
    LOG_GAME = bool(settings["log_game"])
    SHOW_GRAPH = bool(settings["show_graph"])
    DEFAULT_BET = settings["default_bet"]


def open_settings(ui, settings: dict) -> None:
    if edit_settings(ui, settings, SETTINGS_FIELDS, BOT_STRATEGIES):
        with open(SETTINGS_FILE, 'w') as file:
            json.dump(settings, file, indent=4)
        ui.log(f"[green]Settings saved to {SETTINGS_FILE}[/]")
    ui.splash_info = settings_summary(settings)
    ui.refresh()


def ask_player_names(ui, settings: Optional[dict] = None) -> List[str]:
    players = []
    if settings is not None:
        ui.splash_info = settings_summary(settings)
    while True:
        hint = "Enter to start" if players else "at least one player needed"
        if settings is not None:
            hint += " · Tab or /settings for settings"
        name = ui.ask(f"[bold]Player {len(players) + 1}[/] name ([grey62]{hint}[/])",
                      hotkeys={"tab": "/settings"} if settings is not None else None).strip()
        if name.lower() == "/settings" and settings is not None:
            open_settings(ui, settings)
            continue
        if not name or name.lower() == "done":
            if players:
                return players
            ui.error("Must have at least one player.")
        elif name.lower() in (p.lower() for p in players):
            ui.error(f"'{name}' is already at the table.")
        else:
            players.append(name)
            ui.add_splash_name(name)


def make_ui(train_mode: bool, show_output: bool):
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return HeadlessUI(verbose=show_output or not train_mode, progress=train_mode)
    if train_mode and not show_output:
        return HeadlessUI(verbose=False, progress=True)
    return RichUI()


#== Main ==#
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Train AI mode — no player interaction")
    parser.add_argument("--load", type=str, default=None, help="Load a saved game by save name")
    parser.add_argument("--train-counter", type=int, nargs="?", const=8000, default=None, metavar="STEPS",
                        help="Train the card-counting LSTM on simulated shoes (default 8000 steps)")
    args = parser.parse_args()

    if args.train_counter is not None:
        os.makedirs("models", exist_ok=True)
        print(f"Training the card-counting LSTM for {args.train_counter:,} steps on simulated shoes…")
        model = train_counter(steps=args.train_counter)
        save_counter(model, COUNTER_MODEL_PATH)
        print("Average true-count error by shoe size (lower is better):")
        for decks in (1, 2, 6, 8):
            print(f"  {decks} deck{'s' if decks > 1 else ' '}: {evaluate_counter(model, shoes=20, decks=decks):.2f}")
        print(f"Saved to {COUNTER_MODEL_PATH}")
        return

    with open(SETTINGS_FILE, 'r') as file:
        settings = json.load(file)

    global SHOW_GRAPH, TRAIN_MODE
    apply_settings(settings)
    TRAIN_MODE = args.train

    ui = make_ui(TRAIN_MODE, SHOW_OUTPUT)

    if TRAIN_MODE and not settings.get('bots'):
        print("Train mode requires at least one bot configured in game_settings.json.")
        sys.exit(1)

    save_data = None
    if args.load:
        save_name = args.load.replace(".json", "")
        save_path = f"saves/{save_name}.json"
        try:
            with open(save_path, 'r') as f:
                save_data = json.load(f)
            ui.log(f"[green]Loaded save: {esc(save_name)}[/]")
        except FileNotFoundError:
            print(f"Save file '{save_path}' not found.")
            sys.exit(1)

    bj_game = None
    try:
        ui.start()

        if save_data:
            players = [p["name"] for p in save_data["players"]]
        elif TRAIN_MODE:
            players = []
        else:
            players = ask_player_names(ui, settings)

        apply_settings(settings)  # pick up anything changed in the settings editor
        if SHOW_GRAPH and not LOG_GAME:
            ui.log("[yellow]Balance graph needs game logging on — graph will be skipped.[/]")
            SHOW_GRAPH = False

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
            save_data=save_data,
            ui=ui,
            counter_max_spread=int(settings.get("counter", {}).get("max_spread", 8)),
        )

        if TRAIN_MODE:
            rounds = int(settings.get("train_param", {}).get("rounds", 60000))
            bj_game.prepare_training(rounds)
            ui.log(f"[bold]Training for {rounds:,} rounds[/] (deal delay off)")

        bj_game.play_game()
    except (KeyboardInterrupt, EOFError):
        ui.stop()
        if TRAIN_MODE and bj_game is not None:
            bj_game.flush_log()
            bj_game.save_models()
            print(f"\nTraining stopped after {bj_game.game_num:,} rounds — models saved.")
        else:
            print("\nGame interrupted — progress since the last save was not kept.")
        sys.exit(130)
    finally:
        ui.stop()


if __name__ == "__main__":
    main()
