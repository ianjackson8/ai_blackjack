#!myenv/bin/python3.12
'''
Play blackjack in terminal

Author: Ian Jackson
Version v3.0
'''

#== Imports ==#
import random
import os
import json
import argparse
import torch 
import time

import numpy as np
import torch.nn as nn           # type: ignore 
import torch.optim as optim     # type: ignore 
import matplotlib.pyplot as plt 

from typing import List, Union
from datetime import datetime

#== Global Variables ==#
SHOW_OUTPUT = True
LOG_GAME = True
SHOW_GRAPH = True
DEFAULT_BET = 10
TRAIN_MODE = False
GOD_MODE = False

#== Classes ==#
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Card():
    def __init__(self, rank: str, suit: str):
        '''
        initializes a card with rank, suit, and value

        Args:
            rank (str): rank of the card (e.g. "Ace, "2", etc)
            suit (str): suit of the card (e.g. "Hearts", "Diamonds", etc)
        '''
        self.rank = rank
        self.suit = suit
        self.values = self._assign_value(rank)

    def _assign_value(self, rank: str) -> tuple:
        '''
        Assigns a possible value of the card based on the rank

        Args:
            rank (str): rank of the card

        Returns:
            tuple: a tuple of possible values of the card
        '''
        if rank in ["Jack", "King", "Queen"]:
            return (10,)
        elif rank == "Ace":
            return (1,11)
        else:
            return (int(rank),)
    
    def __repr__(self) -> str:
        '''
        string representation of the card

        Returns:
            str: a string showing rank and suit of card
        '''
        return f"{self.rank} of {self.suit}"

class Shoe():
    def __init__(self, num_decks: int = 1):
        '''
        initializes the shoe

        Args:
            num_decks (int, optional): number of decks to use. Defaults to 1
        '''
        self.num_decks = num_decks
        self.cards = self._generate_shoe()
        self.shuffle()

    def _generate_shoe(self) -> List[Card]:
        '''
        creates the shoe with number of decks

        Returns:
            List[Card]: deck list
        '''
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
        shoe = []

        for _ in range(self.num_decks):
            for suit in suits:
                for rank in ranks:
                    shoe.append(Card(rank, suit))
        
        return shoe
    
    def shuffle(self) -> None:
        '''
        shuffles the cards in the shoe
        '''
        random.shuffle(self.cards)

    def draw_card(self) -> Card:
        '''
        Draw a card from the show

        Raises:
            ValueError: if the show is empty

        Returns:
            Card: a card object
        '''
        # check if there is a card to draw
        if not self.cards:
            raise ValueError("The shoe is empty, cannot draw card")
        
        return self.cards.pop()
        
    def __len__(self) -> int:
        '''
        returns the number of cards remaining

        Returns:
            int: int representing number of cards
        '''
        return len(self.cards)

class Hand():
    def __init__(self, hand: list[Card] = None):
        '''
        initializes an empty hand
        ''' 
        if hand == None:
            self.cards = []
        else:
            self.cards = hand

    def add_card(self, card: Card):
        '''
        add a card to the hand

        Args:
            card (Card): to object to be added
        '''
        self.cards.append(card)

    def calculate_value(self) -> int:
        '''
        calculate total value of hand, accounting for aces

        Returns:
            int: best possible value without exceeding 21
        '''
        values = [0]

        # iterate cards in hand
        for card in self.cards:
            new_values = []
            for val in values:
                for card_val in card.values:
                    new_values.append(val + card_val)
            
            # remove duplicates
            values = list(set(new_values))

        # sort values, prioritizing <= 21
        valid_values = [v for v in values if v <= 21]
        return max(valid_values) if valid_values else min(values)

    def is_blackjack(self) -> bool:
        '''
        check if hand is a blackjack

        Returns:
            bool: true if hand is blackjack, false otherwise
        '''
        return len(self.cards) == 2 and self.calculate_value() == 21
    
    def is_busted(self) -> bool:
        '''
        check if hand value exceeds 21

        Returns:
            bool: true if hand is busted, false otherwise
        '''
        return self.calculate_value() > 21 

    def num_cards(self) -> int:
        '''
        returns number of cards in hand

        Returns:
            int: num of cards in hand
        '''
        return len(self.cards)

    def __repr__(self) -> str:
        '''
        string representation of hand

        Returns:
            str: string showing cards in hand and their total value
        '''
        cards_str = ", ".join(str(card) for card in self.cards)
        return f"[{cards_str}] | Value: {self.calculate_value()}"

class Player:
    def __init__(self, name: str, balance: int):
        '''
        initialize a player with name, balance, and empty hand

        Args:
            name (str): name of player
            balance (int): initial balance of player
        '''
        self.name = name
        self.balance = balance
        self.hands = [Hand()]  # A player starts with one hand
        self.current_bet = 0
        self.actions = []
        self.result = ""

    def log_action(self, action: str, dealer_visible_card: str):
        '''
        logs the players action

        Args:
            action (str): action preformed (e.g. "hit")
            dealer_visible_card (str): the dealer's visible card at the time
        '''

        self.actions.append({
            "action": action,
            "player_hand": [str(card) for card in self.hands[-1].cards],
            "hand_value": self.hands[-1].calculate_value(),
            "dealer_visible_card": dealer_visible_card
        })

    def place_bet(self, amount: int) -> None:
        '''
        places a bet for the current round

        Args:
            amount (int): amount to be bet

        Raises:
            ValueError: if bet is invalid (<=0)
            ValueError: if bet exceeds player's balance
        '''
        if amount <= 0:
            raise ValueError(f"{bcolors.FAIL}[ERR] Bet amount must be greater than zero.{bcolors.ENDC}")
        if amount > self.balance:
            raise ValueError(f"{bcolors.FAIL}Bet amount exceeds available balance.{bcolors.ENDC}")
        self.current_bet = amount
        self.balance -= amount

    def hit(self, card: Card) -> None:
        '''
        adds a card to current hand

        Args:
            card (Card): card to be added
        '''
        self.hands[-1].add_card(card)

    def stand(self) -> None:
        '''
        end player's turn
        '''
        pass

    def split(self) -> None:
        '''
        splits the player's hand (if possible)

        Raises:
            ValueError: if the hand cannot be split
        '''
        current_hand = self.hands[-1]
        if len(current_hand.cards) != 2 or current_hand.cards[0].rank != current_hand.cards[1].rank:
            raise ValueError(f"{bcolors.FAIL}[ERR] Cannot split: Hand must have exactly two cards of the same rank.{bcolors.ENDC}")

        # Create two new hands from the split cards
        card1, card2 = current_hand.cards
        self.hands[-1] = Hand()
        self.hands[-1].add_card(card1)
        new_hand = Hand()
        new_hand.add_card(card2)
        self.hands.append(new_hand)

    def double_down(self, card: Card):
        '''
        Double down and take one card

        Args:
            card (Card): card to be taken

        Raises:
            ValueError: if the player has insufficient funds
            ValueError: if the player cannot double down after hit
        '''
        if self.current_bet > self.balance:
            raise ValueError(f"{bcolors.FAIL}[ERR] Insufficient balance to double down.{bcolors.ENDC}")
        if self.hands[0].num_cards() != 2: # FIX: split case
            raise ValueError(f"{bcolors.FAIL}[ERR] Cannot double down after a hit.{bcolors.ENDC}")

        self.balance -= self.current_bet
        self.current_bet *= 2
        self.hit(card)

    def reset_for_new_round(self):
        '''
        reset the player's hands
        '''
        self.hands = [Hand()]
        self.current_bet = 0
        self.actions = []  # Clear the cache for the new round
        self.result = ""

    def __repr__(self) -> str:
        '''
        string rep of player

        Returns:
            str: string showing name, balance, and cur hand
        '''
        # BUG: always shows? set to nothing for now
        hands_str = "".join(str(hand) for hand in self.hands)
        return f""

class Bot(Player):
    def __init__(self, name: str, balance: int, strategy: str):
        '''
        initialize a bot with name, balance, and empty hand

        Args:
            name (str): name of bot
            balance (int): initial balance of bot
        '''
        super().__init__(name, balance)
        self.strategy = strategy

    def play_turn(self, shoe: Shoe, dealer_visible_card: str, deal_delay: int):
        '''
        play a bots turn

        Args:
            shoe (Shoe): the shoe of the game       
            dealer_visible_card (str): dealer's visible card
        '''
        for hand in self.hands:
            print(f"\n{self.name}'s turn:\n\tCurrent Hand: {hand}")
            while not hand.is_blackjack() and not hand.is_busted():
                action = self.make_decision(dealer_visible_card)
                print(f"\t{self.name} chooses to {action}")

                if action == "hit":
                    card = shoe.draw_card()
                    hand.add_card(card)
                    time.sleep(deal_delay)
                    print(f"\t{self.name} hits: {hand}")
                elif action == "stand":
                    break
                elif action == "double":
                    # Ensure enough balance to double AND only on first action
                    if (float(self.balance) >= float(self.current_bet)) and (self.hands[0].num_cards() == 2):  
                        self.balance -= self.current_bet
                        self.current_bet *= 2
                        card = shoe.draw_card()
                        hand.add_card(card)
                        time.sleep(deal_delay)
                        print(f"\t{self.name} doubles: {hand}")
                        break  # Doubling ends the turn
                    else:
                        print(f"\t{bcolors.OKBLUE}[i] cannot double due to insufficient balance.{bcolors.ENDC}")
                        action = "hit"  # Fall back to hitting if double is not possible
                else:
                    print(f"{bcolors.FAIL} Invalid action: {action} {bcolors.ENDC}")

    def make_decision(self, dealer_visible_card: str) -> str:
        '''
        make a decision on the hand, based on strategy

        Args:
            dealer_visible_card (str): dealer's visible card

        Returns:
            str: decision on hand
        '''
        # decision table for by the books
        # TODO: add soft table 
        # TODO: add split table 
        # dealer card: player hand
        DECISION_TABLE_BTB = {
            # player hard
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

        #= default =#
        if self.strategy == "default":
            hand_value = self.hands[0].calculate_value()

            if hand_value <= 16:
                return "hit"
            else:
                return "stand"
            
        #= by the book =#
        elif self.strategy == "by the books":
            dealer_value = get_card_value(dealer_visible_card)
            hand_value = self.hands[0].calculate_value()

            if dealer_value not in DECISION_TABLE_BTB or hand_value not in DECISION_TABLE_BTB[dealer_value]:
                return "stand"  # Default action if no rule exists
            return DECISION_TABLE_BTB[dealer_value][hand_value]

        else:
            raise ValueError(f"{bcolors.FAIL}[ERR] Invalid bot strategy {self.strategy}.{bcolors.ENDC}")

class BlackjackGame:
    def __init__(self, num_decks: int, player_names: List[str], bot_info: List[dict], starting_balance: int, log_file: str,
                 deal_delay: int):
        '''
        init the game with a shoe, dealer, and players
        shuffles deck

        Args:
            num_decks (int): number of decks to use
            player_names (List[str]): list of players
            bot_info: (List[dict]): bot name and play strategy
            starting_balance (int): initial balance for each player
            log_file (str): path of the logfile
            deal_delay (int): time (in seconds) for a card to deal
        '''
        self.num_decks = num_decks
        self.shoe = Shoe(num_decks)
        self.shoe.shuffle()
        self.dealer = Player("Dealer", balance=0)
        self.players = [Player(name, starting_balance) for name in player_names]
        self.deal_delay = deal_delay
        
        # Initialize bots from settings
        self.bots = []
        for bot in bot_info:
            if bot.get('strategy') == 'ai':
                self.bots.append(TrainableBot(bot['name'], starting_balance, 2, 3))
            else:
                self.bots.append(Bot(bot['name'], starting_balance, bot['strategy']))
        
        self.log_file = log_file
        self.round_data = {}
        self.game_num = 0

        # Load AI bot models if they exist
        for bot in self.bots:
            if isinstance(bot, TrainableBot):
                try:
                    additional_data = load_model(bot.q_network, bot.optimizer, "bot_model.pth")
                    bot.epsilon = additional_data.get("epsilon", 1.0)  # Default to 1.0 if not found
                except FileNotFoundError as e:
                    print(f"{bcolors.OKBLUE}[i] AI Save not found, creating new AI. {bcolors.ENDC}")

    def setup_round(self) -> None:
        '''
        set up round
        '''
        # reset players and prepare for bets
        for player in self.players + self.bots:
            player.reset_for_new_round()
        
        # reset dealer
        self.dealer.reset_for_new_round()

        # reset round data entry
        self.round_data = {}
        self.game_num += 1

    def deal_cards(self) -> None:
        '''
        deal two cards to each player who has placed bets and dealer
        '''
        global GOD_MODE
        for player in self.players + self.bots:
            # Only deal cards to players who bet
            if player.current_bet > 0:  
                for _ in range(2):
                    if GOD_MODE and not (isinstance(player, Bot) or isinstance(player, TrainableBot)):
                        player.hands = [Hand([Card("Ace", "Hearts"), Card("King", "Hearts")])]
                        continue

                    player.hit(self.shoe.draw_card())
        for _ in range(2):
            self.dealer.hit(self.shoe.draw_card())

    def display_table(self) -> None:
        '''
        displays the hand of the dealer (first card hidden) as well as the whole table
        '''
        os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal

        print("Dealer's hand:\t[Hidden], {card}".format(card=self.dealer.hands[0].cards[1]))
        for player in self.players + self.bots:
            if player.current_bet > 0:
                print(f"{player.name}'s hand:\t{player.hands[0]}")

    def display_players(self) -> None:
        '''
        display the players and their balances at the beginning
        '''
        print("Players at the table:")
        for player in self.players + self.bots:
            print(f"\t{player.name}: ${player.balance}")

    def player_turn(self, player: Player) -> None:
        '''
        Handles a player's turn

        Args:
            player (Player): player who's turn it is
        '''
        dealer_visible_card = str(self.dealer.hands[0].cards[1])
        print(f"\n{bcolors.BOLD}{player.name}{bcolors.ENDC}, Your Turn:")

        for hand in player.hands:
            while True:
                print(f"\n\tCurrent hand: {hand}")
                if hand.is_blackjack():
                    print(f"\t{bcolors.BOLD}{bcolors.OKGREEN}Blackjack!{bcolors.ENDC}")
                    break
                elif hand.is_busted():
                    print(f"\t{bcolors.BOLD}{bcolors.FAIL}Busted!{bcolors.ENDC}")
                    break

                action = input("\tChoose action (1=hit / 2=stand / 3=double / 4=split): ").lower()

                # Map hotkeys to actions
                if action == "1":
                    action = "hit"
                elif action == "2":
                    action = "stand"
                elif action == "3":
                    action = "double"
                elif action == "4":
                    action = "split"

                if action == "hit":
                    # NOTE: split feature
                    # hand.add_card(Card("2", "Hearts"))
                    hand.add_card(self.shoe.draw_card())
                    time.sleep(self.deal_delay)
                    player.log_action("hit", dealer_visible_card)
                elif action == "stand":
                    player.log_action("stand", dealer_visible_card)
                    break
                elif action == "double":
                    if len(player.hands) > 1:
                        print("Cannot double down after splitting.")
                    else:
                        try:
                            player.double_down(self.shoe.draw_card())
                            player.log_action("double", dealer_visible_card)
                        except ValueError as e:
                            print(f"\t{e}")
                            continue
                        break
                elif action == "split":
                    if len(player.hands) == 1:
                        try:
                            player.split()
                            player.log_action("hit", dealer_visible_card)
                        except ValueError as e:
                            print(f"\t{e}")
                            continue
                        for new_hand in player.hands:
                            new_hand.add_card(self.shoe.draw_card())
                        break
                    else:
                        print("Cannot split more than once.")
                else:
                    print("Invalid action. Try again.")

    def dealer_turn(self) -> None:
        '''
        Handle dealer's turn
        '''
        print(f"\nDealer's turn:\n\tCurrent Hand: {self.dealer.hands[0]}")
        while self.dealer.hands[0].calculate_value() < 17:
            self.dealer.hands[0].add_card(self.shoe.draw_card())
            time.sleep(self.deal_delay)
            print(f"\tDealer hits: {self.dealer.hands[0]}")

        if self.dealer.hands[0].is_busted():
            print(f"\t{bcolors.BOLD}{bcolors.FAIL}Dealer busted!{bcolors.ENDC}")

    def determine_winner(self, player: Player) -> None:
        '''
        determine the winner

        Args:
            player (Player): player object to evaluate
        '''
        dealer_value = self.dealer.hands[0].calculate_value()
        # print(f"Dealer's hand: {dealer_value}")

        for hand in player.hands:
            player_value = hand.calculate_value()
            if hand.is_busted():
                print(f"{player.name}'s hand: {player_value} => {bcolors.BOLD}{bcolors.FAIL}Busted! Lost ${player.current_bet}.{bcolors.ENDC}")
                player.result = "busted"

            elif hand.is_blackjack():
                winnings = player.current_bet * 2.5
                player.balance += winnings
                print(f"{player.name}'s hand: {player_value}. => {bcolors.BOLD}{bcolors.OKGREEN}Blackjack! You win ${winnings}.{bcolors.ENDC}")
                player.result = "blackjack"

            elif dealer_value > 21 or player_value > dealer_value:
                winnings = player.current_bet * 2
                player.balance += winnings
                print(f"{player.name}'s hand: {player_value}. => {bcolors.BOLD}{bcolors.OKGREEN}Win! You win ${winnings}.{bcolors.ENDC}")
                player.result = "win"

            elif player_value == dealer_value:
                player.balance += player.current_bet
                print(f"{player.name}'s hand: {player_value}. => {bcolors.BOLD}{bcolors.OKBLUE}Push. ${player.current_bet} returned.{bcolors.ENDC}")
                player.result = "push"

            else:
                print(f"{player.name}'s hand: {player_value}. => {bcolors.BOLD}{bcolors.FAIL}Lose. Lost ${player.current_bet}.{bcolors.ENDC}")
                player.result = "lose"

    def play_round(self) -> None:
        ''''
        play a single round
        '''
        self.check_shoe()
        self.setup_round()

        # Collect bets
        for player in self.players + self.bots:
            while True:
                try:
                    # bots
                    if isinstance(player, Bot) or isinstance(player, TrainableBot):
                        global DEFAULT_BET
                        bet = min(DEFAULT_BET, player.balance)

                        # if the bot is broke, dont allow play
                        if float(bet) == float(0): break

                        player.place_bet(bet)
                        break
                    else: # player
                        is_bet = False
                        while not is_bet:
                            bet = input(f"{bcolors.BOLD}{player.name}{bcolors.ENDC}, Enter bet amount: ")

                            # if "bet" starts with a / then parse as a command
                            if bet[0] == '/':
                                self.parse_command(bet[1:])
                            else: is_bet = True

                        # parse bet as integer if no command
                        try:
                            bet = int(bet)
                        except:
                            raise ValueError(f"{bcolors.FAIL}[ERR] Bet must be a number.{bcolors.ENDC}")

                        if bet > 0:
                            player.place_bet(bet)
                        break
                except ValueError as e:
                    print(e)

        self.deal_cards()  # Deal cards only to players who placed a bet
        self.display_table()

        # Players take turns
        for player in self.players + self.bots:
            if player.current_bet > 0:  # Only allow turns for players who bet
                if isinstance(player, Bot) or isinstance(player, TrainableBot):
                    player.play_turn(self.shoe, str(self.dealer.hands[0].cards[1]), self.deal_delay)
                else:
                    self.player_turn(player)

        # Dealer's turn
        self.dealer_turn()

        print(f"\n{bcolors.BOLD}{bcolors.UNDERLINE}=== Results ==={bcolors.ENDC}")
        dealer_value = self.dealer.hands[0].calculate_value()
        print(f"Dealer's hand: {dealer_value}")
        # Determine winners
        for player in self.players + self.bots:
            if player.current_bet > 0:
                self.determine_winner(player)
                # Update AI bot with final reward
                if isinstance(player, TrainableBot):
                    player.update_final_reward(player.result)

        global LOG_GAME
        if LOG_GAME: self.log_game()

    def play_game(self):
        '''
        start and play a game of blackjack
        '''
        ai_done = False

        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
            print("Welcome to Blackjack!")
            self.display_players()
            print()

            # training end condition 
            global DEFAULT_BET
            global TRAIN_MODE
            if (self.bots[0].balance <= DEFAULT_BET) and TRAIN_MODE:
                ai_done = True
            else:
                self.play_round()

            if TRAIN_MODE and not ai_done:
                continue

            # Ask if player wants to continue
            while True:
                play_again = input("Play another round? (yes/no): ").lower()
                if play_again in ["yes", "no"]:
                    break
                else:
                    print(f"{bcolors.FAIL}[ERR] Invalid input. Please enter 'yes' or 'no'.{bcolors.ENDC}")

            if play_again != "yes" or ai_done:
                self.end_game()
                break

    def log_game(self):
        '''
        logs the round entry
        '''
        # create entry
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
                "actions": player.actions,
                "final_hand": [str(card) for card in player.hands[0].cards],
                "final_value": player.hands[0].calculate_value(),
                "result": player.result,
                "balance": player.balance
            }
            log_entry["players"].append(player_log)

        # get current entries
        with open(self.log_file, 'r') as file:
            logs = json.load(file)

        logs.append(log_entry)

        # write back
        with open(self.log_file, "w") as file:
            json.dump(logs, file, indent=4)

    def check_shoe(self, force: bool = False):
        '''
        check the shoe to reshuffle

        Args:
            force (bool): force a reshuffle (Default False)
        '''
        if (len(self.shoe) < (len(self.players) + len(self.bots) + 1) * 3) or force:
            print(f"{bcolors.HEADER}[i] Reshuffling Deck{bcolors.ENDC}")
            self.shoe = Shoe(self.num_decks)
            self.shoe.shuffle()

    def end_game(self):
        '''
        End the game
        '''
        print(f"{bcolors.BOLD}{bcolors.UNDERLINE}\n\nPlayer Balance:{bcolors.ENDC}")
        for player in self.players + self.bots:
            print(f"{player.name}: ${player.balance}")
        print()

        # save the AI model
        for bot in self.bots:
            if isinstance(bot, TrainableBot):
                save_model(bot.q_network, bot.optimizer, "bot_model.pth", additional_data={"epsilon": bot.epsilon})

        global SHOW_GRAPH
        if SHOW_GRAPH: parse_log_and_plot(self.log_file)

    def parse_command(self, command: str):
        '''
        Parse user command from bet line

        Args:
            command (str): command to be parsed
        '''
        if command == 'help':
            print(f"{bcolors.BOLD}{bcolors.UNDERLINE}Available commands:{bcolors.ENDC}")
            print('\t/help                                  Prints this message')
            print('\t/exit                                  Quits the game')
            print('\t/graph                                 Displays current player balance graph')
            print('\t/editbalance [player] [new balance]    Modify a players balance')
            print('\t/showbalance                           Display the players balance')
            print('\t/shuffle                               Shuffles and resets the deck')

        elif command == 'exit':
            self.end_game()
            quit()

        elif command == 'graph':
            parse_log_and_plot(self.log_file)

        elif command.split()[0] == 'editbalance':
            # Parse: /editbalance [player name] [new balance]
            # Last token is balance, everything else is player name (allows spaces)
            parts = command.split()
            
            if len(parts) < 3:
                print(f"{bcolors.FAIL}[ERR] Invalid format. Usage: /editbalance [player name] [new balance]{bcolors.ENDC}")
                return
            
            # Last part is balance, everything between is player name
            new_balance_str = parts[-1]
            player_name = ' '.join(parts[1:-1])
            
            # Find the player
            target_player = None
            for player in self.players + self.bots:
                if player.name.lower() == player_name.lower():
                    target_player = player
                    break
            
            if target_player is None:
                print(f"{bcolors.FAIL}[ERR] Player '{player_name}' not found.{bcolors.ENDC}")
                return
            
            # Validate and set new balance
            try:
                new_balance = float(new_balance_str)
                if new_balance < 0:
                    print(f"{bcolors.FAIL}[ERR] Balance cannot be negative.{bcolors.ENDC}")
                    return
                
                old_balance = target_player.balance
                target_player.balance = new_balance
                print(f"{bcolors.OKGREEN}[✓] {target_player.name}'s balance updated: ${old_balance} → ${new_balance}{bcolors.ENDC}")
                
            except ValueError:
                print(f"{bcolors.FAIL}[ERR] Invalid balance amount. Must be a number.{bcolors.ENDC}")
        elif command == 'showbalance':
            print(f"{bcolors.BOLD}{bcolors.UNDERLINE}\nPlayer Balance:{bcolors.ENDC}")
            for player in self.players + self.bots:
                print(f"{player.name}: ${player.balance}")
            print()

        elif command == 'shuffle':
            self.check_shoe(force=True)

        elif command.split()[0] == 'godmode':
            global GOD_MODE
            parts = command.split()
            if parts[-1] == "on":
                GOD_MODE = True
                print(f"{bcolors.HEADER}[i] GOD MODE ON{bcolors.ENDC}")
            elif parts[-1] == "off":
                GOD_MODE = False
                print(f"{bcolors.HEADER}[i] GOD MODE OFF{bcolors.ENDC}")

        else:
            print(f"{bcolors.FAIL}[ERR] Invalid command -- run /help for list of commands.{bcolors.ENDC}")
        return


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
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = 0.95  # Discount factor
        self.memory = []  # Replay buffer

    def get_state(self, hand, dealer_visible_card):
        # Normalize the hand value and dealer card for the state
        hand_value = hand.calculate_value() / 21.0
        # dealer_value = 11 if dealer_visible_card == "Ace" else int(dealer_visible_card) / 11.0
        dealer_value = get_card_value(dealer_visible_card) / 11.0
        return np.array([hand_value, dealer_value])

    def choose_action(self, state):
        if random.random() < self.epsilon:  # Exploration
            return random.choice(["hit", "stand", "double"])
        else:  # Exploitation
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

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def play_turn(self, shoe, dealer_visible_card, deal_delay: int):
        self.last_state = None  # Track last state for final reward update
        self.last_action = None
        
        for hand in self.hands:
            print(f"\n{self.name}'s turn:\n\tCurrent Hand: {hand}")
            while not hand.is_blackjack() and not hand.is_busted():
                state = self.get_state(hand, dealer_visible_card)
                action = self.choose_action(state)
                print(f"\t{self.name} chooses to {action} (epsilon: {self.epsilon:.3f})")
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
                    print(f"\t{self.name} hits: {hand}")
                elif action == "stand":
                    # Store state and action, reward will be updated after game ends
                    self.last_state = state
                    self.last_action = "stand"
                    break
                elif action == "double" and self.balance >= self.current_bet:
                    self.balance -= self.current_bet
                    self.current_bet *= 2
                    card = shoe.draw_card()
                    hand.add_card(card)
                    next_state = self.get_state(hand, dealer_visible_card)
                    if hand.is_busted():
                        reward = -1
                    else:
                        reward = 0
                    self.store_experience(state, "double", reward, next_state, True)
                    self.last_state = next_state
                    self.last_action = "double"
                    break
    
    def update_final_reward(self, result):
        """Update the final reward based on game outcome"""
        if self.last_state is None:
            return
            
        # Map result to reward
        if result == "blackjack":
            reward = 1.5  # Blackjack pays 3:2
        elif result == "win":
            reward = 1.0
        elif result == "push":
            reward = 0.0
        elif result == "lose":
            reward = -1.0
        elif result == "busted":
            reward = -1.0  # Already recorded, but for completeness
        else:
            reward = 0.0
            
        # Store the final experience with actual game outcome
        if self.last_action and result != "busted":  # Don't double-count bust
            self.store_experience(self.last_state, self.last_action, reward, self.last_state, True)
        
        # Train on the experiences
        self.replay()

    def save_model(self, file_path):
        save_model(
            self.q_network,
            self.optimizer,
            file_path,
            additional_data={"epsilon": self.epsilon}
        )

    def load_model(self, file_path):
        additional_data = load_model(self.q_network, self.optimizer, file_path)
        self.epsilon = additional_data.get("epsilon", 1.0)

#== Methods =#
def get_card_value(card_str: str) -> int:
    """
    Extract the numeric value of a card from its string representation.

    Args:
        card_str (str): The string representation of the card (e.g., "2 of Clubs").

    Returns:
        int: The numeric value of the card.
    """
    # Extract the rank (e.g., "2", "Ace", "King") from "2 of Clubs"
    rank = card_str.split(" ")[0]
    
    # Convert the rank to its numeric value
    if rank in ["Jack", "Queen", "King"]:
        return 10  # Face cards are worth 10
    elif rank == "Ace": # FIX: return ace as soft
        return 11  # Default Ace to 11
    else:
        return int(rank)  # Numeric cards retain their values

def parse_log_and_plot(log_file):
    try:
        with open(log_file, 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {log_file} does not exist.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse the JSON file {log_file}.")
        return
    
    # Extract balance and round number
    with open('game_settings.json', 'r') as file:
            settings = json.load(file)
    
    try:
        init_bal = settings["init_balance"]

        # Extract balance and round number for all players
        player_balances = {}

        players = logs[0].get("players", [])
        for player in players:
            name = player.get("name")
            player_balances[name] = []
            player_balances[name].append((0, init_bal))

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

        # Plotting
        plt.figure(figsize=(10, 6))

        for player, rounds_and_balances in player_balances.items():
            rounds, balances = zip(*rounds_and_balances)
            plt.plot(rounds, balances, marker='o', linestyle='-', label=player)

        plt.title("Player Balances Over Rounds")
        plt.xlabel("Round Number")
        plt.ylabel("Balance")
        plt.grid(True)
        plt.legend()
        plt.show()
    except IndexError:
        print(f'{bcolors.WARNING}[W] Cannot create graph, no data{bcolors.ENDC}')

def save_model(model, optimizer, file_path, additional_data=None):
    """
    Save the model, optimizer state, and additional metadata.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer associated with the model.
        file_path (str): Path to the file where the model should be saved.
        additional_data (dict): Any additional data to save (e.g., epsilon, training progress).
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "additional_data": additional_data
    }
    torch.save(checkpoint, file_path)
    print(f"Model saved to {file_path}")

def load_model(model, optimizer, file_path):
    """
    Load the model, optimizer state, and additional metadata.

    Args:
        model: The PyTorch model to load into.
        optimizer: The optimizer associated with the model.
        file_path (str): Path to the file where the model is saved.

    Returns:
        dict: Additional metadata from the saved state.
    """
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Model loaded from {file_path}")
    return checkpoint.get("additional_data", {})


#== Main execution ==#
def main():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action='store_true', help="Train AI mode -- no player interaction")
    args = parser.parse_args()

    # import settings
    with open('game_settings.json', 'r') as file:
        settings = json.load(file)

    # extract show output
    global SHOW_OUTPUT, LOG_GAME, SHOW_GRAPH, DEFAULT_BET, TRAIN_MODE
    SHOW_OUTPUT = settings["show_output"].lower() == "true"
    LOG_GAME = settings["log_game"].lower() == "true"
    SHOW_GRAPH = settings["show_graph"].lower() == "true"
    DEFAULT_BET = settings["default_bet"]
    TRAIN_MODE = args.train

    # get players
    players = []
    i = 1

    while True:
        # if in train mode, skip
        if TRAIN_MODE:
            break

        player_i = input(f"Input player {i}'s name ('done' to continue): ")

        if player_i != "done":
            players.append(player_i)
            i += 1
        else:
            if players != []:
                break
            
            print(f"{bcolors.FAIL}[ERR] Must be at least one player{bcolors.ENDC}")

    # set up log file -- make folder if it doesnt exist
    if LOG_GAME:
        if not os.path.exists("logs"):
            os.mkdir("logs")

        log_file = f'logs/session_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.json'
        with open(log_file, 'w') as file:
            file.write("[]")
    else: log_file = 'logs/session_NULL.json'

    # initiate game
    bj_game = BlackjackGame(
        num_decks = settings['num_decks'], 
        player_names = players,
        bot_info = settings['bots'],
        starting_balance = settings['init_balance'],
        log_file = log_file,
        deal_delay = settings['deal_delay']
    )

    # play game
    bj_game.play_game()

if __name__ == "__main__":
    main()