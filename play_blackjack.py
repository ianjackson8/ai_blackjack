'''
Play blackjack in terminal

Author: Ian Jackson
Version v0.2

'''
#== todo list ==#
# [ ] test funtionality

#== Imports ==#
import random
import os
import json
import argparse

from typing import List, Union
from datetime import datetime
from calc_stats import parse_log_and_plot

#== Global Variables ==#
SHOW_OUTPUT = True
LOG_GAME = True
SHOW_GRAPH = True
DEFAULT_BET = 10

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
    def __init__(self):
        '''
        initializes an empty hand
        ''' 
        self.cards = []

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
        # BUG: doesnt allow hit when it should
        if self.current_bet > self.balance:
            raise ValueError(f"{bcolors.FAIL}[ERR] Insufficient balance to double down.{bcolors.ENDC}")
        if len(self.hands) != 2:
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
    def __init__(self, name: str, balance: int):
        '''
        initialize a bot with name, balance, and empty hand

        Args:
            name (str): name of bot
            balance (int): initial balance of bot
        '''
        super().__init__(name, balance)

    def play_turn(self, shoe: Shoe, dealer_visible_card: str):
        '''
        play a bots turn

        Args:
            shoe (Shoe): the shoe of the game       
            dealer_visible_card (str): dealer's visible card
        '''
        for hand in self.hands:
            while not hand.is_blackjack() and not hand.is_busted():
                action = self.make_decision(dealer_visible_card)

                if action == "hit":
                    card = shoe.draw_card()
                    hand.add_card(card)
                elif action == "stand":
                    break
                else:
                    print(f"{bcolors.FAIL} Invalid action: {action} {bcolors.ENDC}")

    def make_decision(self, dealer_visible_card: str):
        hand_value = self.hands[0].calculate_value()

        if hand_value <= 16:
            return "hit"
        else:
            return "stand"

class BlackjackGame:
    def __init__(self, num_decks: int, player_names: List[str], bot_names: List[str], starting_balance: int, log_file: str):
        '''
        init the game with a shoe, dealer, and players
        shuffles deck

        Args:
            num_decks (int): number of decks to use
            player_names (List[str]): list of players
            bot_names: (List[str]): list of bot names
            starting_balance (int): initial balance for each player
            log_file (str): path of the logfile
        '''
        self.num_decks = num_decks
        self.shoe = Shoe(num_decks)
        self.shoe.shuffle()
        self.dealer = Player("Dealer", balance=0)
        self.players = [Player(name, starting_balance) for name in player_names]
        self.bots = [Bot(name, starting_balance) for name in bot_names]
        self.log_file = log_file
        self.round_data = {}
        self.game_num = 0

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
        for player in self.players + self.bots:
            # Only deal cards to players who bet
            if player.current_bet > 0:  
                for _ in range(2):
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

                action = input("\tChoose action (hit/stand/double/split): ").lower()

                if action == "hit":
                    hand.add_card(self.shoe.draw_card())
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
                    if isinstance(player, Bot):
                        global DEFAULT_BET
                        bet = min(DEFAULT_BET, player.balance)
                        player.place_bet(bet)
                        break
                    else: # player
                        bet = int(input(f"{bcolors.BOLD}{player.name}{bcolors.ENDC}, Enter bet amount: "))
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
                if isinstance(player, Bot):
                    player.play_turn(self.shoe, str(self.dealer.hands[0].cards[1]))
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

        global LOG_GAME
        if LOG_GAME: self.log_game()

    def play_game(self):
        '''
        start and play a game of blackjack
        '''
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Clear the terminal
            print("Welcome to Blackjack!")
            self.display_players()
            print()
            self.play_round()
            if input("Play another round? (yes/no): ").lower() != "yes":
                print(f"{bcolors.BOLD}{bcolors.UNDERLINE}\n\nPlayer Balance:{bcolors.ENDC}")
                for player in self.players + self.bots:
                    print(f"{player.name}: ${player.balance}")

                if SHOW_GRAPH: parse_log_and_plot(self.log_file)
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

    def check_shoe(self):
        '''
        check the shoe to reshuffle
        '''
        if len(self.shoe) < (len(self.players)+1) * 3:
            print(f"{bcolors.HEADER}[i] Reshuffling Deck{bcolors.ENDC}")
            self.shoe = Shoe(self.num_decks)
            self.shoe.shuffle()

#== Methods =#


#== Main execution ==#
def main():
    # import settings
    with open('game_settings.json', 'r') as file:
        settings = json.load(file)

    # extract show output
    global SHOW_OUTPUT, LOG_GAME, SHOW_GRAPH, DEFAULT_BET
    SHOW_OUTPUT = settings["show_output"].lower() == "true"
    LOG_GAME = settings["log_game"].lower() == "true"
    SHOW_GRAPH = settings["show_graph"].lower() == "true"
    DEFAULT_BET = settings["default_bet"]

    # get players
    players = []
    i = 1
    while True:
        player_i = input(f"Input player {i}'s name ('done' to continue): ")

        if player_i != "done":
            players.append(player_i)
            i += 1
        else:
            if players != []:
                break
            
            print(f"{bcolors.FAIL}[ERR] Must be at least one player{bcolors.ENDC}")

    # get bots
    bots = []
    while True:
        is_bots = input(f"Include bots in the game? (y/n): ")
        if is_bots == "n":
            break
        else:
            num_bots = int(input("Enter number of bots: "))
            bots = [f"Bot{i}" for i in range(num_bots)]
            break

    # set up log file -- make folder if it doesnt exist
    if LOG_GAME:
        if not os.path.exists("logs"):
            os.mkdir("logs")

        log_file = f'logs/session_{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.json'
        with open(log_file, 'w') as file:
            file.write("[]")
    else: log_file = 'logs/session_NULL.json'

    bj_game = BlackjackGame(
        num_decks = settings['num_decks'], 
        player_names = players,
        bot_names = bots,
        starting_balance = settings['init_balance'],
        log_file = log_file
    )

    bj_game.play_game()

if __name__ == "__main__":
    main()