'''
Terminal Casino adapter for AI Blackjack (see terminal_casino/docs/GAME_SPEC.md).

The casino imports this file at launch, so keep the top level cheap: the game (and PyTorch)
is only imported when a player sits down at the table.
'''
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PLUGIN_META = {
    "name": "Blackjack",
    "description": "Beat the dealer to 21. Blackjack pays 3:2.",
    "min_bet": 1.0,
    "version": "5.4.1",
    "author": "Ian Jackson",
    # Remote players (GAME_SPEC §3.6): Claude Code sessions take seats through the casino's MCP server
    "multiplayer": {"mode": "seats", "max_remote": 3},
    # Table rules the casino can set per table (GAME_SPEC §3.5). Defaults match game_settings.json.
    "rules": {
        "num_decks": {"type": "int", "label": "Decks", "default": 7, "min": 1, "max": 8,
                      "description": "Decks in the shoe"},
        "allow_double_after_split": {"type": "bool", "label": "Double after split", "default": True,
                                     "description": "Allow doubling down on split hands"},
        "split_limit": {"type": "choice", "label": "Splits per hand", "choices": ["unlimited", "1", "2", "3"],
                        "default": "unlimited", "description": "How many times a hand can be split"},
        "allow_insurance": {"type": "bool", "label": "Insurance", "default": True,
                            "description": "Offer insurance when the dealer shows an Ace"},
        "max_hands": {"type": "int", "label": "Hands", "default": 1, "min": 1, "max": 3,
                      "description": "Hands a player can play at once, each with its own bet"},
        "perfect_pairs": {"type": "bool", "label": "Perfect Pairs", "default": False,
                          "description": "Side bet: first two cards make a pair (6:1 to 25:1)"},
        "perfect_pairs_min": {"type": "int", "label": "Perfect Pairs min", "default": 1, "min": 1, "max": 10000,
                              "description": "Smallest Perfect Pairs bet per hand"},
        "perfect_pairs_max": {"type": "int", "label": "Perfect Pairs max", "default": 100, "min": 1, "max": 10000,
                              "description": "Largest Perfect Pairs bet per hand (raised to the min if lower)"},
        "twenty_one_plus_three": {"type": "bool", "label": "21+3", "default": False,
                                  "description": "Side bet: your 2 cards + dealer up card as a poker hand (5:1 to 100:1)"},
        "twenty_one_plus_three_min": {"type": "int", "label": "21+3 min", "default": 1, "min": 1, "max": 10000,
                                      "description": "Smallest 21+3 bet per hand"},
        "twenty_one_plus_three_max": {"type": "int", "label": "21+3 max", "default": 100, "min": 1, "max": 10000,
                                      "description": "Largest 21+3 bet per hand (raised to the min if lower)"},
    },
}


def run(player_name: str, balance: float, table: dict = None, remote=None) -> float:
    '''Play until the player leaves the table; return their balance. `table` is the casino table
    (name, min_bet, max_bet, rules); without it the game's own settings are used. `remote` seats
    remote players (GAME_SPEC §3.6); None plays exactly as a single-seat table.'''
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from play_blackjack import play_as_plugin   # lazy: pulls in torch/numpy only when playing

    return play_as_plugin(player_name, balance, table, remote)
