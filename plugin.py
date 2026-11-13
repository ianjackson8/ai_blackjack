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
    "version": "5.2.0",
    "author": "Ian Jackson",
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
    },
}


def run(player_name: str, balance: float, table: dict = None) -> float:
    '''Play until the player leaves the table; return their balance. `table` is the casino table
    (name, min_bet, max_bet, rules); without it the game's own settings are used.'''
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from play_blackjack import play_as_plugin   # lazy: pulls in torch/numpy only when playing

    return play_as_plugin(player_name, balance, table)
