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
    "version": "5.1.0",
    "author": "Ian Jackson",
}


def run(player_name: str, balance: float) -> float:
    '''Play until the player leaves the table; return their balance.'''
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from play_blackjack import play_as_plugin   # lazy: pulls in torch/numpy only when playing

    return play_as_plugin(player_name, balance)
