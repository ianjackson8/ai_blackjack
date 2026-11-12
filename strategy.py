'''
Basic strategy for this game's rules: multi-deck, dealer stands on soft 17, double on any
two cards, no surrender. Double-after-split is a parameter because it is a game setting.

Author: Ian Jackson
Version v5.0
'''

from typing import Sequence

TEN_RANKS = {"10", "Jack", "Queen", "King"}


def card_points(rank: str) -> int:
    '''Value of a card for strategy lookups (Ace = 11).'''
    if rank in TEN_RANKS:
        return 10
    if rank == "Ace":
        return 11
    return int(rank)


def _total(ranks: Sequence[str]):
    hard = sum(1 if r == "Ace" else card_points(r) for r in ranks)
    soft = "Ace" in ranks and hard + 10 <= 21
    return (hard + 10, True) if soft else (hard, False)


def _hard(total: int, up: int, can_double: bool) -> str:
    if total >= 17:
        return "stand"
    if 13 <= total <= 16:
        return "stand" if up <= 6 else "hit"
    if total == 12:
        return "stand" if 4 <= up <= 6 else "hit"
    if total == 11:
        return "double" if can_double and up <= 10 else "hit"
    if total == 10:
        return "double" if can_double and up <= 9 else "hit"
    if total == 9:
        return "double" if can_double and 3 <= up <= 6 else "hit"
    return "hit"


def _soft(total: int, up: int, can_double: bool) -> str:
    if total >= 19:
        return "stand"
    if total == 18:
        if 3 <= up <= 6:
            return "double" if can_double else "stand"
        return "stand" if up in (2, 7, 8) else "hit"
    if total == 17:
        return "double" if can_double and 3 <= up <= 6 else "hit"
    if total in (15, 16):
        return "double" if can_double and 4 <= up <= 6 else "hit"
    # soft 13-14
    return "double" if can_double and 5 <= up <= 6 else "hit"


def _should_split(rank: str, up: int, das: bool) -> bool:
    points = card_points(rank)
    if rank == "Ace" or points == 8:
        return True
    if points in (10, 5):
        return False
    if points == 9:
        return up not in (7, 10, 11)
    if points == 7:
        return up <= 7
    if points == 6:
        return up <= 6 if das else 3 <= up <= 6
    if points == 4:
        return das and up in (5, 6)
    # 2s and 3s
    return up <= 7 if das else 4 <= up <= 7


def basic_strategy(ranks: Sequence[str], dealer_up_rank: str, can_double: bool,
                   can_split: bool, das: bool = True) -> str:
    '''Return "hit", "stand", "double" or "split" for a hand given as card ranks.'''
    up = card_points(dealer_up_rank)
    if can_split and len(ranks) == 2 and ranks[0] == ranks[1] and _should_split(ranks[0], up, das):
        return "split"
    total, soft = _total(ranks)
    return _soft(total, up, can_double) if soft else _hard(total, up, can_double)
