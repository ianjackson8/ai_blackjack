'''
Blackjack side bets: Perfect Pairs and 21+3.

Both are settled on the initial deal, independently of how the hand plays out. Each is placed per
hand (spot) and pays "N to 1": a win returns the stake plus N times the stake.

Author: Ian Jackson
'''
from typing import Dict, List, Optional, Tuple

RED_SUITS = {"Hearts", "Diamonds"}
RANK_ORDER = ['Ace', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King']

# outcome name -> pays N:1, best first
PERFECT_PAIRS_PAYS = [("Perfect pair", 25), ("Colored pair", 12), ("Mixed pair", 6)]
TWENTY_ONE_PLUS_THREE_PAYS = [("Suited trips", 100), ("Straight flush", 40), ("Three of a kind", 30),
                              ("Straight", 10), ("Flush", 5)]

Result = Optional[Tuple[str, int]]   # (outcome name, pays N:1), or None for a losing bet


def perfect_pairs(c1, c2) -> Result:
    '''Your first two cards: same rank and suit 25:1, same rank and colour 12:1, same rank 6:1.'''
    if c1.rank != c2.rank:
        return None
    if c1.suit == c2.suit:
        return PERFECT_PAIRS_PAYS[0]
    if (c1.suit in RED_SUITS) == (c2.suit in RED_SUITS):
        return PERFECT_PAIRS_PAYS[1]
    return PERFECT_PAIRS_PAYS[2]


def _is_straight(ranks: List[str]) -> bool:
    '''Three consecutive ranks; an Ace plays high (Q-K-A) or low (A-2-3), but doesn't wrap (K-A-2).'''
    idx = sorted(RANK_ORDER.index(r) for r in ranks)
    if len(set(idx)) != 3:
        return False
    if idx[2] - idx[0] == 2:
        return True
    return idx == [0, 11, 12]        # Q-K-A


def twenty_one_plus_three(c1, c2, up) -> Result:
    '''Your first two cards plus the dealer's up card as a three-card poker hand.'''
    cards = [c1, c2, up]
    ranks = [c.rank for c in cards]
    flush = len({c.suit for c in cards}) == 1
    trips = len(set(ranks)) == 1
    straight = _is_straight(ranks)
    if trips and flush:
        return TWENTY_ONE_PLUS_THREE_PAYS[0]
    if straight and flush:
        return TWENTY_ONE_PLUS_THREE_PAYS[1]
    if trips:
        return TWENTY_ONE_PLUS_THREE_PAYS[2]
    if straight:
        return TWENTY_ONE_PLUS_THREE_PAYS[3]
    if flush:
        return TWENTY_ONE_PLUS_THREE_PAYS[4]
    return None


# key (= the setting / table rule that switches it on) -> how it's shown and settled
SIDE_BETS: Dict[str, dict] = {
    "perfect_pairs": {"label": "Perfect Pairs", "short": "PP", "pays": PERFECT_PAIRS_PAYS,
                      "about": "your first two cards make a pair",
                      "settle": lambda hand, up: perfect_pairs(*hand.cards[:2])},
    "twenty_one_plus_three": {"label": "21+3", "short": "21+3", "pays": TWENTY_ONE_PLUS_THREE_PAYS,
                              "about": "your two cards and the dealer's up card make a poker hand",
                              "settle": lambda hand, up: twenty_one_plus_three(*hand.cards[:2], up)},
}


def paytable(key: str) -> str:
    '''"Perfect pair 25:1 · Colored pair 12:1 · Mixed pair 6:1"'''
    return " · ".join(f"{name} {n}:1" for name, n in SIDE_BETS[key]["pays"])


def side_bet_limits(settings: dict, key: str) -> Tuple[float, float]:
    '''(min, max) per hand for a side bet; a max below the min is raised to it.'''
    lo = max(1.0, float(settings.get(f"{key}_min", 1)))
    hi = max(lo, float(settings.get(f"{key}_max", 100)))
    return lo, hi
