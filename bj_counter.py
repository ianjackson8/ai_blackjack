'''
Card counting with an LSTM.

The LSTM watches every card that becomes visible and predicts, for each of the 13 ranks, what
fraction of that rank is still in the shoe (1.0 = none seen yet, 0.0 = all gone). It is trained
on simulated shoes against the exact answer, so it learns to keep a running tally per rank -
i.e. to count - from a clean signal rather than from noisy wins and losses.

From a predicted composition we derive:
  - a Hi-Lo equivalent true count  (for display and comparison)
  - the player's edge, via standard basic-strategy "effect of removal" values
and the bet is ramped up with the edge.

Author: Ian Jackson
Version v5.1
'''

import random
import time
from typing import Iterable, List, Optional

import numpy as np
import torch
import torch.nn as nn

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'Jack', 'Queen', 'King', 'Ace']
RANK_INDEX = {r: i for i, r in enumerate(RANKS)}
N_RANKS = len(RANKS)
INITIAL_SHARE = np.full(N_RANKS, 1 / N_RANKS)       # every rank is 4 of 52 in a fresh shoe

HILO_TAGS = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1], dtype=np.float64)

# Change in player edge (percent) from removing one card of each rank from a single deck,
# for basic strategy, dealer stands on soft 17. Classic values (Griffin, "Theory of Blackjack").
EOR_PERCENT = np.array([0.38, 0.44, 0.55, 0.69, 0.46, 0.28, 0.00, -0.18, -0.51, -0.51, -0.51, -0.51, -0.61])
BASE_EDGE = -0.005          # neutral-shoe edge assumed for multi-deck basic strategy
EDGE_PER_TRUE_COUNT = 0.005  # rule of thumb used by the plain Hi-Lo tracker
EDGE_PER_UNIT = 0.0025       # bet ramp: one extra unit for every +0.25% of edge

MODEL_ARCH = "counter-v3"
MAX_DECKS = 8


#== Edge / count helpers ==#

def true_count_from_share(share: np.ndarray) -> float:
    '''Hi-Lo true count implied by a remaining composition (share of each rank).'''
    return float(-52 * np.dot(HILO_TAGS, share))


def edge_from_share(share: np.ndarray) -> float:
    '''Estimated player edge for a remaining composition, by linear effect of removal.'''
    removed_per_deck = 52 * (INITIAL_SHARE - share)
    return BASE_EDGE + float(np.dot(EOR_PERCENT, removed_per_deck)) / 100


def bet_units(edge: float, max_spread: int) -> int:
    '''Flat 1 unit at or below even; then one more unit per +0.25% edge, capped.'''
    if edge <= 0:
        return 1
    return int(max(1, min(max_spread, 1 + edge // EDGE_PER_UNIT)))


#== Model ==#

class CardCounterLSTM(nn.Module):
    '''card (one-hot rank) + decks-in-shoe -> LSTM -> fraction of each rank still in the shoe.'''

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=N_INPUTS, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, N_RANKS)
        # start with forget gates wide open so the cell state can act as a running tally
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if name.startswith("bias_ih"):
                    param[hidden:2 * hidden].fill_(3.0)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return torch.sigmoid(self.head(out)), state


N_INPUTS = N_RANKS + 2   # one-hot rank, decks in shoe, fraction of shoe dealt


def _encode(rank_idx: np.ndarray, decks: int, dealt_before: int = 0) -> np.ndarray:
    n = len(rank_idx)
    x = np.zeros((n, N_INPUTS), dtype=np.float32)
    x[np.arange(n), rank_idx] = 1
    x[:, N_RANKS] = decks / MAX_DECKS
    x[:, N_RANKS + 1] = (dealt_before + np.arange(1, n + 1)) / (52 * decks)
    return x


def _simulated_batch(batch: int, decks: int):
    '''Random shuffled shoes with the exact remaining composition after each card.'''
    n = 52 * decks
    xs, ys = [], []
    for _ in range(batch):
        shoe = np.repeat(np.arange(N_RANKS), 4 * decks)
        np.random.shuffle(shoe)
        seen = np.zeros((n, N_RANKS))
        seen[np.arange(n), shoe] = 1
        remaining = 1 - np.cumsum(seen, axis=0) / (4 * decks)   # fraction of each rank left, after each card
        xs.append(_encode(shoe, decks))
        ys.append(remaining.astype(np.float32))
    # drop the final card: nothing remains to predict
    return torch.from_numpy(np.stack(xs)[:, :-1]), torch.from_numpy(np.stack(ys)[:, :-1])


def train_counter(steps: int = 1500, batch: int = 32, log=print, seed: int = 0) -> CardCounterLSTM:
    '''Train the LSTM on simulated shoes of 1-8 decks.'''
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CardCounterLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    start = time.time()
    for step in range(1, steps + 1):
        # curriculum: short shoes first, the full 1-8 deck range by the halfway point
        max_decks = max(1, min(MAX_DECKS, round(MAX_DECKS * step / (steps * 0.5))))
        x, y = _simulated_batch(batch, random.randint(1, max_decks))
        pred, _ = model(x)
        loss = ((pred - y) ** 2).mean() * 100
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            mae = evaluate_counter(model, shoes=4, decks=6)
            log(f"step {step:>5}/{steps}  loss {loss.item():.4f}  true-count error {mae:.3f}  {time.time() - start:.0f}s")
    return model


@torch.no_grad()
def evaluate_counter(model: CardCounterLSTM, shoes: int = 20, decks: int = 6, penetration: float = 0.9) -> float:
    '''Mean absolute error of the LSTM's implied true count vs the exact one.'''
    x, y = _simulated_batch(shoes, decks)
    cut = int(x.shape[1] * penetration)   # very deep in the shoe the true count explodes
    tc_hat = _tc_from_fractions(model(x)[0][:, :cut].numpy())
    tc = _tc_from_fractions(y[:, :cut].numpy())
    return float(np.abs(tc_hat - tc).mean())


def share_from_fractions(left: np.ndarray) -> np.ndarray:
    '''Fraction-of-each-rank-left -> share of the remaining shoe (sums to 1).'''
    left = np.clip(left, 0, 1)
    total = left.sum(-1, keepdims=True)
    return np.divide(left, total, out=np.full_like(left, 1 / N_RANKS), where=total > 1e-9)


def _tc_from_fractions(left: np.ndarray) -> np.ndarray:
    return -52 * (share_from_fractions(left) @ HILO_TAGS)


def save_counter(model: CardCounterLSTM, path: str) -> None:
    torch.save({"arch": MODEL_ARCH, "state_dict": model.state_dict()}, path)


def load_counter(path: str) -> CardCounterLSTM:
    checkpoint = torch.load(path)
    if checkpoint.get("arch") != MODEL_ARCH:
        raise ValueError(f"{path} is not a {MODEL_ARCH} card-counter model")
    model = CardCounterLSTM()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


#== Trackers: what a counter bot uses to "see" the shoe ==#

class Tracker:
    '''Base tracker: keeps exact tallies of what it has seen, knows nothing clever.'''

    name = "flat"

    def __init__(self, num_decks: int):
        self.reset(num_decks)

    def reset(self, num_decks: int) -> None:
        self.num_decks = num_decks
        self.seen = np.zeros(N_RANKS)

    def observe(self, ranks: Iterable[str]) -> None:
        for r in ranks:
            self.seen[RANK_INDEX[r]] += 1

    def _exact_share(self) -> np.ndarray:
        remaining = 4 * self.num_decks - self.seen
        total = remaining.sum()
        return remaining / total if total > 0 else INITIAL_SHARE.copy()

    def true_count(self) -> float:
        return 0.0

    def edge(self) -> float:
        return BASE_EDGE


class HiLoTracker(Tracker):
    '''Classic Hi-Lo: +1 for 2-6, -1 for tens and aces, divided by decks remaining.'''

    name = "hi-lo"

    def true_count(self) -> float:
        left_decks = max((52 * self.num_decks - self.seen.sum()) / 52, 0.5)
        return float(np.dot(HILO_TAGS, self.seen) / left_decks)

    def edge(self) -> float:
        return BASE_EDGE + EDGE_PER_TRUE_COUNT * self.true_count()


class PerfectTracker(Tracker):
    '''Knows the exact remaining composition of what it has seen — the ceiling for the LSTM.'''

    name = "perfect"

    def true_count(self) -> float:
        return true_count_from_share(self._exact_share())

    def edge(self) -> float:
        return edge_from_share(self._exact_share())


class LSTMTracker(Tracker):
    '''Feeds every visible card through the LSTM and reads its predicted composition.'''

    name = "lstm"

    def __init__(self, num_decks: int, model: CardCounterLSTM):
        self.model = model
        super().__init__(num_decks)

    def reset(self, num_decks: int) -> None:
        super().reset(num_decks)
        self.state = None
        self.share = INITIAL_SHARE.copy()

    @torch.no_grad()
    def observe(self, ranks: Iterable[str]) -> None:
        ranks = list(ranks)
        if not ranks:
            return
        super().observe(ranks)   # exact tallies are kept only to show the comparison in the UI
        dealt_before = int(self.seen.sum()) - len(ranks)
        x = torch.from_numpy(_encode(np.array([RANK_INDEX[r] for r in ranks]), self.num_decks, dealt_before)).unsqueeze(0)
        pred, self.state = self.model(x, self.state)
        self.share = share_from_fractions(pred[0, -1].numpy().astype(np.float64))

    def true_count(self) -> float:
        return true_count_from_share(self.share)

    def edge(self) -> float:
        return edge_from_share(self.share)

    def exact_true_count(self) -> float:
        return true_count_from_share(self._exact_share())


def make_tracker(kind: str, num_decks: int, model: Optional[CardCounterLSTM] = None) -> Tracker:
    if kind == "lstm" and model is not None:
        return LSTMTracker(num_decks, model)
    return {"hi-lo": HiLoTracker, "perfect": PerfectTracker}.get(kind, Tracker)(num_decks)
