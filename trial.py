#!myenv/bin/python3
'''
Bot survival trial: every bot starts with the same bankroll, no rebuys, a bot is out when it
hits $0, last one standing wins. Plots balance over rounds and prints a stats table.

    python3 trial.py                      # BTB vs NN vs counter, seed 1
    python3 trial.py --seed 7 --cap 20000
    python3 trial.py --repeat 200         # also run 200 more trials and count who wins

Author: Ian Jackson
Version v5.1
'''

import argparse
import json
import os
import random
from datetime import datetime

import numpy as np
import torch

import play_blackjack as pb
from rich.console import Console
from rich.table import Table
from rich import box

DEFAULT_BOTS = [
    {"name": "BTB Bot", "strategy": "by the books"},
    {"name": "NN Bot", "strategy": "ai-nn"},
    {"name": "Count Bot", "strategy": "counter"},
]
COLOURS = {"BTB Bot": "#4C78A8", "NN Bot": "#E45756", "Count Bot": "#54A24B"}


def run_trial(settings: dict, bots: list, bankroll: float, cap: int, seed: int, record: bool = True) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pb.DEFAULT_BET = settings["default_bet"]
    pb.LOG_GAME = False

    game = pb.BlackjackGame(
        num_decks=settings["num_decks"], player_names=[], bot_info=bots, starting_balance=bankroll,
        log_file="trial", deal_delay=0, allow_double_after_split=settings.get("allow_double_after_split", True),
        split_limit=settings.get("split_limit", -1), counter_max_spread=settings.get("counter", {}).get("max_spread", 8))
    for bot in game.bots:
        if isinstance(bot, pb.TrainableBot):   # fixed skill for the trial: no exploring, no learning
            bot.epsilon = bot.epsilon_min = 0.0
            bot.updates_per_round = 0

    names = [b.name for b in game.bots]
    history = {n: [bankroll] for n in names}
    rounds = {n: [] for n in names}       # per round: (bet, net, results, doubled, split)
    out_at = {}

    while game.game_num < cap:
        alive = [b for b in game.bots if b.balance > 0]
        if len(alive) <= 1:
            break
        before = {b.name: b.balance for b in game.bots}
        game.play_round()
        for b in game.bots:
            if before[b.name] <= 0:
                continue
            history[b.name].append(b.balance)
            if record and b.current_bet > 0:
                rounds[b.name].append((sum(b.hand_bets), b.balance - before[b.name], list(b.results),
                                       any(bet > b.current_bet for bet in b.hand_bets), len(b.hands) > 1))
            if b.balance <= 0:
                out_at[b.name] = game.game_num

    standing = sorted((b for b in game.bots if b.balance > 0), key=lambda b: -b.balance)
    eliminated = sorted(out_at, key=lambda n: -out_at[n])
    order = [b.name for b in standing] + eliminated
    return {"history": history, "rounds": rounds, "out_at": out_at, "order": order,
            "final": {b.name: b.balance for b in game.bots}, "played": game.game_num,
            "stats": {b.name: dict(b.stats) for b in game.bots}}


def streaks(nets):
    best_w = best_l = cur_w = cur_l = 0
    for n in nets:
        cur_w, cur_l = (cur_w + 1, 0) if n > 0 else ((0, cur_l + 1) if n < 0 else (0, 0))
        best_w, best_l = max(best_w, cur_w), max(best_l, cur_l)
    return best_w, best_l


def max_drawdown(balances):
    peak, worst = balances[0], 0.0
    for v in balances:
        peak = max(peak, v)
        worst = max(worst, peak - v)
    return worst


def stats_rows(result: dict, bankroll: float) -> dict:
    out = {}
    for name, hist in result["history"].items():
        rs = result["rounds"][name]
        nets = [r[1] for r in rs]
        bets = [r[0] for r in rs]
        hands = [res for r in rs for res in r[2]]
        n_hands = len(hands) or 1
        wins = [n for n in nets if n > 0]
        losses = [n for n in nets if n < 0]
        wagered = sum(bets)
        peak = max(hist)
        out[name] = {
            "Place": result["order"].index(name) + 1,
            "Status": (f"out @ round {result['out_at'][name]:,}" if name in result["out_at"]
                       else "standing"),
            "Final $": result["final"][name],
            "Peak $": peak,
            "Peak round": hist.index(peak),
            "Low $": min(hist),
            "Rounds played": len(rs),
            "Hand win %": 100 * sum(h in ("win", "blackjack") for h in hands) / n_hands,
            "Hand loss %": 100 * sum(h in ("lose", "busted") for h in hands) / n_hands,
            "Push %": 100 * hands.count("push") / n_hands,
            "Bust %": 100 * hands.count("busted") / n_hands,
            "Blackjacks": hands.count("blackjack"),
            "Avg bet $": float(np.mean(bets)) if bets else 0,
            "Max bet $": max(bets) if bets else 0,
            "Avg win $": float(np.mean(wins)) if wins else 0,
            "Avg loss $": float(np.mean(losses)) if losses else 0,
            "Biggest win $": max(nets) if nets else 0,
            "Biggest loss $": min(nets) if nets else 0,
            "Total wagered $": wagered,
            "Net $": result["final"][name] - bankroll,
            "Return per $ bet": 100 * sum(nets) / wagered if wagered else 0,
            "Doubles": sum(r[3] for r in rs),
            "Splits": sum(r[4] for r in rs),
            "Max drawdown $": max_drawdown(hist),
            "Win streak": streaks(nets)[0],
            "Loss streak": streaks(nets)[1],
        }
    return out


def fmt(key, value):
    if key in ("Place", "Status"):
        return str(value)
    if key.endswith("%") or key == "Return per $ bet":
        return f"{value:+.2f}%" if key == "Return per $ bet" else f"{value:.1f}%"
    if key.endswith("$"):
        sign = "+" if key in ("Net $",) and value > 0 else ""
        return f"{sign}{'-' if value < 0 else ''}${abs(value):,.0f}" if float(value).is_integer() or abs(value) >= 100 \
            else f"{sign}{'-' if value < 0 else ''}${abs(value):,.2f}"
    return f"{value:,}"


def plot(result: dict, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, hist in result["history"].items():
        colour = COLOURS.get(name)
        ax.plot(range(len(hist)), hist, label=name, color=colour, linewidth=1.4)
        if name in result["out_at"]:
            ax.plot(len(hist) - 1, 0, marker="X", markersize=12, color=colour)
            ax.annotate(f"{name} out\nround {result['out_at'][name]:,}", (len(hist) - 1, 0),
                        textcoords="offset points", xytext=(6, 14), fontsize=9, color=colour)
    ax.axhline(result["history"][next(iter(result["history"]))][0], color="grey", linewidth=0.8, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel("Balance ($)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bankroll", type=float, default=500)
    parser.add_argument("--cap", type=int, default=50000, help="stop after this many rounds if 2+ bots remain")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=0, help="extra trials (different seeds) to count wins")
    parser.add_argument("--no-open", action="store_true", help="don't open the graph when done")
    args = parser.parse_args()

    with open(pb.SETTINGS_FILE) as f:
        settings = json.load(f)
    console = Console()

    result = run_trial(settings, DEFAULT_BOTS, args.bankroll, args.cap, args.seed)
    winner = result["order"][0]
    how = "last one standing" if len(result["order"]) - len(result["out_at"]) == 1 else f"highest balance at the {args.cap:,}-round cap"
    console.print(f"\n[bold]Trial (seed {args.seed})[/]: {result['played']:,} rounds — "
                  f"[bold green]{winner} wins[/] ({how}).\n")

    stats = stats_rows(result, args.bankroll)
    names = sorted(stats, key=lambda n: stats[n]["Place"])
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold gold1")
    table.add_column("Stat", style="bold")
    for n in names:
        table.add_column(n, justify="right")
    for key in stats[names[0]]:
        table.add_row(key, *(fmt(key, stats[n][key]) for n in names))
    console.print(table)

    results = pb.HERE / "results"
    results.mkdir(exist_ok=True)
    path = str(results / f"trial_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plot(result, path, f"Bot survival trial — ${args.bankroll:,.0f} each, no rebuys (seed {args.seed}) — {winner} wins")
    console.print(f"Graph saved to [bold]{path}[/]")

    if args.repeat:
        wins, survival = {n: 0 for n in names}, {n: [] for n in names}
        for i in range(args.repeat):
            r = run_trial(settings, DEFAULT_BOTS, args.bankroll, args.cap, 10_000 + i, record=False)
            wins[r["order"][0]] += 1
            for n in names:
                survival[n].append(r["out_at"].get(n, r["played"]))
        rep = Table(title=f"{args.repeat} more trials", box=box.SIMPLE_HEAVY, header_style="bold gold1")
        for col in ("Bot", "Wins", "Win rate", "Median rounds survived"):
            rep.add_column(col, justify="right" if col != "Bot" else "left")
        for n in names:
            rep.add_row(n, str(wins[n]), f"{100 * wins[n] / args.repeat:.0f}%", f"{int(np.median(survival[n])):,}")
        console.print(rep)

    if not args.no_open and os.name == "posix":
        os.system(f"open '{path}' 2>/dev/null || xdg-open '{path}' 2>/dev/null")


if __name__ == "__main__":
    main()
