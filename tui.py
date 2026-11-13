'''
Terminal UI for AI Blackjack, built on Rich.

Two front-ends share one interface:
  RichUI      - full-screen table with card art, live log, and single-key controls
  HeadlessUI  - plain stdin/stdout fallback (tests, pipes, silent training)

Author: Ian Jackson
Version v5.0
'''

from __future__ import annotations

import os
import sys
import time
import weakref
from collections import deque
from types import SimpleNamespace
from typing import Iterable, List, Optional, Sequence, Tuple

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

#== Card art ==#

SUIT_SYMBOLS = {"Hearts": "♥", "Diamonds": "♦", "Clubs": "♣", "Spades": "♠"}
RED_SUITS = {"Hearts", "Diamonds"}
RANK_SHORT = {"Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}

CARD_BG = "on grey93"
BORDER_STYLE = f"grey50 {CARD_BG}"
RED_STYLE = f"bold red3 {CARD_BG}"
BLACK_STYLE = f"bold grey11 {CARD_BG}"
BACK_BORDER = "bright_white on dark_blue"
BACK_FILL = "blue on dark_blue"

CARD_W = 7   # full card width
SLIVER_W = 3  # visible width of an overlapped card

# Hit animation: the card slides in face down, then flips face up
SLIDE_S = 0.35           # seconds sliding in
FLIP_S = 0.30            # seconds flipping
ANIM_S = SLIDE_S + FLIP_S
SLIDE_COLS = 12          # how far to the right the card starts
FLIP_WIDTHS = (5, 3, 1, 3, 5)   # card width through the flip (edge-on in the middle)
ANIM_FPS = 30

RESULT_BADGES = {
    "blackjack": (" BLACKJACK ", "bold black on gold1"),
    "win":       (" WIN ", "bold black on green3"),
    "push":      (" PUSH ", "bold black on grey70"),
    "lose":      (" LOSE ", "bold white on red3"),
    "busted":    (" BUST ", "bold white on red3"),
}

# (keys, value, label) — the first key is the one shown on screen
Option = Tuple[str, str, str]


def _key_matches(key: str, keys) -> bool:
    '''Option keys are either a string of single characters or a tuple of key names like ("up", "k").'''
    if isinstance(keys, str):
        return len(key) == 1 and key in keys
    return key in keys


KEY_LABELS = {"up": "↑", "down": "↓", "left": "←", "right": "→", "enter": "↵", "esc": "Esc", "tab": "Tab"}


def key_label(keys) -> str:
    first = keys[0]
    return KEY_LABELS.get(first, first.upper())


def esc(value) -> str:
    '''Escape a value for use inside Rich markup.'''
    return escape(str(value))


def money(value) -> str:
    value = float(value)
    return f"${value:,.0f}" if value.is_integer() else f"${value:,.2f}"


def signed_money(value) -> str:
    value = float(value)
    return f"+{money(value)}" if value >= 0 else f"-{money(abs(value))}"


def short_rank(card) -> str:
    return RANK_SHORT.get(card.rank, card.rank)


def card_markup(card) -> str:
    '''Inline coloured card label for log lines, e.g. "K♥".'''
    colour = "red" if card.suit in RED_SUITS else "bold white"
    return f"[{colour}]{short_rank(card)}{SUIT_SYMBOLS[card.suit]}[/]"


def hand_total(hand) -> Tuple[int, bool]:
    '''Return (value, is_soft) for a hand.'''
    hard = sum(min(c.values) for c in hand.cards)
    has_ace = any(c.rank == "Ace" for c in hand.cards)
    if has_ace and hard + 10 <= 21:
        return hard + 10, True
    return hard, False


def hand_label(hand) -> str:
    value, soft = hand_total(hand)
    return f"soft {value}" if soft and value != 21 else str(value)


def _card_rows(card, hidden: bool, sliver: bool) -> List[Text]:
    if hidden:
        if sliver:
            return [Text("╭──", BACK_BORDER)] + [Text("│▒▒", BACK_FILL)] * 3 + [Text("╰──", BACK_BORDER)]
        return ([Text("╭─────╮", BACK_BORDER)]
                + [Text.assemble(("│", BACK_BORDER), ("▒▒▒▒▒", BACK_FILL), ("│", BACK_BORDER))] * 3
                + [Text("╰─────╯", BACK_BORDER)])

    rank = short_rank(card)
    suit = SUIT_SYMBOLS[card.suit]
    ink = RED_STYLE if card.suit in RED_SUITS else BLACK_STYLE
    pad = CARD_BG

    if sliver:
        return [
            Text("╭──", BORDER_STYLE),
            Text.assemble(("│", BORDER_STYLE), (f"{rank:<2}", ink)),
            Text.assemble(("│", BORDER_STYLE), (suit, ink), (" ", pad)),
            Text.assemble(("│", BORDER_STYLE), ("  ", pad)),
            Text("╰──", BORDER_STYLE),
        ]
    return [
        Text("╭─────╮", BORDER_STYLE),
        Text.assemble(("│", BORDER_STYLE), (f"{rank:<2}", ink), ("   ", pad), ("│", BORDER_STYLE)),
        Text.assemble(("│", BORDER_STYLE), ("  ", pad), (suit, ink), ("  ", pad), ("│", BORDER_STYLE)),
        Text.assemble(("│", BORDER_STYLE), ("   ", pad), (f"{rank:>2}", ink), ("│", BORDER_STYLE)),
        Text("╰─────╯", BORDER_STYLE),
    ]


def _narrow_rows(card, hidden: bool, w: int) -> List[Text]:
    '''A card squeezed to width w (mid-flip). w == 1 is the card seen edge-on.'''
    border = BACK_BORDER if hidden else BORDER_STYLE
    if w <= 1:
        return [Text("│", border)] * 5
    inner = w - 2
    top, bottom = Text("╭" + "─" * inner + "╮", border), Text("╰" + "─" * inner + "╯", border)
    if hidden:
        mid = [Text.assemble(("│", border), ("▒" * inner, BACK_FILL), ("│", border))] * 3
        return [top] + mid + [bottom]
    ink = RED_STYLE if card.suit in RED_SUITS else BLACK_STYLE
    rank, suit = short_rank(card), SUIT_SYMBOLS[card.suit]
    if inner >= 3:
        cells = [(f"{rank:<{inner}}", ink), (suit.center(inner), ink), (f"{rank:>{inner}}", ink)]
    else:
        cells = [(" " * inner, CARD_BG), (suit.center(inner), ink), (" " * inner, CARD_BG)]
    mid = [Text.assemble(("│", border), cell, ("│", border)) for cell in cells]
    return [top] + mid + [bottom]


def _animated_rows(card, age: float) -> Tuple[List[Text], int]:
    '''Frame of the hit animation at `age` seconds: (card rows, extra left offset).'''
    if age < SLIDE_S:
        p = age / SLIDE_S
        eased = 1 - (1 - p) ** 3
        return _card_rows(card, hidden=True, sliver=False), round(SLIDE_COLS * (1 - eased))
    step = min(len(FLIP_WIDTHS) - 1, int((age - SLIDE_S) / FLIP_S * len(FLIP_WIDTHS)))
    w = FLIP_WIDTHS[step]
    rows = _narrow_rows(card, hidden=step < len(FLIP_WIDTHS) // 2, w=w)
    left, right = (CARD_W - w) // 2, CARD_W - w - (CARD_W - w) // 2
    return [Text(" " * left) + r + Text(" " * right) for r in rows], 0


def _compact_card(card, hidden: bool) -> Text:
    if hidden:
        return Text(" ▒▒ ", BACK_BORDER)
    ink = RED_STYLE if card.suit in RED_SUITS else BLACK_STYLE
    return Text(f" {short_rank(card)}{SUIT_SYMBOLS[card.suit]} ", ink)


def render_cards(cards: Sequence, width: int, hide_first: bool = False, compact: bool = False,
                 last_age: Optional[float] = None) -> Text:
    '''
    Render cards side by side, overlapping or compacting them to fit the width.
    last_age: seconds since the last card arrived, to draw it mid hit-animation (None = settled).
    '''
    if not cards:
        return Text("")
    n = len(cards)
    spread_w = n * (CARD_W + 1) - 1
    overlap_w = (n - 1) * SLIVER_W + CARD_W

    if compact or overlap_w > width:
        line = Text()
        for i, card in enumerate(cards):
            if i:
                line.append(" ")
            line.append_text(_compact_card(card, hide_first and i == 0))
        return line

    overlap = spread_w > width
    columns = [
        _card_rows(card, hide_first and i == 0, sliver=overlap and i < n - 1)
        for i, card in enumerate(cards)
    ]
    offset = 0
    if last_age is not None and last_age < ANIM_S:
        columns[-1], offset = _animated_rows(cards[-1], last_age)
        offset = max(0, min(offset, width - (overlap_w if overlap else spread_w)))
    rows = []
    for r in range(5):
        row = Text()
        for i, col in enumerate(columns):
            if i and not overlap:
                row.append(" ")
            if i == n - 1 and offset:
                row.append(" " * offset)
            row.append_text(col[r])
        rows.append(row)
    return Text("\n").join(rows)


def badge(result: str) -> Text:
    label, style = RESULT_BADGES.get(result, (f" {result.upper()} ", "reverse"))
    return Text(label, style)


#== Chips ==#

# (denomination, label, style) — standard casino colours
CHIP_DENOMS = [
    (1000, "1K",  "bold black on orange1"),
    (500,  "500", "bold white on purple4"),
    (100,  "100", "bold white on grey11"),
    (25,   "25",  "bold white on green4"),
    (5,    "5",   "bold white on red3"),
    (1,    "1",   "bold black on grey85"),
]
CHIP_DIM = "grey50 on grey23"


def chip_breakdown(amount) -> List[Tuple[int, str, str]]:
    '''Greedy split of an amount into chips, largest first (fractions are dropped).'''
    chips = []
    remaining = int(float(amount) + 1e-9)
    for value, label, style in CHIP_DENOMS:
        count, remaining = divmod(remaining, value)
        chips += [(value, label, style)] * count
    return chips


def chip_stack(amount, dim: bool = False) -> Text:
    '''A row of chips, largest first; three or more of a kind collapse to one chip plus "×n".'''
    counts: dict = {}
    for value, label, style in chip_breakdown(amount):
        counts.setdefault(value, [label, style, 0])[2] += 1
    text = Text()
    for label, style, count in counts.values():
        for _ in range(count if count <= 2 else 1):
            if text:
                text.append(" ")
            text.append(f"{label:^3}", CHIP_DIM if dim else style)
        if count > 2:
            text.append(f"×{count}", "grey50" if dim else "bold grey85")
    return text


def chip_line(bet, result: Optional[str], width: int) -> Text:
    '''Chips for one hand's bet; in the results phase shows what happened to them.'''
    line = Text(no_wrap=True, overflow="crop")
    if result in ("lose", "busted"):
        line.append_text(chip_stack(bet, dim=True))
        line.append(f" {money(bet)}", "strike grey50")
        return line
    line.append_text(chip_stack(bet))
    line.append(f" {money(bet)}", "bold")
    payout = {"win": bet, "blackjack": bet * 1.5}.get(result)
    if payout:
        paid = Text.assemble((" + ", "grey62"), chip_stack(payout))
        if line.cell_len + paid.cell_len <= width:
            line.append_text(paid)
        else:
            line.append(f" +{money(payout)}", "bold green")
    return line


# Side-view chip stacks: (body colour, edge-spot colour, label ink) per denomination
STACK_COLOURS = {
    1000: ("orange1", "grey93", "black"),
    500:  ("purple4", "gold1", "white"),
    100:  ("grey11", "grey93", "white"),
    25:   ("green4", "grey93", "white"),
    5:    ("red3", "grey93", "white"),
    1:    ("grey85", "dodger_blue2", "black"),
}
STACK_GAP = "#06261c"   # thin dark line between chips
STACK_MAX = 4           # chips drawn per stack; taller stacks get a "×n" tag
CHIP_W = 5


def _one_stack(value: int, label: str, count: int, dim: bool) -> List[Text]:
    body, spot, ink = STACK_COLOURS[value]
    if dim:
        body, spot, ink = "grey30", "grey42", "grey58"
    rows = []
    if count > STACK_MAX:
        rows.append(Text(f"{'×' + str(count):^{CHIP_W}}", "grey50" if dim else "bold grey85"))
    rows.append(Text(f"{label:^{CHIP_W}}", f"bold {ink} on {body}"))   # top chip face
    for _ in range(min(count, STACK_MAX) - 1):                         # chip edges below it
        edge = Text()
        for j in range(CHIP_W):
            edge.append("▆", f"{spot if j in (1, 3) else body} on {STACK_GAP}")
        rows.append(edge)
    return rows


def hjoin(blocks: Sequence[List[Text]], gap: int = 1) -> Text:
    '''Place blocks of lines side by side, bottom-aligned.'''
    height = max((len(b) for b in blocks), default=0)
    widths = [max((t.cell_len for t in b), default=0) for b in blocks]
    lines = []
    for r in range(height):
        line = Text(no_wrap=True, overflow="crop")
        for k, (block, w) in enumerate(zip(blocks, widths)):
            if k:
                line.append(" " * gap)
            idx = r - (height - len(block))
            cell = block[idx] if idx >= 0 else Text("")
            line.append_text(cell)
            line.append(" " * (w - cell.cell_len))
        lines.append(line)
    return Text("\n").join(lines)


def stack_blocks(amount, dim: bool = False) -> List[List[Text]]:
    counts: dict = {}
    for value, label, _ in chip_breakdown(amount):
        counts.setdefault(value, [label, 0])[1] += 1
    return [_one_stack(value, label, n, dim) for value, (label, n) in counts.items()]


def chip_stacks(bet, result: Optional[str], width: int) -> Text:
    '''Side-view chip stacks for a hand's bet, with the payout pushed alongside on a win.'''
    lost = result in ("lose", "busted")
    blocks = stack_blocks(bet, dim=lost)
    label = Text(money(bet), "strike grey50" if lost else "bold")
    payout = {"win": bet, "blackjack": bet * 1.5}.get(result)
    if payout:
        paid = stack_blocks(payout)
        with_paid = blocks + [[Text("+", "grey62")]] + paid + [[label]]
        if sum(max(t.cell_len for t in b) + 1 for b in with_paid) <= width:
            return hjoin(with_paid)
        label = Text.assemble((money(bet), "bold"), (f" +{money(payout)}", "bold green"))
    return hjoin(blocks + [[label]])


#== Tables used by the game ==#

class _Responsive:
    '''Builds its renderable at render time, once the available width is known.'''

    def __init__(self, build):
        self.build = build

    def __rich_console__(self, console, options):
        yield self.build(options.max_width)


def stats_table(game):
    return _Responsive(lambda width: _stats_table(game, wide=width >= 96))


def _stats_table(game, wide: bool = True) -> Table:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold gold1", expand=False, pad_edge=False)
    table.add_column("Player", style="bold", no_wrap=True)
    table.add_column("Balance", justify="right", no_wrap=True)
    table.add_column("Net", justify="right", no_wrap=True)
    for col in ("W", "L", "P", "Bust", "BJ"):
        table.add_column(col, justify="right")
    table.add_column("Win %", justify="right", no_wrap=True)
    any_rebuys = any(getattr(p, "rebuys", 0) for p in game.players + game.bots)
    if any_rebuys:
        table.add_column("Rebuys", justify="right", style="magenta")
    if wide:
        table.add_column("Best", justify="right", style="green", no_wrap=True)
        table.add_column("Worst", justify="right", style="red", no_wrap=True)

    for player in game.players + game.bots:
        s = player.stats
        history = game.balance_history.get(player.name, [])
        init_bal = history[0] if history else player.balance
        rebuys = getattr(player, "rebuys", 0)
        net = player.balance - init_bal - rebuys * getattr(game, "starting_balance", 0)
        total = sum(s.values())
        win_rate = (s["win"] + s["blackjack"]) / total * 100 if total else 0.0
        deltas = [history[i + 1] - history[i] for i in range(len(history) - 1)]
        best = signed_money(max(deltas)) if deltas and max(deltas) > 0 else "—"
        worst = signed_money(min(deltas)) if deltas and min(deltas) < 0 else "—"
        row = [
            esc(player.name),
            money(player.balance),
            Text(signed_money(net), "green" if net >= 0 else "red"),
            str(s["win"]), str(s["lose"]), str(s["push"]), str(s["bust"]), str(s["blackjack"]),
            f"{win_rate:.1f}%",
        ]
        if any_rebuys:
            row.append(str(rebuys) if rebuys else "")
        if wide:
            row += [best, worst]
        table.add_row(*row)
    return table


def key_value_table(rows: Iterable[Tuple[str, str]], key_style: str = "bold cyan") -> Table:
    table = Table.grid(padding=(0, 3))
    table.add_column(style=key_style, no_wrap=True)
    table.add_column()
    for k, v in rows:
        table.add_row(k, v)
    return table


#== Base / headless UI ==#

class BaseUI:
    interactive = False

    def __init__(self):
        self.game = None
        self.splash_info = ""
        self.active_player = None
        self.active_hand: Optional[int] = None

    # lifecycle
    def attach(self, game) -> None:
        self.game = game

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    # state
    def set_active(self, player=None, hand_index: Optional[int] = None) -> None:
        self.active_player = player
        self.active_hand = hand_index
        self.refresh()

    def refresh(self) -> None:
        pass

    def set_overlay(self, title: str, renderable) -> None:
        pass

    def clear_overlay(self) -> None:
        pass

    def on_round_end(self, game) -> None:
        pass

    def add_splash_name(self, name: str) -> None:
        pass


class HeadlessUI(BaseUI):
    '''Plain-text fallback. Uses builtins.input so tests can patch it.'''

    def __init__(self, verbose: bool = False, progress: bool = False):
        super().__init__()
        self.verbose = verbose
        self.progress = progress
        self.console = Console(highlight=False)

    def log(self, message: str) -> None:
        if self.verbose:
            self.console.print("  " + message)

    def error(self, message: str) -> None:
        if self.verbose:
            self.console.print(f"  [red]{message}[/]")

    def pause(self, seconds: float) -> None:
        if self.verbose and seconds:
            time.sleep(seconds)

    def choose(self, prompt: str, options: Sequence[Option], default: Optional[str] = None) -> str:
        hints = "  ".join(f"[{key_label(keys)}] {label}" for keys, _, label in options)
        raw = input(f"  {Text.from_markup(prompt).plain}  {hints}\n  > ").strip().lower()
        if not raw and default is not None:
            return default
        for keys, value, label in options:
            if _key_matches(raw, keys) or raw == value or raw == label.lower():
                return value
        return raw

    def ask(self, prompt: str, default: str = "", hotkeys: Optional[dict] = None) -> str:
        suffix = f" [{default}]" if default else ""
        raw = input(f"  {Text.from_markup(prompt).plain}{suffix}: ").strip()
        return raw or default

    def wait_key(self, prompt: str = "Press Enter to continue") -> None:
        if self.verbose:
            input(f"  {prompt}")

    def show_info(self, title: str, renderable) -> None:
        if self.verbose:
            self.console.print(Panel(renderable, title=title, expand=False))

    def print_after(self, renderable) -> None:
        if self.progress:
            self.progress = False
            sys.stdout.write("\n")
        self.console.print(renderable)

    def on_round_end(self, game) -> None:
        if not self.progress:
            return
        target = getattr(game, "train_until", None)
        parts = [f"Round {game.game_num:>6,}" + (f"/{target:,}" if target else "")]
        for bot in game.bots:
            part = f"{bot.name} {money(bot.balance)}"
            if hasattr(bot, "epsilon"):
                part += f" (ε {bot.epsilon:.3f})"
            if getattr(bot, "rebuys", 0):
                part += f" rebuys {bot.rebuys}"
            parts.append(part)
        sys.stdout.write("\r  " + "   ".join(parts) + "    ")
        sys.stdout.flush()


#== Rich full-screen UI ==#

class _KeyReader:
    '''Reads single keypresses from the terminal without echo.'''

    def __init__(self):
        self._fd = None
        self._old = None

    def __enter__(self):
        if os.name != "nt":
            import termios
            import tty
            self._fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)  # keeps ISIG, so Ctrl-C still raises KeyboardInterrupt
        return self

    def __exit__(self, *exc):
        if self._old is not None:
            import termios
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            self._old = None

    def read(self) -> str:
        if os.name == "nt":
            import msvcrt
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                return {"H": "up", "P": "down", "K": "left", "M": "right"}.get(msvcrt.getwch(), "")
            if ch == "\x03":
                raise KeyboardInterrupt
            return {"\r": "enter", "\n": "enter", "\x08": "backspace", "\x1b": "esc", "\t": "tab"}.get(ch, ch)

        import select
        first = os.read(self._fd, 1)
        if first == b"\x1b":
            # swallow the rest of an escape sequence (arrow keys etc.)
            seq = b""
            while select.select([self._fd], [], [], 0.02)[0]:
                seq += os.read(self._fd, 1)
            if not seq:
                return "esc"
            return {b"[A": "up", b"[B": "down", b"[C": "right", b"[D": "left",
                    b"OA": "up", b"OB": "down", b"OC": "right", b"OD": "left"}.get(seq, "")
        if first == b"\t":
            return "tab"
        if first in (b"\r", b"\n"):
            return "enter"
        if first in (b"\x7f", b"\x08"):
            return "backspace"
        if first == b"\x04":
            raise EOFError
        # decode multi-byte UTF-8
        lead = first[0]
        extra = 1 if lead >> 5 == 0b110 else 2 if lead >> 4 == 0b1110 else 3 if lead >> 3 == 0b11110 else 0
        data = first + (os.read(self._fd, extra) if extra else b"")
        return data.decode("utf-8", errors="ignore")


class _LogView:
    '''Shows the newest log lines that fit in the region it is given.'''

    def __init__(self, lines: List[str], title: str):
        self.lines = lines
        self.title = title

    def __rich_console__(self, console, options):
        height = options.height or console.height
        inner_w = max(10, options.max_width - 4)
        budget = max(1, height - 2)
        picked: List[Text] = []
        for markup in reversed(self.lines):
            text = Text.from_markup(markup)
            used = len(text.wrap(console, inner_w))
            if used > budget:
                break
            picked.append(text)
            budget -= used
        picked.reverse()
        yield Panel(Group(*picked) if picked else Text("…", "dim"), title=self.title,
                    title_align="left", border_style="grey42", height=height, box=box.ROUNDED)


class RichUI(BaseUI):
    interactive = True

    def __init__(self):
        super().__init__()
        self.console = Console(highlight=False)
        self._live: Optional[Live] = None
        self._keys = _KeyReader()
        self._log: deque = deque(maxlen=300)
        self._prompt: Optional[str] = None
        self._options: Sequence[Option] = ()
        self._default: Optional[str] = None
        self._buffer: Optional[str] = None
        self._placeholder = ""
        self._flash: Optional[str] = None
        self._overlay = None
        self._overlay_title = ""
        self._splash_names: List[str] = []
        self.splash_info = ""
        self.animate = True                      # hit animation on/off (settings: animate_cards)
        self._births = weakref.WeakKeyDictionary()  # card -> time it first appeared mid-hand
        self._anim_until = 0.0
        self._held_log: deque = deque()          # log lines held back until the card flips

    # lifecycle
    def start(self) -> None:
        if self._live:
            return
        self._keys.__enter__()
        self._live = Live(
            get_renderable=self._render,
            console=self.console,
            screen=True,
            auto_refresh=True,
            refresh_per_second=8,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None
        self._keys.__exit__()

    # output
    def log(self, message: str) -> None:
        self.refresh()   # registers any card that just arrived, so its log line can wait for the flip
        if self._held_log or time.monotonic() < self._anim_until:
            self._held_log.append(message)
        else:
            self._log.append(message)
        self.refresh()

    def error(self, message: str) -> None:
        self._flash = message
        self.refresh()

    def refresh(self) -> None:
        if self._live:
            self._live.refresh()

    def pause(self, seconds: float) -> None:
        end = time.monotonic() + (seconds or 0)
        self.refresh()
        end = max(end, self._anim_until)   # always let a hit animation finish
        while (now := time.monotonic()) < end:
            if now < self._anim_until:
                time.sleep(min(1 / ANIM_FPS, end - now))
                self.refresh()
            else:
                self.refresh()
                time.sleep(end - now)
        self.refresh()

    # hit animation
    def _card_age(self, cards: Sequence) -> Optional[float]:
        '''Seconds since the newest card in a hand appeared, while it is still animating.'''
        if not self.animate or len(cards) < 3:     # the initial two-card deal isn't animated
            return None
        card, now = cards[-1], time.monotonic()
        born = self._births.get(card)
        if born is None:
            born = self._births[card] = now
            self._anim_until = max(self._anim_until, now + ANIM_S)
        age = now - born
        return age if age < ANIM_S else None

    def set_overlay(self, title: str, renderable) -> None:
        self._overlay_title = title
        self._overlay = renderable
        self.refresh()

    def clear_overlay(self) -> None:
        self._overlay = None
        self.refresh()

    def show_info(self, title: str, renderable) -> None:
        self.set_overlay(title, renderable)
        self.wait_key("Press any key to return to the table")
        self.clear_overlay()

    def print_after(self, renderable) -> None:
        self.console.print(renderable)

    def add_splash_name(self, name: str) -> None:
        self._splash_names.append(name)
        self.refresh()

    # input
    def _read(self) -> str:
        key = self._keys.read()
        self._flash = None
        return key

    def choose(self, prompt: str, options: Sequence[Option], default: Optional[str] = None) -> str:
        self._prompt, self._options, self._default, self._buffer = prompt, options, default, None
        self.refresh()
        try:
            while True:
                key = self._read()
                if key == "enter" and default is not None:
                    return default
                if len(key) == 1:
                    key = key.lower()
                for keys, value, _ in options:
                    if _key_matches(key, keys):
                        return value
                self.refresh()
        finally:
            self._prompt, self._options, self._default = None, (), None
            self.refresh()

    def ask(self, prompt: str, default: str = "", hotkeys: Optional[dict] = None) -> str:
        self._prompt, self._options, self._buffer, self._placeholder = prompt, (), "", default
        self.refresh()
        try:
            while True:
                key = self._read()
                if hotkeys and key in hotkeys:
                    return hotkeys[key]
                if key == "enter":
                    return self._buffer.strip() or default
                if key == "backspace":
                    self._buffer = self._buffer[:-1]
                elif key == "esc":
                    self._buffer = ""
                elif len(key) == 1 and key.isprintable():
                    self._buffer += key
                self.refresh()
        finally:
            self._prompt, self._buffer, self._placeholder = None, None, ""
            self.refresh()

    def wait_key(self, prompt: str = "Press any key to continue") -> None:
        self._prompt, self._options = prompt, ()
        self.refresh()
        try:
            self._read()
        finally:
            self._prompt = None
            self.refresh()

    # rendering
    def _render(self):
        width, height = self.console.size
        if self._held_log and time.monotonic() >= self._anim_until:
            while self._held_log:
                self._log.append(self._held_log.popleft())
        side_log = width >= 100
        self._inline_log = not side_log and height < 32  # short screen: newest log line goes in the footer
        layout = Layout()
        layout.split_column(
            Layout(self._header(), name="header", size=3),
            Layout(name="body"),
            Layout(self._footer(), name="footer", size=4),
        )
        body = layout["body"]
        log_view = _LogView(list(self._log), "Table Talk")

        if self._overlay is not None:
            body_view = self._overlay if getattr(self._overlay, "fill", False) else Align.center(self._overlay, vertical="middle")
            main = Panel(body_view, title=self._overlay_title,
                         border_style="gold1", box=box.DOUBLE)
        elif self.game is None:
            main = self._splash()
        else:
            main_w = width - (42 if side_log else 0)
            main = self._table(main_w, height - 7 - (0 if side_log or self._inline_log else 7))

        if side_log:
            body.split_row(Layout(main, name="main"), Layout(log_view, name="log", size=42))
        elif self._inline_log:
            body.update(main)
        else:
            body.split_column(Layout(main, name="main"), Layout(log_view, name="log", size=7))
        return layout

    def _header(self):
        grid = Table.grid(expand=True)
        grid.add_column(justify="left")
        grid.add_column(justify="right")
        title = Text.assemble(("♠ ", "bold white"), ("♥ ", "bold red"), ("B L A C K J A C K", "bold gold1"),
                              (" ♦", "bold red"), (" ♣", "bold white"))
        info = Text(style="grey70")
        game = self.game
        if game is not None:
            total = game.num_decks * 52
            info.append(f"Round {game.game_num}", "bold")
            info.append(f"   Shoe {len(game.shoe)}/{total}")
            info.append(f"   {game.num_decks} deck{'s' if game.num_decks != 1 else ''}")
        grid.add_row(title, info)
        return Panel(grid, box=box.HEAVY, border_style="gold1")

    def _footer(self):
        lines = []
        if self._flash:
            lines.append(Text.from_markup(f"[bold red]✖ {self._flash}[/]"))
        elif getattr(self, "_inline_log", False) and self._log:
            latest = Text.from_markup(self._log[-1])
            latest.no_wrap, latest.overflow = True, "ellipsis"
            lines.append(latest)
        if self._prompt is not None:
            line = Text.from_markup(self._prompt)
            if self._buffer is not None:
                line.append("  ")
                if self._buffer:
                    line.append(self._buffer, "bold white")
                elif self._placeholder:
                    line.append(self._placeholder, "grey50")
                line.append("▌", "blink bold gold1")
            if self._options:
                line.append("   ")
                for keys, value, label in self._options:
                    line.append(f" {key_label(keys)} ", "bold black on gold1")
                    if not label:
                        line.append(" ")
                        continue
                    line.append(f" {label}", "bold")
                    line.append(" (↵)  " if value == self._default else "  ", "grey62")
            lines.append(line)
        elif self.active_player is not None and self.game is not None:
            who = "Dealer" if self.active_player is self.game.dealer else esc(self.active_player.name)
            lines.append(Text.from_markup(f"[grey62]{who} is playing…[/]"))
        if not lines:
            lines.append(Text(""))
        return Panel(Group(*lines[-2:]), box=box.ROUNDED, border_style="grey42",
                     subtitle="[grey50]Ctrl-C quits[/]", subtitle_align="right")

    def _splash(self):
        from types import SimpleNamespace
        logo_cards = [SimpleNamespace(rank="Ace", suit="Spades"), SimpleNamespace(rank="King", suit="Hearts")]
        logo = render_cards(logo_cards, width=40)
        title = Text("B L A C K J A C K", "bold gold1")
        subtitle = Text("v5.0  ·  humans vs. bots vs. the house", "grey62")
        names = Table.grid(padding=(0, 2))
        names.add_column(style="gold1")
        names.add_column(style="bold")
        for i, name in enumerate(self._splash_names, 1):
            names.add_row(f"Player {i}", name)
        parts = [Align.center(logo), Text(""), Align.center(title), Align.center(subtitle), Text("")]
        if self._splash_names:
            parts.append(Align.center(names))
        if self.splash_info:
            parts += [Text(""), Align.center(Text.from_markup(self.splash_info, style="grey62", justify="center"))]
        return Panel(Align.center(Group(*parts), vertical="middle"), box=box.ROUNDED, border_style="green4",
                     style="on #0b3d2e")

    def _table(self, width: int, height: int):
        game = self.game
        reveal = getattr(game, "phase", "") in ("dealer", "results", "over")
        seats = game.players + game.bots
        cols = max(1, min(len(seats), (width - 4) // 24)) if seats else 1
        rows = -(-len(seats) // cols) if seats else 0
        max_hands = max((len(p.hands) for p in seats), default=1)
        need = 9 + rows * (4 + 7 * max_hands)
        compact = height < need
        # side-view chip stacks need up to STACK_MAX + 1 rows per hand instead of one
        stacks = height >= 9 + rows * (4 + (6 + STACK_MAX + 1) * max_hands)
        col_w = (width - 4) // cols
        inner_w = col_w - 4

        dealer = self._dealer_panel(width - 6, reveal, compact)

        grid = Table.grid(expand=True, padding=(0, 0))
        for _ in range(cols):
            grid.add_column(ratio=1)
        for r in range(rows):
            row = [self._seat_panel(p, inner_w, compact, stacks) for p in seats[r * cols:(r + 1) * cols]]
            row += [""] * (cols - len(row))
            grid.add_row(*row)

        return Panel(Group(dealer, grid), box=box.ROUNDED, border_style="green4", style="on #0b3d2e",
                     padding=(0, 1))

    def _dealer_panel(self, width: int, reveal: bool, compact: bool):
        dealer = self.game.dealer
        hand = dealer.hands[0]
        active = self.active_player is dealer
        if not hand.cards:
            body = Text("Waiting for bets…", "grey62 italic")
            status = Text("")
        else:
            hide = not reveal and len(hand.cards) >= 2
            age = self._card_age(hand.cards)
            body = render_cards(hand.cards, width, hide_first=hide, compact=compact, last_age=age)
            status = Text()
            if age is not None:   # don't give the total away before the card flips
                status.append(hand_label(SimpleNamespace(cards=hand.cards[:-1])), "bold")
                status.append(" + ?", "grey62")
            elif hide:
                status.append(f"Showing {max(hand.cards[1].values)}", "bold")
            else:
                status.append(hand_label(hand), "bold")
                if hand.is_blackjack():
                    status.append("  ").append_text(badge("blackjack"))
                elif hand.is_busted():
                    status.append("  ").append_text(badge("busted"))
        return Panel(Group(Align.center(body), Align.center(status)), title="[bold]Dealer[/]",
                     subtitle="[grey62]stands on 17[/]", border_style="gold1" if active else "grey50",
                     box=box.ROUNDED, style="on #0b3d2e")

    @staticmethod
    def _chips(bet, result, width: int, stacks: bool):
        if stacks:
            drawn = chip_stacks(bet, result, width)
            if max(line.cell_len for line in drawn.split()) <= width:
                return drawn
        return chip_line(bet, result, width)

    def _seat_panel(self, player, inner_w: int, compact: bool, stacks: bool = False):
        game = self.game
        phase = getattr(game, "phase", "")
        is_active = player is self.active_player
        strategy = "ai-nn" if hasattr(player, "q_network") else getattr(player, "strategy", None)
        sitting_out = player.current_bet == 0 and phase not in ("betting", "")

        title = Text(player.name, "bold")
        if phase == "results" and player.current_bet:
            delta = player.balance - getattr(player, "round_start_balance", player.balance)
            title.append("  ")
            if delta == 0:
                title.append("even", "grey70")
            else:
                title.append(signed_money(delta), "bold green" if delta > 0 else "bold red")
        subtitle = Text(strategy, "grey62") if strategy else None
        info = Text()
        info.append(money(player.balance), "bold green")
        if len(player.hand_bets) > 1:
            info.append("  total bet ", "grey62").append(money(sum(player.hand_bets)), "bold")
        if getattr(player, "insurance_bet", 0):
            info.append("  ins ", "grey62").append_text(chip_stack(player.insurance_bet))

        parts = [info]
        if hasattr(player, "count_summary"):
            summary = Text.from_markup(player.count_summary(), style="cyan")
            summary.no_wrap, summary.overflow = True, "ellipsis"
            parts.append(summary)
        if sitting_out:
            parts.append(Text("sitting out", "grey50 italic"))
        elif not player.hands or not player.hands[0].cards:
            if player.current_bet:
                parts.append(self._chips(player.current_bet, None, inner_w, stacks))
            else:
                parts.append(Text("placing bet…" if is_active else "", "grey62 italic"))
        else:
            multi = len(player.hands) > 1
            for i, hand in enumerate(player.hands):
                hand_active = is_active and (self.active_hand is None or self.active_hand == i)
                age = self._card_age(hand.cards)
                parts.append(render_cards(hand.cards, inner_w - 2, compact=compact, last_age=age))
                status = Text()
                status.append("▶ " if hand_active and phase == "playing" else "  ", "bold gold1")
                if multi:
                    status.append(f"H{i + 1} ", "grey62")
                if age is not None:   # don't give the total away before the card flips
                    status.append(hand_label(SimpleNamespace(cards=hand.cards[:-1])), "bold")
                    status.append(" + ?", "grey62")
                    parts.append(status)
                    bet = player.hand_bets[i] if i < len(player.hand_bets) else player.current_bet
                    parts.append(Padding(self._chips(bet, None, inner_w - 2, stacks), (0, 0, 0, 2)))
                    continue
                status.append(hand_label(hand), "bold")
                bet = player.hand_bets[i] if i < len(player.hand_bets) else player.current_bet
                if bet > player.current_bet:
                    status.append("  doubled", "gold1")
                result = player.results[i] if phase == "results" and i < len(player.results) else None
                if result:
                    status.append("  ").append_text(badge(result))
                elif hand.is_blackjack() and len(player.hands) == 1:
                    status.append("  ").append_text(badge("blackjack"))
                elif hand.is_busted():
                    status.append("  ").append_text(badge("busted"))
                parts.append(status)
                parts.append(Padding(self._chips(bet, result, inner_w - 2, stacks), (0, 0, 0, 2)))

        border = "bold gold1" if is_active else ("grey35" if sitting_out else ("cyan" if strategy else "green3"))
        return Panel(Group(*parts), title=title, title_align="left", subtitle=subtitle, subtitle_align="right",
                     border_style=border, box=box.ROUNDED, style="on #0b3d2e", padding=(0, 1))


#== Settings editor ==#

def _format_setting(kind: str, value) -> Text:
    if kind == "bool":
        return Text("● on", "bold green3") if value else Text("○ off", "grey50")
    if kind == "money":
        return Text(money(value), "bold")
    if kind == "seconds":
        return Text(f"{value:g}s", "bold")
    if kind == "split_limit" and value == -1:
        return Text("unlimited", "bold")
    if value is None:
        return Text("—", "grey50")
    if kind == "int" and isinstance(value, int) and value >= 10000:
        return Text(f"{value:,}", "bold")
    return Text(str(value), "bold")


def _parse_setting(kind: str, raw: str, minimum):
    '''Turn typed text into a setting value, or raise ValueError with a readable message.'''
    try:
        value = float(raw)
    except ValueError:
        raise ValueError("Enter a number.")
    if kind in ("int", "split_limit"):
        if not value.is_integer():
            raise ValueError("Enter a whole number.")
        value = int(value)
    elif value.is_integer():
        value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f"Must be at least {minimum}.")
    if kind == "split_limit" and value == 0:
        raise ValueError("Use -1 for unlimited, or 1 or more.")
    return value


def _get_path(data: dict, key: str):
    for part in key.split("."):
        data = data.get(part, {}) if isinstance(data, dict) else {}
    return None if data == {} else data


def _set_path(data: dict, key: str, value) -> None:
    *parents, last = key.split(".")
    for part in parents:
        data = data.setdefault(part, {})
    data[last] = value


def edit_settings(ui, settings: dict, fields: Sequence[dict], strategies: Sequence[str]) -> bool:
    '''
    Full-screen editor for game settings. Edits a copy; returns True (and updates `settings`
    in place) only when the user saves.

    fields: [{"key", "label", "kind": int|float|money|seconds|bool|split_limit, "desc", "min"}]
    '''
    import copy
    if not ui.interactive:
        ui.error("The settings editor needs an interactive terminal — edit game_settings.json instead.")
        return False

    work = copy.deepcopy(settings)
    work.setdefault("bots", [])
    cursor = 0

    def rows():
        items = [("field", f) for f in fields]
        items += [("bot", i) for i in range(len(work["bots"]))]
        items.append(("add", None))
        return items

    class _View:
        '''Renders the rows that fit the overlay height, scrolled to keep the cursor visible.'''

        fill = True  # takes the overlay's full height instead of being centred

        def __init__(self, items):
            self.items = items

        def __rich_console__(self, console, options):
            items = self.items
            lines = []          # (item index or None, renderable row spec)
            section = None
            for idx, (kind, _) in enumerate(items):
                new_section = "GAME" if kind == "field" else "BOTS"
                if new_section != section:
                    if section is not None:
                        lines.append((None, "blank"))
                    lines.append((None, new_section))
                    section = new_section
                lines.append((idx, None))
            room = max(3, (options.height or console.height) - 2)
            cur_line = next(i for i, (idx, _) in enumerate(lines) if idx == cursor)
            start = max(0, min(cur_line - room // 2, len(lines) - room))
            window = lines[start:start + room]

            table = Table(box=None, show_header=False, expand=True, pad_edge=False, padding=(0, 1))
            table.add_column(width=2, no_wrap=True)
            table.add_column(style="bold", no_wrap=True, min_width=22)
            table.add_column(no_wrap=True, min_width=13)
            table.add_column(style="grey58", ratio=1, no_wrap=True, overflow="ellipsis")
            if start > 0:
                table.add_row("", Text("▲ more", "grey50"), "", "")
            for idx, header in window:
                if idx is None:
                    table.add_row("", Text("" if header == "blank" else header, "bold gold1"), "", "")
                    continue
                kind, item = items[idx]
                selected = idx == cursor
                pointer = Text("▶", "bold gold1") if selected else Text("")
                style = "on grey23" if selected else ""
                if kind == "field":
                    table.add_row(pointer, item["label"], _format_setting(item["kind"], _get_path(work, item["key"])),
                                  item.get("desc", ""), style=style)
                elif kind == "bot":
                    bot = work["bots"][item]
                    hint = "↵ change strategy · R rename · X remove" if selected else ""
                    table.add_row(pointer, esc(bot.get("name", "?")), Text(bot.get("strategy", "?"), "bold cyan"),
                                  hint, style=style)
                else:
                    table.add_row(pointer, Text("+ Add bot", "green3"), "", "↵ or A to add" if selected else "",
                                  style=style)
            if start + room < len(lines):
                table.add_row("", Text("▼ more", "grey50"), "", "")
            yield table

    def render(items):
        return _View(items)

    options = [
        (("up", "k"), "up", ""),
        (("down", "j"), "down", "Move"),
        (("enter", " "), "edit", "Edit"),
        ("a", "add", "Add bot"),
        ("x", "remove", "Remove"),
        ("r", "rename", "Rename"),
        ("s", "save", "Save"),
        (("esc", "q"), "cancel", "Cancel"),
    ]
    try:
        while True:
            items = rows()
            cursor = max(0, min(cursor, len(items) - 1))
            ui.set_overlay("Settings  ·  game_settings.json", render(items))
            action = ui.choose("[bold]Settings[/]", options)
            kind, item = items[cursor]

            if action == "up":
                cursor = (cursor - 1) % len(items)
            elif action == "down":
                cursor = (cursor + 1) % len(items)
            elif action == "save":
                settings.clear()
                settings.update(work)
                return True
            elif action == "cancel":
                if work != settings:
                    confirm = ui.choose("Discard your changes?", [("y", "y", "Discard"), ("n", "n", "Keep editing")])
                    if confirm != "y":
                        continue
                return False
            elif action == "add" or (action == "edit" and kind == "add"):
                name = ui.ask("New bot name", default=f"Bot {len(work['bots']) + 1}")
                if any(b.get("name", "").lower() == name.lower() for b in work["bots"]):
                    ui.error(f"There's already a bot called '{name}'.")
                    continue
                work["bots"].append({"name": name, "strategy": strategies[0]})
                cursor = len(fields) + len(work["bots"]) - 1
            elif kind == "bot" and action == "edit":
                bot = work["bots"][item]
                current = bot.get("strategy")
                nxt = strategies[(strategies.index(current) + 1) % len(strategies)] if current in strategies else strategies[0]
                bot["strategy"] = nxt
            elif kind == "bot" and action == "remove":
                work["bots"].pop(item)
            elif kind == "bot" and action == "rename":
                bot = work["bots"][item]
                name = ui.ask("Rename bot", default=bot.get("name", ""))
                if any(b is not bot and b.get("name", "").lower() == name.lower() for b in work["bots"]):
                    ui.error(f"There's already a bot called '{name}'.")
                    continue
                if bot.get("strategy") == "ai-nn" and name != bot.get("name"):
                    ui.log(f"[yellow]Heads up: '{esc(name)}' will use a new model file — the old one isn't renamed.[/]")
                bot["name"] = name
            elif kind == "field" and action == "edit":
                key = item["key"]
                if item["kind"] == "bool":
                    _set_path(work, key, not _get_path(work, key))
                    continue
                current = _get_path(work, key)
                if current is None:
                    current = item.get("default", "")
                while True:
                    raw = ui.ask(f"[bold]{item['label']}[/]", default=f"{current:g}" if isinstance(current, (int, float)) else str(current))
                    try:
                        _set_path(work, key, _parse_setting(item["kind"], raw, item.get("min")))
                        break
                    except ValueError as e:
                        ui.error(str(e))
            elif action in ("remove", "rename"):
                ui.error("Select a bot first.")
    finally:
        ui.clear_overlay()
