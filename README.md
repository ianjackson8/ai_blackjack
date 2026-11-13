# AI Blackjack
## Play Blackjack in terminal
Run the program
```
python3 play_blackjack.py
```

### Controls
The game runs as a full-screen terminal UI with card art, a pile of casino chips for every bet (the bigger the bet, the bigger the pile; colours run red → green → black → purple → orange as bets grow) that animates as bets go in, winnings get paid and losing bets get swept away, a live table, and a side panel (on terminals 100+ columns wide) with:

- **Scoreboard** — seats ranked by net result, balance sparkline, win %, current streak, biggest win of the session
- **Bot Brains** — why each bot did what it did: the NN bot's Q-values for its bet and each hit/stand/double (and when it explored at random), Count Bot's live count, LSTM vs. real true count, tens+aces left and bet ramp, and the book/basic-strategy play for rule bots
- **Table Talk** — the log of every action, under the brains when there's room; `Tab` switches to a full-height log and back

| Key | Action |
| --- | --- |
| `H` / `1` | Hit |
| `S` / `2` | Stand |
| `D` / `3` | Double (only shown when allowed) |
| `P` / `4` | Split (only shown when allowed) |
| `Enter` | Accept the default: repeat your last bet, deal the next round |
| `Esc` | Clear the text you're typing |
| `Tab` | Switch the side panel between scoreboard/brains and the full log (works while bots play) |
| `Ctrl-C` | Quit immediately |

At the bet prompt, type an amount (`0` to sit out a round) or a `/command`.

## Training Mode
Train the AI bot without human player interaction
```
python3 play_blackjack.py --train
```
Training plays `train_param.rounds` rounds (default 60,000 — about 2 minutes with `show_output` off; editable in the settings menu) with no deal delay. A bot that runs out of money rebuys for `init_balance` so it keeps learning, and `Ctrl-C` stops early and still saves the model. Set `show_output` to false for the fastest run.

### Card-counting bot
The `counter` bot plays basic strategy (including splits) and lets an LSTM count cards for it.
The LSTM watches every card that is shown, predicts what is left in the shoe, and the bot raises its bet when the remaining cards favour the player.
Train the LSTM once (about 2 minutes) before using it:
```
python3 play_blackjack.py --train-counter        # optional: --train-counter 12000 for more steps
```
Its seat shows the LSTM's true count and estimated edge; the log shows its count next to the real one whenever it raises.

### How the AI bot works
The `ai-nn` bot has two small neural networks:
- **Play network** sees its hand total, whether the hand is soft, whether it can double, and the dealer's upcard, and picks hit / stand / double.
- **Bet network** sees the Hi-Lo true count and how deep into the shoe we are, and picks a bet of 1×, 2×, 4×, 6× or 8× `default_bet`.

## Bot Survival Trial
Pit BTB Bot, NN Bot and Count Bot against each other: $500 each, no rebuys, out at $0, last one standing wins.
Prints a stats table and saves a balance graph to `results/`.
```
python3 trial.py                  # one trial (seed 1)
python3 trial.py --seed 7         # a different trial
python3 trial.py --repeat 200     # plus 200 more trials to see who wins most often
```

## Game Settings
Settings can be changed in-game: on the player name screen press `Tab` (or type `/settings`) to open the settings editor.
Use `↑`/`↓` to move, `Enter` to edit or toggle, `A`/`R`/`X` to add, rename or remove bots, `S` to save to `game_settings.json`, and `Esc` to cancel.

They can also be edited directly in `game_settings.json`:
- `num_decks` : \<int\>  number of decks to use in the shoe
- `init_balance` : \<float\> initial balance of all players
- `default_bet` : \<float\> default bet of bots
- `show_output` : \<bool\> show the table UI during `--train` (false = a one-line progress counter, much faster)
- `log_game` : \<bool\> save game information to a log file (true/false)
- `show_graph` : \<bool\> show balance throughout play after quitting game (true/false)
- `deal_delay` : \<float\> time to deal cards (set to 0 for no delay)
- `allow_double_after_split`: \<bool\> allows the player to double on a split (true/false)
- `split_limit`: \<int\> The amount of splits allowed for a player (set to -1 for no limit)
- `allow_insurance`: \<bool\> offer insurance side-bet to human players when dealer shows Ace (true/false)
- `train_param.rounds` : \<int\> rounds played by `--train`
- `counter.max_spread` : \<int\> most a counter bot will bet, in units of `default_bet`
- `bots` : \<List[dict]> information of bots to play alongside player
  - `name` : \<str\> name of the bot
  - `strategy` : \<str\> playing strategy (options below)
    - "default" - hits on anything below 17, stands otherwise
    - "by the books" - plays based on best odds table
    - "ai-nn" - trainable reinforcement learning bot using Q-learning (also sizes its own bets from the card count)
    - "counter" - plays perfect basic strategy and sizes bets with an LSTM card counter (see below)
    - "counter-hilo" - same, but counts with the classic Hi-Lo system (handy for comparison)

> **Note:** `show_graph` requires `log_game: true`. If `log_game` is false, the graph will be skipped with a warning at startup.

## Save & Load
Save a game to resume later:
```
/save [optional name]
```
At the end of a game you will also be prompted to save. If no name is given, the save defaults to `save_YYYY-MM-DD_HHMMSS`. All saves are stored in the `saves/` folder.

Load a saved game:
```
python3 play_blackjack.py --load <save name>
```
Player balances, bot balances, AI training state (epsilon), and balance history are all restored from the save.

## Commands
- `/help` - display available commands
- `/exit` - quit the game
- `/stats` - show win/loss/push/bust counts, net P&L, and win rate for all players
- `/graph` - display player balance graph
- `/editbalance [player] [new balance]` - set a players balance
- `/showbalance` - shows the current balance of all players
- `/shuffle` - shuffle the deck
- `/save [name]` - save the current game state

## AI Bot
The `ai-nn` bot uses deep Q-learning:
- **Play network**: `[hand total, soft, can double, dealer upcard]` → 64 → 32 → Q-values for hit / stand / double (doubling only allowed on two cards)
- **Bet network**: `[Hi-Lo true count, shoe penetration]` → 32 → 16 → Q-values for betting 1×, 2×, 4×, 6× or 8× `default_bet`
- **Rewards**: the hand's result in units of its stake (blackjack +1.5, win +1, push 0, lose/bust −1, doubled hands count twice), credited back through every decision in the hand
- **Training**: ε-greedy exploration, experience replay, a target network, Smooth L1 loss
- **Model persistence**: saved to `models/{bot_name}_model.pth`; models from older versions are rejected and retrained from scratch

# Changelog
- v5.1
  - dealer peeks for blackjack with an Ace or ten showing: a dealer natural ends the round before anyone acts, so you only lose your original bet (#30)
  - fix crash loading a save after bots were added in settings (#29)
  - fix bets placed this round being lost when saving or `/exit`-ing during betting (#31)
  - player names can no longer match a bot's name (#32)
- v5.0
  - full-screen Rich TUI: card art, seat panels, key-driven prompts (`H`/`S`/`D`/`P`), session summary on exit
  - hit animation: cards slide in face down and flip; totals and log lines wait for the flip (`animate_cards` setting)
  - chip piles that grow with the bet, with animations for betting, doubling, payouts and losing bets
  - side panel with a session scoreboard (sparklines, streaks, biggest win) and "Bot Brains" showing each bot's reasoning; `Tab` toggles the full log
  - in-game settings editor (`Tab` / `/settings` on the name screen)
  - rework the `ai-nn` bot: correct doubling rewards, credit across multi-step hands, soft-hand and can-double features, and a bet-sizing network driven by the true count; retrained from scratch
  - add `counter` bot: basic strategy plus an LSTM that learns to count cards from simulated shoes (`--train-counter`) and ramps bets with the estimated edge
  - add `trial.py`: bot survival trial with balance graph and stats table
  - redraw the screen only when something changes (was a full repaint 8×/s); faster startup
  - fix the shoe running out mid-round, bots looping on an unaffordable double, and `insurance_bet` not being initialised
  - fix payouts: 21 after a split pays 1:1 (only a two-card 21 on an unsplit hand is a blackjack), blackjack vs dealer blackjack pushes, and a dealer blackjack beats a multi-card 21
- v4.0
  - add insurance side-bet — offered to human players when dealer shows Ace; pays 2:1; toggleable via `allow_insurance` in settings
  - add `/stats` command — shows W/L/Push/BJ counts, net P&L, win rate, and best/worst round for all players; also printed automatically at end of session
  - refactor terminal UI to match slot machine style: `C` color class, box-draw headers, 2-space indented output, `[N]` menu prompts
  - fix crash when closing `/graph` window mid-game (non-blocking `plt.show`)
  - fix `show_graph: true` silently failing when `log_game: false` — now warns at startup
  - fix `IndexError` in train mode when no bots are configured
  - fix GOD_MODE dealing hand twice and not drawing cards from shoe
  - fix split hands not being logged (all hands now recorded per round)
  - fix double-down on split hand corrupting payout for other hands (per-hand bet tracking)
  - update log format: `hands` (list) and `results` (list) replace single `final_hand`/`result` fields
- v3.0
  - fix split feature
  - add game settings related to splitting mechanics
  - change settings to use JSON booleans instead of strings
  - add saving AI models under their name, under `models/` directory
  - add commands at the betting input (run `/help` to see available commands)
  - add save/load game feature (`/save`, `--load`)
  - fix split targeting wrong hand on second/third split
  - fix post-split dealing sending extra cards to already-played hands
- v2.2
  - fix bot being able to double after a hit
  - fix game ending on first bot going broke
- v2.1 
  - add deal delay to simulate drawing cards
  - add hotkeys for player action
  - fix play again to only take only yes/no
- v2.0
  - add trainable AI bot using Q-learning and PyTorch neural network
  - add support for multiple bot types from settings (regular bots + AI bots)
  - fix bot output display showing actions during gameplay
  - fix AI bot reward system to receive actual game outcomes
  - add training mode flag (--train) for automated AI training
- v1.3
  - fix graph not starting at initial balance on graph
  - fix bug where when bot doubles it crashes
  - remove the old blackjack bot file
  - integrate parse and graph logfile into main file, delete other file
- v1.2
  - add the by the books strategy
  - fix the double feature
- v1.1
  - implement bots into the game, one strategy
- v1.0
  - implement core blackjack game
  - add ending graph
