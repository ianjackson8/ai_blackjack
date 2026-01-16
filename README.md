# AI Blackjack
## Play Blackjack in terminal
Run the program
```
python3 play_blackjack.py
```

## Training Mode
Train the AI bot without human player interaction
```
python3 play_blackjack.py --train
```

## Game Settings
- `num_decks` : \<int\>  number of decks to use in the shoe
- `init_balance` : \<float\> initial balance of all players
- `default_bet` : \<float\> default bet of bots
- `show_output` : \<str\> print to the terminal ("True"/"False")
- `log_game` : \<str\> save game information to a log file ("True"/"False")
- `show_graph` : \<str\> show balance throughout play after quitting game ("True"/"False")
- `deal_delay` : \<float\> time to deal cards (set to 0 for no delay)
- `bots` : \<List[dict]> information of bots to play alongside player
  - `name` : \<str\> name of the bot
  - `strategy` : \<str\> playing strategy (options below)
    - "default" - hits on anything below 17, stands otherwise
    - "by the books" - plays based on best odds table
    - "ai" - trainable reinforcement learning bot using Q-learning

## Commands
- `/help` - display available commands
- `/exit` - quit the game
- `/graph` - display player balance graph
- `/editbalance [player] [new balance]` - set a players balance
- `/showbalance` - shows the current balance of all players

## AI Bot
The AI bot uses reinforcement learning (Q-learning) to learn optimal blackjack strategy:
- **Neural Network**: 3-layer network (128→64→3 neurons) that learns Q-values for actions
- **Actions**: hit, stand, double
- **Epsilon-greedy**: Starts exploring randomly (ε=1.0), gradually exploits learned strategy (ε→0.01)
- **Experience Replay**: Learns from stored game experiences in batches
- **Reward System**: +1.5 for blackjack, +1.0 for win, 0 for push, -1.0 for loss/bust
- **Model Persistence**: Saves/loads from `bot_model.pth` to continue learning across sessions

# Changelog
- v3.0
  - add commands at the betting input (run `/help` to see available commands)
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
