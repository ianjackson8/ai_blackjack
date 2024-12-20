# AI Blackjack
## Play Blackjack in terminal
Run the program
```
python3 play_blackjack.py
```

## Game Settings
- `num_decks` : \<int\>  number of decks to use in the shoe
- `init_balance` : \<float\> initial balance of all players
- `default_bet` : \<float\> default bet of bots (not used)
- `show_output` : \<str\> print to the terminal ("True"/"False")
- `log_game` : \<str\> save game information to a log file ("True"/"False")
- `show_graph` : \<str\> show balance throughout play after quitting game ("True"/"False")
- `bots` " \<List[dict]> information of bots to play alongside player
  - `name` : \<str\> name of the bot
  - `strategy` : \<str\> playing strategy (options below)
    - "default" hits on anything below 17, stands otherwise
    - "by the books" plays based on best odds table

# Changelog
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
