import json
import matplotlib.pyplot as plt #type: ignore

def parse_log_and_plot(log_file):
    try:
        with open(log_file, 'r') as file:
            logs = json.load(file)
    except FileNotFoundError:
        print(f"Error: The file {log_file} does not exist.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse the JSON file {log_file}.")
        return
    
    # Extract balance and round number
    with open('game_settings.json', 'r') as file:
            settings = json.load(file)

    init_bal = settings["init_balance"]

    # Extract balance and round number for all players
    player_balances = {}

    for round_number, entry in enumerate(logs, start=1):
        players = entry.get("players", [])
        for player in players:
            name = player.get("name")
            balance = player.get("balance")
            if name and balance is not None:
                if name not in player_balances:
                    player_balances[name] = []

                if round_number == 0:
                    player_balances[name].append((round_number, init_bal))
                else:
                    player_balances[name].append((round_number, balance))

    # Plotting
    plt.figure(figsize=(10, 6))

    for player, rounds_and_balances in player_balances.items():
        rounds, balances = zip(*rounds_and_balances)
        plt.plot(rounds, balances, marker='o', linestyle='-', label=player)

    plt.title("Player Balances Over Rounds")
    plt.xlabel("Round Number")
    plt.ylabel("Balance")
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    log_file = "logs/session_2024-12-12_003425.json"  # Replace with your log file path
