import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

#importing the urnings algortihm
import main_urnings as mu

#parameters
player_urn_sizes = [6, 10, 14, 18, 50]
change = [0.0001, 0.0005, 0.001, 0.002]

#fixed parameters
n_player = 1000
n_items = 1000
item_urn_sizes = 1000
n_sim = 100
true_value = 0.5

#container for the results
urnings_array = np.zeros((n_player, n_sim + 1, len(change) * len(player_urn_sizes)))

counter = 0

for cg in range(len(change)):
    for pus in range(len(player_urn_sizes)):

        print("Run no ", counter, "Simulation with change: ", change[cg], "per item and urn size: ", player_urn_sizes[pus], "n_adaptive")

        np.random.seed(13181912) #my student number
        #game settings
        starting_score = int(player_urn_sizes[pus]/2)
        player_urn_size = player_urn_sizes[pus]
        

        #true scores for players and items
        item_true_values = np.random.uniform(0.01, 0.99, n_items) 
        item_starting_score = np.zeros(n_items)

        #creating players and items
        players = []
        for i in range(n_player):
            pname = "player" + str(i)
            player = mu.Player(user_id = pname, score = starting_score, urn_size = player_urn_size, true_value = true_value)
            players.append(player)

        items = []
        for i in range(n_items):
            iname = "item" + str(i)
            item_starting_score[i] = np.random.binomial(item_urn_sizes, item_true_values[i])
            item = mu.Player(user_id = iname, score = item_starting_score[i], urn_size = item_urn_sizes, true_value = item_true_values[i])
            items.append(item)


        game_rule = mu.Game_Type(adaptivity="n_adaptive", alg_type="Urnings2", paired_update=True)
        game_sim = mu.Urnings(players = players, items = items, game_type = game_rule)

        for i in range(n_sim):  
            game_sim.play(n_games=1, test = True)
            for pl in players:
                pl.true_value += change[cg]

    
        for pl in range(n_player):
            row = players[pl].estimate_container
            urnings_array[pl,:, counter] = row
        
        counter += 1


np.save("urnings_array_limit_change", urnings_array)

players_file = "players_limit_change.pkl"
items_file = "items_limit_change.pkl"

open_players_file = open(players_file, "wb")
pkl.dump(players, open_players_file)
open_players_file.close()

open_items_file = open(items_file, "wb")
pkl.dump(items, open_players_file)
open_items_file.close()