import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl

#importing the urnings algortihm
import main_urnings as mu

#parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes = [4, 8, 16, 32, 64]

#fixed parameters
n_player = 500
n_items = 1000
item_urn_sizes = 100
n_sim = 300

#container for the results
urnings_array = np.zeros((n_player, n_sim + 1, len(true_values) * len(player_urn_sizes)))

counter = 0

for tv in range(len(true_values)):
    for pus in range(len(player_urn_sizes)):

        print("Run no ", counter, "Simulation with true score: ", true_values[tv], " and urn size: ", player_urn_sizes[pus], 'n_adaptive')

        #np.random.seed(13181918)

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
            player = mu.Player(user_id = pname, score = starting_score, urn_size = player_urn_size, true_value = true_values[tv])
            players.append(player)

        items = []
        for i in range(n_items):
            iname = "item" + str(i)
            item_starting_score[i] = np.random.binomial(item_urn_sizes, item_true_values[i])
            item = mu.Player(user_id = iname, score = item_starting_score[i], urn_size = item_urn_sizes, true_value = item_true_values[i])
            items.append(item)

    
        game_rule = mu.Game_Type(adaptivity="n_adaptive", alg_type="Urnings2", paired_update= True)
        game_sim = mu.Urnings(players = players, items = items, game_type = game_rule)
        game_sim.play(n_games=n_sim, test = True)

        for pl in range(n_player):
            row = players[pl].estimate_container
            urnings_array[pl,:, counter] = row

        counter += 1

np.save("urnings_array_limit", urnings_array)

players_file = "players_limit.pkl"
items_file = "items_limit.pkl"

open_players_file = open(players_file, "wb")
pkl.dump(players, open_players_file)
open_players_file.close()

open_items_file = open(items_file, "wb")
pkl.dump(items, open_players_file)
open_items_file.close()

