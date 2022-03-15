import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the urnings algortihm
import main_urnings as mu

#parameters
true_values = [0.5, 0.6, 0.7, 0.8, 0.9]
player_urn_sizes = [6, 10, 14, 18, 50]

#fixed parameters
n_player = 500
n_items = 1000
item_urn_sizes = 100000
n_sim = 100

#container for the results
urnings_array = np.zeros((n_player, n_sim + 1, len(true_values) * len(player_urn_sizes)))

counter = 0

for tv in range(len(true_values)):
    for pus in range(len(player_urn_sizes)):

        print("Run no ", counter, "Simulation with true score: ", true_values[tv], " and urn size: ", player_urn_sizes[pus])

        np.random.seed(13181912) #my student number
        #game settings
        starting_score = int(player_urn_sizes[pus]/2)
        player_urn_size = player_urn_sizes[pus]
        

        #true scores for players and items
        item_true_values = np.random.uniform(0.01, 0.99, n_items)

        #creating players and items
        players = []
        for i in range(n_player):
            player = mu.Player(user_id = "Student", score = starting_score, urn_size = player_urn_size, true_value = true_values[tv], so_score=10)
            players.append(player)

        items = []
        for i in range(n_items):
            iname = "item" + str(i)

            item = mu.Player(user_id = iname, score = np.round(item_true_values[i], 1) * item_urn_sizes, urn_size = item_urn_sizes, true_value = item_true_values[i])
            items.append(item)


        game_rule = mu.Game_Type(adaptivity="n_adaptive", alg_type="Urnings1")
        game_sim = mu.Urnings(players = players, items = items, game_type = game_rule)
        game_sim.play(n_games=n_sim, test = True)

        for pl in range(n_player):
            row = players[pl].container
            urnings_array[pl,:, counter] = row
        
        counter += 1


np.save("urnings_array_nochange", urnings_array)