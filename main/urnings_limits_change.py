import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the urnings algortihm
import main_urnings as mu

#parameters
player_urn_sizes = [6, 10, 14, 18, 50]
change = [0.0001, 0.0005, 0.001, 0.002]

#fixed parameters
n_player = 500
n_items = 1000
item_urn_sizes = 100000
n_sim = 100
true_value = 0.5

#container for the results
urnings_array = np.zeros((n_player, n_sim + 1, len(change) * len(player_urn_sizes)))

counter = 0

for cg in range(len(change)):
    for pus in range(len(player_urn_sizes)):

        print("Run no ", counter, "Simulation with change: ", change[cg], "per item and urn size: ", player_urn_sizes[pus])

        np.random.seed(13181912) #my student number
        #game settings
        starting_score = int(player_urn_sizes[pus]/2)
        player_urn_size = player_urn_sizes[pus]
        

        #true scores for players and items
        item_true_values = np.random.uniform(0.01, 0.99, n_items)

        #creating players and items
        players = []
        for i in range(n_player):
            player = mu.Player(user_id = "Student", score = starting_score, urn_size = player_urn_size, true_value = true_value, so_score=10)
            players.append(player)

        items = []
        for i in range(n_items):
            iname = "item" + str(i)

            item = mu.Player(user_id = iname, score = np.round(item_true_values[i], 1) * item_urn_sizes, urn_size = item_urn_sizes, true_value = item_true_values[i])
            items.append(item)


        game_rule = mu.Game_Type(adaptivity="n_adaptive", alg_type="Urnings1")
        game_sim = mu.Urnings(players = players, items = items, game_type = game_rule)

        for i in range(n_sim):
            game_sim.play(n_games=1, test = True)
            for pl in players:
                pl.true_value += change[cg]


        for pl in range(n_player):
            row = players[pl].container
            urnings_array[pl,:, counter] = row
        
        counter += 1


np.save("urnings_array_nochange", urnings_array)