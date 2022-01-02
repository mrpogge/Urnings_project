import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.stats as sp
import matplotlib.pyplot as plt

import main_urnings as mu

########################################################################################################################
#setting up the simulation
########################################################################################################################

#game settings
n_player = 10
n_items = 10
starting_score = 50
player_urn_sizes = 100
item_urn_sizes = 100
game_type = "n_adaptive"

#true scores for players and items
player_true_scores = np.random.normal(0, 1, n_player)
item_true_scores = np.random.normal(0, 1, n_items)
player_true_scores = np.exp(player_true_scores) / (1 + np.exp(player_true_scores)) * player_urn_sizes
item_true_scores = np.exp(item_true_scores) / (1 + np.exp(item_true_scores)) * item_urn_sizes



#creating players and items
players = []
items = []
for i in range(n_player):
    pname = "player" + str(i)
    
    player = mu.Player(user_id = pname, score = starting_score, urn_size = player_urn_sizes, true_score = player_true_scores[i])
    players.append(player)

for i in range(n_items):
    iname = "item" + str(i)

    item = mu.Player(user_id = iname, score = starting_score, urn_size = item_urn_sizes, true_score = item_true_scores[i])
    items.append(item)

########################################################################################################################
#setting up the game environment
########################################################################################################################

adaptive_sim = mu.Urnings(game_type= "adaptive", players = players, items = items)

adaptive_sim.play(n_games=10000)

estimated_score = []
for nplayers in range(n_player):

    est = adaptive_sim.players[nplayers].est 
    estimated_score.append(est)

########################################################################################################################
#results
########################################################################################################################

print(estimated_score)
print(player_true_scores / player_urn_sizes)