import numpy as np
import matplotlib.pyplot as plt
import main_urnings as mu
import pandas as pd

#read the data
df = pd.read_csv("df0507.csv")

player_ids = df.iloc[:,2].to_numpy()
item_ids = df.iloc[:,3].to_numpy()
correct_answers = df.iloc[:,5].to_numpy()

data = mu.AlsData(player_ids, item_ids, correct_answers)

players, items, ppunch, ipunch  = data.create_players_items(player_starting_score=10,item_starting_score=50,player_urn_size=20,item_urn_size=100, so_urn_size=10)

adaptive_rule = mu.Game_Type(adaptivity="n_adaptive", alg_type="Urnings2", paired_update=True, adaptive_urn=False, adaptive_urn_type="permutation", window=10, min_urn=8, max_urn=64, permutation_test=True, perm_p_val=0.2)
adaptive_sim = mu.Urnings(players = players, items = items, game_type=adaptive_rule, data=data)
adaptive_sim.play(n_games=None, test = False)