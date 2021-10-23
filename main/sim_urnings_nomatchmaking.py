import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.stats as sp

import main_urnings

#create true objects
player_ids = []
item_ids =  []
numcodes = np.arange(1,101)
for plyrs in range(len(numcodes)):
    player_ids.append("player" + str(numcodes[plyrs]))
    if plyrs < 50:
        item_ids.append("item" + str(numcodes[plyrs]))

player_true_scores = sp.norm.rvs(0,1, size = 100)
item_true_scores = sp.norm.rvs(0,1, size = 50)

