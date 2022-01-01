import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.stats as sp

import main_urnings as mu

Player1 = mu.Player("player1", 0.2, 60, true_score=0.8)
Item1 = mu.Player("item1", 0.3, 60, true_score=0.8)


while Player1.sim_true_y == Item1.sim_true_y: 
    Player1.draw(true_score_logic = True)
    Item1.draw(true_score_logic = True)
result = Player1.sim_true_y

print(result)
print(Item1.sim_true_y)