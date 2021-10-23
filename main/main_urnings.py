import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.stats as sp

class Player:
#class default constructor
    def __init__(self, user_id, score, urn_size): 
        if score > urn_size:
            raise ValueError("The score can't be higher then the urn size.")

        self.user_id = user_id
        self.score = score
        self.urn_size = urn_size
        self.est = score/urn_size
        self.sim_y = []

    def draw(self):
        drawing = sp.bernoulli(self.est)
        sim_y = drawing.rvs(size = 1)
        self.sim_y = sim_y
        return  sim_y



def urnings_game(player, item, result):
    if type(player) != Player:
        raise TypeError("Player needs to be Player type")

    if type(item) != Player:
        raise TypeError("Item needs to be Player type")
    
    if result not in [0,1]:
        raise ValueError("Result is a binary variable: either a win of the player (1), or a loss of the player (0)")
    
    #calculating expected score
    while player.sim_y == item.sim_y:
        player.draw()
        item.draw()
    
    expected_results = player.sim_y

    #updating scores
    player.score = player.score + result - expected_results
    item.score = item.score + (1 - result) - (1 - expected_results)

    return print("Match between ", player.user_id, " and", item.user_id)


#player1 = Player(user_id="player1", score= 50, urn_size = 60)
#player2 = Player(user_id="player2", score= 55, urn_size=60)

#urnings_game(player1, player2, result=1)
#print(player1.score, player2.score)

