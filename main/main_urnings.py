import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.stats as sp

class Player:
#class default constructor
    def __init__(self, user_id, score, urn_size, true_score = None): 
        if score > urn_size:
            raise ValueError("The score can't be higher then the urn size.")

        self.user_id = user_id
        self.score = score
        self.urn_size = urn_size
        self.est = self.score/self.urn_size
        self.true_score = true_score
        self.true_value = self.true_score/self.urn_size
        self.sim_y = 8
        self.sim_true_y = 8

    def draw(self, true_score_logic = False):

        if true_score_logic == False:
            drawing = sp.bernoulli(self.est)
            sim_y = drawing.rvs(size = 1)
            self.sim_y = sim_y
            return  sim_y
        else:
            drawing = sp.bernoulli(self.true_value)
            sim_y = drawing.rvs(size = 1)
            self.sim_true_y = sim_y
            return sim_y


class Urnings:
    def __init__(self, game_type, players, items):
        self.game_type = game_type
        self.standings = []
        self.players = players
        self.items = items
        

    def matchmaking(self):

        if self.game_type == "n_adaptive":
            player_index = np.random.randint(0, len(self.players))
            item_index = np.random.randint(0, len(self.items))
            
            return self.players[player_index], self.items[item_index]
        elif self.game_type == "adaptive":
            None
    
    def urnings_game(self, player, item, result = None):
        if type(player) != Player:
            raise TypeError("Player needs to be Player type")

        if type(item) != Player:
            raise TypeError("Item needs to be Player type")
        
        #simulation or real data analysis
        if result is None and player.true_score is not None and item.true_score is not None:

            while player.sim_true_y == item.sim_true_y:
                player.draw(true_score_logic = True)
                item.draw(true_score_logic = True)
            
            result = player.sim_true_y
            player.sim_true_y = item.sim_true_y = 8

        elif result not in [0,1]:
            raise ValueError("Result is a binary variable: either a win of the player (1), or a loss of the player (0)")
        
        #calculating expected score
        while player.sim_y == item.sim_y:
            player.draw()
            item.draw()
        
        expected_results = player.sim_y
        player.sim_y = item.sim_y = 8

        if self.game_type == "n_adaptive":

            #updating scores
            player_proposal = player.score  + result - expected_results
            item_proposal = item.score  + (1 - result) - (1 - expected_results)

            if player_proposal > player.urn_size:
                player_proposal = player.urn_size
            
            if player_proposal < 0:
                player_proposal = 0
            
            if item_proposal > item.urn_size:
                item_proposal= item.urn_size
            
            if item_proposal < 0:
                item_proposal = 0

            #metropolis step
            old_score = player.score * (player.urn_size - item.score) + (item.urn_size - player.score) * item.score
            new_score = player_proposal * (player.urn_size - item_proposal) + (item.urn_size - player_proposal) * item_proposal
            acceptance = min(1, old_score/new_score)
            u = np.random.uniform()

            if u < min(1, old_score/new_score):
                #accept
                player.score = player_proposal
                item.score = item_proposal
                player.est = player.score / player.urn_size
                item.est = item.score / item.urn_size

        elif self.game_type == "adaptive":
            
            #updating scores
            player_proposal = player.score  + result - expected_results
            item_proposal = item.score  + (1 - result) - (1 - expected_results)

            if player_proposal > player.urn_size:
                player_proposal = player.urn_size
            
            if player_proposal < 0:
                player_proposal = 0
            
            if item_proposal > item.urn_size:
                item_proposal= item.urn_size
            
            if item_proposal < 0:
                item_proposal = 0

            #metropolis step
            old_score = player.score * (player.urn_size - item.score) + (item.urn_size - player.score) * item.score
            new_score = player_proposal * (player.urn_size - item_proposal) + (item.urn_size - player_proposal) * item_proposal
            acceptance = min(1, old_score/new_score)
            u = np.random.uniform()

            if u < min(1, old_score/new_score):
                #accept
                player.score = player_proposal
                item.score = item_proposal
                player.est = player.score / player.urn_size
                item.est = item.score / item.urn_size

        print("Match between ", player.user_id, " and", item.user_id)

    def play(self, n_games):
        for ng in range(n_games):
            current_player, current_item = self.matchmaking()
            self.urnings_game(current_player, current_item)


        


