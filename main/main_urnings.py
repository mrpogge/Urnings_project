import pandas as pd
import numpy as np
import matplotlib as plt
import scipy.stats as sp
import statsmodels.api as sm
from statsmodels.graphics import tsaplots


class Player:
#class default constructor
    def __init__(self, user_id, score, urn_size, true_value): 
        if score > urn_size:
            raise ValueError("The score can't be higher then the urn size.")

        self.user_id = user_id
        self.score = score
        self.urn_size = urn_size
        self.est = self.score/self.urn_size
        self.true_value = true_value
        self.sim_y = 8
        self.sim_true_y = 8

        #creating a container
        self.container = np.array([self.score])
        self.differential_container = np.array([0])

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

    def autocorrelation(self, lag, plots = False):
        
        #calculating autocorrelation for the player's urn chain
        acf_player = sm.tsa.acf(self.container, nlags= lag)

        if plots == True:
            
            fig = tsaplots.plot_acf(self.container, lags = lag)

    
        return acf_player 

    def so_autocorrelation(self, lag, plots = False):
        #calculating autocorrelation for the second order chain
        acf_so = sm.tsa.acf(self.differential_container, nlags = lag)

        if plots == True:
            
            fig = tsaplots.plot_acf(self.differential_container, lags = lag)

    
        return acf_so
    
    def convergence_with_known(self, plots = False):

        true_vec = np.repeat(self.true_value * self.urn_size, repeats=len(self.container))

        diff_vec = self.container - true_vec

        if plots == True:
            #make it prettier
            fig = plt.plot(diff_vec)



        

class Game_Type:
    def __init__(self, adaptivity, alg_type, updating_type = "one_dim"):
        self.adaptivity = adaptivity
        self.alg_type = alg_type
        self.updating_type = updating_type
    
    def draw_rule(self, player, item):
        
        if self.alg_type == "Urnings1":
            #simulating the observed value
            while player.sim_true_y == item.sim_true_y:
                player.draw(true_score_logic = True)
                item.draw(true_score_logic = True)
                
            result = player.sim_true_y
            player.sim_true_y = item.sim_true_y = 8

            #calculating expected score
            while player.sim_y == item.sim_y:
                player.draw()
                item.draw()
            
            expected_results = player.sim_y
            player.sim_y = item.sim_y = 8
        
        elif self.alg_type == "Urnings2":
            #simulating the observed value
            while player.sim_true_y == item.sim_true_y:
                player.draw(true_score_logic = True)
                item.draw(true_score_logic = True)
                
            result = player.sim_true_y
            player.sim_true_y = item.sim_true_y = 8

            #CHECK WHAT HAPPENS EXACTLY IN MARIAS'S PAPER
            #calculating expected value
            player.score += result
            player.est = player.score / player.urn_size
            item.score += 1 - result
            item.est = item.score / item.urn_size

            if player.score > player.urn_size:
                player.score = player.urn_size
                
            if player.score < 0:
                player.score = 0
                
            if item.score > item.urn_size:
                item.score = item.urn_size
                
            if item.score < 0:
                item.score = 0

            while player.sim_y == item.sim_y:
                    player.draw()
                    item.draw()
                
            expected_results = player.sim_y
            player.sim_y = item.sim_y = 8
            
            #returning to the original urn conig
            player.score -= result
            player.est = player.score / player.urn_size
            item.score -= 1 - result
            item.est = item.score / item.urn_size

        return result, expected_results

    
    def updating_rule(self, player, item, result, expected_results):
        
        if self.updating_type == "one_dim":
            #updating scores
            player_prop = player.score  + result - expected_results
            item_prop = item.score  + (1 - result) - (1 - expected_results)

            #Making sure that the urnsize is bigger than the total number of balls obviously
            if player_prop > player.urn_size:
                player_prop = player.urn_size
                
            if player_prop < 0:
                player_prop = 0
                
            if item_prop > item.urn_size:
                item_prop = item.urn_size
                
            if item_prop < 0:
                item_prop = 0
            
        return player_prop, item_prop

    def metropolis_correction(self, player, item, player_proposal, item_proposal):
        
        #algorithm type to provide the first part of the metropolis correction 
        if self.alg_type == "Urnings1":
            old_score = player.score * (player.urn_size - item.score) + (item.urn_size - player.score) * item.score
            new_score = player_proposal * (player.urn_size - item_proposal) + (item.urn_size - player_proposal) * item_proposal

            metropolis_corrector = old_score/new_score
        
        elif self.alg_type == "Urnings2":
            
            metropolis_corrector = 1
        
        return metropolis_corrector

    def adaptivity_correction(self, player, item, player_proposal, item_proposal):
        
        if self.adaptivity == "n_adaptive":
            adaptivity_corrector = 1

        else:
            #change this to be a function or method of some sort
            current_item_prob = np.exp(-2*(np.log((player.score + 1) / (player.urn_size- player.score + 1)) - np.log((item.score + 1) / (item.urn_size - item.score + 1)))**2)
            proposed_item_prob = np.exp(-2*(np.log((player_proposal + 1) / (player.urn_size - player_proposal + 1)) - np.log((item_proposal + 1) / (item.urn_size - item_proposal + 1)))**2)

            adaptivity_corrector = proposed_item_prob/current_item_prob

        return adaptivity_corrector
  
        
class Urnings:
    def __init__(self, players, items, game_type):
        self.standings = []
        self.players = players
        self.items = items
        self.game_type = game_type
    
    #One can define other adaptivity rules, I will add this to gametype later

    def adaptive_rule_normal(self):
        
        adaptive_matrix = np.zeros(shape=(len(self.players), len(self.items)))
        for i in range(len(self.players)):
            for j in range(len(self.items)):

                R_i = self.players[i].score
                R_j = self.items[j].score
                n_i = self.players[i].urn_size
                n_j = self.items[j].urn_size
                prob = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)

                adaptive_matrix[i, j] = prob 

        return adaptive_matrix

    def matchmaking(self):

        if self.game_type.adaptivity == "n_adaptive":
            player_index = np.random.randint(0, len(self.players))
            item_index = np.random.randint(0, len(self.items))
            
            return self.players[player_index], self.items[item_index]

        elif self.game_type.adaptivity == "adaptive":
            adaptive_matrix = self.adaptive_rule_normal()

            player_index = np.random.randint(0, len(self.players))
            item_index = np.random.choice(np.arange(len(self.items)), 1, p = (adaptive_matrix[player_index,:] / np.sum(adaptive_matrix[player_index,:])))

            return self.players[player_index], self.items[int(item_index)]

    def urnings_game(self, player, item):
        if type(player) != Player:
            raise TypeError("Player needs to be Player type")

        if type(item) != Player:
            raise TypeError("Item needs to be Player type")
        
        result, expected_results = self.game_type.draw_rule(player, item)

        player_prop, item_prop = self.game_type.updating_rule(player, item, result, expected_results)

        #adding the metropolis step if needed
        metropolis_corrector = self.game_type.metropolis_correction(player, item, player_prop, item_prop)
        adaptivity_corrector = self.game_type.adaptivity_correction(player, item, player_prop, item_prop)

        acceptance = min(1, metropolis_corrector * adaptivity_corrector)
        u = np.random.uniform()

        player_prev = player.score
        item_prev = item.score

        if u < acceptance:
            player.score = player_prop
            item.score = item_prop
            player.est = player.score / player.urn_size
            item.est = item.score / item.urn_size

        #appending new update to the container
        player.container = np.append(player.container, player.score)
        item.container = np.append(item.container, item.score)

        #calculating the difference
        player_diff = player.score - player_prev
        item_diff = item.score - item_prev

        #appending second order results
        player.differential_container = np.append(player.differential_container, player_diff)
        item.differential_container = np.append(item.differential_container, item_diff)

        #print("Match between ", player.user_id, " and", item.user_id)

    def play(self, n_games):
        for ng in range(n_games):
            current_player, current_item = self.matchmaking()
            self.urnings_game(current_player, current_item)


    


        


