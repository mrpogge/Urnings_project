import pandas as pd
import numpy as np
import matplotlib as plt
from scipy import rand
import scipy.stats as sp
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

class Player:
#class default constructor
    def __init__(self, user_id, score, urn_size, true_value, multiple_urn = False, so_urn_size = 10): 
        if score > urn_size:
            raise ValueError("The score can't be higher then the urn size.")
        if multiple_urn == False:
            #basic attributes
            self.user_id = user_id
            self.score = score
            self.urn_size = urn_size
            self.est = self.score/self.urn_size
            self.true_value = true_value
            self.sim_y = 8
            self.sim_true_y = 8

            #second-order urnings
            self.so_urn_size = so_urn_size
            self.so_score = int(np.round(so_urn_size / 2))
            self.so_est = self.so_score / self.so_urn_size
            
            #creating a container
            self.container = np.array([self.score])
            self.estimate_container = np.array([self.est])
            self.differential_container = np.array([0])
            self.so_container = np.array([self.so_est])
            #save the number of green balls per item 
            #save the urn_size
    
    def __eq__(self, other):
        return self.user_id == other.user_id
    

    def draw(self, true_score_logic = False):

        if true_score_logic == False:
            sim_y = np.random.binomial(1, self.est)
            self.sim_y = sim_y
            return  sim_y
        else:
            sim_y = np.random.binomial(1, self.true_value)
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
            

class Game_Type:
    def __init__(self, 
                adaptivity,
                alg_type, 
                updating_type = "one_dim", 
                paired_update = False, 
                adaptive_urn = False, 
                min_urn = None,
                max_urn = None,
                freq_change = None,
                window = None,
                bound = None):

        self.adaptivity = adaptivity
        self.alg_type = alg_type
        self.updating_type = updating_type
        self.item_pair_update = paired_update
        self.adaptive_urn = adaptive_urn
        self.min_urn = min_urn
        self.max_urn = max_urn
        self.freq_change = freq_change
        self.window = window
        self.bound = bound

        #container for updates
        self.queue_pos = []
        self.queue_neg = []
    
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
            player.est = (player.score + result) / (player.urn_size + 1)
            item.est = (item.score + 1 - result) / (item.urn_size + 1)

            while player.sim_y == item.sim_y:
                    player.draw()
                    item.draw()
                
            expected_results = player.sim_y
            player.sim_y = item.sim_y = 8
            
            #returning to the original urn conig
            player.est = player.score / player.urn_size
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

    #depriciated!!!!!!!!!!!
    def adaptivity_correction(self, player, item, player_proposal, item_proposal, proposed_adaptive_matrix = None):
        
        if self.adaptivity == "n_adaptive":
            adaptivity_corrector = 1

        else:
            #change this to be a function or method of some sort
            current_item_prob = np.exp(-2*(np.log((player.score + 1) / (player.urn_size- player.score + 1)) - np.log((item.score + 1) / (item.urn_size - item.score + 1)))**2)
            proposed_item_prob = np.exp(-2*(np.log((player_proposal + 1) / (player.urn_size - player_proposal + 1)) - np.log((item_proposal + 1) / (item.urn_size - item_proposal + 1)))**2)

            adaptivity_corrector = proposed_item_prob/current_item_prob

        return adaptivity_corrector
       

                
        
    def adaptive_urn_change(self, player):
        
        if self.adaptive_urn == True:
            if len(player.differential_container) >= self.window:
                conv_stat = player.differential_container[-self.window:]

                #check the stats
                if np.sum(conv_stat) >= self.bound and player.urn_size > self.min_urn:
                    change = player.urn_size / self.min_urn
                    player.score = int(np.round(player.score / change))
                    player.urn_size = self.min_urn
                    player.est = player.score / player.urn_size
                elif len(player.differential_container) % self.freq_change == 0 and player.urn_size < self.max_urn:
                    player.urn_size = player.urn_size * 2
                    player.score =  player.score * 2
                    player.est = player.score / player.urn_size

        
class Urnings:
    def __init__(self, players, items, game_type):
        self.standings = []
        self.players = players
        self.items = items
        self.game_type = game_type
        self.queue_pos = {k.user_id : 0 for k in self.items}
        self.queue_neg = {k.user_id : 0 for k in self.items}
        self.adaptive_matrix = self.adaptive_rule_normal()
        
        
        sum_gb_init = 0
        for it in self.items:
            sum_gb_init += it.score
        
        self.item_green_balls = [sum_gb_init]
            
    
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
    
    def adaptive_rule_normal_partial(self, player, item):
        for pl in range(len(self.players)):
            if player.user_id == self.players[pl].user_id:
                player_idx = pl
        
        for it in range(len(self.items)):
            if item.user_id == self.items[it].user_id:
                item_idx = it

        for rw in range(len(self.adaptive_matrix[player_idx,:])):
                if rw != item_idx:
                    R_i = player.score
                    R_j = self.items[rw].score
                    n_i = player.urn_size
                    n_j = self.items[rw].urn_size
                    self.adaptive_matrix[player_idx,rw] = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)
            
        for cl in range(len(self.adaptive_matrix[:,item_idx])):
            if cl != player_idx:
                R_i = self.players[cl].score
                R_j = item.score
                n_i = self.players[cl].urn_size
                n_j = item.urn_size
                self.adaptive_matrix[cl, item_idx] = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)
        
        return self.adaptive_matrix

    def matchmaking(self, ret_adaptivity_matrix = False):

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

        #item and player indexes

        for pl in range(len(self.players)):
            if player.user_id == self.players[pl].user_id:
                player_idx = pl
        
        for it in range(len(self.items)):
            if item.user_id == self.items[it].user_id:
                item_idx = it
        
        result, expected_results = self.game_type.draw_rule(player, item)

        player_proposal, item_proposal = self.game_type.updating_rule(player, item, result, expected_results)

        if self.game_type.adaptivity == "adaptive":
            proposed_adaptive_matrix = self.adaptive_matrix

            #filling the cross
            R_i = player_proposal
            R_j = item_proposal
            n_i = player.urn_size
            n_j = item.urn_size
            prob = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)

            proposed_adaptive_matrix[player_idx, item_idx] = prob

            #filling the cols and rows included in the proposal
            for rw in range(len(proposed_adaptive_matrix[player_idx,:])):
                if rw != item_idx:
                    R_i = player_proposal
                    R_j = self.items[rw].score
                    n_i = player.urn_size
                    n_j = self.items[rw].urn_size
                    proposed_adaptive_matrix[player_idx,rw] = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)
            
            for cl in range(len(proposed_adaptive_matrix[:,item_idx])):
                if cl != player_idx:
                    R_i = self.players[cl].score
                    R_j = item_proposal
                    n_i = self.players[cl].urn_size
                    n_j = item.urn_size
                    proposed_adaptive_matrix[cl, item_idx] = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)
            
            current_item_prob = self.adaptive_matrix[player_idx, item_idx] / np.sum(self.adaptive_matrix[player_idx,:])
            proposed_item_prob = proposed_adaptive_matrix[player_idx, item_idx] / np.sum(proposed_adaptive_matrix[player_idx,:])

            adaptivity_corrector = proposed_item_prob/current_item_prob
        else: 
            adaptivity_corrector = 1



        #adding the metropolis step if needed
        metropolis_corrector = self.game_type.metropolis_correction(player, item, player_proposal, item_proposal)
        #depreciated!!!!!
        #adaptivity_corrector = self.game_type.adaptivity_correction(player, item, player_proposal, item_proposal)
        
        acceptance = min(1, metropolis_corrector * adaptivity_corrector)
        u = np.random.uniform()

        player_prev = player.score
        item_prev = item.score

        if u < acceptance:
            player.score = player_proposal
            item.score = item_proposal
            player.est = player.score / player.urn_size
            item.est = item.score / item.urn_size

        #Paired_update
        #calculating the difference
        player_diff = player.score - player_prev
        item_diff = item.score - item_prev

        if self.game_type.item_pair_update == True:
            if item_diff == 1:
                if all(i < 1 for i in list(self.queue_neg.values())):
                    self.queue_pos[item.user_id] += 1
                    if item.score > 0:
                        item.score -= 1
                        item.est = item.score / item.urn_size 
                else:
                    candidates = {k:v for k,v in self.queue_neg.items() if v >= 1}
                    idx = np.random.randint(0, len(candidates.keys()))
                    candidate_user_id = list(candidates)[idx]

                    self.queue_neg[candidate_user_id] = 0

                    for it in self.items:
                        if it.user_id == candidate_user_id:
                            candidate_item = it 
                    
                    candidate_item.score -= 1
                    candidate_item.est = candidate_item.score / candidate_item.urn_size
            elif item_diff == -1:
                if all(i < 1 for i in list(self.queue_pos.values())):
                    self.queue_neg[item.user_id] += 1
                    if item.score > 0:
                        item.score += 1
                        item.est = item.score / item.urn_size 
                else:
                    candidates = {k:v for k,v in self.queue_pos.items() if v >= 1}
                    idx = np.random.randint(0, len(candidates.keys()))
                    candidate_user_id = list(candidates)[idx]
                    
                    self.queue_pos[candidate_user_id] = 0

                    for it in self.items:
                        if it.user_id == candidate_user_id:
                            candidate_item = it 
                    
                    candidate_item.score += 1
                    candidate_item.est = candidate_item.score / candidate_item.urn_size

        #adaptive matrix recalculation
        if self.game_type.adaptivity == "adaptive":
            self.adaptive_matrix = self.adaptive_rule_normal_partial(player, item)

        #adaptive urn_size
        self.game_type.adaptive_urn_change(player)

        #appending new update to the container
        player.container = np.append(player.container, player.score)
        item.container = np.append(item.container, item.score)

        player.estimate_container = np.append(player.estimate_container, player.est)
        item.estimate_container = np.append(item.estimate_container, item.est)

        #appending second order results
        player.differential_container = np.append(player.differential_container, player_diff)
        item.differential_container = np.append(item.differential_container, item_diff)

        #SECOND ORDER URNINGS PROTOTYPE
        #if np.abs(player_diff) == 1:
            
        #    player.so_score = player.so_score + np.random.binomial(1, player.so_est) - player_diff
        #    player.so_est = player.so_score / player.so_urn_size 
            
        #    if player.so_est > 1:
        #        player.so_est = 1
            
        #    player.so_container = np.append(player.so_container, player.so_est)
        #else:
        #    player.so_score = player.so_score
        #    player.so_container = np.append(player.so_container, player.so_est)

        

    def play(self, n_games, test = False):
        for ng in range(n_games):
            if test == True:
                for pl in range(len(self.players)):
                    #ERROR
                    if self.game_type.adaptivity == "adaptive":
                        item_index = np.random.choice(np.arange(len(self.items)), 1, p = (self.adaptive_matrix[pl,:] / np.sum(self.adaptive_matrix[pl,:])))
                        current_item = self.items[item_index[0]]
                    else:
                        item_index = np.random.randint(0, len(self.items))
                        current_item = self.items[item_index]
                    self.urnings_game(self.players[pl], current_item)
                    #print(self.queue_direction)

                    #calculating the number of green balls in the item urns
                    sum_gb = 0
                    for it in self.items:
                        sum_gb += it.score
                    
                    self.item_green_balls.append(sum_gb)
            else:
                current_player, current_item = self.matchmaking()
                self.urnings_game(current_player, current_item)



    


        


