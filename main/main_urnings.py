import pandas as pd
import numpy as np
import matplotlib as plt
from scipy import rand
import scipy.stats as sp
import statsmodels.api as sm
from statsmodels.graphics import tsaplots
import utilities as util
import typing 
from typing import Type

class Player:
    """
    class Player
        Player object constructor. It is used to create players and items which can play in the adaptive learning system

        attributes:
            user_id: str
                a string indicating the user id
            score: int
                the current number of green balls in the user's urn 
            urn_size: int
                the (current) size of the player's urn
            est: float
                the current estimate of the player's urn
            true_value: float
                the true ability of the student (used in simulation)
            sim_y: int
                helper attribute for draws from player urns
            sim_true_y: int
                helper attribute for draws from player urns (used in simulation)
            so_urn_size: int
                size of the second order urnings (used if Game_Type.adaptive_urn_type = "second order urnings")
            so_score: int
                current number of green balls in the player's second order urn
            so_est: float
                current second order estimate of the player
            container: np.array
                a numpy array containing the scores of the player over multiple games in the system
            estimate_container: np.array
                a numpy array containing the estimates of the player over multiple games in the system
            differential_container: np.array 
                    TODO: check whether it works if the adaptive urnsize algos are online
                a numpy array containing the direction of change of the player's score
            urn_container: np.array
                a numpy array containing the urn size of the player over multiple games in the system (relevant if the adaptive urn algos are online)
            so_container: np.array
                a numpy array containing the second order estimates of the player over multiple games in the system
            idx: int
                an id created when the user enters a game

        methods:
            __eq__(other: Type[Player]):
                an equivalence function which checks whether two player objects are the same
            find(id: int):
                a function with boolean output indicating whether the given player has the inputed id or not
            draw(true_score_logic:bool = False)
                a function governing the draws from urns if the true score logic is true we are drawing with the true probability (used in simulation)
            so_draw():
                a function governing the draws from the second order urns
            autocorrelation(lag: int, plots: bool = False):
                a function which calculates the autocorrelation for the chain of estimates
            so_autocorrelation(lag: int, plots: bool = False):
                a function calculating the autocorrelation for the differential container
    """
    def __init__(self, user_id: str, score: int, urn_size: int, true_value: float, so_urn_size: int = 10): 

        #TODO: implementing player declaration errors
        if score > urn_size:
            raise ValueError("The score can't be higher then the urn size.")

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
        self.urn_container = np.array([self.urn_size])
        self.so_container = np.array([self.so_est])

        #utility attribute
        self.idx = None
    
    def __eq__(self, other):
        return self.user_id == other.user_id
    
    def find(self, id:int):
        return self.user_id == id
    

    def draw(self, true_score_logic: bool = False):
       #drawing the expected result based on the most frequent estimate
        if true_score_logic == False:
            sim_y = np.random.binomial(1, self.est)
            self.sim_y = sim_y
            return  sim_y
        #drawing the simulated outcome
        else:
            sim_y = np.random.binomial(1, self.true_value)
            self.sim_true_y = sim_y
            return sim_y

    def so_draw(self):
        #drawing the expected result based on the most frequent second order estimate
        sim_y = np.random.binomial(1, self.so_est)
        self.sim_y = sim_y
        return sim_y

    def autocorrelation(self, lag: int, plots: bool = False):
        #calculating autocorrelation for the player's urn chain
        acf_player = sm.tsa.acf(self.estimate_container, nlags= lag)

        #if true plot the results
        if plots == True:
            fig = tsaplots.plot_acf(self.container, lags = lag)

        return acf_player 

    def so_autocorrelation(self, lag:int , plots: bool = False):
        #calculating autocorrelation for the second order chain
        acf_so = sm.tsa.acf(self.differential_container, nlags = lag)

        #if true plot the results
        if plots == True: 
            fig = tsaplots.plot_acf(self.differential_container, lags = lag)

        return acf_so
            

class Game_Type:
    """
    class Game_Type:
        a class to setup the different analysis and simulation options for the game environment, ensures the modular architecture

        attributes:
            adaptivity: str
                takes values from ["n_adaptive", "adaptive"], governs the item selection
            alg_type: str
                takes values from ["Urnings1", "Urnings2"], choses the type of algorithm to use
            updating_type: str
                currently only one_dimensional TODO: implement multidimensional, response time and polytomous urning and ELO rating system
            paired_update: bool
                indicates the use of paired update
            adaptive_urn: bool
                indicates the use of adaptive urn algorithms
            adaptive_urn_type: str
                takes values from ["permutation", "second_order_urnings"], choses the type of adaptive urn size algorithm
            min_urn: int
                the minimum urn value (used in adaptive urn size algorithms)
            max_urn: int
                the maximim urn value (used in adaptive urn size algorithms)
            freq_change: int
                the frequency (iteration) to change the urn size (used in adaptive_urn_type = "permutation" witn permutation_test = False)
            window: int
                the size of the moving window (used in adaptive_urn_type = "permutation")
            bound: int
                the number of wins which is considered a winstreak (used in adaptive_urn_type = "permutation" witn permutation_test = False)
            permutation_test: bool
                indicates whether we use permutation test or not
            n_permutations: int
                indicates the number of permutations generated if window < 15 the algorithm uses exact test
            perm_p_val: float
                p value for the permutation test

    """
    def __init__(self, 
                adaptivity: str,
                alg_type: str, 
                updating_type: str = "one_dim", 
                paired_update: bool = False, 
                adaptive_urn: bool = False,
                adaptive_urn_type = None, 
                min_urn = None,
                max_urn = None,
                freq_change = None,
                window: int = 2,
                bound = None,
                permutation_test: bool = False,
                n_permutations: int = 1000,
                perm_p_val: float = 0.05):

        self.adaptivity = adaptivity
        self.alg_type = alg_type
        self.updating_type = updating_type
        self.item_pair_update = paired_update
        self.adaptive_urn = adaptive_urn
        self.adaptive_urn_type = adaptive_urn_type
        self.min_urn = min_urn
        self.max_urn = max_urn
        self.freq_change = freq_change
        self.window = window
        self.bound = bound
        self.permutation_test = permutation_test
        self.n_permutations = n_permutations
        self.perm_p_val = perm_p_val

        #container for updates
        self.queue_pos = []
        self.queue_neg = []

        #all combinations for exact permutation test
        self.all_comb = util.all_binary_combination(self.window)
    
    def draw_rule(self, player, item, data_bool = False, result_data = None):
        
        if self.alg_type == "Urnings1":
            if data_bool == False:
                #simulating the observed value
                while player.sim_true_y == item.sim_true_y:
                    player.draw(true_score_logic = True)
                    item.draw(true_score_logic = True)
                
                result = player.sim_true_y
                player.sim_true_y = item.sim_true_y = 8
            else:
                result = result_data

            #calculating expected score
            while player.sim_y == item.sim_y:
                player.draw()
                item.draw()
            
            expected_results = player.sim_y
            player.sim_y = item.sim_y = 8
        
        elif self.alg_type == "Urnings2":
            if data_bool == False:
                #simulating the observed value
                while player.sim_true_y == item.sim_true_y:
                    player.draw(true_score_logic = True)
                    item.draw(true_score_logic = True)
                
                result = player.sim_true_y
                player.sim_true_y = item.sim_true_y = 8
            else:
                result = result_data


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
            if self.adaptive_urn_type == "permutation":
                if len(player.differential_container) >= self.window and len(player.differential_container) % self.window == 0:
                    conv_stat = player.differential_container[-self.window:]

                    if self.permutation_test == False:
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
                    else:
                        permute_means = np.mean(self.all_comb * conv_stat, axis=1)
                        p_value = 1 - np.sum(permute_means < np.abs(np.mean(conv_stat)))/len(permute_means)

                        if p_value < self.perm_p_val:
                            change = player.urn_size / self.min_urn
                            player.score = int(np.round(player.score / change))
                            player.urn_size = self.min_urn
                            player.est = player.score / player.urn_size
                        elif player.urn_size < self.max_urn:
                            player.urn_size = player.urn_size * 2
                            player.score =  player.score * 2
                            player.est = player.score / player.urn_size

            elif self.adaptive_urn_type == "second_order_urnings":
                draw_urn_control = np.sum(np.random.binomial(1, player.so_container[-1], 2))
                if draw_urn_control != 1 and player.urn_container[-1] > self.min_urn :
                    player.urn_size = player.urn_container[-1] / 2
                    player.score = int(np.round(player.score / 2))
                    player.est = player.score / player.urn_size
                    
                elif draw_urn_control == 1 and player.urn_container[-1] < self.max_urn:
                    player.urn_size = player.urn_container[-1] * 2
                    player.score = player.score * 2
                    player.est = player.score / player.urn_size


class AlsData:
    def __init__(self, player_id, item_id, correct_answer, game_id = None, time_stamp = None):
        self.game_id = game_id
        self.player_id = player_id
        self.item_id = item_id
        self.correct_answer = correct_answer
        self.time_stamp = time_stamp

        self.player_punchcard = self.punchcard()[0]
        self.item_punchcard = self.punchcard()[1]

        if self.game_id is None:
            self.game_id = np.arange(0, len(player_id)-1)
    
    def create_players_items(self, player_starting_score, item_starting_score, player_urn_size, item_urn_size, so_urn_size):

        players = []
        items = []
        player_set = list(set(self.player_id))
        item_set = list(set(self.item_id))
        punchcard_manual_player = np.zeros((len(player_set), 2))
        punchcard_manual_item = np.zeros((len(item_set), 2))

        for pl in range(len(player_set)):
            player = Player(player_set[pl], player_starting_score, player_urn_size, None, so_urn_size)
            player.idx = pl
            players.append(player)

            punchcard_manual_player[pl, :] = player_set[pl], pl
        
        for it in range(len(item_set)):
            item = Player(item_set[it], item_starting_score, item_urn_size, so_urn_size)
            item.idx = it
            items.append(item)

            punchcard_manual_item[it] = item_set[it], it
        
        return players, items, punchcard_manual_player, punchcard_manual_item
    
    #hidden method
    def punchcard(self):
        
        player_set = list(set(self.player_id))
        item_set = list(set(self.item_id))
        punchcard_manual_player = np.zeros((len(player_set), 2))
        punchcard_manual_item = np.zeros((len(item_set), 2))

        for pl in range(len(player_set)):
            punchcard_manual_player[pl, :] = player_set[pl], pl
        
        for it in range(len(item_set)):
            punchcard_manual_item[it] = item_set[it], it

        return punchcard_manual_player, punchcard_manual_item

        
class Urnings:
    """
    class Urnings: 
        A framework which saves all the chosen game options and let the Players and Items play the Urnings Game.
        Both simulation and Data analysis moduls are available.

        attributes: 
            players: list[Player]
                list of Player objects we would like to analyse, representing the students/players in the adaptive learning system
                For details see Player.__doc__().
            items: list[Player]
                list of Player objects we would like to analyse, representing the items in the adaptive learnings system.
                For details see Player.__doc__().
            game_type: Game_Type
                A Game_Type object which save all the options we want to declare before we start the learning system. 
                For details see Game_Type.__doc__()
            data: AlsData
                An AlsData object which contains the serialized, and restructured data, it enables us to create the Player objects and a punchcard which 
                governs the data's item selection procedure. 
                For details see Als Data.__doc__()
            queue_pos: dict
                Dictionary with all the defined item's user id as key. Used in the paired update system. This dictionary contains the items waitning for a positive update.
            queue_neg_ dict
                Dictionary with all the defined item's user id as key. Used in the paired update system. This dictionary contains the items waitning for a negative update.
            adaptive_matrix: np.ndarray
                A numpy array saving the probability of selection of each player item paires (TODO: Refactor it into an Urn based matrix for better computation performance)
            game_count: int
                The number of games we played in the system.
            item_green_balls: list[int]
                A list of integers saving the number of green balls in all item urns. It can be used to check whether there is green ball inflation which is against the 
                assumption of Urnings model. Can be corrected by setting Game.Type.paired_update = True.
            prop_correct: np.ndarray
                A numpy array saving the proportion of correct responses in each player and item urnings. It can be used to investigate model fit by plotting it against 
                the true model predictions using contour plots.
            number_bin_correct: np.ndarray
                A numpy array saving the number of responese in each play and item urnings.
            fit_correct:  np.ndarray
                A numpy array saving the model implied proportions. It can be used to investigate model fit.
        
        methods:
            adaptive_rule_normal(self)
                calculates the probability matrix for adaptive item selection for all player item pairs using the normal method, for details see. Bolsinova et al (2022).
            adaptive_rule_partial(self, player: Type[Player], item: Type[Player])
                helper function for more optimal calculation of the probability matrix for adaptive item selection, by only updating the given column and row based on the 
                player and the item id
            matchmaking(self, ret_adaptive_matrix: bool = False)
                function governing the matchmaking by using either the adaptive or the nonadaptive alternatives, it can retrun the updated probability matrix for adaptive
                item selection
            urnings_game(player: Type(Player), item: Type(Player))
                The summary function which set's up the game environment. It activates after item selection and updates the item and player properties
            play(n_games: int, test: bool = False)
                The function which starts the Urnings game. Test can be used to let each player play the same amount of games. This feature can be useful with simulation studies



    """
    def __init__(self, players: list[Type[Player]], items: list[Type[Player]], game_type: Type[Game_Type], data: Type[AlsData] = None):
        # initial data for the Urnings frameweok
        self.players = players
        self.items = items
        self.game_type = game_type
        self.data = data

        #containers for the paired update queue
        self.queue_pos = {k.user_id : 0 for k in self.items}
        self.queue_neg = {k.user_id : 0 for k in self.items}

        #helper attribute for the paired update queue
        sum_gb_init = 0
        for it in self.items:
            sum_gb_init += it.score
        
        self.item_green_balls = [sum_gb_init]

        #helper attributes for the adaptive item selection
        self.adaptive_matrix = self.adaptive_rule_normal()
        self.game_count = 0

        #arrays to calculate model fit
        if self.game_type.adaptive_urn == False:
            self.game_type.max_urn = self.players[0].urn_size

        self.prop_correct = np.zeros((self.game_type.max_urn, self.items[0].urn_size))
        self.number_per_bin = np.zeros((self.game_type.max_urn, self.items[0].urn_size))
        self.fit_correct = np.zeros((self.game_type.max_urn, self.items[0].urn_size))


    def adaptive_rule_normal(self):
        #preallocating the adaptive matrix with shape = players * items
        adaptive_matrix = np.zeros(shape=(len(self.players), len(self.items)))

        #looping through each player item pairs to calculate the item selection probability
        for i in range(len(self.players)):
            for j in range(len(self.items)):

                R_i = self.players[i].score
                R_j = self.items[j].score
                n_i = self.players[i].urn_size
                n_j = self.items[j].urn_size
                prob = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)

                adaptive_matrix[i, j] = prob 

        #returns an np.ndarray with item selection probabilites
        return adaptive_matrix
    
    def adaptive_rule_normal_partial(self, player: Type(Player), item: Type(Player)):
        #TODO: Find a better solution for finding the player and item ids
        #creating item and player ids: MATURING
        for pl in range(len(self.players)):
            if player.user_id == self.players[pl].user_id:
                player_idx = pl
        
        for it in range(len(self.items)):
            if item.user_id == self.items[it].user_id:
                item_idx = it

        #filling in the row of the player
        for rw in range(len(self.adaptive_matrix[player_idx,:])):
                if rw != item_idx:
                    R_i = player.score
                    R_j = self.items[rw].score
                    n_i = player.urn_size
                    n_j = self.items[rw].urn_size
                    self.adaptive_matrix[player_idx,rw] = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)
        
        #filling in the column of the player
        for cl in range(len(self.adaptive_matrix[:,item_idx])):
            if cl != player_idx:
                R_i = self.players[cl].score
                R_j = item.score
                n_i = self.players[cl].urn_size
                n_j = item.urn_size
                self.adaptive_matrix[cl, item_idx] = np.exp(-2*(np.log((R_i + 1) / (n_i-R_i + 1)) - np.log((R_j + 1) / (n_j-R_j + 1)))**2)
        
        #returning an np.ndarray containing item selection probabilites
        return self.adaptive_matrix

    def matchmaking(self, ret_adaptivity_matrix: bool = False):

        #checking whether the game type is adaptive or non-adaptive
        #randomly selecting an item and a player for a match (used in simulation)
        if self.game_type.adaptivity == "n_adaptive":
            player_index = np.random.randint(0, len(self.players))
            item_index = np.random.randint(0, len(self.items))
            
            return self.players[player_index], self.items[item_index]

        #randomly selecting a player and selecting an item based on the adaptive matrix (used in simulation)
        elif self.game_type.adaptivity == "adaptive":
            adaptive_matrix = self.adaptive_rule_normal()

            player_index = np.random.randint(0, len(self.players))
            item_index = np.random.choice(np.arange(len(self.items)), 1, p = (adaptive_matrix[player_index,:] / np.sum(adaptive_matrix[player_index,:])))

            return self.players[player_index], self.items[int(item_index)]
        #returns an player, item pair

    def urnings_game(self, player: Type(Player), item: Type(Player)):
        #item and player indexes
        #change this but first make the data analysis work !!!!!!!!!!!!!!
        if self.data is None:
            for pl in range(len(self.players)):
                if player.user_id == self.players[pl].user_id:
                    player_idx = pl
            
            for it in range(len(self.items)):
                if item.user_id == self.items[it].user_id:
                    item_idx = it
            
            result, expected_results = self.game_type.draw_rule(player, item)
        
        else:
            player_idx = player.idx
            item_idx = item.idx
            result = self.data.correct_answer[self.game_count]
            result, expected_results = self.game_type.draw_rule(player, item, data_bool = True, result_data = result)

            #updating fit plot's matrices
            #recalculating player score based on the max urn  size
            temp_div = self.game_type.max_urn / player.urn_size
            temp_score = int(player.score * temp_div) - 1 

            self.prop_correct[temp_score, item.score] += result
            self.number_per_bin[temp_score, item.score] += 1
            self.fit_correct[temp_score, item.score] += expected_results

            
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
                if all(i == 0 for i in list(self.queue_neg.values())):
                    self.queue_pos[item.user_id] += 1
                    if item.score > 0:
                        item.score -= 1
                        item.est = item.score / item.urn_size 
                else:
                    candidates = {k:v for k,v in self.queue_neg.items() if v >= 1}
                    idx = np.random.randint(0, len(candidates.keys()))
                    candidate_user_id = list(candidates)[idx]
                    
                    for it in self.items:
                        if it.user_id == candidate_user_id:
                            candidate_item = it 
                    
                    counter = 0
                    while candidate_item.score <= 0:
                        candidates = {k:v for k,v in self.queue_neg.items() if v >= 1}
                        idx = np.random.randint(0, len(candidates.keys()))
                        candidate_user_id = list(candidates)[idx]
                    
                        for it in self.items:
                            if it.user_id == candidate_user_id:
                                candidate_item = it
                        counter +=1 
                        if counter > 100:
                            break
                        
                    self.queue_neg[candidate_user_id] = 0
                    candidate_item.score -= 1
                    candidate_item.est = candidate_item.score / candidate_item.urn_size
            elif item_diff == -1:
                if all(i == 0 for i in list(self.queue_pos.values())):
                    self.queue_neg[item.user_id] += 1
                    if item.score < item.urn_size:
                        item.score += 1
                        item.est = item.score / item.urn_size 
                else:
                    candidates = {k:v for k,v in self.queue_pos.items() if v >= 1}
                    idx = np.random.randint(0, len(candidates.keys()))
                    candidate_user_id = list(candidates)[idx]

                    for it in self.items:
                        if it.user_id == candidate_user_id:
                            candidate_item = it
                    
                    counter = 0
                    while candidate_item.score >= candidate_item.urn_size:
                        candidates = {k:v for k,v in self.queue_pos.items() if v >= 1}
                        idx = np.random.randint(0, len(candidates.keys()))
                        candidate_user_id = list(candidates)[idx]

                        for it in self.items:
                            if it.user_id == candidate_user_id:
                                candidate_item = it
                        counter += 1
                        if counter > 100:
                            break
                    
                    self.queue_pos[candidate_user_id] = 0
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

        player.urn_container = np.append(player.urn_container, player.urn_size)
        item.urn_container = np.append(item.urn_container, item.urn_size)

        #SECOND ORDER URNINGS PROTOTYPE
        
        if player_diff != 0:
            if player_diff == -1:
                player_diff = 0
            if item_diff == -1:
                item_diff = 0

            #drawing estimate for the player and the item 
            player.so_est = (player.so_score + player_diff) / (player.so_urn_size + 1)
            item.so_est = (item.so_score + 1 - player_diff) / (item.so_urn_size + 1)
            while player.sim_y == item.sim_y:
                player.so_draw()
                item.so_draw()    

            expected_results = player.sim_y
            player.sim_y = item.sim_y = 8
            
            #updating 
            player.so_score = player.so_score + player_diff - expected_results
            item.so_score = item.so_score + (1-player_diff) - (1 - expected_results)
        
            if player.so_score > player.so_urn_size:
                player.so_score = player.so_urn_size
                
            if player.so_score <= 0:
                player.so_score = 0
                
            if item.so_score > item.so_urn_size:
                item.so_score = item.so_urn_size
                
            if item.so_score <= 0:
                item.so_score = 0 

            #print(player.so_est, player.so_score, player_diff, expected_results)
            
            player.so_est = player.so_score / player.so_urn_size
            item.so_est = item.so_score / item.so_urn_size
        
        player.so_container = np.append(player.so_container, player.so_est)
        item.so_container = np.append(item.so_container, item.so_est)
            

    def play(self, n_games, test = False):
        if self.data is None:
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
        elif self.data is not None:
            for gm in range(len(self.data.game_id)):
                current_player_id = self.data.player_id[self.game_count]
                current_item_id = self.data.item_id[self.game_count]
                 
                pl_slicer = int(np.where(self.data.player_punchcard[:,0] == current_player_id)[0])
                it_slicer = int(np.where(self.data.item_punchcard[:,0] == current_item_id)[0])

                current_player = self.players[int(self.data.player_punchcard[pl_slicer, 1])]
                current_item = self.items[int(self.data.item_punchcard[it_slicer, 1])]

                self.urnings_game(current_player, current_item)

                self.game_count += 1



    


        


