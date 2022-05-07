from unittest import result
import numpy as np
from pytest import param
import main_urnings as mu
import matplotlib.pyplot as plt
import utilities as util

#creating containers
results_n_change = np.zeros((20, 3))

#recreating parameter space

#changing parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes = [6, 10, 14, 18, 50]

#fixed parameters
n_player = 1000
n_items = 1000
item_urn_sizes = 100
n_sim = 300

parameter_space_limit = np.zeros((len(true_values) * len(player_urn_sizes), 2))

counter = 0
for tv in range(len(true_values)):
    for pus in range(len(player_urn_sizes)):
        parameter_space_limit[counter, :] = true_values[tv], player_urn_sizes[pus]
        counter += 1

#NONADAPTIVE URNINGS WITH NO CHANGE#################################################################
#loading results of the simulation without adaptivity and change
urnings_results_limit = np.load("urnings_array_limit.npy")

for pars in range(len(true_values) * len(player_urn_sizes)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, true_value=parameter_space_limit[pars, 0])
    results_n_change[pars,2] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit[pars, 0])

    #save this to the results
    results_n_change[pars,0] = MSE

    ######################## HITTING TIME #########################################

    #SE_true_value = np.sqrt(parameter_space_limit[pars, 1] * parameter_space_limit[pars, 0] * (1 - parameter_space_limit[pars, 0]) / n_player)
    hitting_time = util.hitting_time(col_means, parameter_space_limit[pars, 0], tol=0.01)
    results_n_change[pars,1] = hitting_time
   
#ADAPTIVE URNINGS WITH NO CHANGE
results_n_change_adaptive = np.zeros((20, 3))
urnings_results_limit = np.load("urnings_array_limit_adaptive.npy", allow_pickle = True)

for pars in range(len(true_values) * len(player_urn_sizes)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, true_value=parameter_space_limit[pars, 0])
    results_n_change_adaptive[pars, 2] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit[pars, 0])
 
    #save this to the results
    results_n_change_adaptive[pars, 0] = MSE

    ######################## HITTING TIME #########################################
    hitting_time = util.hitting_time(col_means, parameter_space_limit[pars, 0])
    #print(hitting_time)
    results_n_change_adaptive[pars, 1] = hitting_time

    ######################## COVERAGE #########################################
    # nominal_coverage_sim = np.random.binomial(parameter_space_limit[pars, 1], parameter_space_limit[pars, 0], (n_player, n_sim)) / parameter_space_limit[pars, 1]
    
    # nominal_coverage = 0
    # for i in range(n_sim):
    #     confint_nc = np.percentile(nominal_coverage_sim[:,i], [5, 95])
    #     if confint_nc[0] < parameter_space_limit[pars, 0] < confint_nc[1]:
    #         nominal_coverage += 1
    
    # nominal_coverage = nominal_coverage / n_sim


#Adaptive urnsize no item selection WITH NO CHANGE#################################################################
#creating containers
results_n_change_aurnsize = np.zeros((24, 3))

#recreating parameter space

#changing parameters
#parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes_start = [4, 8]
player_urn_sizes_end = [16, 32, 64]

#fixed parameters
n_player = 1000
n_items = 1000
item_urn_sizes = 100
n_sim = 300


parameter_space_limit_aurnsize = np.zeros((len(true_values) * len(player_urn_sizes_end) * len(player_urn_sizes_start), 3))

counter = 0
for tv in range(len(true_values)):
    for pus in range(len(player_urn_sizes_start)):
        for pue in range(len(player_urn_sizes_end)):
            parameter_space_limit_aurnsize[counter, :] = true_values[tv], player_urn_sizes_start[pus], player_urn_sizes_end[pue]
            counter += 1

urnings_results_limit = np.load("urnings_array_aurnsize.npy")

for pars in range(len(true_values) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)

    coverage = util.coverage(urnings_matrix, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_aurnsize[pars,2] = coverage

######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_aurnsize[pars,0] = MSE

    ######################## HITTING TIME #########################################
    hitting_time = util.hitting_time(col_means, parameter_space_limit_aurnsize[pars,0])
    results_n_change_aurnsize[pars,1] = hitting_time

#Adaptive urnsize adaptive item selection WITH NO CHANGE#################################################################
#creating containers
results_n_change_aurnsize_adaptive = np.zeros((24, 3))

#recreating parameter space

#changing parameters
#parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes_start = [4, 8]
player_urn_sizes_end = [16, 32, 64]

#fixed parameters
n_player = 1000
n_items = 1000
item_urn_sizes = 100
n_sim = 300


parameter_space_limit_aurnsize = np.zeros((len(true_values) * len(player_urn_sizes_end) * len(player_urn_sizes_start), 3))

counter = 0
for tv in range(len(true_values)):
    for pus in range(len(player_urn_sizes_start)):
        for pue in range(len(player_urn_sizes_end)):
            parameter_space_limit_aurnsize[counter, :] = true_values[tv], player_urn_sizes_start[pus], player_urn_sizes_end[pue]
            counter += 1

urnings_results_limit = np.load("urnings_array_aurnsize_adaptive.npy")

for pars in range(len(true_values) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)

    coverage = util.coverage(urnings_matrix, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_aurnsize_adaptive[pars,2] = coverage

######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_aurnsize_adaptive[pars,0] = MSE

    ######################## HITTING TIME #########################################
    hitting_time = util.hitting_time(col_means, parameter_space_limit_aurnsize[pars,0])
    results_n_change_aurnsize_adaptive[pars,1] = hitting_time


print(parameter_space_limit)
print("                    ")
print(results_n_change)
print("                    ")
print(results_n_change_adaptive)
print("                    ")
print(results_n_change_aurnsize)
print("                    ")
print(results_n_change_aurnsize_adaptive)
print("                    ")

print("Mean hitting time non adaptive: ", np.mean(results_n_change[:,1]),
      "adaptive: ", np.mean(results_n_change_adaptive[:,1]),
      "adaptive urn: ", np.mean(results_n_change_aurnsize[:,1]),
      "adaptive urn adaptive item selection", np.mean(results_n_change_aurnsize_adaptive[:,1]))
print("Mean MSE  non adaptive: ", np.mean(results_n_change[:,0]),
       "adaptive: ", np.mean(results_n_change_adaptive[:,0]),
       "adaptive urn: ", np.mean(results_n_change_aurnsize[:,0]),
       "adaptive urn adaptive item selection", np.mean(results_n_change_aurnsize_adaptive[:,0]))
print("Mean coverage  non adaptive: ", np.mean(results_n_change[:,2]),
      "adaptive: ", np.mean(results_n_change_adaptive[:,2]),
      "adaptive urn: ", np.mean(results_n_change_aurnsize[:,2]),
      "adaptive urn adaptive item selection", np.mean(results_n_change_aurnsize_adaptive[:,2]))

