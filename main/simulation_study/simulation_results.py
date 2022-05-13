from unittest import result
import numpy as np
from pytest import param
import main_urnings as mu
import matplotlib.pyplot as plt
import utilities as util
import pandas as pd

#creating containers
results_n_change = np.zeros((20, 3))

#recreating parameter space

#changing parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes = [4, 8, 16, 32, 64]

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
results_n_change_perm = np.zeros((16, 3))

#recreating parameter space

#changing parameters
#parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes_start = [4, 8]
player_urn_sizes_end = [32, 64]

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

urnings_results_limit = np.load("urnings_array_perm.npy")

for pars in range(len(true_values) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)

    coverage = util.coverage(urnings_matrix, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_perm[pars,2] = coverage

######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_perm[pars,0] = MSE

    ######################## HITTING TIME #########################################
    hitting_time = util.hitting_time(col_means, parameter_space_limit_aurnsize[pars,0])
    results_n_change_perm[pars,1] = hitting_time

#Adaptive urnsize adaptive item selection WITH NO CHANGE#################################################################
#creating containers
results_n_change_perm_adaptive = np.zeros((16, 3))

#recreating parameter space
#changing parameters
#parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes_start = [4, 8]
player_urn_sizes_end = [32, 64]

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

urnings_results_limit = np.load("urnings_array_aurnsize_perm.npy")

for pars in range(len(true_values) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)

    coverage = util.coverage(urnings_matrix, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_perm_adaptive[pars,2] = coverage

######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_perm_adaptive[pars,0] = MSE

    ######################## HITTING TIME #########################################
    hitting_time = util.hitting_time(col_means, parameter_space_limit_aurnsize[pars,0])
    results_n_change_perm_adaptive[pars,1] = hitting_time

#############################################################################################
###############SECOND ORDER URNINGS###############################################
results_n_change_sourn = np.zeros((16, 3))

urnings_results_limit = np.load("urnings_array_sourn.npy")

for pars in range(len(true_values) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)

    coverage = util.coverage(urnings_matrix, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_sourn[pars,2] = coverage

######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_sourn[pars,0] = MSE

    ######################## HITTING TIME #########################################
    hitting_time = util.hitting_time(col_means, parameter_space_limit_aurnsize[pars,0])
    results_n_change_sourn[pars,1] = hitting_time

results_n_change_sourn_adaptive = np.zeros((16, 3))
urnings_results_limit = np.load("urnings_array_sourn_adaptive.npy")

for pars in range(len(true_values) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)

    coverage = util.coverage(urnings_matrix, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_sourn_adaptive[pars,2] = coverage

######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, parameter_space_limit_aurnsize[pars, 0])
    results_n_change_sourn_adaptive[pars,0] = MSE

    ######################## HITTING TIME #########################################
    hitting_time = util.hitting_time(col_means, parameter_space_limit_aurnsize[pars,0])
    results_n_change_sourn_adaptive[pars,1] = hitting_time

print(parameter_space_limit)
print("                    ")
print(results_n_change)
print("                    ")
print(results_n_change_adaptive)
print("                    ")
print(results_n_change_perm)
print("                    ")
print(results_n_change_perm_adaptive)
print("                    ")
print(results_n_change_sourn)
print("                    ")
print(results_n_change_sourn_adaptive)
print("                    ")

print("Mean hitting time non adaptive: ", np.mean(results_n_change[:,1]),
      "adaptive: ", np.mean(results_n_change_adaptive[:,1]),
      "permutation test: ", np.mean(results_n_change_perm[:,1]),
      "permutation test adaptive item selection", np.mean(results_n_change_perm_adaptive[:,1]),
      "second order urnings: ", np.mean(results_n_change_sourn[:,1]),
      "second order urnings adaptive item selection", np.mean(results_n_change_sourn_adaptive[:,1]))
print("Mean MSE  non adaptive: ", np.mean(results_n_change[:,0]),
       "adaptive: ", np.mean(results_n_change_adaptive[:,0]),
       "permutation test: ", np.mean(results_n_change_perm[:,0]),
       "permutation test adaptive item selection", np.mean(results_n_change_perm_adaptive[:,0]),
       "second order urnings: ", np.mean(results_n_change_sourn[:,0]),
      "second order urnings adaptive item selection", np.mean(results_n_change_sourn_adaptive[:,0]))
print("Mean coverage  non adaptive: ", np.mean(results_n_change[:,2]),
      "adaptive: ", np.mean(results_n_change_adaptive[:,2]),
      "permutation test: ", np.mean(results_n_change_perm[:,2]),
      "permutation testadaptive item selection", np.mean(results_n_change_perm_adaptive[:,2]),
      "second order urnings: ", np.mean(results_n_change_sourn[:,2]),
      "second order urnings adaptive item selection", np.mean(results_n_change_sourn_adaptive[:,2]))


#create latex tablea 
params_strings = []
for i in range(parameter_space_limit.shape[0]):
    params_strings.append(str(parameter_space_limit[i,0]) + "/" +str(np.around(parameter_space_limit[i,1])))
    print(params_strings)

table1 = pd.concat([pd.DataFrame(params_strings), pd.DataFrame(results_n_change), pd.DataFrame(results_n_change_adaptive)], axis = 1)

print(table1.to_latex(index=False))

parameter_space_limit_aurnsize = np.zeros((16,3))
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes_start = [4, 8]
player_urn_sizes_end = [32, 64]

counter = 0
for i in true_values:
    for j in player_urn_sizes_start:
        for k in player_urn_sizes_end:
            parameter_space_limit_aurnsize[counter,:] = i,j,k
            counter += 1

params_strings = []
for i in range(parameter_space_limit_aurnsize.shape[0]):
    params_strings.append(str(parameter_space_limit_aurnsize[i,0]) + "/" +str(np.around(parameter_space_limit_aurnsize[i,1])) + "/" + str(parameter_space_limit_aurnsize[i,2]))
    

table2 = pd.concat([pd.DataFrame(params_strings), pd.DataFrame(results_n_change_perm), pd.DataFrame(results_n_change_perm_adaptive)], axis = 1)
print(table2.to_latex(index=False))

table3 = pd.concat([pd.DataFrame(params_strings), pd.DataFrame(results_n_change_sourn), pd.DataFrame(results_n_change_sourn_adaptive)], axis = 1)
print(table3.to_latex(index=False))


########################################################################################
# Analysis with change
########################################################################################

results_change = np.zeros((20, 2))
change = [0.0001, 0.0005, 0.001, 0.002]
player_urn_sizes = [4, 8, 16, 32, 64]
player_urn_sizes_start = [4, 8]
player_urn_sizes_end = [32,64]


parameter_space_limit_change = np.zeros((len(change) * len(player_urn_sizes), 2))

counter = 0
for tv in range(len(change)):
    for pus in range(len(player_urn_sizes)):
        parameter_space_limit_change[counter, :] = change[tv], player_urn_sizes[pus]
        counter += 1

print(parameter_space_limit_change)

#NONADAPTIVE URNINGS WITH CHANGE#################################################################
#loading results of the simulation without adaptivity and change
urnings_results_limit = np.load("urnings_array_limit_change.npy")

for pars in range(len(change) * len(player_urn_sizes)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, 0.5, change=parameter_space_limit_change[pars, 0])
    results_change[pars,1] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, 0.5, parameter_space_limit_change[pars, 0])

    #save this to the results
    results_change[pars,0] = MSE

print(results_change)
#ADAPTIVE URNINGS WITH CHANGE#################################################################
results_change_adaptive = np.zeros((20, 2))
#loading results of the simulation without adaptivity and change
urnings_results_limit = np.load("urnings_array_limit_change_adaptive.npy")

for pars in range(len(change) * len(player_urn_sizes)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, 0.5, change=parameter_space_limit_change[pars, 0])
    results_change_adaptive[pars,1] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means,0.5, parameter_space_limit_change[pars, 0])

    #save this to the results
    results_change_adaptive[pars,0] = MSE

print(results_change_adaptive)

#NONADAPTIVE URNINGS WITH Permutation#################################################################
#loading results of the simulation without adaptivity and change
parameter_space_aurnsize_change = np.zeros((len(change) * len(player_urn_sizes_start) * len(player_urn_sizes_end), 3))

counter = 0
for tv in range(len(change)):
    for pus in range(len(player_urn_sizes_start)):
        for pue in range(len(player_urn_sizes_end)):
            parameter_space_aurnsize_change[counter, :] = change[tv], player_urn_sizes_start[pus], player_urn_sizes_end[pue]
            counter += 1

results_change_perm = np.zeros((16,2))
urnings_results_limit = np.load("urnings_array_perm_change.npy")

for pars in range(len(change) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, 0.5, change=parameter_space_aurnsize_change[pars, 0])
    results_change_perm[pars,1] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, 0.5, parameter_space_aurnsize_change[pars, 0])

    #save this to the results
    results_change_perm[pars,0] = MSE

print(results_change_perm)
#ADAPTIVE URNINGS WITH Permutation#################################################################
#loading results of the simulation without adaptivity and change
results_change_perm_adaptive = np.zeros((16,2))
urnings_results_limit = np.load("urnings_array_perm_adaptive_change.npy")

for pars in range(len(change) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, 0.5, change=parameter_space_aurnsize_change[pars, 0])
    results_change_perm_adaptive[pars,1] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, 0.5, parameter_space_aurnsize_change[pars, 0])

    #save this to the results
    results_change_perm_adaptive[pars,0] = MSE

print(results_change_perm_adaptive)
#NONADAPTIVE URNINGS WITH Second order urnings#################################################################
#loading results of the simulation without adaptivity and change
results_change_sourn= np.zeros((16,2))
urnings_results_limit = np.load("urnings_array_sourn_change.npy")

for pars in range(len(change) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, 0.5, change=parameter_space_aurnsize_change[pars, 0])
    results_change_sourn[pars,1] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, 0.5, parameter_space_aurnsize_change[pars, 0])

    #save this to the results
    results_change_sourn[pars,0] = MSE

print(results_change_sourn)
#ADAPTIVE URNINGS WITH CHANGE and SOURN#################################################################
#loading results of the simulation without adaptivity and change
results_change_sourn_adaptive= np.zeros((16,2))
urnings_results_limit = np.load("urnings_array_sourn_adaptive_change.npy")

for pars in range(len(change) * len(player_urn_sizes_start) * len(player_urn_sizes_end)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0)
    coverage = util.coverage(urnings_matrix, 0.5, change=parameter_space_aurnsize_change[pars, 0])
    results_change_sourn_adaptive[pars,1] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = util.MSE(col_means, 0.5, parameter_space_aurnsize_change[pars, 0])

    #save this to the results
    results_change_sourn_adaptive[pars,0] = MSE

print(results_change_sourn)

print("Mean MSE  non adaptive: ", np.mean(results_change[:,0]),
       "adaptive: ", np.mean(results_change_adaptive[:,0]),
       "permutation test: ", np.mean(results_change_perm[:,0]),
       "permutation test adaptive item selection", np.mean(results_change_perm_adaptive[:,0]),
       "second order urnings: ", np.mean(results_change_sourn[:,0]),
      "second order urnings adaptive item selection", np.mean(results_change_sourn_adaptive[:,0]))
print("Mean coverage  non adaptive: ", np.mean(results_change[:,1]),
      "adaptive: ", np.mean(results_change_adaptive[:,1]),
      "permutation test: ", np.mean(results_change_perm[:,1]),
      "permutation testadaptive item selection", np.mean(results_change_perm_adaptive[:,1]),
      "second order urnings: ", np.mean(results_change_sourn[:,1]),
      "second order urnings adaptive item selection", np.mean(results_change_sourn_adaptive[:,1]))

#create latex tablea 
params_strings_change = []
for i in range(parameter_space_limit_change.shape[0]):
    params_strings_change.append(str(parameter_space_limit_change[i,0]) + "/" +str(np.around(parameter_space_limit_change[i,1])))


table_change = pd.concat([pd.DataFrame(params_strings_change), pd.DataFrame(results_change[:,0]), pd.DataFrame(results_change_adaptive[:,0])], axis = 1)
print(table_change.to_latex(index=False))

params_strings_change_aurnsize = []
for i in range(parameter_space_aurnsize_change.shape[0]):
    params_strings_change_aurnsize.append(str(parameter_space_aurnsize_change[i,0]) + "/" +str(parameter_space_aurnsize_change[i,1]) + "/" + str(parameter_space_aurnsize_change[i,2]))


table_change_aurnsize = pd.concat([pd.DataFrame(params_strings_change_aurnsize),
                          pd.DataFrame(results_change_perm[:,0]),
                          pd.DataFrame(results_change_perm_adaptive[:,0]),
                          pd.DataFrame(results_change_sourn[:,0]),
                          pd.DataFrame(results_change_sourn_adaptive[:,0])], axis = 1)
print(table_change_aurnsize.to_latex(index=False))