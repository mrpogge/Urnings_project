import numpy as np
from pytest import param
import main_urnings as mu
import matplotlib.pyplot as plt

#creating containers
results_n_change = np.zeros((20, 5))

#recreating parameter space

#changing parameters
true_values = [0.6, 0.7, 0.8, 0.9]
player_urn_sizes = [6, 10, 14, 18, 50]

#fixed parameters
n_player = 5000
n_items = 1000
item_urn_sizes = 1000
n_sim = 300

parameter_space_limit = np.zeros((len(true_values) * len(player_urn_sizes), 2))

counter = 0
for tv in range(len(true_values)):
    for pus in range(len(player_urn_sizes)):
        parameter_space_limit[counter, :] = true_values[tv], player_urn_sizes[pus]
        counter += 1


results_n_change[:,3] = parameter_space_limit[:,0]
results_n_change[:,4] = parameter_space_limit[:,1]
#NONADAPTIVE URNINGS WITH NO CHANGE
#loading results of the simulation without adaptivity and change
urnings_results_limit = np.load("urnings_array_limit.npy", allow_pickle = True)

for pars in range(len(true_values) * len(player_urn_sizes)):
    #subsetting for each parameter
    urnings_matrix = urnings_results_limit[:,:,pars]

    #calculating mean, standard error and CI95 per iteration 
    col_means = np.mean(urnings_matrix, axis = 0) / parameter_space_limit[pars, 1]
    col_var = np.std(urnings_matrix, axis = 0) / np.sqrt(n_player)

    col_lower = np.zeros(len(col_means))
    col_upper = np.zeros(len(col_means))
    coverage = 0
    for i in range(len(col_means)):
        confint = np.percentile(urnings_matrix[:,i] / parameter_space_limit[pars, 1], [5, 95])
        col_lower[i] = confint[0]
        col_upper[i] = confint[1]
        #calculating coverage
        if confint[0] < parameter_space_limit[pars, 0] < confint[1]:
            coverage += 1

    coverage = coverage / n_sim

    results_n_change[pars,2] = coverage

    ######################## MEAN SQUARED ERROR #########################################
    MSE = np.sum((parameter_space_limit[pars, 0] - col_means) ** 2) / len(col_means)
    #print(MSE)
    #save this to the results

    results_n_change[pars,0] = MSE

    ######################## HITTING TIME #########################################

    SE_true_value = np.sqrt(parameter_space_limit[pars, 1] * parameter_space_limit[pars, 0] * (1 - parameter_space_limit[pars, 0]) / n_player)

    #print out the iteration when the mean was in this error radius
    error_radius_logical = []
    for clm in col_means:
        error_radius_logical.append((parameter_space_limit[pars, 0] - SE_true_value) < clm < (parameter_space_limit[pars, 0] + SE_true_value))
    
    trues_vec = np.where(error_radius_logical)[0]
    if len(trues_vec) > 0:
        hitting_time = np.min(trues_vec)
    else:
        hitting_time = 999

    results_n_change[pars,1] = hitting_time

    ######################## COVERAGE #########################################

    nominal_coverage_sim = np.random.binomial(parameter_space_limit[pars, 1], parameter_space_limit[pars, 0], (n_player, n_sim)) / parameter_space_limit[pars, 1]
    
    nominal_coverage = 0
    for i in range(n_sim):
        confint_nc = np.percentile(nominal_coverage_sim[:,i], [5, 95])
        if confint_nc[0] < parameter_space_limit[pars, 0] < confint_nc[1]:
            nominal_coverage += 1
    
    nominal_coverage = nominal_coverage / n_sim

# #ADAPTIVE URNINGS WITH NO CHANGE
# urnings_results_limit = np.load("urnings_array_limit_adaptive.npy", allow_pickle = True)
# for pars in range(len(true_values) * len(player_urn_sizes)):
#     #subsetting for each parameter
#     urnings_matrix = urnings_results_limit[:,:,pars]

    
#     #calculating mean, standard error and CI95 per iteration 
#     col_means = np.mean(urnings_matrix, axis = 0) / parameter_space_limit[pars, 1]
#     col_var = np.std(urnings_matrix, axis = 0) / np.sqrt(n_player)

#     col_lower = np.zeros(len(col_means))
#     col_upper = np.zeros(len(col_means))
#     coverage = 0
#     for i in range(len(col_means)):
#         confint = np.percentile(urnings_matrix[:,i] / parameter_space_limit[pars, 1], [5, 95])
#         col_lower[i] = confint[0]
#         col_upper[i] = confint[1]
#         #calculating coverage
#         if confint[0] < parameter_space_limit[pars, 0] < confint[1]:
#             coverage += 1

#     coverage = coverage / n_sim

#     results_n_change[20 + pars, 2] = coverage

#     ######################## MEAN SQUARED ERROR #########################################
#     MSE = np.sum((parameter_space_limit[pars, 0] - col_means) ** 2) / len(col_means)
#     #print(MSE)
#     #save this to the results

#     results_n_change[20 + pars, 0] = MSE

#     ######################## HITTING TIME #########################################

#     SE_true_value = np.sqrt(parameter_space_limit[pars, 1] * parameter_space_limit[pars, 0] * (1 - parameter_space_limit[pars, 0]) / n_player)

#     #print out the iteration when the mean was in this error radius
#     error_radius_logical = []
#     for clm in col_means:
#         error_radius_logical.append((parameter_space_limit[pars, 0] - SE_true_value) < clm < (parameter_space_limit[pars, 0] + SE_true_value))
    
#     trues_vec = np.where(error_radius_logical)[0]
#     if len(trues_vec) > 0:
#         hitting_time = np.min(trues_vec)
#     else:
#         hitting_time = 999

#     #print(hitting_time)
#     results_n_change[20 + pars, 1] = hitting_time

#     ######################## COVERAGE #########################################

#     nominal_coverage_sim = np.random.binomial(parameter_space_limit[pars, 1], parameter_space_limit[pars, 0], (n_player, n_sim)) / parameter_space_limit[pars, 1]
    
#     nominal_coverage = 0
#     for i in range(n_sim):
#         confint_nc = np.percentile(nominal_coverage_sim[:,i], [5, 95])
#         if confint_nc[0] < parameter_space_limit[pars, 0] < confint_nc[1]:
#             nominal_coverage += 1
    
#     nominal_coverage = nominal_coverage / n_sim


print(parameter_space_limit)
print(results_n_change)