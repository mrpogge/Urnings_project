# Minimial Requirement to Build an Adaptive Learning System Based on Urnings Algorithm

## Project summary
In educational measurement, a new algorithm (Urnings) was developed that allows to track student ability and item difficulty in Adaptive Learning System (ALS). The Urnings algorithm is currently the only algorithm for ALS that provides reliable standard error estimates. However, this is only applicable when the model parameters converge. Currently there is no method that indicate whether the parameters are stable within a time window. Therefore the first aim of this project, is to construct a method indicating convergence, and test it using simulation. 

Second, we will use this method to optimise the current algorithm. Using our convergence statistics and the student behaviour we will allow the algorithm to optimise the speed to which it can adapt to new data. This will result in more reliable inferences about the model parameters and in a more adaptive ALS, therefore a better experience for the students. 

## Code base

The code base contains the necessarry classes for performing simulation with different versions of the Urnings algorithm. These are Urnings 1 (Bolsinova et al 2021), Urnings 2 (Bolsinova et al, 2022), and the afforementioned algorithms operate with adaptive and nonadaptive item selection. Currently the adaptivity is based on Hofmann et al (2021, normal quantiles method).

ADD A VIGNETTE LATER

## Goal 

1. Convergence diagnostics (Bayesian convergence diagnostics online?)
2. Change-point detection
3. Adaptive Urn-size algorithm based on second order tracking 
