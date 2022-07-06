# Warming up the Cold Start Problem: Adaptive Step Size Methods for the Urnings Algorithm

## Project summary
Adaptive learning systems aim to tailor the item selection to the needs of the student. 
Recently a new algorithm (Urnings) was developed to track student ability and item difficulty in Adaptive Learning Systems (ALS). Urnings algorithm improves upon the Elo rating system since Urnings have a known stationary distribution, which makes it possible to calculate the standard error of the estimates, and enables statistical analysis of the Urnings of the students in the system.

If a student enters such an adaptive learning system, we have no prior information about their ability, so we cannot tailor the item selection. This issue is called the cold start problem. While the Urnings algorithm provides a straightforward and flexible solution for the adaptive item selection, it has a fixed step size for parameter updating, making it susceptible to the cold start problem.

In this thesis project, we develop a version of the Urnings algorithm capable of changing the step size based on the student's behaviour in the system. In contrast to the standard solutions in the Elo-based systems, we are not focusing on the duration of the students' practising periods but on the direction and rate of change of the ratings. 

We also introduce a framework to create different adaptive step size algorithms and two possible solutions. We investigate these solutions using simulation.

## Code base

The code base contains the necessarry classes for performing simulation with different versions of the Urnings algorithm. These are Urnings 1 (Bolsinova et al 2021), Urnings 2 (Bolsinova et al, 2022), and the afforementioned algorithms operate with adaptive and nonadaptive item selection. Currently the adaptivity is based on Hofmann et al (2021, normal quantiles method).

The analysis presented in the article can be found in the plots.ipynb and plots2.ipynb notebooks. The other analysis presented here were to validate the software itself.




