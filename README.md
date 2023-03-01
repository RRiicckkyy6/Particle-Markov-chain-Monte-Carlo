# Particle-Markov-chain-Monte-Carlo
Python implementation of Particle Markov chain Monte Carlo (PMCMC) algorithms  
Include codes that generate synthetic data and simple visualization code that plot the results

Three specific versions of PMCMC algorithms are implmented:  
1. Particle independent Metropolis Hastings (PIMH)  pimh.py  
2. Particle marginal Metropolis Hastings (PMMH) with Gaussian random walk proposal  pmmh.py  
3. Particle Gibbs (PG)  pg.py  

The sequential Monte Carlo (SMC) / particle filter components use the prior / state transition / system dynamics proposal and the most basic resampling scheme, multinomial resampling
