# implementation of particle independent metropolis-hastings sampler
import numpy as np
import math
import matplotlib.pyplot as plt
import sample_from_parameter_posterior as sfpp
import particle_sampler as ps

def simulation(T,v,w):
    # simulate and state and observation sequence
    m0 = 0
    P0 = 5
    variance_x = v
    variance_y = w
    sigma_x = math.sqrt(variance_x)
    sigma_y = math.sqrt(variance_y)
    X = np.zeros(T+1)
    Y = np.zeros(T+1)
    Y[0] = 0
    X[0] = np.random.normal(m0,P0)
    for i in range(T):
        vn = np.random.normal(0,sigma_x)
        X[i+1] = X[i] / 2 + 25 * X[i] / (1 + X[i]**2) + 8 * math.cos(1.2 * (i + 1)) + vn
    for n in range(T):
        wn  =np.random.normal(0,sigma_y)
        Y[n+1] = X[n+1]**2 / 20 + wn
    return X, Y

def particle_independent_metropolis_hastings(num,T,N,v,w):
    # Generate simulated data
    X, Y = simulation(T,v,w)

    # initialization of parameters and states
    variance_x = np.zeros(num+1)
    variance_y = np.zeros(num+1)
    variance_x[0] = 10
    variance_y[0] = 10
    sampled_X = np.zeros((num+1,T+1))
    prior_b = 0.01
    prior_a = 0.01

    # initialization of marginal likelihood
    marginal_likelihood = np.zeros(num+1)
    X_sample, log_ml = ps.particle_sample(T,N,variance_x[0],variance_y[0],X,Y)
    marginal_likelihood[0] = log_ml
    sampled_X[0] = X_sample
    reject = 0

    # PIMH recursion
    # first sample from p(theta|X,Y), then sample from p(X|Y,theta) using particle filter
    for i in range(1,num+1):
        new_v, new_w = sfpp.theta_sampler(sampled_X[i-1],Y,prior_b,prior_a)
        print(i,new_w)
        variance_x[i] = new_v
        variance_y[i] = new_w
        new_X, new_ml = ps.particle_sample(T,N,new_v,new_w,X,Y)
        log_acceptance_probability = new_ml - marginal_likelihood[i-1]
        if log_acceptance_probability < -500:
            acceptance_probability = 0
        elif log_acceptance_probability > 500:
            acceptance_probability = 2
        else:
            acceptance_probability = math.exp(log_acceptance_probability)
        u = np.random.uniform()
        if u < acceptance_probability:
            sampled_X[i] = new_X
            marginal_likelihood[i] = new_ml
        else:
            sampled_X[i] = sampled_X[i-1]
            marginal_likelihood[i] = marginal_likelihood[i-1]
    return variance_x, variance_y, sampled_X, reject

def test():
    T = 100
    N = 200
    v = 10
    w = 2.5
    num = 2000
    variance_x, variance_y, sampled_X, reject = particle_independent_metropolis_hastings(num,T,N,v,w)

    plt.scatter(variance_x,variance_y)
    plt.show()
    plt.hist(variance_y,bins=200)
    plt.show()

test()







