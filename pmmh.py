# implementation of particle marginal metropolis hastings
import numpy as np
import matplotlib.pyplot as plt
import math
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

def sample_proposal(v,w,v_proposal,w_proposal):
    new_v = np.random.normal(v,v_proposal)
    new_w = np.random.normal(w,w_proposal)
    return new_v, new_w

def acceptance_calculator(v_new,w_new,v_old,w_old,prior_a,prior_b):
    p_v_new = prior_b**prior_a / math.gamma(prior_a) / v_new**(1+prior_a) * math.exp(-prior_b / v_new)
    p_w_new = prior_b**prior_a / math.gamma(prior_a) / w_new**(1+prior_a) * math.exp(-prior_b / w_new)
    p_v_old = prior_b**prior_a / math.gamma(prior_a) / v_old**(1+prior_a) * math.exp(-prior_b / v_old)
    p_w_old = prior_b**prior_a / math.gamma(prior_a) / w_old**(1+prior_a) * math.exp(-prior_b / w_old)

    temp = np.log(p_v_new) + np.log(p_w_new) - np.log(p_v_old) - np.log(p_w_old)
    return temp

def particle_marginal_metropolis_hastings(num,T,N,v,w):

    # generate simulation data
    X, Y = simulation(T,v,w)

    # standard deviation for the theta proposal
    v_proposal = 0.2
    w_proposal = 0.2

    # prior parameter for theta
    prior_a = 0.01
    prior_b = 0.01

    # initialization of parameters and states
    variance_x = np.zeros(num+1)
    variance_y = np.zeros(num+1)
    variance_x[0] = 10
    variance_y[0] = 10
    sampled_X = np.zeros((num+1,T+1))
    marginal_likelihood = np.zeros(num+1)
    X_sample, log_ml = ps.particle_sample(T,N,variance_x[0],variance_y[0],X,Y)
    sampled_X[0] = X_sample
    marginal_likelihood[0] = log_ml
    accept = 0

    # PMMH iteration
    # First sample new theta from the proposal, then sample X from the SMC approximation
    for i in range(1,num+1):
        new_v, new_w = sample_proposal(variance_x[i-1],variance_y[i-1],v_proposal,w_proposal)
        while new_v < 0 or new_w < 0:
            new_v, new_w = sample_proposal(variance_x[i-1],variance_y[i-1],v_proposal,w_proposal)
        X_sample, log_ml = ps.particle_sample(T,N,new_v,new_w,X,Y)
        log_acc = log_ml - marginal_likelihood[i-1]
        log_acc = log_acc + acceptance_calculator(new_v,new_w,variance_x[i-1],variance_y[i-1],prior_a,prior_b)
        if log_acc < -500:
            acceptance_probability = 0
        elif log_acc > 500:
            acceptance_probability = 2
        else:
            acceptance_probability = math.exp(log_acc)
        u = np.random.uniform()
        if u < acceptance_probability:
            sampled_X[i] = X_sample
            variance_x[i] = new_v
            variance_y[i] = new_w
            marginal_likelihood[i] = log_ml
            print(i,new_w,'accept')
            accept = accept + 1
        else:
            sampled_X[i] = sampled_X[i-1]
            variance_x[i] = variance_x[i-1]
            variance_y[i] = variance_y[i-1]
            marginal_likelihood[i] = marginal_likelihood[i-1]
            print(i,'reject')
    print("rate",accept/num)
    return variance_x, variance_y, sampled_X
        
def test():
    T = 100
    N = 200
    v = 10
    w = 5
    num = 50000
    variance_x, variance_y, sampled_X = particle_marginal_metropolis_hastings(num,T,N,v,w)

    plt.hist(variance_y,100)
    plt.show()
    TT = np.arange(num+1)
    plt.plot(TT[1:],variance_y[1:])
    plt.show()


test()

