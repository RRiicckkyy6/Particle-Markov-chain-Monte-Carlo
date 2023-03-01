# implementation of particle Gibbs Metropolis Hastings sampler
import numpy as np
import matplotlib.pyplot as plt
import math
import sample_from_parameter_posterior as sfpp
import conditional_particle_sampler_initial as cpsi
import conditional_particle_sampler as cps

def simulation(T,v,w):
    # simulate state and observation sequence
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
        wn = np.random.normal(0,sigma_y)
        Y[n+1] = X[n+1]**2 / 20 + wn
    return X, Y

def particle_gibbs_sampler(num,T,N,v,w):

    # generate simulation data
    X, Y = simulation(T,v,w)

    # prior parameter for theta
    prior_a = 0.01
    prior_b = 0.01

    # initialization of parameters, states, and ancestral lineage
    variance_x = np.zeros(num+1)
    variance_y = np.zeros(num+1)
    variance_x[0] = 10
    variance_y[0] = 10
    sampled_X = np.zeros((num+1,T+1))
    sampled_B = np.zeros((num+1,T+1))
    sample_X, sample_B = cpsi.particle_sample(T,N,variance_x[0],variance_y[0],X,Y)
    sampled_X[0] = sample_X
    sampled_B[0] = sample_B

    # PG iterations
    # First sample p(theta|X,Y), then sample p(X|theta,Y,path,lineage)
    for i in range(1,num+1):
        new_v, new_w = sfpp.theta_sampler(sampled_X[i-1],Y,prior_b,prior_a)
        variance_x[i] = new_v
        variance_y[i] = new_w

        new_X, new_B = cps.particle_sample(T,N,new_v,new_w,X,Y,sampled_X[i-1],sampled_B[i-1])
        sampled_X[i] = new_X
        sampled_B[i] = new_B
        print(i,new_w)
    return variance_x, variance_y, sampled_X, X

def test():
    T = 200
    N = 50
    v = 2
    w = 2
    num = 50
    variance_x, variance_y, sampled_X, X = particle_gibbs_sampler(num,T,N,v,w)
    sampled_X = sampled_X.transpose()

    expected_X = np.zeros(T+1)
    for i in range(T+1):
        expected_X[i] = np.sum(sampled_X[i]) / (num + 1)
    plt.hist(variance_y,100)
    plt.show()
    TT = np.arange(num+1)
    plt.plot(TT[1:],variance_y[1:])
    plt.show()
    T = np.arange(T+1)
    plt.plot(T,X)
    plt.plot(T,expected_X)
    plt.show()

    sampled_X = sampled_X.transpose()
    for j in range(num+1):
        plt.plot(T,sampled_X[j],color='red',marker='o',linestyle='dashed',markersize='3')
    plt.plot(T,X)
    plt.show()

test()

"""color='red',marker='o',linestyle='dashed',markersize='5'"""