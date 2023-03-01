import numpy as np
import matplotlib.pyplot as plt
import math

# Sample from p(theta|X,Y) in a parametric state space model
# Assumes additive Gaussian noise for both state transition and observation
# version 1: only allow scalar states and observations

def state_transition(x0,T):
    # input: scalar x0 initial state, scalar T length
    # output: python array X_0_T
    # To be modified according to problem specification
    # implemented here is the popular nonlinear SSM in the sequential Monte Carlo community
    X_0_T = np.zeros(T+1)
    X_0_T[0] = x0
    variance = 10
    sigma_v = math.sqrt(variance)
    for n in range(T):
        vn = np.random.normal(0,sigma_v)
        xn = X_0_T[n] / 2 + 25 * X_0_T[n] / (1 + X_0_T[n]**2) + 8 * math.cos(1.2 * (n + 1)) + vn
        X_0_T[n+1] = xn
    return X_0_T    

def observation(X):
    # input: numpy array X_0_T states
    # output: numpy array Y_1_T observations
    T = X.size
    Y_0_T = np.zeros(T)
    variance = 10
    sigma_w = math.sqrt(variance)
    Y_0_T[0] = 0
    for n in range(T-1):
        wn = np.random.normal(0,sigma_w)
        yn = X[n+1]**2 / 20 + wn
        Y_0_T[n+1] = yn
    return Y_0_T

def generate(T):
    # generate simulation data
    # input: scalar T length
    # ouput: numpy arrays as simulated X and Y
    mean0 = 0
    variance0 = 5
    sigma0 = math.sqrt(variance0)
    x0 = np.random.normal(mean0,sigma0)
    l = T
    X = state_transition(x0,l)
    Y = observation(X)
    return X, Y

def beta_calculate(X,Y,p):
    # calculate the beta parameter for the posterior inverse gamma parameter distribution
    # input: numpy array X, Y state and observation sequence
    # input: scalar p prior beta value
    # output: scalar pp_v, pp_w posterior beta value
    T = X.size
    pp_v = p
    for n in range(T-1):
        f_X = X[n] / 2 + 25 * X[n] / (1 + X[n]**2) + 8 * math.cos(1.2 * (1 + n))
        pp_v = pp_v + (X[n+1] - f_X)**2 / 2
    pp_w = p
    for n in range(T-1):
        g_X = X[n+1]**2 / 20
        pp_w = pp_w + (Y[n+1] - g_X)**2 / 2
    return pp_v, pp_w

def sample(X,Y,p_b,p_a):
    # sample from p(theta|X,Y)
    # input: X, Y numpy array, states and observations
    # input: p_b, p_a, prior parameters
    # output: a sample drawn from IG(alpha,beta)
    T = X.size - 1
    beta_v, beta_w = beta_calculate(X,Y,p_b)
    alpha_v = p_a + T / 2
    alpha_w = p_a + T / 2
    theta_v_inverse = np.random.gamma(alpha_v,1/beta_v)
    theta_v = 1 / theta_v_inverse
    theta_w_inverse = np.random.gamma(alpha_w,1/beta_w)
    theta_w = 1 / theta_w_inverse
    return theta_v, theta_w

def test():
    T = 100
    X, Y = generate(T)
    prior_beta = 0.01
    prior_alpha = 0.01
    N = 100
    sampled_v = np.zeros(N)
    sampled_w = np.zeros(N)
    for i in range(N):
        sampled_v[i], sampled_w[i] = sample(X,Y,prior_alpha,prior_beta)
    print(sampled_v)
    print(sampled_w)

    plt.hist(sampled_v,bins=50,range=(0,20))
    plt.show()

def theta_sampler(X,Y,p_b,p_a):
    theta_v, theta_w = sample(X,Y,p_b,p_a)
    return theta_v, theta_w
