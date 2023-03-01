import numpy as np
import matplotlib.pyplot as plt
import math

def sample_prior(N,m0,P0):
    # draw N samples from the prior p(x0)
    m = m0
    variance = P0
    sigma = math.sqrt(variance)
    X0 = np.zeros(N)
    for i in range(N):
        x = np.random.normal(m,sigma)
        X0[i] = x
    return X0

def sample_importance(x,index,P,N):
    # draw a sample from the importance distribution
    # The state transition distribution is used as the proposal
    # input: x, numpy array of current state (all particles)
    x_new = np.zeros(N)
    variance = P
    sigma = math.sqrt(variance)
    for i in range(N):
        m = x[i] / 2 + 25 * x[i] / (1 + x[i]**2) + 8 * math.cos(1.2 * index)
        x_new[i] = np.random.normal(m,sigma)
    return x_new

def update_weight(y,x,P,N):
    # update the weight of a single particle
    weight_new = np.zeros(N)
    for i in range(N):
        weight_new[i] = math.exp(-(y - x[i]**2 / 20)**2 / 2 / P) / math.sqrt(2 * math.pi * P)
        if weight_new[i] != weight_new[i]:
            print("fuck off",y,x[i],P)
    return weight_new

def resample(X,w,N):
    # perform resampling
    # input: 2d numpy array X, all generated sequences
    # input: numpy array w, weights, scalar N number of particles, scalar T length of state sequence
    w_normalize = w / np.sum(w)
    re = np.random.choice(N,N,p=w_normalize)
    T = X[0].size
    new_particles = np.zeros((N,T))
    for i in range(N):
        new_particles[i] = X[re[i]]
    return new_particles
    
def particle_filter(T,N,v,w,X,Y):
    # perform particle filter on simulated data
    m0 = 0
    P0 = 5
    variance_x = v
    variance_y = w
    sigma_x = math.sqrt(variance_x)
    sigma_y = math.sqrt(variance_y)
    X0 = sample_prior(N,m0,P0)
    weights = np.zeros((T+1,N))
    temp = np.ones(N)
    temp = temp / N
    weights[0] = temp
    particles = np.zeros((T+1,N))
    particles[0] = X0

    for i in range(1,T+1):
        X_n = sample_importance(particles[i-1],i,variance_x,N)
        particles[i] = X_n
        weight_n = update_weight(Y[i],X_n,variance_y,N)
        weights[i] = weight_n
        particles_resampled = resample(particles.transpose(),weight_n,N)
        particles = particles_resampled.transpose()
    
    return particles, weights

def marginal_likelihood(T,N,v,w,X,Y):
    # particle estimate of the marginal likelihood
    particles, weights = particle_filter(T,N,v,w,X,Y)
    log_ml = 0
    for i in range(T):
        ml = np.sum(weights[i+1]) / N
        lml = np.log(ml)
        log_ml = log_ml + lml
    return log_ml, particles

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

def particle_sample(T,N,v,w,X,Y):
    log_ml, particles = marginal_likelihood(T,N,v,w,X,Y)
    particles_pool = particles.transpose()
    sampler = np.random.choice(N,1)[0]
    sample_particle = particles_pool[sampler]
    return sample_particle, log_ml

def test():
    T = 100
    N = 200
    v = 10
    w = 10
    num = 5
    X, Y = simulation(T,v,w)

    ML = np.zeros(num+1)
    X_sample, log_ml = particle_sample(T,N,v,w,X,Y)
    ML[0] = log_ml
    sampled_X = np.zeros((num+1,T+1))
    sampled_X[0] = X_sample
    accept = 0
    total = 0
    for i in range(1,num+1):
        new_X, new_ml = particle_sample(T,N,v,w,X,Y)
        acceptance_probability = math.exp(new_ml - ML[i-1])
        print(acceptance_probability)
        u = np.random.uniform()
        total = total + 1
        if u < acceptance_probability:
            sampled_X[i] = new_X
            ML[i] = new_ml
            accept = accept + 1
        else:
            sampled_X[i] = sampled_X[i-1]
            ML[i] = ML[i-1]



