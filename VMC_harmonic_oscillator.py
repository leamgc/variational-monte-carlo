#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 10:41:27 2026

@author: leagallien-currier
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import uniform_filter1d



#TRIAL WAVEFUNCTION OF FORM e^-alpha r^2
def phi_trial(r, alpha):
    return np.exp(-alpha * r**2)

#PDF = phi^2 - for true pdf should have integrand normalising factor 
#however not needed here as ratio / comparing magnitudes only ever needed - avoids having to do integration 
#one of the uses of metropolis sampling/monte carlo integration as possible to sample from non-integrable pdfs. 
def rho(r, alpha):
    return phi_trial(r, alpha)**2



def metropolis(alpha, x_0, sig):
    samples = np.array([])
    samples = np.append(samples, x_0)
    total_runs = 0 
    rejections = 0

    while len(samples) < 5000: 
        total_runs += 1
        
        x_prev = samples[-1] #previous accepted sample
        
        #GENERATE CANDIDATE FOR NEXT SAMPLE -> sample from gaussian with mean = x_prev. Std of gaussian is tuned to control accept_rate
        x_cand = np.random.normal(x_prev, sig)
        
        
        #DECIDE IF X_CAND IS ACCEPTED
        if rho(x_cand, alpha) >= rho(x_prev, alpha):
            samples = np.append(samples, x_cand) # ACCEPT MOVE -> x_cand always accepted if it exists in higher prob. region of pdf
        
        else: #x_cand exists in lower prob. region
            acceptance_prob = rho(x_cand, alpha)/rho(x_prev, alpha) # CALCULATE ACCEPTANCE PROBABILITY -> probability of accepting the move to lower prob. region = p(x_cand)/p(x_prev)
            
            if random.random() <= acceptance_prob: 
                samples = np.append(samples, x_cand) #ACCEPT MOVE WITH PROBABILITY acceptance_prob
                
            else:
                rejections += 1 
                continue
            
    accept_rate = (total_runs - rejections)/total_runs #CALCULATE ACCEPTANCE RATE OF CANDIDATE POINTS -> should aim for 20-50% for best efficiency. 
    
    return samples, accept_rate



points, rate = metropolis(2, -4, 1.2)

points_clean = points[1:] #REMOVE BURN IN PHASE -> initial points do not properly sample target pdf as are affected by choice of initial point x_0.

moving_average = uniform_filter1d(points, size=50)  # same length as a

plt.plot(moving_average)

#plt.hist(points_clean, bins=30)
#plt.show()
plt.plot(points)
plt.show()




#%%

points = np.array([])
for i in range(10000):
    points = np.append(points, np.random.normal(0,1))

plt.hist(points, bins=30)
plt.show()


