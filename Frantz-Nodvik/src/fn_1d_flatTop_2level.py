import numpy as np
import matplotlib.pyplot as plt
import os


#################
# This file explores the 1D Frantz-Nodvik Equations for a two
# level system given a flat top pulse.
# This shows how the pulse responds after a gain medium.
# Also assuming a constant population inversion through out the entire medium.
#################


def calculate_photon_density(n0, sigma, delta0, eta, tau, c, x, t):
    # Parameters: sigma - absorption cross section
    #             delta0 - const inversion
    #             eta - total number of photons/unit area
    #             tau - pulse width
    #             c - speed of light in vacuum
    
    # Variables: x - spatial
    #            t - time
    
    n = n0/(1-(1-np.exp(-sigma*delta0*x))*np.exp(-2*sigma*eta*(t-x/c)/tau))
    
    return n

def main(save_path):
    # define standard parameters (from paper)
    eta = 4*10**18 #photons/cm^2
    L = 10 # 10 cm long medium
    tau = 6*10**(-9) #6ns long pulse...pretty big but going with paper
    delta0 = 8*10**8 #c 1/cm^3 ompletey excited initial population density 
    sigma = 2.5*10**(-20) # cm^2 cross section for ruby at room temp
    c = 30 #cm/ns
    n0 = 1 #normalize
    
    #Calculate time profile of pulse at x = 0 (beginning of medium)
    x = 0
    times_x0 = np.arange(0+x/c, tau+x/c, tau/100)
    n_x0 = np.zeros(len(times_x0))
    for i in range(len(n_x0)):
        t = times_x0[i]
        n_x0[i] = calculate_photon_density(n0, sigma, delta0, eta, tau, c, x, t)
    plt.plot(times_x0, n_x0,'-o')
    plt.show()
    
    #Calculate time profile of pulse at x = 10 cm (beginning of medium)
    x = L
    times_x10 = np.arange(0+x/c, tau+x/c, tau/100)
    n_x10 = np.zeros(len(times_x10))
    for i in range(len(n_x10)):
        t = times_x10[i]
        n_x10[i] = calculate_photon_density(n0, sigma, delta0, eta, tau, c, x, t)
    plt.plot(times_x10, n_x10, '-o')
    plt.show()
    
    #Calculate time profile of pulse at x = 10 cm (beginning of medium)
    x = L/2
    times_x5 = np.arange(0+x/c, tau+x/c, tau/100)
    n_x5 = np.zeros(len(times_x5))
    for i in range(len(n_x10)):
        t = times_x5[i]
        n_x5[i] = calculate_photon_density(n0, sigma, delta0, eta, tau, c, x, t)
    plt.plot(times_x5, n_x5, '-o')
    plt.show()
    
    
    #end_time = tau+L/c
    #times_x5 = np.arange(0, end_time, tau/100)
    #n_x5 = np.zeros(len(times_x5))
    #for i in range(len(n_x10)):
     #   t = times_x5[i]
    #    n_x5[i] = calculate_photon_density(n0, sigma, delta0, eta, tau, c, x, t)
    #plt.plot(times_x5, n_x5, '-o')
    #plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    path = os.getcwd()
    main(save_path='path')