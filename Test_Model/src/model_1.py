import numpy as np
import sys
import os
import re
import math
import matplotlib.pyplot as plt

'''
This code is the initial version of the toy model laser modeling. 
Based on work from Springer, Alexeev, et al form Numerical simulation of short laser pulse amplification
'''

class Amplifier:
    
    '''
    (Much of the description taken from the paper at beginning of code)
    This class sets the parameters for the simulation in the amplifier.
    First a proper mesh must be constructed. This requires a mesh spacing parameter
    h where h = L/(nx - 1), L = medium length, nx = number of discrete points of simulation domain.
    This also requires a time step, tau, given by h/c where c is the speed of light. 
    The mesh is separated into point and cell data. The point data is related to prop photons in crystal
    and entry and exit of photons can be calculated between cells and on end faces of crystal yielding
    simulation domain boundary of n^k_(i-1) and n^(k+1)_i. Time and space dep pop inversion delta^k_(i-1)
    is characterized as cell data. i is spacial step, k is time step.
    
    This is solving for (1) dn/dt + c dn/dx = sigma*c*n*delta
    and (2) ddelta/dt = -2*sigma*c*n*delta
    '''

    
    def __init__(self, discrete_points, medium_length, light_speed_medium, stim_emission_crossec):
        self.nx = discrete_points
        self.L = medium_length
        self.h = self.L/(self.nx-1)
        self.c = light_speed_medium
        self.sigma = stim_emission_crossec
        self.tau = self.h/self.c
        
    def forward_euler_approach_step(self, n_prevSpace_curTime, delta_prevSpace_curTime, option=1):
        '''
        Propagates photons from grid point i-1 one step further in space and time. Calculates amplification
        based on inversion in cell i-1. Updated photon density (n^(k+1)_i). Inversion in cell i-1
        updated to consider photons that have passed (delta^(k+1)_(i-1))

        Parameters
        ----------
        n_prevSpace_curTime : TYPE
            DESCRIPTION.
        delta_prevSpace_curTime : TYPE
            DESCRIPTION.
        option : TYPE

        Returns
        -------
        n_curSpace_nextTime : TYPE
            DESCRIPTION.
        delta_prevSpace_nextTime : TYPE
            DESCRIPTION.

        '''
        if (option==1):
            n_curSpace_nextTime = n_prevSpace_curTime*(1+self.sigma*self.h*delta_prevSpace_curTime)
            delta_prevSpace_nextTime = delta_prevSpace_curTime * (1-2*self.sigma*self.h*n_prevSpace_curTime)
        elif (option==2):
            n_curSpace_nextTime = n_prevSpace_curTime/(1-self.sigma*self.h*delta_prevSpace_curTime)
            delta_prevSpace_nextTime = delta_prevSpace_curTime / (1+self.sigma*self.h*n_curSpace_nextTime)
        else :
            n_curSpace_nextTime = n_prevSpace_curTime*(1+self.sigma*self.h*delta_prevSpace_curTime)
            delta_prevSpace_nextTime = delta_prevSpace_curTime * (1-2*self.sigma*self.h*n_prevSpace_curTime)
            print("Invalid option selected, explicit approach being used.")
        
        return n_curSpace_nextTime, delta_prevSpace_nextTime
    
    def prediction_correction_approach_step(self, n_prevSpace_curTime, delta_prevSpace_curTime):
        '''
        This function uses the prediction-correction approach which utilizes Heun's method to build
        on the Euler approach (using the results from the explicit euler calculation as predicted values).
        A correction is then applied on top of the predicted values
        Parameters
        ----------
        n_prevSpace_curTime : TYPE
            DESCRIPTION.
        delta_prevSpace_curTime : TYPE
            DESCRIPTION.

        Returns
        -------
        n_curSpace_nextTime : TYPE
            DESCRIPTION.
        delta_prevSpace_nextTime : TYPE
            DESCRIPTION.

        '''
        
     
        n_curSpace_nextTime_predicted, delta_prevSpace_nextTime_predicted =\
            self.forward_euler_approach_step( n_prevSpace_curTime, delta_prevSpace_curTime, option=1)
            
        n_curSpace_nextTime = 0.5*n_prevSpace_curTime+0.5*\
            (n_curSpace_nextTime_predicted+self.h*self.sigma*n_curSpace_nextTime_predicted*\
             delta_prevSpace_nextTime_predicted)
                
        delta_prevSpace_nextTime = 0.5*delta_prevSpace_curTime+0.5*delta_prevSpace_nextTime_predicted*\
            (1+self.h*self.sigma*n_curSpace_nextTime_predicted)
            
        return n_curSpace_nextTime,delta_prevSpace_nextTime
    
    def FN_analytic(self, n0, beta, T, delta0, t, x):
        '''
        This anlaytic expression for the transport equation (two level system) for amplified lorentzian 
        pulse. Describes growth of pulse in solid-state amplifier with uniform initial inversion in medium
        Originally developed by Frantz and Nodvik
        

        Parameters
        ----------
        n0 : TYPE
            initial photon density.
        beta : TYPE
            pulse duration.
        T : TYPE
            half width.
        delta0 : TYPE
            initial population inversion.
        t : TYPE
            current time.
        x : TYPE
            current position.

        Returns
        -------
        n_x_t : TYPE
            current photon density.

        '''
        n_x_t = n0*beta*T/np.pi * 1/((t-x/self.c)**2 + T**2) * (1-(1-np.exp(-self.sigma*delta0*x))*\
                                                   np.exp(-self.sigma*n0*self.c*beta*\
                                                          (1+2/np.pi*np.arctan((t-x/self.c)/T))))**(-1)
        return n_x_t
    
    
    def l2_norm_calculation(self, approx_solution, analytical_solution, num_grid_points):
        return np.sqrt(1/num_grid_points*np.sum(((approx_solution-analytical_solution)/analytical_solution)**2,\
                                                keepdims=True))
    def max_norm_calculation(self, approx_solution, analytical_solution):
        return np.max((approx_solution-analytical_solution)/analytical_solution)
    
    
def main():
    
    #test functions, define parameters
    n0 = 3.92e17 # 1/cm^3 , initial photon density
    delta0 = 1.8e18 # 1/cm^3, initial population inversion
    sigma = 2.5e-20 # cm^2, emission cross section
    beta = 1e-9 # s, pulse duration
    T = 15e-12 # s, pulse half width
    L = 10 # cm, length of crystal
    c = 1.7e10 # cm/s, speed of light in medium
    
    amp1 = Amplifier(30, L, c, sigma)
    t_list = np.linspace(0,2e-9,num=30)
    x_list = np.linspace(0,L,num=30)
    solution_analytic_list = np.zeros((len(x_list), len(t_list)))
    ii = 0
    jj = 0
    fig, axs = plt.subplots()
    for t in t_list:
        ii = 0
        for x in x_list:
            solution_analytic_list[ii,jj]=amp1.FN_analytic(n0, beta, T, delta0, t, x)
            ii+=1
        #if (t//10==0):
        #    axs.plot(x_list, solution_list[:,jj], label="t="+str(t*1e9)+"ns")
        #else:
        #    axs.plot(x_list, solution_list[:,jj])
        axs.plot(x_list, solution_analytic_list[:,jj])
        jj+=1
    axs.set_title('Photon Density')
    axs.set_xlabel('x (cm)')
    axs.set_ylabel('photon density (1/cm^3)')
    axs.legend()
    
    fig,axs = plt.subplots()
    for ii in range(0,2):
        axs.plot(t_list[0:len(t_list)//3]*10**9, solution_analytic_list[ii,0:len(t_list)//3])
    axs.set_title('Photon Density')
    axs.set_xlabel('t (ns)')
    axs.set_ylabel('photon density (1/cm^3)')
    axs.legend()
    plt.show()
    
    
    solution_pc_list = np.zeros((len(x_list), len(t_list), 2))
    solution_pc_list[:,0,0] = n0
    solution_pc_list[:,:,1] = delta0
    solution_explicit_list = np.zeros((len(x_list), len(t_list),2))
    solution_explicit_list[:,0,0] = n0
    solution_explicit_list[:,0,1] = delta0
    solution_explicit_list[0,:,0] = n0
    solution_explicit_list[0,:,1] = delta0
    solution_implicit_list = np.zeros((len(x_list), len(t_list),2))
    
    
    while(True):
        for jj in range(1, len(t_list)):
            for ii in range(1,len(x_list)):
                solution_explicit_list[ii,jj,0],solution_explicit_list[ii-1,jj,1] =\
                    amp1.forward_euler_approach_step(solution_explicit_list[ii-1,jj-1,0], solution_explicit_list[ii-1,jj-1,1], option=1)
            
        
       
            
    
    return
    
    
if __name__ == "__main__":
    main()
    
                
                                                                                                      
        
        
        
        
