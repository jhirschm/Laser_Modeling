#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:35:39 2020

@author: jackhirschman
"""


import numpy as np
import sys
import os
import re
import math
import matplotlib.pyplot as plt
import scipy.fftpack
import pylab
import h5py
from scipy.interpolate import interp1d
import csv
import pandas as pd
from dazzler_class import *
import random

c = 299792458.0 # m/s
lam0 = 800e-9 #central wavelength
w0 = 2*np.pi*c/(lam0) #central frequency

pulse_dur = 9e-15 #time duration of pulse
delta_freq = 0.44/pulse_dur
delta_lam = (delta_freq*lam0**2)/c

#Amplitude params:
position = lam0 #link to dazzler position param
width = delta_lam #link to dazzler width param


model_num = 1000
hole_positions = np.zeros((model_num,))
hole_widths = np.zeros((model_num,))
hole_depths= np.zeros((model_num,))

#Phase params:
delays=np.zeros((model_num,))#4250*1e-15#4250e-15
sec_orders=np.zeros((model_num,))#22000*(1e-15)**2
third_orders=np.zeros((model_num,))#0*10*(1e-15)**3
fourth_orders=np.zeros((model_num,))#0*(1e-15)**4

num_points = 100000
time_vector = np.linspace(-5000e-15,5000e-15,num=num_points, endpoint=True)
input_E = 1*np.exp(-2*np.log(2)*(time_vector/pulse_dur)**2)
input_E.astype(complex)
input_E=input_E*np.exp(1j*w0*time_vector)

#Manual assigning of first few parameter combos
hole_positions[0:16] = 780e-9
hole_widths[2] = 20e-9
hole_depths[3] = 1
hole_positions[4:5] = 780e-9
hole_widths[4:6] = 20e-9
hole_depths[5:6] = 1

delays[7] = 4250*1e-15
sec_orders[8] = 22000*(1e-15)**2
third_orders[9] = 10*(1e-15)**3
#fourth_orders[10] = (1e-15)**4
delays[11:13] = 4250*1e-15
sec_orders[11:13] = 22000*(1e-15)**2
third_orders[11:12] = 10*(1e-15)**3
#fourth_orders[13] = (1e-15)**4

hole_positions[14:15] = 780e-9
hole_widths[14:15] = 20e-9
hole_depths[14:15] = 1
delays[14:15] = 4250*1e-15
sec_orders[14:15] = 22000*(1e-15)**2
third_orders[14:15] = 10*(1e-15)**3
#fourth_orders[15] = (1e-15)**4

random.seed(0)
for jj in range(16,model_num):
    if (jj <= model_num//2):
        hole_positions[jj] = lam0 + random.randint(-100,100)*1e-9
        hole_widths[jj] = random.randint(0,50)*1e-9
        hole_depths[jj] = random.randint(0,1)
        delays[jj] = random.randint(0,5000)*1e-15
        sec_orders[jj] = random.randint(-22000,22000)*1e-9*(1e-15)**2
        third_orders[jj] = random.randint(0,30)*(1e-15)**3
        #fourth_orders[jj] = random.randint(0,30)*(1e-15)**4
    else:
       hole_positions[jj] = lam0 + random.randint(-100,100)*1e-9
       hole_widths[jj] = random.randint(0,50)*1e-9
       hole_depths[jj] = random.randint(0,1)
       delays[jj] = 0
       sec_orders[jj] = 0
       third_orders[jj] = 0
       fourth_orders[jj] = 0
       
    if (hole_positions[jj] == 0):
        print(jj)


dazzlerParamFileName = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/dazParams_0_10272020.csv"
with open (dazzlerParamFileName,'w') as f_w:
    for ii in range(model_num):
        hole_position = hole_positions[ii]
        hole_width = hole_widths[ii]
        hole_depth= hole_depths[ii]

        delay=delays[ii]
        sec_order=sec_orders[ii]
        third_order=third_orders[ii]
        fourth_order=fourth_orders[ii]
        
        
        f_w.write(str(hole_position))
        f_w.write("\t")
        f_w.write(str(hole_width))
        f_w.write("\t")
        f_w.write(str(hole_depth))
        f_w.write("\t")
        f_w.write(str(delay))
        f_w.write("\t")
        f_w.write(str(sec_order))
        f_w.write("\t")
        f_w.write(str(third_order))
        f_w.write("\t")
        f_w.write(str(fourth_order))
        f_w.write("\n")

Eparams = np.array([[position],[width]])
    
E_file_name = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/inputE_0_10272020.out"
np.savetxt(E_file_name,input_E)
E_paramsFile_name=  "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/inputE_params_0_10272020.out"
np.savetxt(E_paramsFile_name, Eparams)
time_file_name = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/timeVector_0_10272020.out"
np.savetxt(time_file_name, time_vector)  

saveFilePath = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/"
saveFileName = "run0_10272020"


prepare_input_output_pair_ensemble(dazzlerParamFileName, E_file_name, time_file_name, 1, E_paramsFile_name, saveFilePath=saveFilePath, saveFileName=saveFileName)






