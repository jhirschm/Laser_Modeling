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
hole_position_range = 260#e-9
hole_width_range = 260#e-9
hole_depth_range = 1

delays_range = 10**4#*1e-15
sec_order_range = 2*10**5#*(1e-15)**2
third_order_range = 2*10**6#*(1e-15)**3
fourth_orders_range = 2*10**8#*(1e-15)**4


random.seed(0)
for jj in range(model_num):
    if (jj%10==0):
        print("generating model: " + str(jj))
    hole_positions[jj] = lam0 + random.randint(-hole_position_range/2,hole_position_range/2)*1e-9
    hole_widths[jj] = random.randint(0,hole_width_range)*1e-9
    hole_depths[jj] = random.randint(0,hole_depth_range*100)/100
    delays[jj] = random.randint(0,delays_range)*1e-15
    sec_orders[jj] = random.randint(-1*sec_order_range/2,sec_order_range/2)*(1e-15)**2
    third_orders[jj] = random.randint(-1*third_order_range/2,third_order_range/2)*(1e-15)**3
    fourth_orders[jj] = random.randint(-1*fourth_orders_range/2,fourth_orders_range/2)*(1e-15)**4
       

extension = "1_10292020"
dazzlerParamFileName = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/dazParams_"+extension+".csv"
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


    
E_file_name = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/inputE_"+extension+".out"
np.savetxt(E_file_name,input_E)
E_paramsFile_name=  "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/inputE_params_"+extension+".out"
np.savetxt(E_paramsFile_name, Eparams)
time_file_name = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/timeVector_"+extension+".out"
np.savetxt(time_file_name, time_vector)  

saveFilePath = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/"
saveFileName = "run"+extension


prepare_input_output_pair_ensemble(dazzlerParamFileName, E_file_name, time_file_name, 1, E_paramsFile_name, saveFilePath=saveFilePath, saveFileName=saveFileName)






