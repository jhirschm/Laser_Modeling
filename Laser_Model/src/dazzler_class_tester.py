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

c = 299792458.0 # m/s
lam0 = 800e-9 #central wavelength
w0 = 2*np.pi*c/(lam0) #central frequency

pulse_dur = 9e-15 #time duration of pulse
delta_freq = 0.44/pulse_dur
delta_lam = (delta_freq*lam0**2)/c

#Amplitude params:
position = lam0 #link to dazzler position param
width = delta_lam #link to dazzler width param
hole_position = 780e-9
hole_width = 20e-9#3e-9
hole_depth=1

#Phase params:
delay=4250*1e-15#4250e-15
sec_order=22000*(1e-15)**2
third_order=0*10*(1e-15)**3
fourth_order=0*(1e-15)**4

num_points = 100000
time_vector = np.linspace(-5000e-15,5000e-15,num=num_points, endpoint=True)
input_E = 1*np.exp(-2*np.log(2)*(time_vector/pulse_dur)**2)
input_E.astype(complex)
input_E=input_E*np.exp(1j*w0*time_vector)

dazzlerParamFileName = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/dazParams_0_10252020.csv"
with open (dazzlerParamFileName,'w') as f_w:
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

Eparams = np.array([[position],[width]])
    
E_file_name = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/inputE_0_10252020.out"
np.savetxt(E_file_name,input_E)
E_paramsFile_name=  "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/inputE_params_0_10252020.out"
np.savetxt(E_paramsFile_name, Eparams)
time_file_name = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/timeVector_0_10252020.out"
np.savetxt(time_file_name, time_vector)  

saveFilePath = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/"
saveFileName = "run0_10252020"


prepare_input_output_pair_ensemble(dazzlerParamFileName, E_file_name, time_file_name, 1, E_paramsFile_name, saveFilePath=saveFilePath, saveFileName=saveFileName)






