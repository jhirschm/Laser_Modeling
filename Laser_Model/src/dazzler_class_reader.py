#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:05:42 2020

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
import ast



saveFilePath = "/Users/jackhirschman/Documents/Stanford/PhD_Project/Laser_Modeling/Laser_Model/Data/Dazzler/ITO_Data/"
saveFileName = "run1_10272020"
file = saveFilePath+saveFileName+".hdf5"
data = h5py.File(file,'r')
dazzlerParams = ast.literal_eval(data['Runs']['run0']['DazzlerParams'][()])


