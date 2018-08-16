# -*- coding: utf-8 -*-
import os
import numpy as np 
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
filename1 = '500_sigma0.000000_lambda0.980000_model0_task0_everything.mat'
filename2 = 'BASIC_COMP_SI.mat'

if os.path.isfile(filename1):
    data1 = loadmat(filename1)
    data2 = loadmat(filename2)
else:
    exit

hidr = data1['hidResps'] # batch_size x time x nneuron
SI = data2['SI']
#   infloss_test = data['infloss_test']
infloss_test = data1['frac_rmse_test']    
