#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:11:00 2024

@author: maxzhelyeznyakov
"""

import numpy as np
import scipy
from waveprop import rs
import matplotlib.pyplot as plt

field = scipy.io.loadmat('/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/phase_profile_2.mat')['phase_profile']
#%%
F = 10000
nf = field[0,:,:]
nf1d = field[0,894,:]
ff = np.abs(rs.angular_spectrum(nf, 0.4, 5.2, F)[0])**2
ff1d = np.abs(rs.angular_spectrum(nf1d.reshape([-1,1]), 0.4, 5.2, F)[0])**2