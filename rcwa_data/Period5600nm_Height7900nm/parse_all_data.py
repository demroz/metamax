#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:36:30 2024

@author: demroz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
thetas = [0,5,10,15]
waves = np.arange(3000,5100,100)

rads = []
T = []
phase = []
wave_ar = []
theta_ar = []
phi_ar = []
for wave in waves:
    for theta in thetas:
        data = np.loadtxt('theta_{}_wavelength_{}nm.txt'.format(theta,wave))
        rads.append(data[:,0])
        T.append(data[:,2])
        phase.append(data[:,1])
        wave_ar.append(np.ones_like(data[:,0])*wave/1000)
        theta_ar.append(np.ones_like(data[:,0])*theta)
        phi_ar.append(np.ones_like(data[:,0])*0)
        
d = {}
d['r1'] = np.concatenate(rads)
d['wave'] = np.concatenate(wave_ar)
d['theta'] = np.concatenate(theta_ar)
d['t'] = np.concatenate(T)
phase = np.concatenate(phase)
phase += np.min(phase)
phase = phase % ( 2 * np.pi) - np.pi
#phase = phase % np.pi

d['p'] = phase
d['phi'] = np.concatenate(phi_ar)

df = pd.DataFrame(d)
df.to_csv('mwir-wave-rad-angle-sweep.csv')