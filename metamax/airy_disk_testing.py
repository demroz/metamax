#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 14:15:24 2024

@author: maxzhelyeznyakov
"""

import numpy as np
from waveprop import fraunhofer
import matplotlib.pyplot as plt

D = 50
F = 50
dx = 1
wave = 0.1
x = np.linspace(-D/2,D/2,500)
y = x
xx,yy = np.meshgrid(x,y)

aperture = np.zeros_like(xx,dtype=np.complex128)
aperture[(xx**2+yy**2) < (D/2)**2 ] = 1+0j


out = np.abs(fraunhofer.fraunhofer(aperture,wave,x[1]-x[0],F)[0])**2
#%%
plt.figure()
plt.imshow(out)