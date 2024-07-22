#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 16:00:39 2024

@author: maxzhelyeznyakov
"""

from waveprop import rs
import numpy as np
import matplotlib.pyplot as plt

D = 100
F = 100

wave = 0.633

dx = 0.3

x = np.arange(0,D/2, dx)
y = x

xx,yy = np.meshgrid(x,y)

phase = 2*np.pi/wave * (F-np.sqrt(xx**2+yy**2+F**2))

angle = 15.0 * np.pi/180
amplitude = np.ones_like(phase,dtype=np.complex128)

k = 2*np.pi/wave
kx = k* np.sin(angle) / np.sqrt(2)
ky = k* np.sin(angle) / np.sqrt(2)


amplitude = np.exp(1j*(kx*xx+ky*yy))
ff = np.abs(rs.angular_spectrum(amplitude*np.exp(1j*phase), wave, dx, F)[0])**2
ix,iy = np.unravel_index(np.argmax(ff),ff.shape)


locx,locy = x[ix],y[iy]
theory = F*np.tan(angle) / np.sqrt(2)

plt.figure()
plt.imshow(ff)
plt.colorbar()
plt.show()