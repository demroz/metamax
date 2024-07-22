#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:10:49 2024

@author: maxzhelyeznyakov
"""

import numpy as np
import matplotlib.pyplot as plt
#from pyhank import qdht, iqdht, HankelTransform
from hankel_torch import HankelTransform
from hankel_propagator import HankelPropagator
from waveprop import rs
import torch
from angular_propagate_pytorch import Propagator
from MetamaterialUtitilyFunctions import *
import scipy 

p = 1
D = 100
F = D

x1 = np.arange(0,D/2,p)
r = x1
x = np.concatenate((-1*x1[::-1],x1[1:]))
y = x
xx,yy = np.meshgrid(x,y)
apert = np.zeros_like(xx)
apert[xx**2+yy**2 < (D/2)**2] = 1
rr = np.sqrt(xx**2+yy**2)

kx = np.fft.fftshift(np.fft.fftfreq(len(x),p)) * 2 * np.pi
ky = np.fft.fftshift(np.fft.fftfreq(len(x),p)) * 2 * np.pi
kxx,kyy = np.meshgrid(kx,ky)
krr = np.sqrt(kxx**2+kyy**2)

fr = np.sin(r+0.001)/(r+0.001)
f = np.interp(rr,r,fr)*apert

ffft = np.abs(np.fft.fftshift(np.fft.fft2(f)))
ffft /= np.max(ffft)
ht = HankelTransform(order=0,radial_grid=torch.tensor(r).cuda(),device='cuda')
frh = ht.to_transform_r(torch.tensor(fr).cuda())
frkh = torch.abs(ht.qdht(frh))
frkh /= torch.max(frkh)
frkh = frkh.cpu().numpy()

# plt.figure()
# plt.plot(ht.kr,frkh)
# plt.plot(krr[500,500:],ffft[500,500:])
# plt.xlim([0,5])
#%%
wave = 0.78
phi_r = 2*np.pi/wave*(F-np.sqrt(F**2+r**2)) % (2*np.pi)
phi = f = np.interp(rr,r,phi_r)*apert

ff = np.abs(rs.angular_spectrum(np.exp(1j*phi), wave, p, F)[0])**2
ff /= np.max(ff)
hp = HankelPropagator(wave, radial_grid=torch.tensor(r).cuda(), dz=F)
ffr = torch.abs(hp.propagate(torch.exp(1j*torch.tensor(phi_r)).cuda()))**2
ffr /= torch.max(ffr)
plt.figure()
plt.plot(torch.abs(ffr).cpu()[0:100])
plt.plot(ff[len(ffr),len(ffr):][0:100])
plt.legend(['hankel prop', 'fft prop'])
plt.show()
