#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 09:09:26 2024

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

p = 5.6
D = 10000
F = 5*D
R = D/2
wavelength = 3
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

psf = hyperboloid_psf_2D(R, F, int(D/p), wavelength)
mtf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
# f = np.interp(rr,r,fr)*apert

# ffft = np.abs(np.fft.fftshift(np.fft.fft2(f)))
# ffft /= np.max(ffft)
# ht = HankelTransform(order=0,radial_grid=torch.tensor(r).cuda(),device='cuda')
# frh = ht.to_transform_r(torch.tensor(fr).cuda())
# frkh = torch.abs(ht.qdht(frh))
# frkh /= torch.max(frkh)
# frkh = frkh.cpu().numpy()

# plt.figure()
# plt.plot(f[892,892:])