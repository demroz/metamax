#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:27:53 2024

@author: maxzhelyeznyakov
"""

import torch
import numpy as np
import pandas as pd
from System import *
from Layers import *
from MetamaterialUtitilyFunctions import *
import pandas as pd
import mat73
from objective_functions import *
from Optimization import *
from hankel_torch import HankelTransform
import scipy
from scipy import interpolate
#%%
rcwa_filename = '/home/maxzhelyeznyakov/Documents/code/metamax/src/rcwa_data/zhihao-lookup-table.csv'
rcwa_data = pd.read_csv(rcwa_filename)
waves = np.linspace(3,5,3)
angles = np.array([0.0])
nwaves = len(waves)

D = 10000
R = D/2
F = D
rmin = 1.5
rmax = 2.4
dx = 5.6
#rho = np.linspace(0,D/2,int(D/2/dx))
rx = np.arange(0,D/2-dx,dx)
x = np.concatenate([rx[::-1],rx[1:]])
y = x
xx,yy = np.meshgrid(x,y)
rr = np.sqrt(xx**2+yy**2)
init_r = mat73.loadmat('pillar.mat')['pillar']
fi = scipy.interpolate.interp1d(rx, init_r, kind='nearest', fill_value="extrapolate")
init_r = fi(rr)
init_r[init_r > rmax] = rmin
#init_r = np.random.uniform(low=rmin,high=rmax,size=xx.shape)
# init_r = initialize_single_wavelength_metasurface(waves[11], D, F, dx, 
#                                                rcwa_data)
init_rt = torch.tensor(init_r,dtype=torch.float32).cuda()

ms_layer1 = DifferentiableDiffractiveLayer(parameters = init_rt,
                                          pos = torch.tensor([0,0,0]),
                                          pmin = rmin,
                                          pmax = rmax)

detector = Detector([0,0,F])

layer_list = [ms_layer1, detector]
system = System(layer_list = layer_list, 
                diameter=D, 
                periodicity=dx, 
                wavelengths = waves, 
                angles = angles, 
                rcwa_filename = rcwa_filename,
                symmetry=None)
system.forward()
#%%
# ideal_mtfs = {}
# ideal_psfs = {}

# for wave in waves:
#     for angle in angles:
#         ideal_psfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_psf_2D(R, F, int(D/dx)-2, wave)).cuda()
#         ideal_mtfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_mtf_2D(R, F, int(D/dx)-2, wave)).cuda()
        
# optimizer = torch.optim.SGD([system.layers[0].parameters], lr = 1)
# objective = MTF_AND_PSF_CORRELATION_2D(ideal_psfs=ideal_psfs, ideal_mtfs=ideal_mtfs, D=D, npix = int(R/dx))
# objective.forward(system.layers[-1])        
# opt = Optimization(system, optimizer, objective, rmax, rmin)
# opt.run()

# #%%
# to_dump = np.zeros([3,xx.shape[0],xx.shape[1]],dtype=np.complex128)
# i = 0
# for wave in waves:
#     for angle in angles:
        
#         field = system.layers[0].get_transmitted_field(wave,angle).cpu().detach().numpy()
#         to_dump[i,:,:] = field
#         i+=1
        
# d={}
# d['phase_profile'] = to_dump
# fname = '/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/phase_profile_4.mat'
# scipy.io.savemat(fname,d)
