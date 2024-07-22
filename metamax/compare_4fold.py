#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 10:47:07 2024

@author: maxzhelyeznyakov
"""

import torch
import numpy as np
import pandas as pd
from system import *
from Layers import *
from MetamaterialUtitilyFunctions import *
import pandas as pd
import mat73
from objective_functions import *
from Optimization import *
from hankel_torch import HankelTransform
import scipy
from scipy import interpolate
import torch_dct as dct
#%%
rcwa_filename = '/home/maxzhelyeznyakov/Documents/code/metamax/src/rcwa_data/zhihao-lookup-table.csv'
rcwa_data = pd.read_csv(rcwa_filename)
waves = np.linspace(3,5,21)
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
x = rx
y = -x
xx,yy = np.meshgrid(x,y)
rr = np.sqrt(xx**2+yy**2)
init_r = mat73.loadmat('pillar.mat')['pillar']
init_r = mat73.loadmat('/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/structure_var_new.mat')['structure_var_new']
fi = scipy.interpolate.interp1d(rx, init_r, kind='nearest', fill_value="extrapolate")
init_r = fi(rr)
init_r[init_r > rmax] = rmin

#init_r= rcwa_data['r1'].unique()
#init_r = np.random.uniform(low=rmin,high=rmax,size=xx.shape)
# init_r = initialize_single_wavelength_metasurface(waves[11], D, F, dx, 
#                                                rcwa_data)
init_rt = torch.tensor(init_r,dtype=torch.float32).cuda()
# ddf = DifferentiableDiffraction(waves, angles, rcwa_datafile=rcwa_filename)
# f = ddf.forward(init_rt)
# f3m = f[0][0].cpu().detach().numpy()

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
field3um = system.layers[-1].get_transmitted_field(3.0,0.0).cpu().detach().numpy()
field3um[xx**2+yy**2 > R**2]=0
zf = mat73.loadmat('/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/phase_profile.mat')['phase_profile']

#field3um = fields['3.00.0'].cpu().detach().numpy()
zf_3um = zf[0,893:,893:]

plt.figure()
plt.subplot(1,2,1)
plt.plot(np.angle(field3um[0,:]))
plt.subplot(1,2,2)
plt.plot(np.angle(zf_3um[0,:]))