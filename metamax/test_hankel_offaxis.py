#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 12:59:38 2024

@author: maxzhelyeznyakov
"""

print('import torch')
import torch
print('import numpy as np')
import numpy as np
print('import pandas as pd')
import pandas as pd
print('from System import *')
from System import *
print('from Layers import *')
from Layers import *
print('from MetamaterialUtitilyFunctions import *')
from MetamaterialUtitilyFunctions import *
print('import pandas as pd')
import pandas as pd
print('import mat73')
import mat73
print('from objective_functions import *')
from objective_functions import *
print('from Optimization import *')
from Optimization import *
print('from hankel_torch import HankelTransfom')
from hankel_torch import HankelTransform
print('import scipy')
import scipy
print('from scipy import interpolate')
from scipy import interpolate
print('import torch_dct as dct')
import torch_dct as dct
import time
#%%
rcwa_filename = '../rcwa_data/zhihao-lookup-table.csv'
rcwa_data = pd.read_csv(rcwa_filename)
waves = [3.,4.,5.]# np.linspace(3,5,21)
angles = [0.0,5.0,10.0,15.0] # np.array([0.,5.,10.,15.0])

device_hash = {}
gpu_id = 0
n_gpu = 2
for wave in waves:
    for angle in angles:
        gpu_id = gpu_id % n_gpu
        if gpu_id == 0:
            device_hash[str(wave)+str(angle)] = 'cuda'
        if gpu_id == 1:
            device_hash[str(wave)+str(angle)] = 'cuda'
        gpu_id += 1

            
        
#%%
nwaves = len(waves)

D = 10000
R = D/2
F = D
rmin = 1.5
rmax = 2.4
dx = 5.6
#rho = np.linspace(0,D/2,int(D/2/dx))
rx = np.arange(0,D/2-dx,dx)
x = np.concatenate((-rx[::-1],rx[1:])) # rx
y = x
xx,yy = np.meshgrid(x,y)
rr = np.sqrt(xx**2+yy**2)
init_r = scipy.io.loadmat('pillar_8cm.mat')['pillar'][:,0]
init_r = mat73.loadmat('pillar.mat')['pillar'][0:len(rx)]
init_r = initialize_circularly_symmetric_metsurface(4, D, F, dx, 
                                               rcwa_data)[:len(rx)]
# #init_r = mat73.loadmat('/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/structure_var_new.mat')['structure_var_new']
fi = scipy.interpolate.interp1d(rx, init_r, kind='nearest', fill_value="extrapolate")
# init_r = fi(rr)
# init_r[init_r > rmax] = rmin
#init_r = np.random.uniform(low=rmin,high=rmax,size=xx.shape)
# init_r = initialize_single_wavelength_metasurface(waves[11], D, F, dx, 
#                                                rcwa_data)
# init_rt = rmin + torch.rand(xx.shape,dtype=torch.float32).cuda()*(rmax-rmin)#torch.tensor(init_r,dtype=torch.float32).cuda()
init_rt_full = torch.tensor(fi(rr),dtype=torch.float32).cuda()
init_rt = torch.tensor(init_r,dtype=torch.float32).cuda()
ms_layer1 = DifferentiableDiffractiveLayer(parameters = init_rt,
                                          pos = torch.tensor([0,0,0]),
                                          pmin = rmin,
                                          pmax = rmax,
                                          device_hash = device_hash)

detector = Detector([0,0,F])

layer_list = [ms_layer1, detector]
system = System(layer_list = layer_list, 
                diameter=D, 
                periodicity=dx, 
                wavelengths = waves, 
                angles = angles, 
                rcwa_filename = rcwa_filename,
                device_hash = device_hash,
                symmetry='c')
system.forward()
ff_hankel = system.layers[-1].get_incident_field()
for key in ff_hankel.keys():
    ff_hankel[key] = ff_hankel[key].cpu().detach().numpy()
ms_layer1_full = DifferentiableDiffractiveLayer(parameters = init_rt_full,
                                          pos = torch.tensor([0,0,0]),
                                          pmin = rmin,
                                          pmax = rmax,
                                          device_hash = device_hash)
layer_list2 = [ms_layer1_full, Detector([0,0,F])]
system2 = System(layer_list = layer_list2, 
                diameter=D, 
                periodicity=dx, 
                wavelengths = waves, 
                angles = angles, 
                rcwa_filename = rcwa_filename,
                device_hash = device_hash)
system2.forward()
ff = system2.layers[-1].get_incident_field()

for key in ff.keys():
    ff[key] = ff[key].cpu().detach().numpy()
#%%
for key in ff.keys():
    plt.figure()
    plt.plot(np.abs(ff[key][891:,891])**2/np.max(np.abs(ff[key][891:,891])**2))
    plt.plot(np.abs(ff_hankel[key])**2/np.max(np.abs(ff_hankel[key])**2))
    plt.title(key)
    plt.legend(['norm psf asm', 'norm psf hankel'])
    plt.show()
#%%
offset = 100
for key in ff.keys():
    plt.figure()
    plt.plot(np.abs(ff[key][891+offset:,891])**2/np.max(np.abs(ff[key][891+offset:,891])**2))
    plt.plot(np.abs(ff_hankel[key][offset:])**2/np.max(np.abs(ff_hankel[key][offset:])**2))
    plt.title(key)
    plt.legend(['norm psf asm', 'norm psf hankel'])
    plt.show()