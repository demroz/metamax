#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:09:02 2024

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
#%%
rcwa_filename = '/home/maxzhelyeznyakov/Documents/code/metamax/src/rcwa_data/zhihao-lookup-table.csv'
#rcwa_filename = '/home/maxzhelyeznyakov/Documents/code/metamax/src/rcwa_data/Period5600nm_Height7900nm/mwir-wave-rad-angle-sweep.csv'
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
rho = torch.linspace(dx,D/2,int(D/2/dx))
init_ms = []

init_r = torch.tensor(initialize_circularly_symmetric_metsurface(4, D, F, dx, 
                                               rcwa_data)).reshape([-1,1]).cuda()[1:]
#init_r = torch.tensor(rcwa_data['r1'].unique()).reshape([-1,1]).cuda()
ddf = DifferentiableDiffraction(waves, angles, rcwa_datafile=rcwa_filename)
#%%
# field = ddf.forward(torch.tensor(rcwa_data['r1'].unique()).cuda())
# hr_r = np.linspace(rmin,rmax,1000)
# fhr = ddf.forward(torch.tensor(hr_r).cuda())
# plt.figure()
# plt.plot(rcwa_data['r1'].unique(),torch.abs(field[1][0]).cpu(),'*')
# plt.plot(hr_r,torch.abs(fhr[1][0]).cpu(),'.')
#plt.xlim([2.2,2.4])
# init_r = (rmax-rmin)*torch.rand(rho.shape,dtype=torch.float64).reshape([-1,1]).cuda()+rmin # torch.rand(rho.shape).reshape([-1,1]).cuda()#torch.tensor(mat73.loadmat('pillar.mat')['pillar']).reshape([-1,1]).cuda()
init_r = torch.tensor(mat73.loadmat('pillar.mat')['pillar'],dtype=torch.float32).reshape([-1,1]).cuda()
#init_r = torch.tensor(mat73.loadmat('/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/structure_var_new.mat')['structure_var_new']).reshape([-1,1]).cuda()
ms_layer1 = DifferentiableDiffractiveLayer(parameters = init_r,
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
                symmetry='c')
system.forward()

ideal_mtfs = {}
ideal_psfs = {}

for wave in waves:
    for angle in angles:
        ideal_psfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_psf_1D(R, F, int(R/dx), wave)).cuda()
        ideal_mtfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_mtf_1D(R, F, int(R/dx), wave)).cuda()
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(ideal_psfs[str(wave)+str(angle)].cpu())
        plt.subplot(2,1,2)
        plt.plot(ideal_mtfs[str(wave)+str(angle)].cpu())
        plt.show()
        
        
optimizer = torch.optim.SGD([system.layers[0].parameters], lr = 1e-2)
objective = MTF_AND_PSF_CORRELATION(ideal_psfs=ideal_psfs, ideal_mtfs=ideal_mtfs, D=D, npix = int(R/dx))
opt = Optimization(system, optimizer, objective, rmax, rmin)
opt.run()