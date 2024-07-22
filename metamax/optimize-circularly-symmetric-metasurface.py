#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:09:02 2024

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
#%%
rcwa_filename = '/home/maxzhelyeznyakov/Documents/code/metamax/src/rcwa_data/Period2um_Height3um/2ump_3to5umwave.csv'
rcwa_data = pd.read_csv(rcwa_filename)
waves = np.linspace(3,5,21)
angles = np.array([0.0])
nwaves = len(waves)

D = 10000
R = D/2
F = D
rmin = 0.6/2
rmax = 1.4/2
dx = 2.0
rho = torch.linspace(dx,D/2,int(D/2/dx))
init_ms = []
#init_rx = mat73.loadmat('pillar.mat')['pillar'][0:len(rho)]
init_r = torch.tensor(initialize_circularly_symmetric_metsurface(4.5, D, F, dx, rcwa_data),
                       dtype=torch.float32).reshape([-1,1]).cuda()
#init_r = rmin + torch.rand(len(rho))*(rmax-rmin)
#init_r = init_r.reshape([-1,1]).cuda()
ddf = DifferentiableDiffraction(waves, angles, rcwa_datafile=rcwa_filename)
#%%

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
        ideal_psfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_psf_radial_symmetry_angle(R, F, 
                                                                                                      int(R/dx),
                                                                                                      wave,
                                                                                                      angle)).cuda()
        ideal_mtfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_mtf_radial_symmetry_angle(R, F,
                                                                                                      int(R/dx),
                                                                                                      wave, 
                                                                                                      angle)).cuda()
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(ideal_psfs[str(wave)+str(angle)].cpu())
        plt.subplot(2,1,2)
        plt.plot(ideal_mtfs[str(wave)+str(angle)].cpu())
        plt.show()
        
        
optimizer = torch.optim.Adam([system.layers[0].parameters], lr = 1e-2)
objective = MTF_AND_PSF_CORRELATION(ideal_psfs=ideal_psfs, ideal_mtfs=ideal_mtfs, D=D, npix = int(R/dx))
opt = Optimization(system, optimizer, objective, rmax, rmin)
opt.run(plot=True)
#%%
outdir = "/home/maxzhelyeznyakov/Documents/code/metamax/src/data/hankel_transform_optimized_metasurfaces/"
np.savetxt(outdir+"params_3to5um_planewave_F1_hankel.txt",init_r.detach().cpu().numpy())