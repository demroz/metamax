#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:23:14 2024

@author: maxzhelyeznyakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:27:53 2024

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
rcwa_filename = '../rcwa_data/Period2um_Height3um/2ump_3to5umwave.csv'
rcwa_data = pd.read_csv(rcwa_filename)
waves = np.linspace(3,5,21)
angles = np.array([0.,5.,10.,15.0])

device_hash = {}
gpu_id = 0
n_gpu = 1
for wave in waves:
    for angle in angles:
        gpu_id = gpu_id % n_gpu
        device_hash[str(wave)+str(angle)] = 'cuda:'+str(gpu_id)
        #if gpu_id == 0:
        #    device_hash[str(wave)+str(angle)] = 'cuda'
        #if gpu_id == 1:
        #    device_hash[str(wave)+str(angle)] = 'cuda'
        gpu_id += 1

            
        
#%%
nwaves = len(waves)

D = 10000
R = D/2
F = D
rmin = 0.6/2
rmax = 1.4/2
dx = 2.0
#rho = np.linspace(0,D/2,int(D/2/dx))
rx = np.arange(0,D/2-dx,dx)
x = rx
y = -x
xx,yy = np.meshgrid(x,y)
rr = np.sqrt(xx**2+yy**2)
#init_r = scipy.io.loadmat('pillar_8cm.mat')['pillar'][:,0]
# init_r = mat73.loadmat('pillar.mat')['pillar'][0:len(rx)]
# #init_r = mat73.loadmat('/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/structure_var_new.mat')['structure_var_new']
# fi = scipy.interpolate.interp1d(rx, init_r, kind='nearest', fill_value="extrapolate")
# init_r = fi(rr)
# init_r[init_r > rmax] = rmin
#init_r = np.random.uniform(low=rmin,high=rmax,size=xx.shape)
init_r = initialize_single_wavelength_metasurface(waves[15], D, F, dx, 
                                                rcwa_data)
#init_rt = rmin + torch.rand(xx.shape,dtype=torch.float32).cuda()*(rmax-rmin)#torch.tensor(init_r,dtype=torch.float32).cuda()
# init_rt = torch.tensor(init_r,dtype=torch.float32).cuda()
# ms_layer1 = DifferentiableDiffractiveLayer(parameters = init_rt,
#                                           pos = torch.tensor([0,0,0]),
#                                           pmin = rmin,
#                                           pmax = rmax,
#                                           device_hash = device_hash)

# detector = Detector([0,0,F])

# layer_list = [ms_layer1, detector]
# system = System(layer_list = layer_list, 
#                 diameter=D, 
#                 periodicity=dx, 
#                 wavelengths = waves, 
#                 angles = angles, 
#                 rcwa_filename = rcwa_filename,
#                 device_hash = device_hash)
# system.forward()

# #%%
# ideal_mtfs = {}
# ideal_psfs = {}

# for wave in waves:
#     for angle in angles:
#         key = str(wave)+str(angle)
#         st = time.time()
#         ideal_psfs[key] = torch.tensor(diffraction_limited_psf_4fold(R, F, len(x), wave, angle)).to(torch.device(device_hash[key]))
#         ideal_mtfs[key] = torch.tensor(diffraction_limited_mtf_4fold(R, F, len(x), wave, angle)).to(torch.device(device_hash[key]))
#         et = time.time()
#         print(et-st)
# optimizer = torch.optim.Adam([system.layers[0].parameters], lr = 1e-2)
# objective = MTF_AND_PSF_CORRELATION_4FOLD(ideal_psfs=ideal_psfs, 
#                                           ideal_mtfs=ideal_mtfs, 
#                                           D=D, 
#                                           npix = int(R/dx))
# objective.forward(system.layers[-1])        
# opt = Optimization(system, optimizer, objective, rmax, rmin)
# opt.run()
# #%%
# np.savetxt('optimized_parameters.txt',init_rt.cpu().detach().numpy())

