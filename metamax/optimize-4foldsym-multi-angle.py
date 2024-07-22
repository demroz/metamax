#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 13:46:54 2024

@author: maxzhelyeznyakov
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:27:53 2024

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

D = 80000
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
#init_r = mat73.loadmat('/home/maxzhelyeznyakov/Downloads/12 my_design/8 strehl_ratio_inverse/structure_var_new.mat')['structure_var_new']
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

psf = diffraction_limited_psf_4fold(10, 1, len(x), waves[0])
dx = 10/len(x)
psf_half = np.concatenate([np.fliplr(psf),psf],axis=1)
psf_full = np.concatenate([psf_half[::-1,:],psf_half])
plt.figure()
plt.imshow(psf_full)
plt.show()

mtf_full = np.abs(np.fft.fftshift(np.fft.fft2(psf_full)))
mtf_4fold = np.abs(np.fft.fftshift(np.fft.fft2(psf)))

psf_rfft = scipy.fftpack.dctn(psf,type=2)

plt.figure()
plt.plot(psf_rfft[0,0:50])
plt.plot(mtf_full[892,892:920])

#%%
ideal_mtfs = {}
ideal_psfs = {}

for wave in waves:
    for angle in angles:
        ideal_psfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_psf_4fold(R, F, len(x), wave)).cuda()
        ideal_mtfs[str(wave)+str(angle)] = torch.tensor(diffraction_limited_mtf_4fold(R, F, len(x), wave)).cuda()