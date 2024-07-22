#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:09:02 2024

@author: maxzhelyeznyakov
"""

import torch
import numpy as np
import pandas as pd
from metamax.system import *
from Layers import *
from MetamaterialUtitilyFunctions import *
import pandas as pd
import mat73
rcwa_filename = '/home/maxzhelyeznyakov/Documents/code/metamax/src/rcwa_data/zhihao-lookup-table.csv'
rcwa_data = pd.read_csv(rcwa_filename)
waves = np.linspace(3,5,21)
angles = np.array([0.0])
nwaves = len(waves)

D = 10000
F = 10000
rmin = 1.5
rmax = 2.4
dx = 5.2

init_ms = []

init_r = torch.tensor(initialize_circularly_symmetric_metsurface(4, D, F, dx, 
                                               rcwa_data)).reshape([-1,1]).cuda()

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
                rcwa_filename = rcwa_filename)

nz = 1000
www = system.vis_system(4.0,angles[0],nz)