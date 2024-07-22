#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:27:38 2024

@author: maxzhelyeznyakov
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from hankel_torch import HankelTransform
import logging

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("../logs/optimization_log.log"),
#         logging.StreamHandler()
#     ]
# )
logger = logging.getLogger(__name__)
class Optimization():
    def __init__(self, system, 
                 optimizer,
                 objective, 
                 dmax,
                 dmin,
                 tol = 1e-1, 
                 maxiter = 300):
        
        self.system = system
        #self.optimizer = torch.optim.Adam([self.system.layers[0].parameters], lr = 1e-4)
        self.optimizer = optimizer
        self.objective = objective
        self.tol = tol
        self.maxiter = maxiter
        self.dmax = dmax
        self.dmin = dmin
    
    def run(self, plot=False):
        lar = []
        iar = []
        if plot:
            plt.figure()
        for i in range(self.maxiter):
            self.optimizer.zero_grad()
            self.system.forward()
            loss = self.objective.forward(self.system.layers[-1])
            lar.append(loss.item())
            iar.append(i)
            logger.info("loss = %f", loss.item())
            if plot:
                plt.scatter(iar,lar)
                plt.pause(0.1)
            loss.backward()
            self.optimizer.step()
            for j in range(len(self.system.layers)-1):
                with torch.no_grad():
                    self.system.layers[0].parameters.clamp_(self.dmin,self.dmax)
            
            #outdir = "/home/maxzhelyeznyakov/Documents/code/metamax/src/data/hankel_transform_optimized_metasurfaces/"
            #np.savetxt(outdir+"params_3to5um_planewave_F1_hankel.txt",self.system.layers[0].parameters.detach().cpu().numpy())
            if plot:
                if i % 10 == 0:
                    plt.figure()
                    plt.plot(lar)
                    plt.show()
            
            
            
            
            
            