#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:44:01 2024

@author: noise
"""

from hankel_torch import HankelTransform
import torch
class HankelPropagator:
    def __init__(self, wave, radial_grid, dz):
        self.device = radial_grid.device.type
        self.ht = HankelTransform(order=0, radial_grid=radial_grid,device=self.device)
        self.k0 = 2*torch.pi/wave
        self.dz = dz
        
    def propagate(self, Er):
        ErH = self.ht.to_transform_r(Er) 
        EkrH = self.ht.qdht(ErH)
        kz = torch.sqrt(self.k0 ** 2 + 0j - self.ht.kr ** 2).to(torch.device(self.device))

        phi_z = kz * self.dz  # Propagation phase
        EkrHz = EkrH * torch.exp(1j * phi_z)  # Apply propagation
        ErHz = self.ht.iqdht(EkrHz)  # iQDHT
        Erz =  self.ht.to_original_r(ErHz)
        
        return Erz