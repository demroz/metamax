#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:53:24 2024

@author: maxzhelyeznyakov
"""

import torch
import numpy as np
import logging
from Layers import *
import pandas as pd
from waveprop import rs
from DifferentiableDiffraction import *
from hankel_propagator import HankelPropagator
import torch.utils.checkpoint as checkpoint
import logging
logger = logging.getLogger(__name__)

class prop_helper():
    def __init__(self, wave, dx, z, device):
        self.wave = wave
        self.dx = dx
        self.z = z
        self.device = torch.device(device)
    def prop(self, field):
        logger.info('propagating %f on device %s', self.wave, self.device)
        return rs.angular_spectrum(field, self.wave, self.dx, self.z,
                                   device=self.device)[0]
class System():
    def __init__(self, layer_list: list = None,
                 diameter: float = None,
                 periodicity: float = None,
                 wavelengths: list = None,
                 angles: list = None,
                 symmetry: str = None,
                 rcwa_filename: str='../rcwa_data/visible_data.csv',
                 device_hash: dict = None):
        
        self.rcwa_filename = rcwa_filename
        self.D = diameter
        self.dx = periodicity
        self.wavelengths = wavelengths
        self.angles = angles
        self.layers = layer_list
        first_diffractive_layer = next(obj for obj in layer_list if type(obj) == DifferentiableDiffractiveLayer)
        self.paramsize = first_diffractive_layer.parameters.size()
        self.zlist = []
        for layer in self.layers:
            self.zlist.append(layer.pos[2])
            
        self.diffraction_model = DifferentiableDiffraction(self.wavelengths, 
                                                           self.angles,
                                                           rcwa_datafile=rcwa_filename,
                                                           device_hash=device_hash)
        
        self.symmetry = symmetry
        if self.symmetry == 'c':
            self.rho = torch.linspace(0,diameter/2,int(diameter/2/periodicity)).cuda()
        self.device_hash = device_hash
        
    
    def planewave(self, wave, angle):
        if self.symmetry == 'c':
            x = torch.linspace(0,self.D/2,self.paramsize[0])
            y = torch.tensor(0.0)
            xx,yy = torch.meshgrid(x,y)
            k = 2*np.pi/wave            
            angle_rad = angle*np.pi/180
            
            kx = k*torch.sin(torch.tensor(angle_rad)) #/ np.sqrt(2)
            ky = 0 #k*torch.sin(torch.tensor(angle_rad)
            
        elif self.symmetry == '4fold':
            x = torch.linspace(0,self.D/2,self.paramsize[0])
            y = x
            xx,yy = torch.meshgrid(x,y)
            k = 2*np.pi/wave            
            angle_rad = angle*np.pi/180
            kx = k*torch.sin(torch.tensor(angle_rad)) / np.sqrt(2)
            ky = k*torch.sin(torch.tensor(angle_rad) / np.sqrt(2))
            
            
        else:
            x = torch.linspace(-self.D/2, self.D/2, self.paramsize[0])
            y = x
            xx,yy = torch.meshgrid(x,y)
            k = 2*np.pi/wave            
            angle_rad = angle*np.pi/180
            kx = k*torch.sin(torch.tensor(angle_rad)) # / np.sqrt(2)
            ky = 0 #k*torch.sin(torch.tensor(angle_rad) / np.sqrt(2))
            
        
        amp = torch.ones_like(xx)
        phase = torch.exp(1j*(kx*xx+ky*yy))
        
        return (amp*phase).cuda()
    
    def forward(self):
        self.propagation_distances = np.diff(self.zlist)
        amp = torch.zeros_like(self.planewave(self.wavelengths[0], 0.0))
        # x = torch.linspace(0, self.D/2, self.paramsize[0])
        # y = x
        # xx,yy = torch.meshgrid(x,y)
        # amp[xx**2+yy**2 < (self.D/2)**2] = 1
        for k, z in enumerate(self.propagation_distances):
            tfun = self.diffraction_model.forward(self.layers[k].parameters)
            for i, wave in enumerate(self.wavelengths):
                for j, angle in enumerate(self.angles):
                    key = str(wave)+str(angle)
                    if k==0:
                        incidence = self.planewave(wave, angle)
                        if self.device_hash:
                            incidence = incidence.to(torch.device(self.device_hash[key]))
                            amp = amp.to(torch.device(self.device_hash[key]))
                        self.layers[k].set_incident_field(incidence, wave, angle)
                    if isinstance(self.layers[k], DifferentiableDiffractiveLayer):
                        incident_field = self.layers[k].get_incident_field_at_wave_and_angle(wave,angle)
                        transmitted_field = incident_field*tfun[key].reshape(incident_field.shape) #*amp
                        if self.symmetry == 'c':
                            prop = HankelPropagator(wave, radial_grid=self.rho, dz=z)
                            prop_field = checkpoint.checkpoint(prop.propagate,transmitted_field[:,0],use_reentrant=True)
                        else:
                            if self.device_hash:
                                prop = prop_helper(wave, self.dx, z, self.device_hash[key])
                            else:
                                prop = prop_helper(wave, self.dx, z, transmitted_field.device)
                            prop_field = checkpoint.checkpoint(prop.prop,transmitted_field, use_reentrant=True)
                            # prop_field = rs.angular_spectrum(transmitted_field, 
                            #                                   wave, 
                            #                                   self.dx, 
                            #                                   z,
                            #                                   bandlimit=True,
                            #                                   pad=True,
                            #                                   device=torch.device('cuda'))[0]
                        self.layers[k].set_transmitted_field(transmitted_field, wave, angle)
                        self.layers[k+1].set_incident_field(prop_field, wave, angle)
        
        