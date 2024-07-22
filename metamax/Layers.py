#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:05:45 2024

@author: maxzhelyeznyakov
"""

import logging
from constants import *
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../logs/log_file.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
import numpy as np
import torch
from DifferentiableDiffraction import *



class BasicLayer():
    def __init__(self, parameters: torch.Tensor=None, 
                 pos: torch.Tensor=torch.tensor([0,0,0]),
                 differentiable: bool=True,
                 device_hash: dict=None):
        '''
        

        Parameters
        ----------
        parameters : torch.Tensor, optional
            DESCRIPTION. metasurface design parameters
            
        pos : torch.Tensor, optional
            DESCRIPTION. The default is [0,0,0].
            layer position
        differentiable : bool, optional
            DESCRIPTION. defines if we want to optimize over the layer

        Returns
        -------
        None.

        '''
        
        self.parameters = parameters
        self.pos = pos
        self.differentiable = differentiable
        self.incident_field = {}
        self.transmitted_field = {}
        self.device_hash = device_hash
        
    def set_incident_field(self, field,
                  wave: float,
                  angle: float):
        self.incident_field[str(wave)+str(angle)] = field
    
    def get_incident_field_at_wave_and_angle(self,wave,angle):
        return self.incident_field[str(wave)+str(angle)]
    
    def get_incident_field(self):
         return self.incident_field
    
    def set_transmitted_field(self, field: dict,
                  wave: float,
                  angle: float):
        self.transmitted_field[str(wave)+str(angle)] = field
    def get_transmitted_field(self,
                  wave: float,
                  angle: float):
        return self.transmitted_field[str(wave)+str(angle)]

class DifferentiableDiffractiveLayer(BasicLayer):
    def __init__(self, parameters: torch.Tensor=None, 
                 pos: torch.Tensor = None,
                 pmin: float=None,
                 pmax: float=None,
                 NP: int=None,
                 device_hash: dict = None):
        '''
        

        Parameters
        ----------
        parameters : torch.Tensor, optional
            DESCRIPTION. The default is None.
        pos : torch.Tensor, optional
            DESCRIPTION. The default is torch.tensor([0,0,0]).
        pmin : float, not actually optional. must be specified
            DESCRIPTION. The default is None. minimum parameter constraint
        pmax : float, not actually optional. must be specified
            DESCRIPTION. The default is None. max parameter constraint
        NP : int, optional
            DESCRIPTION. Must be provided if parameters are not initialized
        Returns
        -------
        None.

        '''
        BasicLayer.__init__(self,parameters=parameters, pos=pos, differentiable=True)
        self.differentiable = True
        self.parameters = parameters
        self.pos = pos
        if self.parameters is not None:
            logger.info('parameter intiialization provided')
            self.NP = len(parameters)
        if self.parameters is None:
            logging.info('parameter initialization not provided')
            if NP is None or pmin is None or pmax is None:
                logger.critical('must provide parameter dimension NP, and range (pmin pmax), if parameters are not initialized')
            self.NP = NP
            self.pmin = pmin
            self.pmax = pmax
            self._init_parameters_random_uniform()
        
        self.parameters.requires_grad_(True)
        self.parameters.retain_grad()
        self.device_hash = device_hash
        #self.incident_field = {}
        #self.transmitted_field = {}
        
    def _init_parameters_random_uniform(self):
        self.parameters = (self.pmin - self.pmax) * torch.rand(size=(self.NP,)) + self.pmax
        
        
class Detector(BasicLayer):
    def __init__(self, pos):
        BasicLayer.__init__(self, pos=pos, differentiable=False)
        self.pos = pos
        
    def forward(self, field):
        return torch.abs(field)**2
    
class RefractiveElement(BasicLayer):
    def __init__(self, pos: torch.Tensor = None,
                 R1: torch.Tensor = None,
                 R2: torch.Tensor = None,
                 tkn: torch.Tensor = None,
                 refractive_index: torch.Tensor = None,
                 max_r1: torch.Tensor = None,
                 min_r1: torch.Tensor = None,
                 max_r2: torch.Tensor = None,
                 min_r2: torch.Tensor = None,
                 differentiable = False
                 ):
        self.differentiable = differentiable
        
        self.R1 = R1
        self.R2 = R2
        self.tkn = tkn
        
        self.max_r1 = max_r1
        self.min_r1 = min_r1
        
        self.max_r2 = max_r2
        self.min_r2 = min_r2
        
        if self.differentiable:
            self.R1.requires_grad_(differentiable)
            self.R1.retain_grad()
            self.R2.requires_grad_(differentiable)
            self.R2.retain_grad()
            self.tkn.requires_grad_(differentiable)
            self.tkn.retain_grad()
    def compute_thickness_function(self,xx,yy):
        self.delta = self.tkn - self.R1*(1-torch.sqrt(1-(xx**2+yy**2)/R1**2))+self.R2*(1-torch.sqrt(1-(xx**2+yy**2)/R2**2))
        
            