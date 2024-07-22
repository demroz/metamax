#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 13:25:03 2024

@author: maxzhelyeznyakov
"""

import logging
from constants import *
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("../logs/log_file.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)
import numpy as np
import pandas as pd
import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
from torch_interpolations import RegularGridInterpolator
import matplotlib.pyplot as plt
import torch.utils.checkpoint as checkpoint
DEBUG = True

class diffraction_helper():
    def __init__(self, diffractor, Y1):
        self.diffractor = diffractor
        self.Y1 = Y1
    def diffract(self, X1):
        return self.diffractor([X1,self.Y1])
    
class DifferentiableDiffraction():
    def __init__(self, 
                 waves: np.array, 
                 angles: np.array, 
                 rcwa_datafile: str = "../rcwa_data/visible-data.csv", 
                 device: str='cuda',
                 device_hash: dict = None
                 ):
        '''

        Parameters
        ----------
        waves : np.array
            DESCRIPTION. wavelengths to simulate
        angles : np.array
            DESCRIPTION. angles to simulate
        rcwa_datafile : str, optional
            DESCRIPTION. The default is "../rcwa_data/mwir-wave-rad-angle-sweep.csv".
            rcwa lookup table. must have the following keys
            
            r1 [ scatterer geometry parameter], 
            wave [ wavelength ], 
            theta [angle of incidence ], 
            t [ transmission ], 
            p [ phase ]
            
        device : str, optional
            DESCRIPTION. The default is 'cuda'.
            interpolate on GPU by default

        Returns
        -------
        None.

        '''
        self.device = torch.device(device)
        self.device_hash = device_hash
        self.rcwa_data = pd.read_csv(rcwa_datafile)
        
        self.waves_to_simuilate = waves
        self.angles_to_simulate = angles
        
        self.unique_waves = self.rcwa_data['wave'].unique()
        self.unique_angles = self.rcwa_data['theta'].unique()
        self.construct_differentiable_model()
        
    def construct_differentiable_model(self):
        # convert non differentiable parameters to integer hash
        # logger.info("constructing wavelength hash")
        self.wavelength_hash = np.arange(len(self.waves_to_simuilate))
        self.angle_hash = np.arange(len(self.angles_to_simulate))
        self.unique_radii = self.rcwa_data['r1'].unique()
        self.unique_radii = torch.tensor(self.unique_radii, dtype=torch.float32) #.to(self.device)
        
        # logger.info("construcing spline hash")
        self.t_spline_hash = {}
        self.p_spline_hash = {}
        
        self.re_spline_hash = {}
        self.im_spline_hash = {}
        # logger.info("spline hash %i unique wavelengths and %i unique angles", len(self.waves_to_simuilate), len(self.angles_to_simulate))
        for i, wave in enumerate(self.waves_to_simuilate):
            # self.t_spline_hash[i] = {}
            # self.p_spline_hash[i] = {}
            # self.re_spline_hash[i] = {}
            # self.im_spline_hash[i] = {}
            for j, angle in enumerate(self.angles_to_simulate):
                key = str(wave)+str(angle)
                # logger.info("wave %f angle %f", wave, angle)
                # if len(self.unique_waves) > 1:
                #     logger.info("multiple unique wavelengths")
                #     idx_w = np.argpartition(np.abs(self.unique_waves-wave), 2)
                    
                #     nearest_waves = self.unique_waves[ [idx_w[0],idx_w[1]] ]
                #     nearest_wave_l = np.min(nearest_waves)
                #     nearest_wave_u = np.max(nearest_waves)
                    
                #     dw = nearest_wave_u-nearest_wave_l
                #     logger.info("constructing interpolation coeffs for wavelength data")
                #     wave_alpha = 1-np.abs(wave-nearest_wave_l)/dw
                #     wave_beta = 1-np.abs(wave-nearest_wave_u)/dw
                    
                # if len(self.unique_angles) > 1:
                #     logger.info("multiple unique angles")
                #     idx_a = np.argpartition(np.abs(self.unique_angles-angle), 2)
                    
                #     nearest_angles = self.unique_angles[ [idx_a[0],idx_a[1]] ]
                #     nearest_angle_l = np.min(nearest_angles)
                #     nearest_angle_u = np.max(nearest_angles)
                    
                #     dtheta = nearest_angle_u-nearest_angle_l
                #     logger.info("constructing interpolation coeffs for angle data")
                #     angle_alpha = 1-np.abs(angle-nearest_angle_l)/dtheta
                #     angle_beta = 1-np.abs(angle-nearest_angle_u)/dtheta
                
                # if (len(self.unique_waves) > 1) and (len(self.unique_angles) > 1):
                #     logger.info("multiple unique wavelengths and angles in rcwa file")
                #     logger.info("constructing bilinear interpolation for wavelength/angle data")
                #     df_wl_angle_l = self.rcwa_data.loc[(self.rcwa_data['wave']==nearest_wave_l) & 
                #                (self.rcwa_data['theta']==nearest_angle_l) & 
                #                (self.rcwa_data['phi']==0.0)]
                    
                #     df_wl_angle_u = self.rcwa_data.loc[(self.rcwa_data['wave']==nearest_wave_l) & 
                #                (self.rcwa_data['theta']==nearest_angle_u) & 
                #                (self.rcwa_data['phi']==0.0)]
                    
                #     df_wu_angle_l = self.rcwa_data.loc[(self.rcwa_data['wave']==nearest_wave_u) & 
                #                (self.rcwa_data['theta']==nearest_angle_l) & 
                #                (self.rcwa_data['phi']==0.0)]
                    
                #     df_wu_angle_u = self.rcwa_data.loc[(self.rcwa_data['wave']==nearest_wave_u) & 
                #                (self.rcwa_data['theta']==nearest_angle_l) & 
                #                (self.rcwa_data['phi']==0.0)]
                    
                #     tvals = (angle_alpha*(wave_alpha*df_wl_angle_l['t'].to_numpy()+wave_beta*df_wu_angle_l['t'].to_numpy()) +
                #              angle_beta*(wave_alpha*df_wl_angle_u['t'].to_numpy()+wave_beta*df_wu_angle_u['t'].to_numpy()))
                #     pvals = (angle_alpha*(wave_alpha*df_wl_angle_l['p'].to_numpy()+wave_beta*df_wu_angle_l['p'].to_numpy()) +
                #              angle_beta*(wave_alpha*df_wl_angle_u['p'].to_numpy()+wave_beta*df_wu_angle_u['p'].to_numpy()))
                
                # if (len(self.unique_waves) == 1) and (len(self.unique_angles)==1):
                #     logger.info("single angle/wavelength")
                #     logger.info("no interpolation coefficients constructed")
                #     tvals = self.rcwa_data['t']
                #     pvals = self.rcwa_data['p']
                    
                # if len(self.unique_waves == 1) and (len(self.unique_angles) > 1):
                #     logger.info("single uniquye wavelength, multiple unique angles")
                #     logger.info("constructing linear interpolation table for angles")
                #     df_angle_l = self.rcwa_data.loc[(self.rcwa_data['wave']==wave) & 
                #                (self.rcwa_data['theta']==nearest_angle_l) & 
                #                (self.rcwa_data['phi']==0.0)]
                #     df_angle_u = self.rcwa_data.loc[(self.rcwa_data['wave']==wave) & 
                #                (self.rcwa_data['theta']==nearest_angle_u) & 
                #                (self.rcwa_data['phi']==0.0)]
                    
                #     tvals = angle_alpha*df_angle_l['t'].to_numpy()+angle_beta*df_angle_u['t'].to_numpy()
                #     pvals = angle_alpha*df_angle_l['p'].to_numpy()+angle_beta*df_angle_u['p'].to_numpy()
                
                # if (len(self.unique_waves) > 1) and (len(self.unique_angles)==1):
                #     logger.info("multiple unique wavelengths single unique angle")
                #     logger.info("wavelength interpolation table being constructed")
                #     df_wl = self.rcwa_data.loc[(self.rcwa_data['wave']==nearest_wave_l) & 
                #                (self.rcwa_data['theta']==angle) & 
                #                (self.rcwa_data['phi']==0.0)]
                #     df_wu = self.rcwa_data.loc[(self.rcwa_data['wave']==nearest_wave_u) & 
                #                (self.rcwa_data['theta']==angle) & 
                #                (self.rcwa_data['phi']==0.0)]
                #     tvals = wave_alpha*df_wl['t'].to_numpy()+wave_beta*df_wu['t'].to_numpy()
                #     pvals = wave_alpha*df_wl['p'].to_numpy()+wave_beta*df_wu['p'].to_numpy()
                idx_w = np.argmin(np.abs(self.unique_waves-wave))
                idx_a = np.argmin(np.abs(self.unique_angles-angle))
                
                nearest_wave = self.unique_waves[ idx_w ] 
                nearest_angle = self.unique_angles[ idx_a ]
                
                df_wl_angle = self.rcwa_data.loc[(self.rcwa_data['wave']==nearest_wave) & 
                                (self.rcwa_data['theta']==nearest_angle) & 
                                (self.rcwa_data['phi']==0.0)]
                
                tvals = df_wl_angle['t'].to_numpy()
                pvals = df_wl_angle['p'].to_numpy()
                
                if self.device_hash:
                    tvals = torch.tensor(tvals,dtype=torch.float32).reshape(-1,1).to(torch.device(self.device_hash[key]))
                    pvals = torch.tensor(pvals,dtype=torch.float32).reshape(-1,1).to(torch.device(self.device_hash[key]))
                    field = tvals*torch.exp(1j*pvals)
                    self.re_spline_hash[key] = RegularGridInterpolator([self.unique_radii.to(torch.device(self.device_hash[key])),
                                                                        torch.tensor([0.]).to(torch.device(self.device_hash[key]))],
                                                                       torch.real(field))
                    self.im_spline_hash[key] = RegularGridInterpolator([self.unique_radii.to(torch.device(self.device_hash[key])),
                                                                        torch.tensor([0.]).to(torch.device(self.device_hash[key]))],
                                                                       torch.imag(field))
                    
                else:
                    tvals = torch.tensor(tvals,dtype=torch.float32).reshape(-1,1).to(self.device)
                    pvals = torch.tensor(pvals,dtype=torch.float32).reshape(-1,1).to(self.device)
                    field = tvals*torch.exp(1j*pvals)
                    self.re_spline_hash[key] = RegularGridInterpolator([self.unique_radii.to(self.device),
                                                                        torch.tensor([0.]).cuda()], torch.real(field))
                    self.im_spline_hash[key] = RegularGridInterpolator([self.unique_radii.to(self.device),
                                                                        torch.tensor([0.]).cuda()], torch.imag(field))
                
                #t_coeffs = natural_cubic_spline_coeffs(self.unique_radii, tvals)
                #p_coeffs = natural_cubic_spline_coeffs(self.unique_radii, pvals)
                #self.t_spline_hash[i][j] = NaturalCubicSpline(t_coeffs)
                #self.p_spline_hash[i][j] = NaturalCubicSpline(p_coeffs)
                #self.t_spline_hash[i][j] = RegularGridInterpolator([self.unique_radii,torch.tensor([0]).cuda()], tvals)
                #self.p_spline_hash[i][j] = RegularGridInterpolator([self.unique_radii, torch.tensor([0]).cuda()], pvals)
                # self.re_spline_hash[i][j] = RegularGridInterpolator([self.unique_radii,torch.tensor([0.]).cuda()], torch.real(field))
                # self.im_spline_hash[i][j] = RegularGridInterpolator([self.unique_radii,torch.tensor([0.]).cuda()], torch.imag(field))
                
    def computeTransmission(self,radii):
        At = {}
        Pt = {}
        # logger.info('computing interpolation table for wavelengths/angle')
        X,Y = torch.meshgrid(radii.flatten(),torch.tensor([0.],dtype=torch.float32).cuda(),indexing='ij')
        X1 = X.flatten()
        Y1 = Y.flatten()
        for i, wave in enumerate(self.waves_to_simuilate):
            for j, angle in enumerate(self.angles_to_simulate):
                key = str(wave)+str(angle)
                if self.device_hash:
                    X1 = X1.to(torch.device(self.device_hash[key]))
                    Y1 = Y1.to(torch.device(self.device_hash[key]))
                diffractor_re = diffraction_helper(self.re_spline_hash[key], Y1)
                diffractor_im = diffraction_helper(self.im_spline_hash[key], Y1)
                At[key] = checkpoint.checkpoint(diffractor_re.diffract, X1)
                Pt[key] = checkpoint.checkpoint(diffractor_im.diffract, X1)
                # At[wavehash][anglehash] = self.re_spline_hash[wavehash][anglehash]([X1,Y1]) #.evaluate(radii)
                # Pt[wavehash][anglehash] = self.im_spline_hash[wavehash][anglehash]([X1,Y1]) #.evaluate(radii)
                # logger.info("interpolation table for %f wavelength and %f angle constructed", self.waves_to_simuilate[wavehash],
                #            self.angles_to_simulate[anglehash])
        return At, Pt
    
    def forward(self, radii):
        At,Pt = self.computeTransmission(radii)
        t = {}
        # logger.info('computing forward transmission coefficients')
        for i, wave in enumerate(self.waves_to_simuilate):
            for j, angle in enumerate(self.angles_to_simulate):
                key = str(wave)+str(angle)
                t[key] = At[key]+1j*Pt[key]#*torch.exp(1j*Pt[wavehash][anglehash])
        # logger.info('successfully constructed transmission table')
        return t
#d = DifferentiableDiffraction(waves=np.linspace(0.43,0.6,20), angles=np.array([0.0]))
        