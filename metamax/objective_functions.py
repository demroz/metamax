#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:47:11 2024

@author: maxzhelyeznyakov
"""
import torch
import matplotlib.pyplot as plt
from hankel_torch import HankelTransform
import torch_dct as dct
PLOT = True
import logging
logger = logging.getLogger(__name__)

class MTF_Correlation():
    def __init__(self, ideal_mtfs = None, D=None, npix=None):
        self.ideal_mtfs = ideal_mtfs
        self.radial_grid = torch.linspace(0, D/2, npix).cuda()
        self.H = HankelTransform(order=0,radial_grid = self.radial_grid, device='cuda')
        
    def normalized_corr(self, x, y):
        if len(x.shape) > 1:
            x = x[:,0]
        if len(y.shape) > 1:
            y = y[:,0]
        return torch.sum(x*y) / torch.sqrt( torch.sum(x*x) * torch.sum(y*y))
    
    def forward(self, detector):
        incident_field = detector.get_incident_field()
        ret = []
        for key in incident_field.keys():
            psf_design = torch.abs(incident_field[key])
            
            psf_design_norm = psf_design / torch.sum(psf_design)
            
            IrH = self.H.to_transform_r(psf_design_norm[:,0] )
            mtf = torch.abs(self.H.qdht(IrH))
            
            mtf_design = torch.abs(torch.fft.fft(psf_design_norm))
            mtf_design_norm = mtf_design / torch.max(mtf_design)
            
            plt.figure()
            plt.plot(mtf_design_norm.detach().cpu().numpy())
            plt.plot(self.ideal_mtfs[key].detach().cpu().numpy())
            plt.show()
            # ret.append(torch.sum(mtf_design_norm)/torch.sum(self.ideal_mtfs[key].cuda()))
            ret.append(self.normalized_corr(mtf_design_norm, self.ideal_mtfs[key].cuda()))
            pass
            #ret.append(self.normalized_corr(mtf, self.ideal_mtfs[key]))
            #ret.append(psf_design[int(len(psf_design)/2)])
        return -1*torch.prod(torch.stack(ret))
            
class MTF_AND_PSF_CORRELATION():
    def __init__(self, ideal_mtfs = None, ideal_psfs = None, D = None, npix = None):
        self.ideal_mtfs = ideal_mtfs
        self.ideal_psfs = ideal_psfs
        self.radial_grid = torch.linspace(0, D/2, npix).cuda()
        self.H = HankelTransform(order=0,radial_grid = self.radial_grid, device='cuda')
        
    def normalized_corr(self, x, y):
        if len(x.shape) > 1:
            x = x[:,0]
        if len(y.shape) > 1:
            y = y[:,0]
        return torch.sum(x*y) / torch.sqrt( torch.sum(x*x) * torch.sum(y*y))
    
    def forward(self, detector):
        incident_field = detector.get_incident_field()
        ret = []
        
        for key in incident_field.keys():
            psf_design = torch.abs(incident_field[key])**2
            psf_design_norm = psf_design / torch.sum(psf_design)
            IrH = self.H.to_transform_r(psf_design_norm[:,0] )
            mtf_design = torch.abs(self.H.qdht(IrH))
            mtf_design_norm = mtf_design / torch.max(mtf_design)
            mtf_design_norm = mtf_design_norm
            psf_corr = self.normalized_corr(psf_design_norm, self.ideal_psfs[key])
            mtf_corr = self.normalized_corr(mtf_design_norm, self.ideal_mtfs[key])
            ret.append(torch.sum(mtf_design_norm) / torch.sum(self.ideal_mtfs[key]))
            # ret.append(mtf_corr)
            logger.info('psf_corr %f mtf_corr %f strehl_ratio %f',
                        psf_corr, mtf_corr, ret[-1])
            
            plt.figure()
            plt.plot(mtf_design_norm.detach().cpu())
            plt.plot(self.ideal_mtfs[key].cpu())
            plt.title(key)
            plt.show()
        return torch.max(-1*torch.log(torch.stack(ret))) #-1*torch.log(torch.prod(torch.stack(ret))) #torch.log(torch.prod(torch.stack(ret)))
    
class MTF_AND_PSF_CORRELATION_2D():
    def __init__(self, ideal_mtfs = None, ideal_psfs = None, D = None, npix = None):
        self.ideal_mtfs = ideal_mtfs
        self.ideal_psfs = ideal_psfs
        self.radial_grid = torch.linspace(0, D/2, npix).cuda()
       
        
    def normalized_corr(self, x, y):
        return torch.sum(x*y) / torch.sqrt( torch.sum(x*x) * torch.sum(y*y))
    
    def forward(self, detector):
        incident_field = detector.get_incident_field()
        ret = []
        
        for key in incident_field.keys():
            psf_design = torch.abs(incident_field[key])**2
            psf_design_norm = psf_design / torch.sum(psf_design)
            
            mtf_design = torch.abs(torch.fft.fftshift(torch.fft.fft2(psf_design_norm)))
            mtf_design_norm = mtf_design / torch.max(mtf_design)
            
            mtf_corr = self.normalized_corr(mtf_design_norm, self.ideal_mtfs[key])
            strehl = torch.sum(mtf_design_norm) / torch.sum(self.ideal_mtfs[key])
            ret.append(strehl)
        if PLOT:
            ret_plot = torch.stack(ret).cpu().detach().numpy()
            plt.figure()
            plt.plot(ret_plot)
            plt.show()
            
        return -1*torch.log(torch.prod(torch.stack(ret)))
        
class MTF_AND_PSF_CORRELATION_4FOLD():
    def __init__(self, ideal_mtfs = None, ideal_psfs = None, D = None, npix = None, device = 'cuda'):
        self.ideal_mtfs = ideal_mtfs
        self.ideal_psfs = ideal_psfs
        self.device=torch.device(device)
        
    def normalized_corr(self, x, y):
        return torch.sum(x*y) / torch.sqrt( torch.sum(x*x) * torch.sum(y*y))
    
    def forward(self, detector):
        incident_field = detector.get_incident_field()
        ret = []
        strehl = []
        for key in incident_field.keys():
            psf_design = torch.abs(incident_field[key])**2
            psf_design_norm = psf_design / torch.sum(psf_design)
            ix,iy = torch.unravel_index(torch.argmax(self.ideal_psfs[key]),psf_design.shape)
            mtf_design = torch.abs(dct.dct_2d(psf_design_norm))
            mtf_design_norm = mtf_design / torch.max(mtf_design)
            
            psf_corr = self.normalized_corr(psf_design_norm, self.ideal_psfs[key])
            mtf_corr = self.normalized_corr(mtf_design_norm, self.ideal_mtfs[key])
            sr = torch.sum(mtf_design_norm) / torch.sum(self.ideal_mtfs[key])
            strehl.append(sr.to(self.device))
            #ret.append(psf_design_norm[ix,iy])
            ret.append(sr.to(self.device))
            logger.info('key %s psf_corr %f mtf_corr %f strehl_ratio %f',
                        key,
                        psf_corr,
                        mtf_corr,
                        torch.sum(mtf_design_norm) / torch.sum(self.ideal_mtfs[key]))
            #ret.append(mtf_corr)
        if PLOT:
            strehl_plot = torch.stack(strehl).cpu().detach().numpy()
            plt.figure()
            plt.plot(strehl_plot)
            plt.show()
            
        return torch.max(-1*torch.log(torch.stack(ret).cuda()))
        