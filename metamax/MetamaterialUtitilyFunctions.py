#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:15:24 2024

@author: maxzhelyeznyakov
"""
import torch
import numpy as np
import time
import torch.utils.checkpoint as checkpoint
from scipy import special
import matplotlib.pyplot as plt
import scipy
from waveprop import rs
from hankel_torch import HankelTransform
from hankel_propagator import HankelPropagator
from numba import jit

def airy_disk(r):
    '''
    generated airy disk with
        \lim_{r->0} J1(r)/r = 0.5

    Parameters
    ----------
    r : float or numpy array
        radii

    Returns
    -------
    TYPE
        J1(r)/r.

    '''
    tol = 1e-7
    ret = np.zeros_like(r)
    ret[np.abs(r) < tol] = 0.5
    ret[np.abs(r) > tol] = special.j1(r[np.abs(r)>tol])/r[np.abs(r)>tol]
    return ret
def _padNumpy(source, pad_x, pad_y):
    '''
    

    Parameters
    ----------
    source : numpy array
        nxn numpy array
    pad_x, pad_y : int
        integer padding in x,y
    Returns
    -------
    numpy array
        padded numpy array

    '''
    source = torch.tensor(source)
    padded = torch.nn.functional.pad(input=source,
                                     pad=(pad_x, pad_x, pad_y, pad_y),
                                     mode='constant', value=0)
    return padded.numpy()

def _padTorch(source, pad=None):
    '''
    

    Parameters
    ----------
    source : nxn torch tensor
        to be padded
    pad : tuple, optional
        DESCRIPTION. The default is None.
        pad tuple for x,y

    Returns
    -------
    TYPE
        padded torch toensor

    '''
    Nx, Ny = source.shape
    if pad is None:
        pad_x = int (Nx // 2)
        pad_y = int (Ny // 2)
    else:
        pad_x = int(pad[0])
        pad_y = int(pad[1])
    return torch.nn.functional.pad(input=source,
                                   pad = (pad_x, pad_x, pad_y, pad_y),
                                   mode='constant', value = 0)

def unpad(source, pad_factor=1.):
    '''
    

    Parameters
    ----------
    source : torch tensor
        to be upadded
    pad_factor : TYPE, optional
        DESCRIPTION. The default is 1..

    Returns
    -------
    TYPE
        upadded torch tensor

    '''
    if pad_factor == 0.:
        return source
    
    *_, n_x, n_y = source.size()
    pad_x = int(n_x * pad_factor / (2 + 2 * pad_factor))
    pad_y = int(n_y * pad_factor / (2 + 2 * pad_factor))
    return source[pad_x:-pad_x, pad_y:-pad_y]    
def ideal_lens_phase_2D(wave, D, F, npix):
    '''
    coputes ideal lens phase for hyperboloid lens
    Parameters
    ----------
    wave : float
        wavelength
    D : float
        lens diamater
    F : float
        lens focal length
    npix : int
        number of pixels

    Returns
    -------
    phase : numpy float array
        ideal lens phase

    '''
    x = np.linspace(-D/2, D/2, npix)
    y = x
    [xx,yy] = np.meshgrid(x,y)
    phase = np.mod( 2*np.pi/wave * ( F - np.sqrt(xx**2 + yy**2 + F**2) ), 2*np.pi ) - np.pi
    
    return phase
def ideal_lens_phase_circular_symmetry(wave, D, F, p):
    '''
    

    coputes ideal lens phase for hyperboloid lens
    with circular symmetry
    Parameters
    ----------
    wave : float
        wavelength
    D : float
        lens diamater
    F : float
        lens focal length
    p : float
        grid resolution (or metalens periodicity)

    Returns
    -------
    phase : numpy float array
        ideal lens phase


    '''
    x = np.arange(0, D/2, p)
    phase = np.mod( - 2 * np.pi/wave * (np.sqrt(x**2+F**2) - F), 2*np.pi) - np.pi
    return phase

def initialize_circularly_symmetric_metsurface(wave, D, F, p, 
                                               rcwa_data):
    '''
    create a circularly symmetric metasurface lens
    Parameters
    ----------
    wave : float
        wavelength
    D : float
        lens diamater
    F : float
        lens focal length
    p : float
        grid resolution (or metalens periodicity)
    rcwa_data : pandas dataframe
        lookup table
    Returns
    -------
    r : python list
        list of metasurface parameters

    '''
    
    phase = ideal_lens_phase_circular_symmetry(wave, D, F, p)
    r = []
    df = rcwa_data.loc[(rcwa_data['wave']==wave) &
                       (rcwa_data['theta']==0.0) &
                       (rcwa_data['phi']==0.0)]
    for phi in phase:
        idp = np.argmin(np.abs(phi-df['p']))
        r.append(df['r1'].to_numpy()[idp])
    return r

def initialize_single_wavelength_metasurface(wave, D, F, p, 
                                               rcwa_data):
    '''
    create a lens
    Parameters
    ----------
    wave : float
        wavelength
    D : float
        lens diamater
    F : float
        lens focal length
    p : float
        grid resolution (or metalens periodicity)
    rcwa_data : pandas dataframe
        lookup table

    Returns
    -------
    r : numpy array
        list of metasurface parameters
    '''
    npix = int(D/p)
    phase = ideal_lens_phase_2D(wave, D, F, npix)
    phase = phase.flatten()
    r = []
    df = rcwa_data.loc[(rcwa_data['wave']==wave) & 
                       (rcwa_data['theta']==0.0) & 
                       (rcwa_data['phi']==0.0)]
    for phi in phase:
        idp = np.argmin(np.abs(phi-df['p']))
        r.append(df['r1'].to_numpy()[idp])
    r = np.array(r)
    r = r.reshape([npix,npix])
    return r

    
def diffraction_limited_psf_2D(R, F, npix, wavelength):
    '''
    
    computes theoretically ideal point spread function
    Parameters
    ----------
    R : Float
        lens radius
    F : float
        lens focal length
    npix : int
        number of pixels along a direction
    wavelength : float
        wavelength
    scale : int
        rescale psf by integer if working with super-wavelength periodicity

    Returns
    -------
    norm_intensity : 2d numpy array
        psf normalized to integral 1

    '''
    
    xmax = 2 * np.pi / wavelength * R * np.sin(np.arctan(R/F))
    x = np.linspace(-xmax,xmax, npix)
    y = np.linspace(-xmax,xmax, npix)
    xv,yv = np.meshgrid(x,y)
    r = np.sqrt(xv**2 + yv**2)
    
    intensity = 4 * airy_disk(r) ** 2 # 4 * (special.j1(r) / (r+0.001)) ** 2
    energy = np.sum(intensity)
    norm_intensity = intensity / energy
    return norm_intensity

def diffraction_limited_psf_1D(R, F, npix, wavelength):
    '''
    

    Parameters
    ----------
    R : numpy array
        lens radius
    F : float
        focal length
    npix : int
        number of pixels along a lens diameter (2 * npix radius)
    wavelength : float
        wavelength

    Returns
    -------
    norm_intensity : numpy array
        psf along a diameter of the lens

    '''
    xmax = 2 * np.pi / wavelength * R * np.sin(np.arctan(R/F))
    x = np.linspace(0,xmax, npix)
    
    r = np.sqrt(x**2)
    
    intensity = 4 * airy_disk(r) ** 2 # 4 * (special.j1(r) / (r+0.0001)) ** 2
    energy = np.sum(intensity)
    norm_intensity = intensity / energy
    return norm_intensity

def diffraction_limited_mtf_2D(R,F,npix, wavelength):
    '''
    

    Parameters
    ----------
    R : float
        radius
    F : float
        diameter
    npix : int
        pixels along lens diameter
    wavelength : float
        wavelength

    Returns
    -------
    mtf_norm : 2d numpy array
        fft of psf

    '''
    psf = diffraction_limited_psf_2D(R,F,npix,wavelength)
    mtf = np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    mtf_max = np.max(mtf)
    mtf_norm = mtf / mtf_max
    return mtf_norm

def diffraction_limited_mtf_1D(R,F,npix, wavelength):
    '''
    Parameters
    ----------
    R : float
        radius
    F : float
        diameter
    npix : int
        pixels along lens diameter
    wavelength : float
        wavelength

    Returns
    -------
    mtf_norm : 1d numpy array
        fft of psf

    '''
    psf = diffraction_limited_psf_1D(R,F,npix,wavelength)
    radial_grid = torch.linspace(0,R,npix)
    H = HankelTransform(order=0,radial_grid = radial_grid)
    
    IrH = H.to_transform_r(torch.tensor(psf) )
    mtf = np.abs(H.qdht(IrH).numpy())
    mtf_max = np.max(mtf)
    mtf_norm = mtf / mtf_max
    return mtf_norm

def hyperboloid_psf_2D(R,F,npix, wavelength):
    phase = ideal_lens_phase_2D(wavelength, 2*R, F, npix)
    apert = np.zeros_like(phase)
    x = np.linspace(-R,R,npix)
    y = x
    xx,yy = np.meshgrid(x,y)
    apert[xx**2+yy**2 < R**2] = 1
    #apert = 1
    nf = apert*np.exp(1j*phase)
    ff = rs.angular_spectrum(nf, wavelength, 2*R/npix, F)[0]
    psf = np.abs(ff)**2
    psf_norm = psf/np.sum(psf)
    return psf_norm

def diffraction_limited_psf_4fold(R,F,npix, wavelength, angle):
    xmax = 2 * np.pi / wavelength * R * np.sin(np.arctan(R/F))
    x = np.linspace(0,xmax, npix)
    y = x
    xx,yy = np.meshgrid(x,y)
    
    # angle with respect to normal incidence
    angle_rad = angle*np.pi/180
    
    shift = F*np.tan(angle_rad)
    r = np.sqrt((xx-shift)**2+(yy-shift)**2)
    
    intensity = 4 * airy_disk(r) ** 2 # 4 * (special.j1(r) / (r+0.0001)) ** 2
    energy = np.sum(intensity)
    norm_intensity = intensity / energy
    return norm_intensity

def diffraction_limited_mtf_4fold(R,F,npix,wavelength,angle):
    psf = diffraction_limited_psf_4fold(R,F,npix,wavelength,angle)
    mtf = np.abs(scipy.fftpack.dctn(psf,type=2))#np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    mtf_max = np.max(mtf)
    mtf_norm = mtf / mtf_max
    return mtf_norm
    
def diffraction_limited_psf_radial_symmetry_angle(R,F,npix, wavelength, angle):
    xmax = 2 * np.pi / wavelength * R * np.sin(np.arctan(R/F))
    x = np.linspace(0,xmax, npix)
    y = 0
    xx,yy = np.meshgrid(x,y)
    
    # angle with respect to normal incidence
    angle_rad = angle*np.pi/180
    
    shift = F*np.tan(angle_rad)
    r = x # np.sqrt((x-shift)**2)
    
    intensity = 4 * airy_disk(r - shift) ** 2 # 4 * (special.j1(r) / (r+0.0001)) ** 2
    energy = np.sum(intensity)
    norm_intensity = intensity / energy
    return norm_intensity

def diffraction_limited_mtf_radial_symmetry_angle(R,F,npix,wavelength,angle):
    psf = diffraction_limited_psf_radial_symmetry_angle(R,F,npix,wavelength,angle)
    #mtf = np.abs(scipy.fftpack.dctn(psf,type=2))#np.abs(np.fft.fftshift(np.fft.fft2(psf)))
    radial_grid = torch.linspace(0,R,npix)
    H = HankelTransform(order=0,radial_grid = radial_grid)
    
    IrH = H.to_transform_r(torch.tensor(psf) )
    mtf = np.abs(H.qdht(IrH).numpy())
    mtf_max = np.max(mtf)
    mtf_norm = mtf / mtf_max
    return mtf_norm
def dump_cs_detector_field_to_mat_file(system,diffraction_model, fname):
    import scipy.io as sio
    R = system.D/2
    dx = system.dx
    waves = system.wavelengths
    angles = system.angles
    r = np.linspace(0,R,int(R/dx))
    x = np.concatenate([r[::-1],r[1:]])
    y = x
    xx,yy = np.meshgrid(x,y)
    rr = np.sqrt(xx**2+yy**2)
    aperture = np.zeros_like(rr)
    aperture[rr**2 < R**2] = 1
    fields_to_dump = np.zeros([len(waves),rr.shape[0],rr.shape[1]],dtype=np.complex128)
    i = 0
    radii = system.layers[0].parameters.detach().cpu().numpy()
    rads2d = np.interp(rr,r,radii[:,0])
    fields = diffraction_model.forward(torch.tensor(rads2d).cuda())
    # import copy
    # system2 = copy.deepcopy(system)
    for i, wave in enumerate(waves):
        for j, angle in enumerate(angles):
             propagator = HankelPropagator(wave, radial_grid=torch.tensor(r).cuda(), dz=system.zlist[-1])
             field = system.layers[0].get_transmitted_field(wave,angle).cpu().detach().numpy()
             field2d = aperture*fields[i][j].cpu().numpy().reshape(rads2d.shape)
             psf2d = np.abs(rs.angular_spectrum(field2d, wave, dx, system.zlist[-1])[0])**2
             psf2d /= np.max(psf2d)
             print(np.unravel_index(np.argmax(psf2d),psf2d.shape))
             psf1d = torch.abs(propagator.propagate(system.layers[0].get_transmitted_field(wave,angle)[:,0]))**2
             psf1d /= torch.max(psf1d)
             
             fields_to_dump[i,:,:] = field2d
             
    mdict = {}
    mdict['phase_profile'] = fields_to_dump
    sio.savemat(fname,mdict)
def strehl_plot(system, diffraction_model):
    import scipy.io as sio
    R = system.D/2
    dx = system.dx
    waves = system.wavelengths
    angles = system.angles
    r = np.linspace(0,R,int(R/dx))
    x = np.concatenate([r[::-1],r[1:]])
    y = x
    xx,yy = np.meshgrid(x,y)
    rr = np.sqrt(xx**2+yy**2)
    aperture = np.zeros_like(rr)
    aperture[rr**2 < R**2] = 1
    fields_to_dump = np.zeros([len(waves),rr.shape[0],rr.shape[1]],dtype=np.complex128)
    i = 0
    radii = system.layers[0].parameters.detach().cpu().numpy()
    #rads2d = np.interp(rr,r,radii[:,0])
    import scipy
    f = scipy.interpolate.interp1d(r,radii[:,0],kind='nearest',fill_value="extrapolate")
    rads2d = f(rr)
    fields = diffraction_model.forward(torch.tensor(rads2d).cuda())
    # import copy
    # system2 = copy.deepcopy(system)
    ht = HankelTransform(order=0,radial_grid=torch.tensor(r).cuda(),device='cuda')
    
    kx = np.fft.fftshift(np.fft.fftfreq(len(x),dx)) * 2 *np.pi
    ky = np.fft.fftshift(np.fft.fftfreq(len(x),dx))* 2 *np.pi
    kxx,kyy = np.meshgrid(kx,ky)
    krr = np.sqrt(kxx**2+kyy**2)
    for i, wave in enumerate(waves):
        for j, angle in enumerate(angles):
             propagator = HankelPropagator(wave, radial_grid=torch.tensor(r).cuda(), dz=system.zlist[-1])
             field = system.layers[0].get_transmitted_field(wave,angle).cpu().detach().numpy()
             field2d = aperture*fields[i][j].cpu().numpy().reshape(rads2d.shape)
             psf2d = np.abs(rs.angular_spectrum(field2d, wave, dx, system.zlist[-1])[0])**2
             psf2d /= np.max(psf2d)
             mtf2d = np.abs(np.fft.fftshift(np.fft.fft2(psf2d)))
             mtf2d /= np.max(mtf2d)
             print(np.unravel_index(np.argmax(psf2d),psf2d.shape))
             psf1d = torch.abs(propagator.propagate(system.layers[0].get_transmitted_field(wave,angle)[:,0]))**2
             psf1d /= torch.max(psf1d)
             mtfr = ht.to_transform_r(torch.tensor(psf1d[:,0]).cuda())
             mtfkr = torch.abs(ht.qdht(mtfr)).cpu().detach().numpy()
             mtfkr /= np.max(mtfkr)
             plt.figure()
             plt.plot(psf1d.cpu().detach())
             plt.plot(psf2d[len(r)-1,len(r)-1:])
             plt.show()
             
             plt.figure()
             plt.plot(ht.kr.cpu().numpy(),mtfkr)
             plt.plot(krr[len(r)-1,len(r)-1:],mtf2d[len(r)-1,len(r)-1:])
             plt.show()
    
        
        
    