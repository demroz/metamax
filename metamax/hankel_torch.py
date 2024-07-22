#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:09:53 2024

@author: noise
"""

import torch
import torch.special as torch_special
import scipy.special as scipy_special

from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)

import matplotlib.pyplot as plt

class HankelTransform:
    
    def __init__(self, order : int,
                 max_radius: float = None,
                 n_points: int = None,
                 radial_grid: torch.Tensor = None,
                 k_grid: torch.Tensor = None,
                 device: str = 'cpu'):
        """Constructor"""
        self._device = torch.device(device)
        usage = 'Either radial_grid or k_grid or both max_radius and n_points must be supplied'
        if radial_grid is None and k_grid is None:
            if max_radius is None or n_points is None:
                raise ValueError(usage)
        elif k_grid is not None:
            if max_radius is not None or n_points is not None or radial_grid is not None:
                raise ValueError(usage)
            assert k_grid.ndim == 1, 'k grid must be a 1d torch.Tensor'
            n_points = k_grid.size()[0]
        elif radial_grid is not None:
            if max_radius is not None or n_points is not None:
                raise ValueError(usage)
            assert radial_grid.ndim == 1, 'Radial grid must be a 1d torch.Tensor'
            max_radius = torch.max(radial_grid)
            n_points = radial_grid.size()[0]
        else:
            raise ValueError(usage)  # pragma: no cover - backup case: cannot currently be reached

        self._order = order
        self._n_points = n_points
        self._original_radial_grid = radial_grid
        self._original_k_grid = k_grid
        
        # Calculate N+1 roots must be calculated before max_radius can be derived from k_grid
        alpha = scipy_special.jn_zeros(self._order, self._n_points+1)
        self.alpha = torch.Tensor(alpha[0:-1])
        self.alpha_n1 = alpha[-1]

        if k_grid is not None:
            v_max = torch.max(k_grid) / (2 * torch.pi)
            max_radius = self.alpha_n1 / (2 * torch.pi * v_max)
        self._max_radius = float(max_radius)
        
        # Calculate co-ordinate vectors
        self.r = self.alpha * self._max_radius / self.alpha_n1
        self.v = self.alpha / (2 * torch.pi * self._max_radius)
        self.kr = 2 * torch.pi * self.v
        self.v_max = self.alpha_n1 / (2 * torch.pi * self._max_radius)
        self.S = self.alpha_n1
        
        # Calculate hankel matrix and vectors
        jp = scipy_special.jv(order, (self.alpha[:, None] @ self.alpha[None, :]) / self.S)
        jp1 = torch.abs(scipy_special.jv(order + 1, self.alpha))
        self.T = (2 * jp / ((jp1[:, None] @ jp1[None, :]) * self.S)).to(self._device)
        self.JR = (jp1 / self._max_radius).to(self._device)
        self.JV = (jp1 / self.v_max).to(self._device)
        
        self.T = self.T.type(torch.complex128)
        self.JR = self.JR.type(torch.complex128)
        self.JV = self.JV.type(torch.complex128)
        self.r = self.r.to(self._device)
        pass
    
    def to_transform_r(self, function: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """Interpolate a function, assumed to have been given at the original radial
        grid points used to construct the ``HankelTransform`` object onto the grid required
        of use in the QDHT algorithm.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in radius, then it needs the function to transform to be sampled at a specific
        grid before it can be passed to :meth:`.HankelTransform.qdht`. This method provides
        a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            :attr:`~.HankelTransform.original_radial_grid`.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the radial dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function suitable to passing to
            :meth:`HankelTransform.qdht` (sampled at ``self.r``)
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self._original_radial_grid, function, self.r, axis)

    def to_original_r(self, function: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """Interpolate a function, assumed to have been given at the Hankel transform points
        ``self.r`` (as returned by :meth:`HankelTransform.iqdht`) back onto the original grid
        used to construct the ``HankelTransform`` object.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in radius, it may be useful to convert back to this grid after a IQDHT.
        This method provides a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            ``self.r``.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the radial dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_radial_grid`.
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self.r, function, self._original_radial_grid, axis)

    def to_transform_k(self, function: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """Interpolate a function, assumed to have been given at the original k
        grid points used to construct the ``HankelTransform`` object onto the grid required
        of use in the IQDHT algorithm.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in :math:`k`, then it needs the function to transform to be sampled at a specific
        grid before it can be passed to :meth:`.HankelTransform.iqdht`. This method provides
        a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the k points
            :attr:`~.HankelTransform.original_k_grid`.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the frequency dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function suitable to passing to
            :meth:`HankelTransform.qdht` (sampled at ``self.kr``)
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self._original_k_grid, function, self.kr, axis)

    def to_original_k(self, function: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """Interpolate a function, assumed to have been given at the Hankel transform points
        ``self.k`` (as returned by :meth:`HankelTransform.qdht`) back onto the original grid
        used to construct the ``HankelTransform`` object.

        If the the ``HankelTransform`` object was constructed with a (say) equally-spaced
        grid in :math:`k`, it may be useful to convert back to this grid after a QDHT.
        This method provides a convenient way of doing this.

        :parameter function: The function to be interpolated. Specified at the radial points
            ``self.k``.
        :type function: :class:`numpy.ndarray`
        :parameter axis: Axis representing the frequency dependence of `function`.
        :type axis: :class:`int`

        :return: Interpolated function at the points held in :attr:`~.HankelTransform.original_k_grid`.
        :rtype: :class:`numpy.ndarray`
        """
        if function.ndim == 1:
            axis = 0
        return _spline(self.kr, function, self._original_k_grid, axis)
    
    def qdht(self, fr: torch.Tensor) -> torch.Tensor:
        r"""QDHT: Quasi Discrete Hankel Transform

        Performs the Hankel transform of a function of radius, returning
        a function of frequency.

        .. math::
            f_v(v) = \mathcal{H}^{-1}\{f_r(r)\}

        .. warning:
            The input function must be sampled at the points ``self.r``, and the output
            will be sampled at the points ``self.v`` (or equivalently ``self.kr``)

        :parameter fr: Function in real space as a function of radius (sampled at ``self.r``)
        :type fr: :class:`numpy.ndarray`
        :parameter axis: Axis over which to compute the Hankel transform.
        :type axis: :class:`int`

        :return: Function in frequency space (sampled at ``self.v``)
        :rtype: :class:`numpy.ndarray`
        """
        #if (fr.ndim == 1) or (axis == -2):
        #jr, jv = self._get_scaling_factors(fr)

        fv = (self.JV + 0j) * torch.matmul( (self.T+0j), (fr / (self.JR[:,None]+0j)))[:,0]
        return fv
    
    def iqdht(self, fv: torch.Tensor) -> torch.Tensor:
        r"""IQDHT: Inverse Quasi Discrete Hankel Transform

        Performs the inverse Hankel transform of a function of frequency, returning
        a function of radius.

        .. math::
            f_r(r) = \mathcal{H}^{-1}\{f_v(v)\}

        :parameter fv: Function in frequency space (sampled at self.v)
        :type fv: :class:`numpy.ndarray`
        :parameter axis: Axis over which to compute the Hankel transform.
        :type axis: :class:`int`

        :return: Radial function (sampled at self.r) = IHT(fv)
        :rtype: :class:`numpy.ndarray`
        """
        
        fr = (self.JR+0j) * torch.matmul(self.T, (fv / (self.JV+0j))[:,None])[:,0]
        return fr

def _spline(x0: torch.Tensor, y0: torch.Tensor, x: torch.Tensor, axis: int) -> torch.Tensor:
    yre = torch.real(y0[:,None])
    if torch.is_complex(y0):
        yim = torch.imag(y0[:,None])
    else:
        yim = 0*yre
    
    coeffs_re = natural_cubic_spline_coeffs(x0, yre)
    coeffs_im = natural_cubic_spline_coeffs(x0, yim)
    
    spline_re = NaturalCubicSpline(coeffs_re)
    spline_im = NaturalCubicSpline(coeffs_im)
    return spline_re.evaluate(x)+1j*spline_im.evaluate(x)
