import sys
import numpy as np
from astropy.io import fits
from os.path import basename
from copy import deepcopy as copy
import os
from astropy.wcs import WCS
import time

from lvmdap.pyFIT3D.common.io import ReadArguments
from lvmdap.pyFIT3D.common.io import get_data_from_fits, array_to_fits
from lvmdap.pyFIT3D.common.constants import __FWHM_to_sigma__
#from lvmdap.pyFIT3D.common.tools import flux_elines_cube_EW,momana_spec_wave
from lvmdap.pyFIT3D.common.tools import flux_elines_cube_EW,momana_spec_wave

import numpy as np
from copy import deepcopy as copy
from os.path import basename, isfile
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from astropy.io.fits.verify import VerifyWarning

from lvmdap.pyFIT3D.common.stats import median_filter as st_median_filter
from lvmdap.pyFIT3D.common.stats import hyperbolic_fit_par, std_m, pdl_stats, _STATS_POS
from lvmdap.pyFIT3D.common.io import trim_waves, get_data_from_fits, get_wave_from_header
from lvmdap.pyFIT3D.common.constants import __c__, __sigma_to_FWHM__, __indices__, _INDICES_POS
from lvmdap.pyFIT3D.common.io import output_spectra, array_to_fits, write_img_header, print_verbose
from lvmdap.pyFIT3D.common.constants import __Hubble_constant__, __Omega_matter__, __Omega_Lambda__
from lvmdap.pyFIT3D.common.constants import __solar_luminosity__, __solar_metallicity__, _figsize_default
from scipy.ndimage import gaussian_filter1d
from lvmdap.pyFIT3D.common.tools import vel_eline
from lvmdap.pyFIT3D.common.io import array_to_fits, get_data_from_fits, trim_waves
from astropy.table import Table
from astropy.table import vstack as vstack_table
from lvmdap.dap_tools import read_tab_EL


from matplotlib import use as mpl_use
mpl_use('Agg')
import matplotlib.pyplot as plt


def flux_elines_RSS_EW(flux__wyx, input_header, n_MC, elines_list, vel__yx, sigma__yx,
                        eflux__wyx=None, flux_ssp__wyx=None, w_range=60, plot=0):
    nx, nw = flux__wyx.shape
    print(nx,nw,plot)
    crpix = input_header['CRPIX1']
    crval = input_header['CRVAL1']
    cdelt = input_header['CDELT1']
    w_min=crval+cdelt*(0-crpix)
    w_max=crval+cdelt*(nw-crpix)
    flux__wyx[np.isnan(flux__wyx)] = 0
    if eflux__wyx is not None:
        median_data = np.nanmedian(eflux__wyx)
        np.clip(eflux__wyx, -5*median_data, 5*median_data, out=eflux__wyx)
        eflux__wyx[np.isnan(eflux__wyx)] = 5*median_data
    else:
        eflux__wyx = np.zeros_like(flux__wyx)
    if flux_ssp__wyx is not None:
        mean_data = np.nanmean(flux_ssp__wyx)
        flux_ssp__wyx[np.isnan(flux_ssp__wyx)] = mean_data
    if not isinstance(vel__yx, np.ndarray):
        vel__yx = vel__yx*np.ones([nx])    
    [nx1] = vel__yx.shape
    if not isinstance(sigma__yx, np.ndarray):
        sigma__yx = sigma__yx*np.ones(nx1)
    if nx1 < nx:
        nx = nx1
    wavelengths = np.array([])
    name_elines = np.array([])
    ne = 0
    with open(elines_list) as fp:
        line = fp.readline()
        while line:
            if not line.startswith('#'):
                tmp_line = line.strip().split()
                if len(tmp_line) > 1:
                    wave_now=float(tmp_line[0])
                    if ((wave_now>w_min)&(wave_now<w_max)):
                        wavelengths = np.append(wavelengths, float(tmp_line[0]))
                        name_elines = np.append(name_elines, tmp_oline[1])
                        ne += 1
#                else:
#                    wavelengths = np.append(wavelengths, 0)
#                    name_elines = np.append(name_elines, ' ')
 
            line = fp.readline()
    NZ_out = ne * 4 * 2
    out = np.zeros([NZ_out, nx])
    print('{} emission lines'.format(ne))
    labels = ['flux', 'vel', 'disp', 'EW','e_flux', 'e_vel', 'e_disp', 'e_EW']
#    print('PASO fe')
    for i, name in enumerate(name_elines):
        print(i,name)
        _tmp = [i, i + ne, i + 2*ne, i + 3*ne, i + 4*ne, i + 5*ne, i + 6*ne, i + 7*ne]
        for j, I in enumerate(_tmp):
            header_label = 'NAME{}'.format(I)
            wavelen_label = 'WAVE{}'.format(I)
            units_label = 'UNIT{}'.format(I)
            if ('vel'==labels[j]) | ('e_vel'==labels[j]):
                units = 'km/s'
            if ('disp'==labels[j]) | ('e_disp'==labels[j]):
                units = 'km/s'
            if ('flux'==labels[j]) | ('e_flux'==labels[j]):
                units = '10^-16 erg/s/cm^2'
            if ('EW'==labels[j]) | ('e_EW'==labels[j]):
                units = 'Angstrom'
            input_header[header_label] = '{} {}'.format(labels[j], name)
            input_header[wavelen_label] = '{}'.format(wavelengths[i])
            input_header[units_label] = "{}".format(units)
    for k in np.arange(0, ne):
        f_m = 1 + vel__yx / __c__
        start_w_m = wavelengths[k]*f_m - 1.5*__sigma_to_FWHM__*sigma__yx
        end_w_m = wavelengths[k]*f_m + 1.5*__sigma_to_FWHM__*sigma__yx
        start_i_m = ((start_w_m - crval)/cdelt).astype(int)
        end_i_m = ((end_w_m - crval)/cdelt).astype(int)
        d_w_m = (end_w_m - start_w_m)/4
        start_i_lim_m = ((start_w_m - crval - w_range)/cdelt).astype(int)
        end_i_lim_m = ((end_w_m - crval + w_range)/cdelt).astype(int)
        mask1 = (start_i_m < 0) | (end_i_m < 0)
        mask2 = (start_i_m >= nw - 1) | (end_i_m >= nw - 1)
        mask3 = (start_i_lim_m >= nw - 1) | (end_i_lim_m >= nw - 1)
        sigma_mask = sigma__yx > 0
        mask = (~(mask1 | mask2 | mask3)) & sigma_mask
        [i_m] = np.where(mask)
        for i in i_m:
            I0, vel_I1, I2, EW, s_I0, s_vel_I1, s_I2, e_EW = momana_spec_wave(
                gas_flux__w=flux__wyx[i, :],
                egas_flux__w=eflux__wyx[i, :],
                wave=wavelengths[k],
                vel=vel__yx[i],
                sigma=sigma__yx[i],
                crval=crval, cdelt=cdelt,
                n_MC=n_MC, flux_ssp__w=flux_ssp__wyx[i, :],
                f_width=2,
            )
            I2=1.217*(I2/2.354)**2
            sig_I2 = I2#/2.354
            w_m_guess = wavelengths[k]*(1+vel_I1/__c__)
            #
            # Aperture correction
            #
            w_plot = crval+np.arange(0,flux__wyx[i, :].shape[0])*cdelt
            mask_wave = (w_plot>start_w_m[0]) & (w_plot<end_w_m[0])
            G_f = np.exp(-0.5*((w_plot-wavelengths[k]*(1+vel_I1/__c__))/sig_I2)**2)
            sum_G = sig_I2 * np.sqrt(2*np.pi)
            sum_G_obs = np.sum(G_f[mask_wave]*cdelt)
            A_corr = sum_G/sum_G_obs
            #print(f'# A_corr = {A_corr}')
            if (np.isfinite(A_corr)):
                I0=I0*A_corr
            G = (I0/(sig_I2*np.sqrt(2*np.pi)))*G_f            
            res_NP = flux__wyx[i, :]-G            
            if (plot==1):
                fig,ax = plt.subplots(1,1,figsize=(8,4.5))
                ax.plot(w_plot[mask_wave],\
                         flux__wyx[i, :][mask_wave],color='black',linewidth=1.5,label='obs.')
                ax.plot([w_m_guess,w_m_guess],[0.8*np.max(flux__wyx[i, :][mask_wave]),np.max(flux__wyx[i, :][mask_wave])],color='red',alpha=0.8)
                ax.plot(w_plot[mask_wave],\
                         G[mask_wave],color='blue',alpha=0.8,label='NP mod.')
                ax.plot(w_plot[mask_wave],\
                         res_NP[mask_wave],color='orange',alpha=0.8,label='NP res.')
                ax.set_title(f'NP analysis: {wavelengths[k]}')
                ax.set_xlabel(r'wavelength ($\AA$)')
                ax.set_ylabel(r'flux')
                ax.legend()
                plt.tight_layout()
                plt.show(block=True)
            out[k, i] = I0 #*cdelt*np.sqrt(2*np.pi)
            out[ne + k, i] = vel_I1
            out[2*ne + k, i] = I2
            out[3*ne + k, i] = EW
            out[4*ne + k, i] = s_I0
            out[5*ne + k, i] = s_vel_I1
            out[6*ne + k, i] = s_I2
            out[7*ne + k, i] = e_EW
            print('# :',I0,vel_I1, 1.217*(I2/2.354)**2, EW, s_I0, s_vel_I1, s_I2, e_EW)
        
        print('{}/{}, {},{} DONE'.format(k + 1, ne, wavelengths[k], name_elines[k]))
    
    return out, input_header




def flux_elines_RSS_EW_cl_250427(flux__wyx, input_header, n_MC, tab_el, vel__yx, sigma__yx,
                          eflux__wyx=None, flux_ssp__wyx=None, w_range=60, plot=0, figsize=(12,4.5), verbose=False):
    nx, nw = flux__wyx.shape
    crpix = input_header['CRPIX1']
    crval = input_header['CRVAL1']
    cdelt = input_header['CDELT1']
    wl__w = crval+cdelt*np.arange(nw)
    m_flux_rss = flux__wyx
    m_flux=m_flux_rss[0,:]
    #tab_el=read_tab_EL(elines_list)

    
    fe_m_data, fe_m_hdr = flux_elines_RSS_EW_tab(flux__wyx, input_header, n_MC, tab_el, vel__yx,\
                                              sigma__yx, eflux__wyx=eflux__wyx, flux_ssp__wyx=flux_ssp__wyx,\
                                                 w_range=w_range, plot=plot, verbose=verbose)


     
    if (plot==1):
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(wl__w,\
                m_flux_rss[0,:],color='black',linewidth=1.5,label='obs.')
        ne_np = int(fe_m_data.shape[0]/8)
        for k in np.arange(ne_np):
            I0 = fe_m_data[k, 0] #*cdelt*np.sqrt(2*np.pi)
            vel_I1 = fe_m_data[ne_np + k, 0] 
            I2 = fe_m_data[2*ne_np + k, 0]
            w_m_guess = tab_el[k]['wl']*(1+vel_I1/300000)
            G_f = np.exp(-0.5*((wl__w - w_m_guess)/I2)**2)
            if (k==0): 
                G = (I0/(I2*np.sqrt(2*np.pi)))*G_f
            else:
                G = G + (I0/(I2*np.sqrt(2*np.pi)))*G_f                
        ax.plot([w_m_guess,w_m_guess],[0.8*np.nanmax(m_flux),np.nanmax(m_flux)],color='red',alpha=0.8)
        ax.plot(wl__w,G,color='blue',alpha=0.8,label='NP mod.')
        ax.plot(wl__w,m_flux_rss[0,:]-G,color='orange',alpha=0.8,label='NP res.')
        ax.set_title(f'NP analysis')
        ax.set_xlabel(r'wavelength ($\AA$)')
        ax.set_ylabel(r'flux')
        ax.legend()
        plt.tight_layout()
        plt.show(block=True)

    return fe_m_data, fe_m_hdr



def flux_elines_RSS_EW_tab_250427(flux__wyx, input_header, n_MC, tab_elines, vel__yx, sigma__yx,
                           eflux__wyx=None, flux_ssp__wyx=None, w_range=60, plot=0,verbose=False):
    nx, nw = flux__wyx.shape
    print(nx,nw,plot)
    crpix = input_header['CRPIX1']
    crval = input_header['CRVAL1']
    cdelt = input_header['CDELT1']
    w_min=crval+cdelt*(0-crpix)
    w_max=crval+cdelt*(nw-crpix)
    flux__wyx[np.isnan(flux__wyx)] = 0
    if eflux__wyx is not None:
        median_data = np.nanmedian(eflux__wyx)
        np.clip(eflux__wyx, -5*median_data, 5*median_data, out=eflux__wyx)
        eflux__wyx[np.isnan(eflux__wyx)] = 5*median_data
    else:
        eflux__wyx = np.zeros_like(flux__wyx)
    if flux_ssp__wyx is not None:
        mean_data = np.nanmean(flux_ssp__wyx)
        flux_ssp__wyx[np.isnan(flux_ssp__wyx)] = mean_data
    if not isinstance(vel__yx, np.ndarray):
        vel__yx = vel__yx*np.ones([nx])    
    [nx1] = vel__yx.shape
    if not isinstance(sigma__yx, np.ndarray):
        sigma__yx = sigma__yx*np.ones(nx1)
    if nx1 < nx:
        nx = nx1
   # wavelengths = np.array([])
   # name_elines = np.array([])
   # ne = 0
   
    wavelengths = tab_elines['wl']

    name_elines = tab_elines['id']
    ne = len(tab_elines)
#    print(f'*** ne = {ne}')
    NZ_out = ne * 4 * 2
    out = np.zeros([NZ_out, nx])
    print('{} emission lines'.format(ne))
    labels = ['flux', 'vel', 'disp', 'EW','e_flux', 'e_vel', 'e_disp', 'e_EW']
#    print('PASO fe')
    for i, name in enumerate(name_elines):
        print(i,name,wavelengths[i])
        _tmp = [i, i + ne, i + 2*ne, i + 3*ne, i + 4*ne, i + 5*ne, i + 6*ne, i + 7*ne]
        for j, I in enumerate(_tmp):
            header_label = 'NAME{}'.format(I)
            wavelen_label = 'WAVE{}'.format(I)
            units_label = 'UNIT{}'.format(I)
            if ('vel'==labels[j]) | ('e_vel'==labels[j]):
                units = 'km/s'
            if ('disp'==labels[j]) | ('e_disp'==labels[j]):
                units = 'km/s'
            if ('flux'==labels[j]) | ('e_flux'==labels[j]):
                units = '10^-16 erg/s/cm^2'
            if ('EW'==labels[j]) | ('e_EW'==labels[j]):
                units = 'Angstrom'
            input_header[header_label] = '{} {}'.format(labels[j], name)
            input_header[wavelen_label] = '{}'.format(str(wavelengths[i]))
            input_header[units_label] = "{}".format(units)
    for k in np.arange(0, ne):
        f_m = 1 + vel__yx / __c__
        start_w_m = wavelengths[k]*f_m - 1.5*__sigma_to_FWHM__*sigma__yx
        end_w_m = wavelengths[k]*f_m + 1.5*__sigma_to_FWHM__*sigma__yx
        start_i_m = ((start_w_m - crval)/cdelt).astype(int)
        end_i_m = ((end_w_m - crval)/cdelt).astype(int)
        d_w_m = (end_w_m - start_w_m)/4
        start_i_lim_m = ((start_w_m - crval - w_range)/cdelt).astype(int)
        end_i_lim_m = ((end_w_m - crval + w_range)/cdelt).astype(int)
        mask1 = (start_i_m < 0) | (end_i_m < 0)
        mask2 = (start_i_m >= nw - 1) | (end_i_m >= nw - 1)
        mask3 = (start_i_lim_m >= nw - 1) | (end_i_lim_m >= nw - 1)
        sigma_mask = sigma__yx > 0
        mask = (~(mask1 | mask2 | mask3)) & sigma_mask
        [i_m] = np.where(mask)
        for i in i_m:
            I0, vel_I1, I2, EW, s_I0, s_vel_I1, s_I2, e_EW = momana_spec_wave(
                gas_flux__w=flux__wyx[i, :],
                egas_flux__w=eflux__wyx[i, :],
                wave=wavelengths[k],
                vel=vel__yx[i],
                sigma=sigma__yx[i],
                crval=crval, cdelt=cdelt,
                n_MC=n_MC, flux_ssp__w=flux_ssp__wyx[i, :],
                f_width=2,
            )
            I2=1.217*(I2/2.354)**2
            sig_I2 = I2#/2.354
            w_m_guess = wavelengths[k]*(1+vel_I1/__c__)
            #
            # Aperture correction
            #
            w_plot = crval+np.arange(0,flux__wyx[i, :].shape[0])*cdelt
            mask_wave = (w_plot>start_w_m[0]) & (w_plot<end_w_m[0])
            G_f = np.exp(-0.5*((w_plot-wavelengths[k]*(1+vel_I1/__c__))/sig_I2)**2)
            sum_G = sig_I2 * np.sqrt(2*np.pi)
            sum_G_obs = np.sum(G_f[mask_wave]*cdelt)
            A_corr = sum_G/sum_G_obs
            #print(f'# A_corr = {A_corr}')
            if (np.isfinite(A_corr)):
                I0=I0*A_corr
            G = (I0/(sig_I2*np.sqrt(2*np.pi)))*G_f            
            res_NP = flux__wyx[i, :]-G            
            if (plot==1):
                fig,ax = plt.subplots(1,1,figsize=(8,4.5))
                ax.plot(w_plot[mask_wave],\
                         flux__wyx[i, :][mask_wave],color='black',linewidth=1.5,label='obs.')
                ax.plot([w_m_guess,w_m_guess],[0.8*np.max(flux__wyx[i, :][mask_wave]),np.max(flux__wyx[i, :][mask_wave])],color='red',alpha=0.8)
                ax.plot(w_plot[mask_wave],\
                         G[mask_wave],color='blue',alpha=0.8,label='NP mod.')
                ax.plot(w_plot[mask_wave],\
                         res_NP[mask_wave],color='orange',alpha=0.8,label='NP res.')
                ax.set_title(f'NP analysis: {wavelengths[k]}')
                ax.set_xlabel(r'wavelength ($\AA$)')
                ax.set_ylabel(r'flux')
                ax.legend()
                plt.tight_layout()
                plt.show(block=True)
            out[k, i] = I0 #*cdelt*np.sqrt(2*np.pi)
            out[ne + k, i] = vel_I1
            out[2*ne + k, i] = I2
            out[3*ne + k, i] = EW
            out[4*ne + k, i] = s_I0
            out[5*ne + k, i] = s_vel_I1
            out[6*ne + k, i] = s_I2
            out[7*ne + k, i] = e_EW
            if (verbose==True):
                print('# :',I0,vel_I1, 1.217*(I2/2.354)**2, EW, s_I0, s_vel_I1, s_I2, e_EW)
        
        print('{}/{}, {},{} DONE'.format(k + 1, ne, str(wavelengths[k]), name_elines[k]))
    
    return out, input_header




##############################################
# Modification of the non-parametric analysis
##############################################

"""Mask-aware moment analysis of an emission line with Gaussian Monte Carlo errors.

Independent of lvmdap/astropy: requires only numpy.

NOTE: the old Gaussian-weighted I0 statistic is deliberately replaced by the
physical wavelength integral of continuum-subtracted flux density.  The old
'sigma_we' parameter is accepted for call compatibility but not applied:
that weighting biased the flux normalization and second moment.
"""

import numpy as np

_C_KMS = 299792.458
_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))


def momana_spec_wave_corrected(
    gas_flux__w, wave, vel, sigma, crval, cdelt,
    n_MC=200, egas_flux__w=None, flux_ssp__w=None,
    sigma_we=0.8, f_width=1.5,
    *,
    bad_pixel_mask=None,
    eflux_ssp__w=None,
    subtract_gas_baseline=True,
    baseline_sidebands=(30.0, 60.0),
    min_valid_fraction=0.90,
    reject_masked_core=True,
    continuum_systematic=0.0,
    flux_systematic_fraction=0.0,
    velocity_systematic_kms=0.0,
    dispersion_systematic_kms=0.0,
    ssp_continuum_fractional_error=0.0,
    random_state=None,
):
    """Measure line flux, velocity, and observed velocity dispersion by moments.

    Parameters
    ----------
    gas_flux__w : 1D array
        Gas-only flux DENSITY at each wavelength (e.g., 1e-16 erg/s/cm2/A).
    wave : float
        Rest-frame line wavelength, Angstrom.
    vel : float
        Approximate line velocity, km/s; defines the window only.
    sigma : float
        Approximate Gaussian sigma in ANGSTROM (not FWHM or km/s); defines
        the measurement window, not the measured second moment.
    crval, cdelt : float
        Wavelength at first sample and positive wavelength increment, Angstrom.
    n_MC : int
        Independent Gaussian MC realizations; original spectrum is NOT used
        as an MC draw. At least 2 for errors; 100-300 recommended.
    egas_flux__w : 1D array, optional
        1-sigma error spectrum in flux-density units. If absent, a robust
        sideband noise estimate is used; errors are NaN if unavailable.
    flux_ssp__w : 1D array, optional
        Stellar continuum flux density for emission equivalent width.
    sigma_we : float
        RETAINED FOR API COMPATIBILITY, NOT USED. The original Gaussian
        weighting does not yield an unbiased integrated flux/moment width.
    f_width : float
        Half-window in units of FWHM, i.e., +/- f_width*2.35482*sigma.
    bad_pixel_mask : 1D array, optional
        Boolean or integer mask; all nonzero values are flagged and excluded.
    eflux_ssp__w : 1D array, optional
        1-sigma stellar-continuum error. Independently randomized for EW MC.
    subtract_gas_baseline : bool
        Estimate a constant residual gas baseline from external sidebands;
        recompute it on each MC realization so its uncertainty propagates.
    baseline_sidebands : (inner, outer)
        Offsets (Angstrom) outside each measurement-window edge, default
        30-60 A, used for gas baseline and stellar continuum.
    min_valid_fraction : float
        Reject if fraction of usable pixels in the line window is smaller.
    reject_masked_core : bool
        Reject if any missing/invalid pixel lies within +/- sigma of the
        predicted line center. Do not silently interpolate a line peak.
    continuum_systematic : float
        Optional 1-sigma correlated residual gas baseline uncertainty in
        flux-density units, added by one offset per MC draw.
    flux_systematic_fraction : float
        Optional fractional flux-calibration systematic, quadrature added.
    velocity_systematic_kms, dispersion_systematic_kms : float
        Optional velocity/width systematic floors added in quadrature.
    ssp_continuum_fractional_error : float
        Optional 1-sigma fractional stellar continuum systematic for EW MC.
    random_state : int or numpy.random.Generator, optional
        MC random seed/generator; set an integer for reproducible diagnostics.

    Returns
    -------
    (F, v, sigma_v, EW, e_F, e_v, e_sigma_v, e_EW) : tuple of float
        F in flux-density*A, v and sigma_v in km/s, EW in Angstrom.
        EW < 0 for emission. Unavailable/invalid measurements are NaN.
        sigma_v is OBSERVED width, without instrument-LSF subtraction.

    Caveats
    -------
    - Assumes independent pixel errors; correlated resampling/sky residuals
      require covariance modelling or an empirically estimated noise factor.
    - Uses *signed*, continuum-subtracted flux moments; low-S/N or blended
      lines may produce invalid (negative) second moments, returned as NaN.
    - Masked samples are NEVER replaced by synthetic measurements. Partial
      window coverage may still bias the moments; masked line cores are
      rejected, and larger gaps require a profile fit to valid pixels.
    - The original function's arbitrary factors 2.35482 and fixed 10%, 2%
      floors have been removed; optional systematic floors are explicit.
    """
    nan_result = (np.nan,) * 8

    gas = np.asarray(gas_flux__w, dtype=float)
    if gas.ndim != 1 or len(gas) < 3:
        return nan_result
    if not (np.isfinite(wave) and wave > 0 and
            np.isfinite(vel) and np.isfinite(sigma) and sigma > 0 and
            np.isfinite(crval) and np.isfinite(cdelt) and cdelt > 0 and
            np.isfinite(f_width) and f_width > 0):
        raise ValueError('Invalid wavelength grid, center, line sigma, or f_width')
    if n_MC < 2:
        raise ValueError('n_MC must be at least 2 for uncertainty estimation')
    if not 0 < min_valid_fraction <= 1:
        raise ValueError('min_valid_fraction must be in (0, 1]')
    if len(baseline_sidebands) != 2 or not (0 <= baseline_sidebands[0] < baseline_sidebands[1]):
        raise ValueError('baseline_sidebands must be (inner, outer), 0 <= inner < outer')
    floors = (continuum_systematic, flux_systematic_fraction,
              velocity_systematic_kms, dispersion_systematic_kms,
              ssp_continuum_fractional_error)
    if not all(np.isfinite(f) and f >= 0 for f in floors):
        raise ValueError('Systematic terms must be finite and nonnegative')

    nw = gas.size
    wl = crval + cdelt * np.arange(nw)
    bad = np.zeros(nw, dtype=bool) if bad_pixel_mask is None else np.asarray(bad_pixel_mask, dtype=bool)
    if bad.shape != gas.shape:
        raise ValueError('bad_pixel_mask must match the gas spectrum shape')

    centre = wave * (1.0 + vel / _C_KMS)
    half_width = f_width * _SIGMA_TO_FWHM * sigma
    window = (wl >= centre - half_width) & (wl <= centre + half_width)
    if not np.any(window):
        return nan_result

    side_inner, side_outer = baseline_sidebands
    sides = (((wl >= centre-half_width-side_outer) &
              (wl <= centre-half_width-side_inner)) |
             ((wl >= centre+half_width+side_inner) &
              (wl <= centre+half_width+side_outer)))
    valid_flux = ~bad & np.isfinite(gas)

    # Use provided uncertainties; if missing, estimate only from clean
    # sidebands. The sidebands' scatter is a noise estimate, not automatically
    # an uncertainty on the mean continuum level.
    if egas_flux__w is None:
        side_values = gas[sides & valid_flux]
        if side_values.size < 5:
            return nan_result
        med = np.median(side_values)
        noise = 1.4826 * np.median(np.abs(side_values - med))
        if not np.isfinite(noise) or noise <= 0:
            return nan_result
        eflux = np.full(nw, noise, dtype=float)
    else:
        eflux = np.asarray(egas_flux__w, dtype=float)
        if eflux.shape != gas.shape:
            raise ValueError('egas_flux__w must match the gas spectrum shape')

    measured = valid_flux & np.isfinite(eflux) & (eflux > 0)
    line_valid = window & measured
    if line_valid.sum() < 5 or line_valid.sum() / window.sum() < min_valid_fraction:
        return nan_result
    if reject_masked_core and np.any(window & ~measured & (np.abs(wl-centre) <= sigma)):
        return nan_result

    # Constant residual gas baseline from clean external sidebands.
    side_valid = sides & measured
    use_baseline = bool(subtract_gas_baseline and side_valid.sum() >= 5)
    baseline = float(np.median(gas[side_valid])) if use_baseline else 0.0

    lw = wl[line_valid]
    y = gas[line_valid]
    ey = eflux[line_valid]
    dx = lw - centre                     # stable centering for short wavelength differences
    dw = cdelt                           # integral for regularly sampled RSS

    def _moments(draws, offsets):
        """Vectorized, signed-moment estimator for N draws."""
        line = draws - offsets[:, None]
        total = np.sum(line, axis=1) * dw
        first = np.full(total.size, np.nan, dtype=float)
        second = np.full(total.size, np.nan, dtype=float)
        good_total = np.isfinite(total) & (total > 0)
        with np.errstate(divide='ignore', invalid='ignore'):
            first[good_total] = np.sum(line[good_total] * dx, axis=1) * dw / total[good_total]
            second[good_total] = (np.sum(line[good_total] * dx**2, axis=1) * dw /
                                  total[good_total] - first[good_total]**2)
        good_sigma = good_total & np.isfinite(second) & (second > 0)
        v = np.full(total.size, np.nan, dtype=float)
        sv = np.full(total.size, np.nan, dtype=float)
        v[good_sigma] = _C_KMS * ((centre + first[good_sigma]) / wave - 1.0)
        sv[good_sigma] = _C_KMS * np.sqrt(second[good_sigma]) / wave
        # Keep signed flux in every MC draw, even if <=0.  Only the
        # kinematic moments are undefined in non-positive-flux draws.
        return total, v, sv

    F0, v0, sigv0 = _moments(y[None, :], np.array([baseline]))
    F, velocity, sigv = float(F0[0]), float(v0[0]), float(sigv0[0])
    if not np.isfinite(sigv):
        return nan_result

    # Independent Gaussian pixel perturbations. Do NOT include the unperturbed
    # spectrum among the MC realizations (avoids biasing the ensemble).
    rng = (random_state if isinstance(random_state, np.random.Generator)
           else np.random.default_rng(random_state))
    draws = y[None, :] + rng.normal(size=(n_MC, len(y))) * ey[None, :]
    if use_baseline:
        bg_draws = gas[side_valid][None, :] + (
            rng.normal(size=(n_MC, side_valid.sum())) * eflux[side_valid][None, :])
        baseline_draws = np.median(bg_draws, axis=1)
    else:
        baseline_draws = np.zeros(n_MC, dtype=float)
    if continuum_systematic:
        baseline_draws += rng.normal(0.0, continuum_systematic, size=n_MC)

    F_mc, v_mc, sigv_mc = _moments(draws, baseline_draws)
    e_F = float(np.std(F_mc[np.isfinite(F_mc)], ddof=1)) if np.isfinite(F_mc).sum() >= 2 else np.nan
    # Discard pathological realizations with negative variances / F <= 0.
    valid_kin = np.isfinite(v_mc) & np.isfinite(sigv_mc)
    e_v = float(np.std(v_mc[valid_kin], ddof=1)) if valid_kin.sum() >= max(2, int(.5*n_MC)) else np.nan
    e_sigv = float(np.std(sigv_mc[valid_kin], ddof=1)) if valid_kin.sum() >= max(2, int(.5*n_MC)) else np.nan

    # Calibration and instrument-related floors are separate from MC noise.
    if np.isfinite(e_F):
        e_F = float(np.hypot(e_F, abs(F)*flux_systematic_fraction))
    if np.isfinite(e_v):
        e_v = float(np.hypot(e_v, velocity_systematic_kms))
    if np.isfinite(e_sigv):
        e_sigv = float(np.hypot(e_sigv, dispersion_systematic_kms))

    EW = e_EW = np.nan
    if flux_ssp__w is not None:
        ssp = np.asarray(flux_ssp__w, dtype=float)
        if ssp.shape != gas.shape:
            raise ValueError('flux_ssp__w must match the gas spectrum shape')
        ssp_valid = ~bad & np.isfinite(ssp)
        if eflux_ssp__w is not None:
            essp = np.asarray(eflux_ssp__w, dtype=float)
            if essp.shape != gas.shape:
                raise ValueError('eflux_ssp__w must match the gas spectrum shape')
            ssp_valid &= np.isfinite(essp) & (essp >= 0)
        else:
            essp = None
        cont_valid = ssp_valid & sides
        if cont_valid.sum() < 5:
            cont_valid = ssp_valid & window
        if cont_valid.sum() >= 3:
            C = float(np.median(ssp[cont_valid]))
            if np.isfinite(C) and C != 0:
                EW = -F / abs(C)
                if essp is not None:
                    sim_ssp = ssp[cont_valid][None, :] + (
                        rng.normal(size=(n_MC, cont_valid.sum())) * essp[cont_valid][None, :])
                    C_mc = np.median(sim_ssp, axis=1)
                else:
                    C_mc = np.full(n_MC, C, dtype=float)
                if ssp_continuum_fractional_error:
                    C_mc = C_mc * (1 + rng.normal(
                        0.0, ssp_continuum_fractional_error, size=n_MC))
                with np.errstate(divide='ignore', invalid='ignore'):
                    EW_mc = -F_mc / np.abs(C_mc)
                good_ew = np.isfinite(EW_mc)
                if good_ew.sum() >= 2:
                    e_EW = float(np.std(EW_mc[good_ew], ddof=1))
                    if flux_systematic_fraction:
                        # The flux-calibration systematic may cancel with a
                        # consistent continuum calibration; conservative here.
                        e_EW = float(np.hypot(e_EW, abs(EW)*flux_systematic_fraction))

    return F, velocity, sigv, EW, e_F, e_v, e_sigv, e_EW




def _fit_emission_components_mc(
    wl, flux, eflux, bad, centers_rest, input_vel, input_sigma,
    n_MC, ssp=None, essp=None, min_snr=None, rng=None,
    velocity_range=60.0, width_range=(0.6, 1.8),
    full_refit_mc=False,
):
    """Joint Gaussian-template measurement, for blends or unstable moments.

    Flux amplitudes are signed integrated fluxes (no detection cutoff).
    Velocity and sigma are fitted jointly and re-fitted in *every* Gaussian
    MC realization for full_refit_mc=True (slow).  By default, draw Gaussian
    noise and propagate it through the local Jacobian (fast linearized MC).
    All pixels flagged by the original DRP mask are excluded.
    Returns eight quantities per component, with dispersion in ANGSTROM.

    NOTE: this is an explicitly parametric fallback/deblender, not a pure
    non-parametric moment.  The line widths and velocities are common within
    a blended group.  Use the PM result for more complex line profiles.
    """
    from scipy.optimize import least_squares

    rng = np.random.default_rng() if rng is None else rng
    centers_rest = np.asarray(centers_rest, dtype=float)
    nlines = len(centers_rest)
    result = np.full((8, nlines), np.nan, dtype=float)
    if not (np.isfinite(input_sigma) and input_sigma > 0 and
            np.isfinite(input_vel)):
        return result

    centers = centers_rest * (1 + input_vel / _C_KMS)
    lo = centers.min() - 4.5 * input_sigma
    hi = centers.max() + 4.5 * input_sigma
    win = (wl >= lo) & (wl <= hi)
    measured = win & ~bad & np.isfinite(flux) & np.isfinite(eflux) & (eflux > 0)
    # A core flag is not a nondetection; insufficient information is.
    if measured.sum() < max(6, nlines * 4) or measured.sum() < 0.65 * win.sum():
        return result
    x, y, ey = wl[measured], flux[measured], eflux[measured]

    # Residual gas baseline from adjacent pixels, avoiding bright line cores.
    side = ((wl >= lo - 35) & (wl < lo - 8)) | ((wl > hi + 8) & (wl <= hi + 35))
    side &= ~bad & np.isfinite(flux) & np.isfinite(eflux) & (eflux > 0)
    bg = float(np.median(flux[side])) if side.sum() >= 8 else 0.0
    z = y - bg
    root2pi = np.sqrt(2 * np.pi)

    def templates(v, sigma_a):
        cc = centers_rest * (1 + v / _C_KMS)
        d = (x[:, None] - cc[None, :]) / sigma_a
        return np.exp(-0.5 * d*d) / (root2pi * sigma_a)

    # Signed weighted least-squares initialization: do not clamp faint or
    # absorption-like noise fluctuations to nonnegative integrated flux.
    X0 = templates(input_vel, input_sigma)
    f_init, *_ = np.linalg.lstsq(X0 / ey[:, None], z / ey, rcond=None)
    sigma_lo, sigma_hi = (width_range[0]*input_sigma, width_range[1]*input_sigma)
    p0 = np.r_[f_init, input_vel, input_sigma]
    lower = np.r_[np.full(nlines, -np.inf), input_vel - velocity_range, sigma_lo]
    upper = np.r_[np.full(nlines, np.inf), input_vel + velocity_range, sigma_hi]

    def resid(p, ys):
        return (templates(p[-2], p[-1]) @ p[:nlines] - ys) / ey

    try:
        best = least_squares(resid, p0, args=(z,), bounds=(lower, upper),
                             x_scale='jac', max_nfev=90, ftol=1e-6,
                             xtol=1e-6, gtol=1e-6)
    except (ValueError, RuntimeError, FloatingPointError):
        return result
    fit = best.x

    # Formal flux errors are used only if too few MC fits succeed.
    # No fitted component is discarded because of its flux S/N.
    # min_snr is retained in the signature solely for call compatibility.
    Xf = templates(fit[-2], fit[-1])
    cov = np.linalg.pinv((Xf/ey[:, None]).T @ (Xf/ey[:, None]))
    formal_eF = np.sqrt(np.maximum(np.diag(cov), 0))

    n_MC = max(2, int(n_MC))
    params = np.full((n_MC, nlines+2), np.nan)
    if full_refit_mc:
        # Exact (more expensive) parametric Monte Carlo.
        for m in range(n_MC):
            noise_draw = rng.normal(size=len(y)) * ey
            bg_draw = bg
            if side.sum() >= 8:
                bg_draw = float(np.median(
                    flux[side] + rng.normal(size=side.sum())*eflux[side]))
            ys = y + noise_draw - bg_draw
            try:
                sol = least_squares(resid, fit, args=(ys,), bounds=(lower, upper),
                                    x_scale='jac', max_nfev=65, ftol=1e-5,
                                    xtol=1e-5, gtol=1e-5)
                params[m] = sol.x
            except (ValueError, RuntimeError, FloatingPointError):
                continue
    else:
        # Linearized Gaussian MC: one nonlinear fit and an inexpensive
        # Jacobian propagation for all MC noise realizations.  This is
        # first-order exact under the local Gaussian model; near hard
        # parameter bounds the full_refit_mc=True option is preferable.
        J_pinv = np.linalg.pinv(best.jac)
        epsilon = rng.normal(size=(n_MC, len(y)))
        if side.sum() >= 8:
            side_eps = rng.normal(size=(n_MC, side.sum())) * eflux[side][None,:]
            side_draws = np.median(flux[side][None,:] + side_eps, axis=1)
            epsilon -= ((side_draws-bg)[:,None] / ey[None,:])
        params[:] = fit[None,:] + epsilon @ J_pinv.T
    good_mc = np.isfinite(params).all(axis=1)
    if good_mc.sum() >= 2:
        errs = np.std(params[good_mc], axis=0, ddof=1)
    else:
        errs = np.full(nlines+2, np.nan)
        errs[:nlines] = formal_eF

    for j in range(nlines):
        F = float(fit[j])
        result[0, j] = F
        result[1, j] = fit[-2]  # shared velocity, km/s
        result[2, j] = fit[-1]  # observed gaussian sigma, Angstrom
        result[4, j] = errs[j]
        result[5, j] = errs[-2]
        result[6, j] = errs[-1]
        if ssp is not None:
            csel = side & np.isfinite(ssp)
            if csel.sum() < 3:
                csel = measured & np.isfinite(ssp)
            if csel.sum() >= 3:
                C = float(np.median(ssp[csel]))
                if np.isfinite(C) and C != 0:
                    result[3, j] = -F / abs(C)
                    if good_mc.sum() >= 2:
                        C_mc = np.full(good_mc.sum(), C)
                        if essp is not None and np.isfinite(essp[csel]).all():
                            C_mc = np.median(ssp[csel][None, :] +
                                rng.normal(size=(good_mc.sum(), csel.sum()))*
                                essp[csel][None, :], axis=1)
                        ew_mc = -params[good_mc,j] / np.maximum(abs(C_mc), 1e-30)
                        result[7,j] = np.std(ew_mc, ddof=1)
                    elif np.isfinite(errs[j]):
                        result[7,j] = errs[j] / abs(C)
    return result


def flux_elines_RSS_EW_tab(
    flux__wyx, input_header, n_MC, tab_elines, vel__yx, sigma__yx,
    eflux__wyx=None, flux_ssp__wyx=None, w_range=60, plot=0,
    figsize=(8, 4), f_width=2, sigma_we=2, verbose=False,
    *, bad_pixel_mask=None, eflux_ssp__wyx=None,
    min_valid_fraction=0.90, reject_masked_core=True,
    subtract_gas_baseline=True, random_state=None,
    weak_line_snr=None, min_template_snr=None,
    full_refit_mc=False,
):
    """NP moment measurements, with MC Gaussian deblending and weak-line fallback.

    Strong isolated lines: direct moments and full Gaussian-noise MC.
    Blended lines: simultaneous signed Gaussian-component fit and MC.
    Weak lines with unstable moments: Gaussian fallback and MC.

    Output blocks: F, vel [km/s], sigma [Angstrom], EW [Angstrom],
    eF, e_vel [km/s], e_sigma [Angstrom], eEW.
    """
    flux = np.asarray(flux__wyx, dtype=float)
    if flux.ndim != 2:
        raise ValueError('flux__wyx must be a 2D RSS')
    nx, nw = flux.shape
    crval = float(input_header['CRVAL1'])
    cdelt = float(input_header['CDELT1'])
    if not np.isfinite(cdelt) or cdelt <= 0:
        raise ValueError('CDELT1 must be positive')
    wl = crval + cdelt * np.arange(nw)
    err = (None if eflux__wyx is None else np.asarray(eflux__wyx, float))
    ssp = None if flux_ssp__wyx is None else np.asarray(flux_ssp__wyx, float)
    essp = None if eflux_ssp__wyx is None else np.asarray(eflux_ssp__wyx, float)
    for arr, name in ((err,'eflux'), (ssp,'stellar'), (essp,'stellar error')):
        if arr is not None and arr.shape != flux.shape:
            raise ValueError(f'{name} RSS shape mismatch')
    bad = (np.zeros(flux.shape, bool) if bad_pixel_mask is None else
           np.asarray(bad_pixel_mask, dtype=bool).copy())
    if bad.shape != flux.shape:
        raise ValueError('bad_pixel_mask RSS shape mismatch')
    bad |= ~np.isfinite(flux)
    if err is not None:
        bad |= ~np.isfinite(err) | (err <= 0)

    def per_fiber(value, name):
        a = np.asarray(value, float)
        if a.ndim == 0:
            return np.full(nx, float(a))
        if a.shape != (nx,):
            raise ValueError(f'{name} must be scalar or one value per fiber')
        return a

    vel = per_fiber(vel__yx, 'vel__yx')
    sig = per_fiber(sigma__yx, 'sigma__yx')
    waves = np.asarray(tab_elines['wl'], float)
    names = [str(t) for t in tab_elines['id']]
    ne = len(waves)
    # Use the blended flags supplied by dap.py.  They were previously lost.
    try:
        flags = np.asarray(tab_elines['blended'], bool)
    except (KeyError, IndexError):
        flags = np.zeros(ne, dtype=bool)

    out = np.full((ne * 8, nx), np.nan)
    labels = ('flux','vel','disp','EW','e_flux','e_vel','e_disp','e_EW')
    units = ('10^-16 erg/s/cm^2','km/s','Angstrom','Angstrom',
             '10^-16 erg/s/cm^2','km/s','Angstrom','Angstrom')
    hdr = input_header.copy()
    for k in range(ne):
        for m,(label,unit) in enumerate(zip(labels,units)):
            q = m*ne+k
            hdr[f'NAME{q}'] = f'{label} {names[k]}'
            hdr[f'WAVE{q}'] = f'{float(waves[k]):.2f}'
            hdr[f'UNIT{q}'] = unit
    rng = np.random.default_rng(random_state)

    for i in range(nx):
        if not (np.isfinite(sig[i]) and sig[i] > 0 and np.isfinite(vel[i])):
            continue
        # Connect close neighbors only when the original pipeline flagged them.
        order = np.argsort(waves)
        groups = []
        group = [int(order[0])] if ne else []
        for previous, current in zip(order[:-1], order[1:]):
            prev, cur = int(previous), int(current)
            linked = (flags[prev] and flags[cur] and
                      waves[cur]-waves[prev] < 4.0*sig[i])
            if linked:
                group.append(cur)
            else:
                groups.append(group)
                group = [cur]
        if group:
            groups.append(group)
        for indices in groups:
            # Both branches use the same mask and MC error spectrum.
            # No artificially high errors or repaired pixels enter either fit.
            good = np.isfinite(waves[indices]).all()
            if not good:
                continue
            if len(indices) >= 2:
                if err is None:
                    # No uncertainty input: robust measured-sideband estimate.
                    v = flux[i,~bad[i]]
                    if len(v) < 5:
                        continue
                    med = np.median(v)
                    std = 1.4826*np.median(abs(v-med))
                    ei = np.full(nw, max(std,1e-30))
                else:
                    ei = err[i]
                estimated = _fit_emission_components_mc(
                    wl, flux[i], ei, bad[i], waves[indices],
                    vel[i], sig[i], n_MC, ssp=None if ssp is None else ssp[i],
                    essp=None if essp is None else essp[i],
                    min_snr=min_template_snr, rng=rng,
                    full_refit_mc=full_refit_mc)
                if verbose:
                    print(f'# NP blend: {[(names[k], waves[k]) for k in indices]} '
                          f'accepted={np.isfinite(estimated[0]).sum()}')
                for j,k in enumerate(indices):
                    out[[m*ne+k for m in range(8)], i] = estimated[:,j]
                continue

            k = indices[0]
            wave = waves[k]
            cen = wave*(1+vel[i]/_C_KMS)
            half = f_width*_SIGMA_TO_FWHM*sig[i]
            if cen-half < wl[0] or cen+half > wl[-1]:
                continue
            F, vout, sv, EW, eF, ev, esv, eEW = momana_spec_wave_corrected(
                flux[i], wave, vel[i], sig[i], crval, cdelt,
                n_MC=int(n_MC), egas_flux__w=None if err is None else err[i],
                flux_ssp__w=None if ssp is None else ssp[i],
                eflux_ssp__w=None if essp is None else essp[i],
                sigma_we=0, f_width=f_width, bad_pixel_mask=bad[i],
                min_valid_fraction=min_valid_fraction,
                reject_masked_core=reject_masked_core,
                subtract_gas_baseline=subtract_gas_baseline,
                random_state=rng)
            # Even detected weak lines can have *negative signed variance*;
            # do not discard their flux or silently omit the line profile.
            use_template = (not (np.isfinite(F) and np.isfinite(vout) and
                                 np.isfinite(sv) and sv > 0 and
                                 np.isfinite(eF) and eF > 0) or
                            not (0.5*sig[i] <= sv*wave/_C_KMS <= 2.*sig[i]))
            if use_template:
                if err is None:
                    vals = flux[i,~bad[i]]
                    if len(vals)<5: continue
                    med = np.median(vals)
                    noise = 1.4826*np.median(abs(vals-med))
                    ei = np.full(nw, max(noise,1e-30))
                else:
                    ei = err[i]
                estimated = _fit_emission_components_mc(
                    wl, flux[i], ei, bad[i], [wave], vel[i], sig[i],
                    n_MC, ssp=None if ssp is None else ssp[i],
                    essp=None if essp is None else essp[i],
                    min_snr=min_template_snr, rng=rng,
                    full_refit_mc=full_refit_mc)
                out[[m*ne+k for m in range(8)], i] = estimated[:,0]
                if verbose:
                    print(f'# NP template fallback: {names[k]} '
                          f'flux={estimated[0,0]:.3g}')
            else:
                fac = wave/_C_KMS
                out[[m*ne+k for m in range(8)],i] = (
                    F,vout,sv*fac,EW,eF,ev,esv*fac,eEW)
    return out, hdr


def flux_elines_RSS_EW_cl(
    flux__wyx, input_header, n_MC, tab_el, vel__yx, sigma__yx,
    eflux__wyx=None, flux_ssp__wyx=None, w_range=60, plot=0, f_width=2,
    sigma_we=2, figsize=(12, 4.5), verbose=False,
    *, bad_pixel_mask=None, eflux_ssp__wyx=None,
    min_valid_fraction=0.90, reject_masked_core=True,
    subtract_gas_baseline=True, random_state=None,
    weak_line_snr=None, min_template_snr=None,
    full_refit_mc=False,
):
    """Wrapper retaining the LVM-DAP FE data/header return convention."""
    out, hdr = flux_elines_RSS_EW_tab(
        flux__wyx, input_header, n_MC, tab_el, vel__yx, sigma__yx,
        eflux__wyx=eflux__wyx, flux_ssp__wyx=flux_ssp__wyx,
        w_range=w_range, plot=plot, figsize=figsize,
        f_width=f_width, sigma_we=sigma_we, verbose=verbose,
        bad_pixel_mask=bad_pixel_mask, eflux_ssp__wyx=eflux_ssp__wyx,
        min_valid_fraction=min_valid_fraction,
        reject_masked_core=reject_masked_core,
        subtract_gas_baseline=subtract_gas_baseline,
        random_state=random_state,
        weak_line_snr=weak_line_snr, min_template_snr=min_template_snr,
        full_refit_mc=full_refit_mc,
    )
    if plot == 1:
        import matplotlib.pyplot as plt
        obs = np.asarray(flux__wyx, dtype=float)
        if obs.ndim == 2 and obs.shape[0] and len(tab_el):
            crval = float(input_header['CRVAL1'])
            cdelt = float(input_header['CDELT1'])
            wl = crval + cdelt * np.arange(obs.shape[1])
            model = np.zeros_like(wl)
            ne = len(tab_el)
            for k in range(ne):
                F = out[k, 0]
                v = out[ne + k, 0]
                sigma_A = out[2*ne + k, 0]
                if not all(np.isfinite(t) for t in (F, v, sigma_A)) or sigma_A <= 0:
                    continue
                center = float(tab_el['wl'][k]) * (1 + v / _C_KMS)
                model += F / (sigma_A * np.sqrt(2*np.pi)) * (
                    np.exp(-0.5*((wl-center)/sigma_A)**2))
            fig, ax = plt.subplots(figsize=figsize)
            obs0 = obs[0].copy()
            if bad_pixel_mask is not None:
                flagged = np.asarray(bad_pixel_mask, dtype=bool)
                if flagged.shape == obs.shape:
                    obs0[flagged[0]] = np.nan
            ax.plot(wl, obs0, label='Gas spectrum', color='black')
            ax.plot(wl, model, label='Moment Gaussian sum', color='royalblue')
            ax.plot(wl, obs0 - model, label='Residual', color='orange')
            ax.set_xlabel('Wavelength [Angstrom]')
            ax.set_ylabel('Flux density')
            ax.legend()
            fig.tight_layout()
            plt.show(block=True)
            plt.close(fig)
    return out, hdr
