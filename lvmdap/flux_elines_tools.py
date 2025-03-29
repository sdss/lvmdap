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
                        name_elines = np.append(name_elines, tmp_line[1])
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
                                                 w_range=w_range, plot=plot, verbose=False)


     
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

def flux_elines_RSS_EW_tab(flux__wyx, input_header, n_MC, tab_elines, vel__yx, sigma__yx,
                           eflux__wyx=None, flux_ssp__wyx=None, w_range=60, plot=0, 
                           figsize=(8,4), f_width=2, sigma_we = 2, verbose=False):
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
        start_w_m = wavelengths[k]*f_m - f_width*__sigma_to_FWHM__*sigma__yx
        end_w_m = wavelengths[k]*f_m + f_width*__sigma_to_FWHM__*sigma__yx
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
            try:
                sigma_we_k = np.sqrt(tab_elines['lsf'][k]**2+cdelt**2)
#                sigma_we_k = tab_elines['lsf'][i]
            except:
                sigma_we_k = sigma_we

            #
            # We apply a weight only if the lines are two near
            #
            #            blended = 'No'
            try:    
                sigma_we_k = sigma_we_k * tab_elines['blended'][k]
                blended = True
            except:
                blended = None
            I0, vel_I1, I2, EW, s_I0, s_vel_I1, s_I2, e_EW = momana_spec_wave(
                gas_flux__w=flux__wyx[i, :],
                egas_flux__w=eflux__wyx[i, :],
                wave=wavelengths[k],
                vel=vel__yx[i],
                sigma=sigma__yx[i],
                crval=crval, cdelt=cdelt,
                n_MC=n_MC, flux_ssp__w=flux_ssp__wyx[i, :],
                f_width=f_width, sigma_we = sigma_we_k
            )

            if (sigma_we_k>0):
                scaling = 1.19
                I0 = I0 * np.sqrt(scaling) 
                I2 = I2 * scaling
            #I2=1.217*(I2/2.354)**2
            I2 = I2 / 2.354
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
            #if (np.isfinite(A_corr)):
            #    I0=I0*A_corr
            G = (I0/(sig_I2*np.sqrt(2*np.pi)))*G_f            
            res_NP = flux__wyx[i, :]-G            
            if (plot == 11):
                fig,ax = plt.subplots(1,1,figsize=figsize)
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
        
        print('{}/{}, {},{} blended: {} DONE'.format(k + 1, ne, str(wavelengths[k]), name_elines[k], tab_elines['blended'][k] ))
    
    return out, input_header

def flux_elines_RSS_EW_cl(flux__wyx, input_header, n_MC, tab_el, vel__yx, sigma__yx,
                          eflux__wyx=None, flux_ssp__wyx=None, w_range=60, plot=0, f_width=2, 
                          sigma_we = 2, figsize=(12,4.5), verbose=False):
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
                                                 w_range=w_range, plot=plot, f_width=f_width, 
                                                 sigma_we = sigma_we, verbose=False)


     
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

#############################################################
