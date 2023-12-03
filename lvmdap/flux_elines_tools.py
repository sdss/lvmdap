import sys
import numpy as np
from astropy.io import fits
from os.path import basename
from copy import deepcopy as copy
import os
from astropy.wcs import WCS

from pyFIT3D.common.io import ReadArguments
from pyFIT3D.common.io import get_data_from_fits, array_to_fits
from pyFIT3D.common.constants import __FWHM_to_sigma__
from pyFIT3D.common.tools import flux_elines_cube_EW,momana_spec_wave
import numpy as np
from copy import deepcopy as copy
from os.path import basename, isfile
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from astropy.io.fits.verify import VerifyWarning

from pyFIT3D.common.stats import median_filter as st_median_filter
from pyFIT3D.common.stats import hyperbolic_fit_par, std_m, pdl_stats, _STATS_POS
from pyFIT3D.common.io import trim_waves, get_data_from_fits, get_wave_from_header
from pyFIT3D.common.constants import __c__, __sigma_to_FWHM__, __indices__, _INDICES_POS
from pyFIT3D.common.io import output_spectra, array_to_fits, write_img_header, print_verbose
from pyFIT3D.common.constants import __Hubble_constant__, __Omega_matter__, __Omega_Lambda__
from pyFIT3D.common.constants import __solar_luminosity__, __solar_metallicity__, _figsize_default
from scipy.ndimage import gaussian_filter1d
from pyFIT3D.common.tools import vel_eline
from pyFIT3D.common.io import array_to_fits, get_data_from_fits, trim_waves
from astropy.table import Table
from astropy.table import vstack as vstack_table

def flux_elines_RSS_EW(flux__wyx, input_header, n_MC, elines_list, vel__yx, sigma__yx,
                        eflux__wyx=None, flux_ssp__wyx=None, w_range=60):
    nx, nw = flux__wyx.shape
    print(nx,nw)
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
    print('PASO fe')
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
        print(k,ne)
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
            )
            #print(f'vel {vel_I1} vs {vel__yx[i]}')

            out[k, i] = I0
            out[ne + k, i] = vel_I1
            out[2*ne + k, i] = I2
            out[3*ne + k, i] = EW
            out[4*ne + k, i] = s_I0
            out[5*ne + k, i] = s_vel_I1
            out[6*ne + k, i] = s_I2
            out[7*ne + k, i] = e_EW
        print('{}/{}, {},{} DONE'.format(k + 1, ne, wavelengths[k], name_elines[k]))
    return out, input_header

