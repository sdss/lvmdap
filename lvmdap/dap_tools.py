import io
import sys
import warnings
import itertools
import numpy as np
from astropy.io import fits, ascii
from copy import deepcopy as copy
from os.path import basename, isfile
import os
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from astropy.io.fits.verify import VerifyWarning

from matplotlib import use as mpl_use
mpl_use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from lvmdap.pyFIT3D.common.stats import median_filter as st_median_filter
from lvmdap.pyFIT3D.common.stats import hyperbolic_fit_par, std_m, pdl_stats, _STATS_POS
from lvmdap.pyFIT3D.common.io import trim_waves, get_data_from_fits, get_wave_from_header
from lvmdap.pyFIT3D.common.constants import __c__, __sigma_to_FWHM__
from lvmdap.pyFIT3D.common.io import output_spectra, array_to_fits, write_img_header, print_verbose
from lvmdap.pyFIT3D.common.constants import __Hubble_constant__, __Omega_matter__, __Omega_Lambda__
from lvmdap.pyFIT3D.common.constants import __solar_luminosity__, __solar_metallicity__, _figsize_default
from lvmdap.pyFIT3D.common.tools import peak_finder

from astropy.table import Table,join
from astropy.table import vstack as vstack_table
from lvmdap.cmaps_CLC import vel_map_CLC
import re
from collections import Counter

import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mpl_colors

from matplotlib import rcParams as rc

from scipy.ndimage import gaussian_filter1d, median_filter

from astropy.convolution import convolve
#from scipy.ndimage import generic_filter

from numpy.polynomial.legendre import Legendre

from scipy.interpolate import UnivariateSpline

from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator,    RBFInterpolator

#from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
#
# Fit emission line systems given emission lines
#

#import numpy as np
#import matplotlib.pyplot as plt

from pyFIT3D.common.gas_tools import (
    fit_elines_main,
    ConfigEmissionModel,
)
from pyFIT3D.common.constants import (
    _MODELS_ELINE_PAR,
    __n_models_params__,
    __c__,
)


def fit_emission_line_system(
    wl,
    flux,
    eflux,
    line_wavelengths,
    redshift=0.0,
    velocity_range=200.0,
    fit_padding=25.0,
    local_window=50.0,
    sigma_guess=1.2,
    sigma_bounds=(0.5, 2.0),
    error_scale=0.5,
    tie_kinematics=True,
    add_background=True,
    chi_goal=1.0,
    chi_step=0.05,
    n_mc=30,
    n_loops=15,
    run_mode="both",
    redefine_max=True,
    max_factor=2.0,
    redefine_min=True,
    min_factor=0.012,
    fine_search=False,
    scale_ini=0.15,
    plot=False,
    ax=None,
    bg_order=2,
):
    """
    Fit a system of emission lines using pyFIT3D.

    Parameters
    ----------
    wl : array-like
        Observed wavelength array.

    flux : array-like
        Flux array. It must have the same shape as ``wl``.

    eflux : array-like
        One-sigma flux uncertainty. It must have the same shape as ``wl``.

    line_wavelengths : array-like
        Rest-frame central wavelengths of the lines to fit.

    redshift : float, optional
        Initial redshift of the emission-line system.

    velocity_range : float, optional
        Allowed velocity interval around ``redshift * c``, in km/s.

    fit_padding : float, optional
        Wavelength padding added before the first line and after the last
        line when defining the fitted interval, in Angstrom.

    local_window : float, optional
        Half-width of the interval used to estimate the initial flux of
        each line, in Angstrom.

    sigma_guess : float, optional
        Initial Gaussian sigma, in Angstrom.

    sigma_bounds : tuple of float, optional
        Minimum and maximum allowed Gaussian sigma, in Angstrom.

    error_scale : float, optional
        Multiplicative factor applied to ``eflux``. The original cells
        used ``0.5 * abs(eflux)``.

    tie_kinematics : bool, optional
        When True, all lines share the same velocity and sigma.

    add_background : bool, optional
        Add a first-order polynomial background.

    chi_goal, chi_step : float, optional
        Parameters passed to ``ConfigEmissionModel``.

    n_mc, n_loops : int, optional
        Number of Monte Carlo iterations and fitting loops.

    run_mode : {"RND", "LM", "both"}, optional
        pyFIT3D fitting mode.

    redefine_max, redefine_min : bool, optional
        Allow pyFIT3D to redefine the line-flux boundaries.

    max_factor, min_factor : float, optional
        Factors used when redefining the flux boundaries.

    fine_search : bool, optional
        Activate the pyFIT3D fine search.

    scale_ini : float, optional
        Initial scale used by the fitting algorithm.

    plot : bool, optional
        Plot the input spectrum and fitted model.

    ax : matplotlib.axes.Axes, optional
        Existing axis on which to draw the result.

    Returns
    -------
    result : dict
        Dictionary containing:

        ``fit``
            Result returned by ``fit_elines_main``.

        ``config``
            Generated ``ConfigEmissionModel``.

        ``mask``
            Boolean mask applied to the original arrays.

        ``wavelength``
            Wavelength array passed to the fit.

        ``flux``
            Flux array passed to the fit.

        ``eflux``
            Effective uncertainty passed to the fit.

        ``figure``, ``axis``
            Matplotlib objects, or None when ``plot=False``.
    """

    # --------------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------------
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    eflux = np.asarray(eflux, dtype=float)
    line_wavelengths = np.asarray(line_wavelengths, dtype=float)
    if (bg_order==0):
        add_background=False
    if wl.ndim != 1 or flux.ndim != 1 or eflux.ndim != 1:
        raise ValueError("wl, flux, and eflux must be one-dimensional arrays.")

    if not (wl.shape == flux.shape == eflux.shape):
        raise ValueError("wl, flux, and eflux must have the same shape.")

    if line_wavelengths.ndim != 1 or line_wavelengths.size == 0:
        raise ValueError(
            "line_wavelengths must contain at least one wavelength."
        )

    if not np.all(np.isfinite(line_wavelengths)):
        raise ValueError("line_wavelengths contains non-finite values.")

    if velocity_range <= 0:
        raise ValueError("velocity_range must be greater than zero.")

    if sigma_bounds[0] <= 0 or sigma_bounds[1] <= sigma_bounds[0]:
        raise ValueError(
            "sigma_bounds must satisfy 0 < minimum < maximum."
        )

    # Sorting is important because linked models refer to the
    # immediately preceding model.
    line_wavelengths = np.sort(line_wavelengths)

    # Approximate observed positions, used only to select the data.
    observed_line_wavelengths = line_wavelengths * (1.0 + redshift)

    w_min = np.min(observed_line_wavelengths) - fit_padding
    w_max = np.max(observed_line_wavelengths) + fit_padding

    fit_mask = (
        (wl > w_min)
        & (wl < w_max)
        & np.isfinite(wl)
        & np.isfinite(flux)
        & np.isfinite(eflux)
    )

    if np.count_nonzero(fit_mask) < 5:
        raise ValueError(
            f"Insufficient valid data between {w_min:.2f} and "
            f"{w_max:.2f} Angstrom."
        )

    fit_wl = wl[fit_mask]
    fit_flux = flux[fit_mask]
    fit_eflux = error_scale * np.abs(eflux[fit_mask])

    # Replace zero or invalid uncertainties with a representative
    # positive value.
    valid_errors = fit_eflux[
        np.isfinite(fit_eflux) & (fit_eflux > 0)
    ]

    if valid_errors.size == 0:
        raise ValueError("eflux does not contain positive uncertainties.")

    representative_error = np.nanmedian(valid_errors)

    fit_eflux = np.where(
        np.isfinite(fit_eflux) & (fit_eflux > 0),
        fit_eflux,
        representative_error,
    )

    # --------------------------------------------------------------
    # pyFIT3D parameter indices
    # --------------------------------------------------------------
    i_wave = _MODELS_ELINE_PAR["central_wavelength"]
    i_flux = _MODELS_ELINE_PAR["flux"]
    i_sigma = _MODELS_ELINE_PAR["sigma"]
    i_v0 = _MODELS_ELINE_PAR["v0"]

    central_velocity = redshift * __c__

    config_model = ConfigEmissionModel(
        filename=None,
        chi_goal=chi_goal,
        chi_step=chi_step,
    )

    # --------------------------------------------------------------
    # Add the emission-line models
    # --------------------------------------------------------------
    for line_number, (rest_wl, observed_wl) in enumerate(
        zip(line_wavelengths, observed_line_wavelengths)
    ):
        local_mask = (
            (wl > observed_wl - local_window)
            & (wl < observed_wl + local_window)
            & np.isfinite(flux)
        )

        local_flux = flux[local_mask]

        if local_flux.size == 0:
            raise ValueError(
                f"No valid data around the line at {rest_wl:.2f} Angstrom."
            )

        local_max = np.nanmax(local_flux)
        local_min = np.nanmin(local_flux)
        local_scale = max(
            np.abs(local_max),
            np.abs(local_min),
            np.nanstd(local_flux),
            np.finfo(float).eps,
        )

        # Preserve the original boundary prescription where possible,
        # while ensuring that the interval is valid.
        flux_bound_0 = local_min
        flux_bound_1 = 1.5 * local_max

        flux_lower = min(flux_bound_0, flux_bound_1)
        flux_upper = max(flux_bound_0, flux_bound_1)

        if np.isclose(flux_lower, flux_upper):
            flux_lower -= local_scale
            flux_upper += local_scale

        flux_guess = np.clip(
            local_max,
            flux_lower,
            flux_upper,
        )

        guess = np.zeros(__n_models_params__, dtype=float)
        pars_0 = np.zeros(__n_models_params__, dtype=float)
        pars_1 = np.zeros(__n_models_params__, dtype=float)
        links = -np.ones(__n_models_params__, dtype=int)
        to_fit = np.zeros(__n_models_params__, dtype=bool)

        # Rest-frame central wavelength remains fixed.
        guess[i_wave] = rest_wl

        guess[i_flux] = flux_guess
        pars_0[i_flux] = flux_lower
        pars_1[i_flux] = flux_upper
        to_fit[i_flux] = True

        if tie_kinematics and line_number > 0:
            # Same sigma as the preceding line:
            # sigma_current = 1 * sigma_previous.
            guess[i_sigma] = 1.0
            pars_0[i_sigma] = 1.0
            pars_1[i_sigma] = 1.0
            links[i_sigma] = 1
            to_fit[i_sigma] = True

            # Same velocity as the preceding line:
            # v_current = 0 + v_previous.
            guess[i_v0] = 0.0
            pars_0[i_v0] = 0.0
            pars_1[i_v0] = 0.0
            links[i_v0] = 1
            to_fit[i_v0] = True

        else:
            guess[i_sigma] = sigma_guess
            pars_0[i_sigma] = sigma_bounds[0]
            pars_1[i_sigma] = sigma_bounds[1]
            to_fit[i_sigma] = True

            guess[i_v0] = central_velocity
            pars_0[i_v0] = central_velocity - velocity_range
            pars_1[i_v0] = central_velocity + velocity_range
            to_fit[i_v0] = True

        config_model.add_model(
            model_type="eline",
            guess=guess,
            to_fit=to_fit,
            pars_0=pars_0,
            pars_1=pars_1,
            links=links,
        )

    # --------------------------------------------------------------
    # Add a first-order polynomial background
    # --------------------------------------------------------------
    if add_background:
        guess = np.zeros(__n_models_params__, dtype=float)
        pars_0 = np.zeros(__n_models_params__, dtype=float)
        pars_1 = np.zeros(__n_models_params__, dtype=float)
        links = -np.ones(__n_models_params__, dtype=int)
        to_fit = np.zeros(__n_models_params__, dtype=bool)

        flux_scale = max(
            np.abs(np.nanmin(fit_flux)),
            np.abs(np.nanmax(fit_flux)),
            np.nanstd(fit_flux),
            np.finfo(float).eps,
        )

        mean_wavelength = np.nanmean(fit_wl)

        # Constant coefficient.
        guess[0] = np.nanmedian(fit_flux)
        pars_0[0] = -1.5 * flux_scale
        pars_1[0] = 1.5 * flux_scale
        to_fit[0] = True

        # Linear coefficient.
        if (bg_order == 2):
            guess[1] = 0.0
            pars_0[1] = -1.5 * flux_scale / mean_wavelength
            pars_1[1] = 1.5 * flux_scale / mean_wavelength
            to_fit[1] = True

        config_model.add_model(
            model_type="poly1d",
            guess=guess,
            to_fit=to_fit,
            pars_0=pars_0,
            pars_1=pars_1,
            links=links,
        )

    # --------------------------------------------------------------
    # Perform the fit
    # --------------------------------------------------------------
    emission_line_fit = fit_elines_main(
        wavelength=fit_wl,
        flux=fit_flux,
        sigma_flux=fit_eflux,
        config=config_model,
        plot=0,
        n_MC=n_mc,
        n_loops=n_loops,
        redefine_max=redefine_max,
        max_factor=max_factor,
        redefine_min=redefine_min,
        min_factor=min_factor,
        run_mode=run_mode,
        fine_search=fine_search,
        scale_ini=scale_ini,
    )

    # --------------------------------------------------------------
    # Optional plot
    # --------------------------------------------------------------
    figure = None

    if plot:
        if ax is None:
            figure, ax = plt.subplots(figsize=(16, 5))
        else:
            figure = ax.figure

        ax.plot(
            fit_wl,
            fit_flux,
            label="Data",
        )
        ax.plot(
            fit_wl,
            emission_line_fit.model,
            label="Model",
        )

        for observed_wl in observed_line_wavelengths:
            ax.axvline(
                observed_wl,
                linestyle=":",
                alpha=0.4,
            )

        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Flux")
        ax.legend()

        figure.tight_layout()

    return {
        "fit": emission_line_fit,
        "config": config_model,
        "mask": fit_mask,
        "wavelength": fit_wl,
        "flux": fit_flux,
        "eflux": fit_eflux,
        "figure": figure,
        "axis": ax if plot else None,
    }


warnings.simplefilter('ignore', category=VerifyWarning)

__indices__ = {
    'Hd':     [4083.500, 4122.250, 4041.600, 4079.750, 4128.500, 4161.000],
    'Hb':     [4847.875, 4876.625, 4827.875, 4847.875, 4876.625, 4891.625],
    'Mgb':    [5160.125, 5192.625, 5142.625, 5161.375, 5191.375, 5206.375],
    'Fe5270': [5245.650, 5285.650, 5233.150, 5248.150, 5285.650, 5318.150],
    'Fe5335': [5312.125, 5352.125, 5304.625, 5315.875, 5353.375, 5363.375],
    'D4000':  [4050.000, 4250.000, 3750.000, 3950.000, 0.000,    1.000],
    'Hdmod':  [4083.500, 4122.250, 4079,     4083,     4128.500, 4161.000],
    'Hg':     [4319.75,  4363.50,  4283.50,  4319.75,  4367.25,  4419.75]
}
_INDICES_POS_ = {
    'OL1': 0,
    'OL2': 1,
    'OLb1': 2,
    'OLb2': 3,
    'OLr1': 4,
    'OLr2': 5,
    'OLb': 6,
    'OLr': 7
}

__indices_sky_ = {
    'OI5577':  [5557, 5597, 5507, 5547, 5607, 5647],
    'NaID':    [5873, 5893, 5823, 5863, 5903, 5943],
    'OI6300':  [6280, 6320, 6230, 6270, 6330, 6370],
}


class scatter():
    def __init__(self,x,y,ax,size=1,**kwargs):
        self.n = len(x)
        self.ax = ax
        self.ax.figure.canvas.draw()
        self.size_data=size
        self.size = size
        self.sc = ax.scatter(x,y,s=self.size,**kwargs)
        self._resize()
        self.cid = ax.figure.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self,event=None):
        ppd=72./self.ax.figure.dpi
        trans = self.ax.transData.transform
        s =  ((trans((1,self.size_data))-trans((0,0)))*ppd)[1]
        if s != self.size:
            self.sc.set_sizes(s**2*np.ones(self.n))
            self.size = s
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.ax.figure.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.ax.figure.canvas.draw_idle())
        self.timer.start()
        


def gamma_scale(flux,gamma):
    return (flux/np.abs(flux))*(np.abs(flux))**gamma

def gamma_scale_abs(flux,gamma):
    return (np.abs(flux))**gamma

def my_medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    import numpy as np

    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.median (y, axis=1)


def my_meanfilt (x, k):
    """Apply a length-k mean filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    
    import numpy as np

    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:,i] = x[:-j]
        y[:j,i] = x[0]
        y[:-j,-(i+1)] = x[j:]
        y[-j:,-(i+1)] = x[-1]
    return np.mean (y, axis=1)

def replace_nan_inf_with_adjacent_avg(arr):
    # Ensure the input is a numpy array of floats
    arr = np.array(arr, dtype=float)

    # Define a kernel for convolution (4-connected neighbors)
    kernel = np.array([[1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 0, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1]], dtype=float)

    # Handle NaN and inf values
    mask = ~np.isfinite(arr) 
    arr[mask] = 0  # Temporarily set NaN/inf to 0 for convolution

    # Convolve the array with the kernel to get the sum of adjacent values
    convolved_sum = convolve(arr, kernel, boundary='extend', normalize_kernel=False)

    # Convolve the mask to get the count of finite adjacent values
#    convolved_mask = convolve(~mask.astype(float), kernel, boundary='extend', normalize_kernel=False)
    mask_r = ~mask
    convolved_mask = convolve(mask_r.astype(float), kernel, boundary='extend', normalize_kernel=False)
    
    # Compute the average of adjacent values where finite adjacent values are available
    adjacent_avg = convolved_sum / convolved_mask
    adjacent_avg[convolved_mask == 0] = np.nan  # Handle cells with no adjacent finite values

    # Replace NaN and inf values with the computed adjacent average
    arr[mask] = adjacent_avg[mask]

    return arr



        
def nanaverage(A,weights,axis):
    return np.nansum(A*weights,axis=axis)/((~np.isnan(A))*weights).sum(axis=axis)

        
def map_plot_DAP(tab_DAP,line='flux_Halpha_6562.85', \
                 vmin=0, vmax=0, title=None, filename='junk',\
                 cmap='Spectral', fsize=5, figs_dir='.',fig_type='png',\
                 gamma=1.0, sf=1.0, tab_pt=None):

    rc.update({'font.size': 19,\
               'font.weight': 900,\
               'text.usetex': True,\
               'path.simplify'           :   True,\
               'xtick.labelsize' : 19,\
               'ytick.labelsize' : 19,\
               'axes.linewidth'  : 2.0,\
               'xtick.major.size'        :   6,\
               'ytick.major.size'        :   6,\
               'xtick.minor.size'        :   3,\
               'ytick.minor.size'        :   3,\
               'xtick.major.width'       :   1,\
               'ytick.major.width'       :   1,\
               'lines.markeredgewidth'   :   1,\
               'legend.numpoints'        :   1,\
               'xtick.minor.width'       :   1,\
               'ytick.minor.width'       :   1,\
               'legend.frameon'          :   False,\
               'legend.handletextpad'    :   0.3,\
               'font.family'    :   'serif',\
               'mathtext.fontset'        :   'stix',\
               'axes.facecolor' : "w",\
               })
    
    X=tab_DAP['ra']
    Y=tab_DAP['dec']
    C=tab_DAP[line]    
    if (title== None):
        title=line.replace("_"," ")
        title=line.replace("_"," ")
    if (vmin==vmax):
        vmin=np.nanmin(C)
        vmax=np.nanmax(C)
    print(line,vmin,vmax)
    fig,ax = plt.subplots(1,1,figsize=(fsize,fsize))
    norm=mpl_colors.PowerNorm(vmin=vmin,vmax=vmax,gamma=gamma)
    scat=scatter(X, Y, ax, size=sf*35.6/3600, c=C, cmap=cmap, norm=norm)
#    scat=scatter(X, Y, ax, size=35.6/3600, c=C,vmin = vmin, vmax = vmax, cmap=cmap, norm=colors.PowerNorm(gamma=gamma))
    ax.set_xlabel('Ra [deg]')#,fontsize=21)
    ax.set_ylabel('Dec [deg]')#,fontsize=21)
    xx = ax.get_xlim()
    ax.set_xlim(xx[1],xx[0])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(scat.sc, cax=cax,label=fr'{title}')#,fontsize=21)
    if (tab_pt != None):
        for tap_pt_now in tab_pt:
            ax.text(tab_pt['ra'],tab_pt['dec'],tab_pt['id'])
#    ax.set_aspect('equal', adjustable='box')
    try:
        plt.show()
    except:
        plt_show=False
    fig.tight_layout()
    fig.savefig(f'{figs_dir}/{filename}.{fig_type}')
    plt.close()


def map_plot_DAP_rgb(tab_DAP,\
                     rgb_key=('flux_pe_6583.45','flux_pe_6562.85','flux_pe_5006.84'), \
                     rgb_scale=(1.0,1.0,1.0),\
                 vmin=0, vmax=0, titles=('r','g','b'), filename='junk',\
                fsize=5, figs_dir='.',fig_type='png',\
                     gamma=1.0, sf=1.0, tab_pt=None,fs=18, equal=True):

    rc.update({'font.size': fs*fsize/5,\
               'font.weight': 900,\
               'text.usetex': True,\
               'path.simplify'           :   True,\
               'xtick.labelsize' : fs*fsize/5,\
               'ytick.labelsize' : fs*fsize/5,\
               'axes.linewidth'  : 2.0,\
               'xtick.major.size'        :   6,\
               'ytick.major.size'        :   6,\
               'xtick.minor.size'        :   3,\
               'ytick.minor.size'        :   3,\
               'xtick.major.width'       :   1,\
               'ytick.major.width'       :   1,\
               'lines.markeredgewidth'   :   1,\
               'legend.numpoints'        :   1,\
               'xtick.minor.width'       :   1,\
               'ytick.minor.width'       :   1,\
               'legend.frameon'          :   False,\
               'legend.handletextpad'    :   0.3,\
               'font.family'    :   'serif',\
               'mathtext.fontset'        :   'stix',\
               'axes.facecolor' : "w",\
               })
    
    X= tab_DAP['ra'].value
    Y= tab_DAP['dec'].value
    R=tab_DAP[rgb_key[2]].value  
    G=tab_DAP[rgb_key[1]].value    
    B=tab_DAP[rgb_key[0]].value 
    R = R * rgb_scale[2]
    G = G * rgb_scale[1]
    B = B * rgb_scale[0]
    RGB = np.stack((R, G, B), axis=1)
    if (vmin==vmax):
        vmin=np.nanmin(RGB)
        vmax=np.nanmax(RGB)
    print(f'min/max:{vmin}/{vmax}')
    r = (R-vmin)/(vmax-vmin)
    g = (G-vmin)/(vmax-vmin)
    b = (B-vmin)/(vmax-vmin)

    r[r>1]=1
    r[r<0]=0
    b[b>1]=1
    b[b<0]=0
    r[r>1]=1
    r[r<0]=0    
    
    r = np.ma.filled(r, fill_value=0.0)
    g = np.ma.filled(g, fill_value=0.0)
    b = np.ma.filled(b, fill_value=0.0)


    r=gamma_scale(r,gamma)
    g=gamma_scale(g,gamma)
    b=gamma_scale(b,gamma)

#    r=r**gamma
#    g=g**gamma
#    b=b**gamma
    
#    print (r)
    r = np.ma.compressed(r)
    g = np.ma.compressed(g)
    b = np.ma.compressed(b)   


    #print(r,g,b)
    colors = np.stack((r, g, b), axis=1)
    colors[colors>1]=1
    colors[colors<0]=0
#    print(colors)
    fig,ax = plt.subplots(1,1,figsize=(fsize,fsize*0.95))
#    norm=mpl_colors.PowerNorm(vmin=0,vmax=1,gamma=gamma)
    scatter(X, Y, ax, color=colors, size=sf*35.6/3600)#, color=colors)
    ax.set_xlabel('Ra [deg]')#,fontsize=21)
    ax.set_ylabel('Dec [deg]')#,fontsize=21)
    xx = ax.get_xlim()
    yy = ax.get_ylim()
    ax.set_xlim(xx[1],xx[0])
    ax.text(xx[1]-0.03*(xx[1]-xx[0]),yy[1]-0.06*(yy[1]-yy[0]),titles[0],fontsize=fs*(fsize/5),color='tomato')
    ax.text(xx[1]-0.03*(xx[1]-xx[0]),yy[1]-0.12*(yy[1]-yy[0]),titles[1],fontsize=fs*(fsize/5),color='forestgreen')
    ax.text(xx[1]-0.03*(xx[1]-xx[0]),yy[1]-0.18*(yy[1]-yy[0]),titles[2],fontsize=fs*(fsize/5),color='steelblue')

    if (tab_pt != None):
        for tap_pt_now in tab_pt:
            ax.text(tab_pt['ra'],tab_pt['dec'],tab_pt['id'])
    if (equal == True):
        ax.set_aspect('equal', adjustable='box')
    try:
        plt.show()
    except:
        plt_show=False
    fig.tight_layout()
    fig.savefig(f'{figs_dir}/{filename}.{fig_type}', facecolor='white')
    plt.close()

    
    
def list_columns(obj, cols=4, columnwise=True, gap=2):
    """
    Print the given list in evenly-spaced columns.

    Parameters
    ----------
    obj : list
        The list to be printed.
    cols : int
        The number of columns in which the list should be printed.
    columnwise : bool, default=True
        If True, the items in the list will be printed column-wise.
        If False the items in the list will be printed row-wise.
    gap : int
        The number of spaces that should separate the longest column
        item/s from the next column. This is the effective spacing
        between columns based on the maximum len() of the list items.
    """

    sobj = [str(item) for item in obj]
    if cols > len(sobj): cols = len(sobj)
    max_len = max([len(item) for item in sobj])
    if columnwise: cols = int(math.ceil(float(len(sobj)) / float(cols)))
    plist = [sobj[i: i+cols] for i in range(0, len(sobj), cols)]
    if columnwise:
        if not len(plist[-1]) == cols:
            plist[-1].extend(['']*(len(sobj) - len(plist[-1])))
        plist = zip(*plist)
    printer = '\n'.join([
        ''.join([c.ljust(max_len + gap) for c in p])
        for p in plist])
    print(printer)

def dap_indices_spec(wave__w, flux_ssp__w, res__w, redshift, n_sim, plot=0, wl_half_range=200, verbose=False,\
                      __indices__=__indices__, _INDICES_POS=_INDICES_POS_):
    if plot:
        from matplotlib import use as mpl_use
        mpl_use('Agg')
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt

        #plt.style.use('dark_background')
        f, ax = plt.subplots()
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Flux')

    nw = wave__w.size
    cdelt = wave__w[1] - wave__w[0]
    names__i = list(__indices__.keys())
    _ind = np.array(list(__indices__.values()))
    wmin = _ind.T[_INDICES_POS['OLb1']].min() - wl_half_range
    wmax = _ind.T[_INDICES_POS['OLr2']].max() + wl_half_range
    n_ind = _ind.shape[0]

    FWHM = __sigma_to_FWHM__*3*cdelt

    stflux = pdl_stats(flux_ssp__w)
    if (stflux[_STATS_POS['mean']] == 0) and (stflux[_STATS_POS['min']] == stflux[_STATS_POS['max']]):
        flux_ssp__w = np.ones_like(flux_ssp__w)
        res__w = 100*np.ones_like(flux_ssp__w)

    med_res__w = median_filter(res__w, size=int(3*FWHM/cdelt), mode='reflect')

    _ind_corr = _ind*(1 + redshift)
    _ind_corr.T[_INDICES_POS['OLb']] = (_ind_corr.T[_INDICES_POS['OLb1']] + _ind_corr.T[_INDICES_POS['OLb2']])/2
    _ind_corr.T[_INDICES_POS['OLr']] = (_ind_corr.T[_INDICES_POS['OLr1']] + _ind_corr.T[_INDICES_POS['OLr2']])/2
    wmin = _ind_corr.T[_INDICES_POS['OLb1']].min() - wl_half_range
    wmax = _ind_corr.T[_INDICES_POS['OLr2']].max() + wl_half_range

    med_flux = np.median(flux_ssp__w[np.isfinite(flux_ssp__w)])
    std_flux = np.std(flux_ssp__w[np.isfinite(flux_ssp__w)])
    EW__ki = np.zeros((n_sim, n_ind))
    for k in range(n_sim):
        noise = np.random.normal(size=nw)
        noise *= res__w
        _flux__w = flux_ssp__w + noise
        _med_flux = np.median(_flux__w)
        _min_flux = -0.1*np.abs(_med_flux)
        _max_flux = 3.5*_med_flux
        for ii in range(n_ind):
            if plot:
                plt.cla()
                ax = plt.gca()
                ax.set_xlim(wmin, wmax)
                ax.set_ylim(_min_flux - 0.2*np.abs(_min_flux), _max_flux + 0.2*np.abs(_max_flux))
                ax.plot(wave__w, _flux__w, 'k-')
            name = names__i[ii]
            Lb1 = _ind_corr[ii][_INDICES_POS['OLb1']]
            Lb2 = _ind_corr[ii][_INDICES_POS['OLb2']]
            Lr1 = _ind_corr[ii][_INDICES_POS['OLr1']]
            Lr2 = _ind_corr[ii][_INDICES_POS['OLr2']]
            L1 = _ind_corr[ii][_INDICES_POS['OL1']]
            L2 = _ind_corr[ii][_INDICES_POS['OL2']]
            Lb = _ind_corr[ii][_INDICES_POS['OLb']]
            Lr = _ind_corr[ii][_INDICES_POS['OLr']]

            str_verbose = f'L1:{L1}, L2:{L2}\nblue: Lb:{Lb}, Lb1:{Lb1}, Lb2:{Lb2}\nred: Lr:{Lr}, Lr1:{Lr1}, Lr2:{Lr2}'
            print_verbose(str_verbose, verbose=verbose, level=1)

            if name != 'D4000':
                iw1_b = (Lb1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2_b = (Lb2 - wave__w[0] - 0.5*cdelt)/cdelt
                Sb = 0
                nb = 0
                for iw in range(int(iw1_b + 1), int(iw2_b)):
                    Sb += _flux__w[iw]*cdelt
                    nb += 1
                iw = int(iw1_b)
                ff = iw + 1 - iw1_b
                Sb += (_flux__w[iw]*ff*cdelt)
                iw = int(iw2_b)
                ff = iw2_b - iw
                Sb += (_flux__w[iw]*ff*cdelt)
                Sb = Sb/(Lb2 - Lb1)

                iw1_r = (Lr1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2_r = (Lr2 - wave__w[0] - 0.5*cdelt)/cdelt
                Sr = 0
                nr = 0
                for iw in range(int(iw1_r + 1), int(iw2_r)):
                    Sr += _flux__w[iw]*cdelt
                    nr += 1
                iw = int(iw1_r)
                ff = iw + 1 - iw1_r
                Sr += (_flux__w[iw]*ff*cdelt)
                iw = int(iw2_r)
                ff = iw2_r - iw
                Sr += (_flux__w[iw]*ff*cdelt)
                Sr = Sr/(Lr2 - Lr1)
                EW = 0
                CK = []
                waveK = []
                iw1 = (L1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2 = (L2 - wave__w[0] - 0.5*cdelt)/cdelt
                for iw in range(int(iw1 + 1), int(iw2)):
                    C = Sb*((Lr - wave__w[iw])/(Lr - Lb)) + Sr*((wave__w[iw] - Lb)/(Lr - Lb))
                    EW = EW + (1 - _flux__w[iw]/C)*(wave__w[iw] - wave__w[iw - 1])
                    CK.append(C)
                    waveK.append(wave__w[iw])
                iw = int(iw1)
                ff = iw + 1 - iw1
                C = Sb*((Lr - wave__w[iw])/(Lr - Lb)) + Sr*((wave__w[iw] - Lb)/(Lr - Lb))
                EW = EW + (1 - _flux__w[iw]/C)*(wave__w[iw] - wave__w[iw - 1])*ff
                iw = int(iw2)
                ff = iw2 - iw
                C = Sb*((Lr - wave__w[iw])/(Lr - Lb)) + Sr*((wave__w[iw] - Lb)/(Lr - Lb))
                EW = EW + (1 - _flux__w[iw]/C)*(wave__w[iw] - wave__w[iw - 1])*ff
                EW = EW/(1 + redshift)
                if plot:
                    _xy, dx, dy = (Lb1, _max_flux*0.5), Lb2-Lb1, _max_flux*0.3
                    rectLb1Lb2 = patches.Rectangle(_xy, dx, dy, edgecolor='b', facecolor='none')
                    ax.add_patch(rectLb1Lb2)
                    _xy, dx, dy = (Lr1, _max_flux*0.5), Lr2-Lr1, _max_flux*0.3
                    rectLr1Lr2 = patches.Rectangle(_xy, dx, dy, edgecolor='r', facecolor='none')
                    ax.add_patch(rectLr1Lr2)
                    _xy, dx, dy = (L1, _max_flux*0.5), L2-L1, _max_flux*0.3
                    rectL1L2 = patches.Rectangle(_xy, dx, dy, edgecolor='g', facecolor='none')
                    ax.add_patch(rectL1L2)
                    ax.plot(waveK, CK, ls='--', c='gray')
                    ax.scatter(Lb, Sb, marker='x', color='b', s=50)
                    ax.scatter(Lr, Sr, marker='x', color='r', s=50)
                    ax.set_xlim(Lb1 - 0.02*Lb1, Lr2 + 0.02*Lr2)
                    ax.text(0.98, 0.98, name, transform=ax.transAxes, va='top', ha='right')
                    plt.pause(0.001)
            else:
                Sb = 0
                nb = 0
                iw1_b = (Lb1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2_b = (Lb2 - wave__w[0] - 0.5*cdelt)/cdelt
                for iw in range(int(iw1_b + 1), int(iw2_b)):
                    Sb += _flux__w[iw]*cdelt
                    nb += 1
                iw = int(iw1_b)
                ff = iw + 1 - iw1_b
                Sb += _flux__w[iw]*ff*cdelt
                iw = int(iw2_b)
                ff = iw2_b - iw
                Sb += _flux__w[iw]*ff*cdelt
                Sb = Sb/(Lb2 - Lb1)

                S = 0
                K = 0
                iw1 = (L1 - wave__w[0] - 0.5*cdelt)/cdelt
                iw2 = (L2 - wave__w[0] - 0.5*cdelt)/cdelt
                for iw in range(int(iw1 + 1), int(iw2)):
                    S += _flux__w[iw]*cdelt
                    K += 1
                iw = int(iw1)
                ff = iw + 1 - iw1
                S += _flux__w[iw]*ff*cdelt
                iw = int(iw2)
                ff = iw2 - iw
                S += _flux__w[iw]*ff*cdelt
                S = S/(L2 - L1)
                if Sb != 0:
                    EW = S/Sb
                else:
                    EW = 1e16
                if plot:
                    _xy, dx, dy = (Lb1, _max_flux*0.5), Lb2-Lb1, _max_flux*0.3
                    rectLb1Lb2 = patches.Rectangle(_xy, dx, dy, edgecolor='b', facecolor='none')
                    ax.add_patch(rectLb1Lb2)
                    # _xy, dx, dy = (Lr1, _max_flux*0.5), Lr2-Lr1, _max_flux*0.3
                    # rectLr1Lr2 = patches.Rectangle(_xy, dx, dy, edgecolor='r', facecolor='none')
                    # ax.add_patch(rectLr1Lr2)
                    _xy, dx, dy = (L1, _max_flux*0.5), L2-L1, _max_flux*0.3
                    rectL1L2 = patches.Rectangle(_xy, dx, dy, edgecolor='r', facecolor='none')
                    ax.add_patch(rectL1L2)
                    # ax.plot(waveK, CK, '--k')
                    ax.scatter(Lb, Sb, marker='x', color='b', s=50)
                    ax.scatter((L1 + L2)/2, S, marker='x', color='r', s=50)
                    ax.set_xlim(Lb1 - 0.02*Lb1, L2 + 0.02*L2)
                    ax.text(0.98, 0.98, name, transform=ax.transAxes, va='top', ha='right')
                    plt.pause(0.001)
            EW__ki[k, ii] = EW
    if plot:
        plt.close(f)
    return EW__ki, med_flux, std_flux


#
# data = FLUX_RSS ['FLUX'] extension
# sdata = SKY_RSS ['SKY'] extension
# hdr = header extension of flux extension
# Band: wavelengthg of the emission line, and wavelegnth of two non-emission line regimes
# This could be refined, of course
#
def sky_hack_f(data, sdata, hdr, m2a=10e9, band=np.array((7238,7242,7074,7084,7194,7265))):
    #nx=hdr['NAXIS1']
    #ny=hdr['NAXIS2']
    ny=hdu['FLUX'].data.shape[0]
    nx=hdu['FLUX'].data.shape[1]

    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    i_band = ((band-crval)/cdelt).astype(int)

    map_b=np.mean(data[:,i_band[0]:i_band[1]],axis=1)
    map_0=np.mean(data[:,i_band[2]:i_band[3]],axis=1)
    map_1=np.mean(data[:,i_band[4]:i_band[5]],axis=1)
    map_c = map_b-0.5*(map_0+map_1)

    smap_b=np.mean(sdata[:,i_band[0]:i_band[1]],axis=1)
    smap_0=np.mean(sdata[:,i_band[2]:i_band[3]],axis=1)
    smap_1=np.mean(sdata[:,i_band[4]:i_band[5]],axis=1)
    smap_c = smap_b-0.5*(smap_0+smap_1)

    scale=map_c/smap_c
    data_c=data-np.transpose(np.transpose(sdata)*scale)
    return data_c

def eline(w,W,F,D,V,dap_fwhm=2.354):
  c = 299792.00
  w0 = W*(1+(V/c))
  sigma=D/dap_fwhm
  e1=np.exp(-0.5*((w-w0)/sigma)**2)
  return F*e1/(sigma*((2*3.1416)**0.5))     
#
# Load an standard LVM RSS file
#
# m2a=10e9 : Pre v1.0 reduction!
def load_LVM_rss(lvm_file, m2a=1, flux_scale=1e16, ny_range=None, sky_hack= True,nx_range=None, get_lsf=False):
    """Return the RSS from the given and LVM filename in the parsed command line arguments"""
    hdu = fits.open(lvm_file, memmap=False)
    rss_0_hdr = hdu[0].header
    rss_f_spectra = hdu['FLUX'].data
    rss_f_hdr = hdu['FLUX'].header
    mask = hdu['MASK'].data
    if (get_lsf == True):
        LSF_rss = np.float32(hdu['LSF'].data/2.354)
    else:
        LSF_rss = np.float32(rss_f_spectra*0.0)

#    rss_f_spectra[mask==1]=np.nan
    
    try:
        crval1=rss_f_hdr['CRVAL1']
    except:
        rss_f_hdr = hdu[0].header
    #    rss_f_hdr = hdu['FLUX'].header
    #    rss_sky = hdu['SKY'].data
    
    try:
        rss_ivar = hdu['IVAR'].data
        hdu_e = fits.ImageHDU(data=1/np.sqrt(rss_ivar),header=hdu[1].header,name='ERROR')
        hdu.append(hdu_e)
        Cframe=False
    except:
        Cframe=True

    try: 
        rss_e_spectra = hdu['ERR'].data
        hdu_e = fits.ImageHDU(data=rss_e_spectra,header=hdu[1].header,name='ERROR')
    except:
        print('# No ERR extension') 

    try:
        rss_e_spectra = hdu['ERROR'].data
 #       rss_e_spectra[mask==1]=np.nan


    except:
        rss_e_spectra = np.ones(rss_f_spectra.shape)
        std_spectra = 0.1*np.nanstd(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51)),axis=1)
        for I,std_now in enumerate(std_spectra):
            rss_e_spectra[I,:]=std_now*rss_e_spectra[I,:]
    wl__w = np.array([rss_f_hdr["CRVAL1"] + i*rss_f_hdr["CDELT1"] for i in range(hdu['FLUX'].data.shape[1])])
    wl__w = wl__w*m2a
    #
    # We need to revise the errors!
    #
    mean_flux = np.nanmedian(rss_f_spectra[:,5000:6000])
    #print(f'# mean_flux = {mean_flux}')
    if (mean_flux>1e-3):
        flux_scale=1
        print(f'# mean_flux = {mean_flux} does not seem to be in 10^-16 cgs')
    rss_f_spectra=rss_f_spectra*flux_scale
    rss_e_spectra=rss_e_spectra*flux_scale
    rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]*m2a
    rss_f_hdr["CDELT1"]=rss_f_hdr["CDELT1"]*m2a
    rss_f_hdr['NAXIS1']=hdu['FLUX'].data.shape[1]
    rss_f_hdr['NAXIS2']=hdu['FLUX'].data.shape[0]
    #ny=hdr['NAXIS2']
    
    if (sky_hack == True):
        print(f'# sky re-evaluated')
        try:
            rss_sky = hdu['SKY'].data
            rss_f_spectra=sky_hack_f(rss_f_spectra, rss_sky, rss_f_hdr)
        except:
            print('# sky-hack did not work')
            sky_hack = False

    if (nx_range != None):
        print(f'# X-axis trimmed: {nx_range}')
        rss_f_spectra=rss_f_spectra[:,nx_range[0]:nx_range[1]]
        rss_e_spectra=rss_e_spectra[:,nx_range[0]:nx_range[1]]
        LSF_rss=LSF_rss[:,nx_range[0]:nx_range[1]]
        wl__w=wl__w[nx_range[0]:nx_range[1]]
        rss_f_hdr['NAXIS1']=nx_range[1]-nx_range[0]+1
        rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]+nx_range[0]*rss_f_hdr["CDELT1"]
    
    if (ny_range != None):
        print(f'# Y-axis trimmed: {ny_range}')
        rss_f_spectra=rss_f_spectra[ny_range[0]:ny_range[1],:]
        rss_e_spectra=rss_e_spectra[ny_range[0]:ny_range[1],:]
        LSF_rss=LSF_rss[ny_range[0]:ny_range[1],:]
        rss_f_hdr['NAXIS2']=ny_range[1]-ny_range[0]+1

    rss_e_spectra=np.abs(rss_e_spectra)
    if (get_lsf == True):
        return wl__w, rss_f_spectra, rss_e_spectra, rss_f_hdr, rss_0_hdr,LSF_rss
    else:
        return wl__w, rss_f_spectra, rss_e_spectra, rss_f_hdr, rss_0_hdr


def load_LVM_rss_old(lvm_file, m2a=1, flux_scale=1e16, ny_range=None, sky_hack= True,nx_range=None):
    """Return the RSS+lsf from the given and LVM filename in the parsed command line arguments"""
    hdu = fits.open(lvm_file, memmap=False)
    rss_0_hdr = hdu[0].header
    rss_f_spectra = hdu['FLUX'].data
    rss_f_hdr = hdu['FLUX'].header
    mask = hdu['MASK'].data


#    rss_f_spectra[mask==1]=np.nan
    
    try:
        crval1=rss_f_hdr['CRVAL1']
    except:
        rss_f_hdr = hdu[0].header
    #    rss_f_hdr = hdu['FLUX'].header
    #    rss_sky = hdu['SKY'].data
    
    try:
        rss_ivar = hdu['IVAR'].data
        hdu_e = fits.ImageHDU(data=1/np.sqrt(rss_ivar),header=hdu[1].header,name='ERROR')
        hdu.append(hdu_e)
        Cframe=False
    except:
        Cframe=True

    try:
        rss_e_spectra = hdu['ERROR'].data
 #       rss_e_spectra[mask==1]=np.nan


    except:
        rss_e_spectra = np.ones(rss_f_spectra.shape)
        std_spectra = 0.1*np.nanstd(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51)),axis=1)
        for I,std_now in enumerate(std_spectra):
            rss_e_spectra[I,:]=std_now*rss_e_spectra[I,:]
    wl__w = np.array([rss_f_hdr["CRVAL1"] + i*rss_f_hdr["CDELT1"] for i in range(hdu['FLUX'].data.shape[1])])
    wl__w = wl__w*m2a
    #
    # We need to revise the errors!
    #
    mean_flux = np.nanmedian(rss_f_spectra[:,5000:6000])
    #print(f'# mean_flux = {mean_flux}')
    if (mean_flux>1e-3):
        flux_scale=1
        print(f'# mean_flux = {mean_flux} does not seem to be in 10^-16 cgs')
    rss_f_spectra=rss_f_spectra*flux_scale
    rss_e_spectra=rss_e_spectra*flux_scale
    rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]*m2a
    rss_f_hdr["CDELT1"]=rss_f_hdr["CDELT1"]*m2a
    rss_f_hdr['NAXIS1']=hdu['FLUX'].data.shape[1]
    rss_f_hdr['NAXIS2']=hdu['FLUX'].data.shape[0]
    #ny=hdr['NAXIS2']
    
    if (sky_hack == True):
        print(f'# sky re-evaluated')
        try:
            rss_sky = hdu['SKY'].data
            rss_f_spectra=sky_hack_f(rss_f_spectra, rss_sky, rss_f_hdr)
        except:
            print('# sky-hack did not work')
            sky_hack = False

    if (nx_range != None):
        print(f'# X-axis trimmed: {nx_range}')
        rss_f_spectra=rss_f_spectra[:,nx_range[0]:nx_range[1]]
        rss_e_spectra=rss_e_spectra[:,nx_range[0]:nx_range[1]]
        wl__w=wl__w[nx_range[0]:nx_range[1]]
        rss_f_hdr['NAXIS1']=nx_range[1]-nx_range[0]+1
        rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]+nx_range[0]*rss_f_hdr["CDELT1"]
    
    if (ny_range != None):
        print(f'# Y-axis trimmed: {ny_range}')
        rss_f_spectra=rss_f_spectra[ny_range[0]:ny_range[1],:]
        rss_e_spectra=rss_e_spectra[ny_range[0]:ny_range[1],:]
        rss_f_hdr['NAXIS2']=ny_range[1]-ny_range[0]+1
    rss_e_spectra=np.abs(rss_e_spectra)
    return wl__w, rss_f_spectra, rss_e_spectra, rss_f_hdr, rss_0_hdr


#
# Load an standard LVM RSS file
#
def load_LVMSIM_rss(lvm_file, m2a=1, flux_scale=1e16, ny_range=None, nx_range=None):
    """Return the RSS from the given and LVM filename in the parsed command line arguments"""
    hdu = fits.open(lvm_file, memmap=False)
    rss_0_hdr = hdu[0].header
    rss_f_spectra = hdu['TARGET'].data
    rss_f_hdr = hdu['TARGET'].header
    rss_sky = hdu['SKY'].data
    try:
        rss_e_spectra = hdu['ERR'].data#+np.abs(rss_f_spectra)
    except:
        rss_e_spectra = 3*np.nanstd(np.abs(rss_f_spectra))*np.ones(rss_f_spectra.shape)
        #rss_e_spectra = 3*np.nanstd(np.abs(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51))))*np.ones(rss_f_spectra.shape)
        #rss_e_spectra = np.abs(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51)))  
    # We force an arbitrary (?) error      
    #rss_e_spectra = np.abs(0.07*rss_f_spectra)+0.07*np.nanmean(np.abs(rss_f_spectra[rss_f_spectra>0]))
    #rss_e_spectra = np.abs(0.07*rss_f_spectra)+0.07*np.nanmean(np.abs(rss_f_spectra[rss_f_spectra>0]))
#    rss_e_spectra = 2*median_filter(np.abs(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51))),size=(1,51))        
    wl__w = np.array([rss_f_hdr["CRVAL1"] + i*rss_f_hdr["CDELT1"] for i in range(rss_f_hdr["NAXIS1"])])
    wl__w = wl__w*m2a
    rss_f_spectra=rss_f_spectra*flux_scale
    #
    # We need to revise the errors!
    #
    rss_e_spectra=rss_e_spectra*flux_scale
    rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]*m2a
    rss_f_hdr["CDELT1"]=rss_f_hdr["CDELT1"]*m2a

    if (nx_range != None):
        print(f'# X-axis trimmed: {nx_range}')
        rss_f_spectra=rss_f_spectra[:,nx_range[0]:nx_range[1]]
        rss_e_spectra=rss_e_spectra[:,nx_range[0]:nx_range[1]]
        wl__w=wl__w[nx_range[0]:nx_range[1]]
        rss_f_hdr['NAXIS1']=nx_range[1]-nx_range[0]+1
        rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]+nx_range[0]*rss_f_hdr["CDELT1"]
    
    if (ny_range != None):
        print(f'# Y-axis trimmed: {ny_range}')
        rss_f_spectra=rss_f_spectra[ny_range[0]:ny_range[1],:]
        rss_e_spectra=rss_e_spectra[ny_range[0]:ny_range[1],:]
        rss_f_hdr['NAXIS2']=ny_range[1]-ny_range[0]+1
    return wl__w, rss_f_spectra, rss_e_spectra, rss_f_hdr, rss_0_hdr


#
# Load an standard LVM RSS file
#
def load_in_rss(lvm_file, nHDU=0, m2a=1, flux_scale=1e16, ny_range=None, nx_range=None):
    """Return the RSS from the given and LVM filename in the parsed command line arguments"""
    hdu = fits.open(lvm_file, memmap=False)
    rss_0_hdr = hdu[0].header
    rss_f_spectra = hdu[nHDU].data
    rss_f_hdr = hdu[nHDU].header
    try:
        rss_e_spectra = hdu['ERR'].data#+np.abs(rss_f_spectra)
    except:
        rss_e_spectra = np.ones(rss_f_spectra.shape)
        std_spectra = 0.1*np.nanstd(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51)),axis=1)
        for I,std_now in enumerate(std_spectra):
            rss_e_spectra[I,:]=std_now*rss_e_spectra[I,:]
        #0.03*np.abs(median_filter(rss_f_spectra,size=(1,51)))+3*np.nanstd(np.abs(rss_f_spectra))*np.ones(rss_f_spectra.shape)
#        rss_e_spectra = 3*np.nanstd(np.abs(rss_f_spectra))*np.ones(rss_f_spectra.shape)
        #rss_e_spectra = np.nanstd(rss_f_spectra-median_filter(rss_f_spectra,size=(1,51)))*np.ones(rss_f_spectra.shape)
        #rss_e_spectra = 0.03*np.abs(median_filter(rss_f_spectra,size=(1,51)))  
    wl__w = np.array([rss_f_hdr["CRVAL1"] + i*rss_f_hdr["CDELT1"] for i in range(rss_f_hdr["NAXIS1"])])
    wl__w = wl__w*m2a
    rss_f_spectra=rss_f_spectra*flux_scale
    #
    # We need to revise the errors!
    #
    rss_e_spectra=rss_e_spectra*flux_scale
    rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]*m2a
    rss_f_hdr["CDELT1"]=rss_f_hdr["CDELT1"]*m2a

    if (nx_range != None):
        print(f'# X-axis trimmed: {nx_range}')
        rss_f_spectra=rss_f_spectra[:,nx_range[0]:nx_range[1]]
        rss_e_spectra=rss_e_spectra[:,nx_range[0]:nx_range[1]]
        wl__w=wl__w[nx_range[0]:nx_range[1]]
        rss_f_hdr['NAXIS1']=nx_range[1]-nx_range[0]+1
        rss_f_hdr["CRVAL1"]=rss_f_hdr["CRVAL1"]+nx_range[0]*rss_f_hdr["CDELT1"]
    
    if (ny_range != None):
        print(f'# Y-axis trimmed: {ny_range}')
        rss_f_spectra=rss_f_spectra[ny_range[0]:ny_range[1],:]
        rss_e_spectra=rss_e_spectra[ny_range[0]:ny_range[1],:]
        rss_f_hdr['NAXIS2']=ny_range[1]-ny_range[0]+1
    return wl__w, rss_f_spectra, rss_e_spectra, rss_f_hdr, rss_0_hdr



PLATESCALE = 112.36748321030637


def rotate(xx,yy,angle):
    # rotate x and y cartesian coordinates by angle (in degrees)
    # about the point (0,0)
    if (angle == None):
        angle=0.0
    theta = -1.*angle * np.pi / 180. # in radians
    xx1 = np.cos(theta) * xx - np.sin(theta) * yy
    yy1 = np.sin(theta) * xx + np.cos(theta) * yy
    return xx1, yy1

def make_radec(xx0,yy0,ra,dec,pa):
    PLATESCALE = 112.36748321030637
    if (pa == None):
        pa = 0.0
    if (ra == None):
        ra = 0.0
    if (dec == None):
        dec = 0.0
    xx, yy = rotate(xx0,yy0,pa)
    ra_fib = ra + xx*PLATESCALE/3600./np.cos(dec*np.pi/180.) 
    dec_fib = dec - yy*PLATESCALE/3600. 
    return ra_fib, dec_fib

    
def make_line(wave, r1, sci, wl_shift_vel, whichone):

    # Halpha
    wl_ha = (wave > 6560 + wl_shift_vel/3e5*6560) & (wave < 6570+ wl_shift_vel/3e5*6570)
    iis_ha = np.where(wl_ha)[0]
    ha = np.sum(r1[:,iis_ha],axis=1)[sci]

    wl_ha_cont = (wave > 6600+ wl_shift_vel/3e5*6600) & (wave < 6610+ wl_shift_vel/3e5*6610)
    iis_ha_cont = np.where(wl_ha_cont)[0]
    ha_cont = np.sum(r1[:,iis_ha_cont],axis=1)[sci]
        
        
    # SII doublet

    wl_sii1 = (wave > 6715+ wl_shift_vel/3e5*6715) & (wave < 6725+ wl_shift_vel/3e5*6725)
    iis_sii1 = np.where(wl_sii1)[0]
    sii1 = np.sum(r1[:,iis_sii1],axis=1)[sci]

    wl_sii2 = (wave > 6730+ wl_shift_vel/3e5*6730) & (wave < 6740+ wl_shift_vel/3e5*6740)
    iis_sii2 = np.where(wl_sii2)[0]
    sii2 = np.sum(r1[:,iis_sii2],axis=1)[sci]


    wl_sii_cont = (wave > 6700+ wl_shift_vel/3e5*6700) & (wave < 6710+ wl_shift_vel/3e5*6710)
    iis_sii_cont = np.where(wl_sii_cont)[0]
    sii_cont = np.sum(r1[:,iis_sii_cont],axis=1)[sci]
        
        
    # [SIII]9069

    wl_siii9068 = (wave > 9065+ wl_shift_vel/3e5*9065) & (wave < 9075+ wl_shift_vel/3e5*9075)
    iis_siii9068 = np.where(wl_siii9068)[0]
    siii9068 = np.sum(r1[:,iis_siii9068],axis=1)[sci]
    
    # [OIII]5007
    wl_oiii = (wave > 5000 + wl_shift_vel/3e5*5000) & (wave < 5015+ wl_shift_vel/3e5*5015)
    iis_oiii = np.where(wl_oiii)[0]
    oiii = np.sum(r1[:,iis_oiii],axis=1)[sci]
    
    
    # [OIII]4363    
    wl_oiii4363 = (wave > 4360 + wl_shift_vel/3e5*4360) & (wave < 4365+ wl_shift_vel/3e5*4365)
    iis_oiii4363 = np.where(wl_oiii4363)[0]
    oiii4363 = np.sum(r1[:,iis_oiii4363],axis=1)[sci]

    wl_oiii4363_cont = (wave > 4375+ wl_shift_vel/3e5*4375) & (wave < 4380+ wl_shift_vel/3e5*4380)
    iis_oiii4363_cont = np.where(wl_oiii4363_cont)[0]
    oiii4363_cont = np.sum(r1[:,iis_oiii4363_cont],axis=1)[sci]
   
    # [OII]3727
    wl_oii = (wave > 3720 + wl_shift_vel/3e5*3720) & (wave < 3732+ wl_shift_vel/3e5*3732)
    iis_oii = np.where(wl_oii)[0]
    oii = np.sum(r1[:,iis_oii],axis=1)[sci]
    
    #[NII]5755
    wl_nii5755 = (wave > 5752 + wl_shift_vel/3e5*5752) & (wave < 5758+ wl_shift_vel/3e5*5758)
    iis_nii5755 = np.where(wl_nii5755)[0]
    nii5755 = np.sum(r1[:,iis_nii5755],axis=1)[sci]

    wl_nii5755_cont = (wave > 5710+ wl_shift_vel/3e5*5710) & (wave < 5715+ wl_shift_vel/3e5*5715)
    iis_nii5755_cont = np.where(wl_nii5755_cont)[0]
    nii5755_cont = np.sum(r1[:,iis_nii5755_cont],axis=1)[sci]
    
    wl_nii5755_cont2 = (wave > 5770+ wl_shift_vel/3e5*5770) & (wave < 5775+ wl_shift_vel/3e5*5775)
    iis_nii5755_cont2 = np.where(wl_nii5755_cont2)[0]
    nii5755_cont2 = np.sum(r1[:,iis_nii5755_cont2],axis=1)[sci]
   
    #r band
    wl_r = (wave > 5500 + wl_shift_vel/3e5*5500) & (wave < 6900+ wl_shift_vel/3e5*6900)
    iis_r = np.where(wl_r)[0]
    r = np.sum(r1[:,iis_r],axis=1)[sci]


    if whichone == 'ha':
        return ha
    if whichone == 'ha_sub':
        return ha - ha_cont
    if whichone == 'sii':
        return sii1 + sii2
    if whichone == 'sii_ratio':
        ratio = sii1/sii2
        ratio[sii1+sii2 < 10] = np.nan
        return ratio
    if whichone == 'siii':
        return siii9068
    
    if whichone == 'oiii': 
        return oiii
    
    if whichone == 'oii':
        return oii
    
    if whichone == 'oiii4363':
        return oiii4363 - oiii4363_cont
    
    if whichone == 'nii5755':
        return nii5755 - (nii5755_cont + nii5755_cont2)/2.
   
    if whichone == 'r_cont':
        return r


#
# Routine to aveage all rows in a table
#
def Table_mean_rows(input_table):
    mean_values = {}
    for col in input_table.colnames:
        if np.issubdtype(input_table[col].dtype, np.number):
            mean_values[col] = np.nanmean(input_table[col])
        else:
            mean_values[col] = input_table[col][0]  # Handle non-numeric columns if necessary
    mean_row = {col: [mean_values[col]] for col in input_table.colnames if mean_values[col] is not None}
    output_table = Table(mean_row)
    return output_table


#
# routine to read into an astropy table the Emission line list file
#

def read_tab_EL(el_file='_legacy/emission_lines_long_list.LVM'):
    tab_el=Table(names=('id','wl'), dtype=('S2', 'f4'))
    with open(el_file) as my_file:
        for line in my_file:
            line=line.replace('\n','')
            a_line= re.split(' +', line)#.rstrip('\n'))
            tab_el.add_row((a_line[1],a_line[0]))
    return tab_el


    
def read_file(file_ID, mjd, whichone = 'ha', wl_shift_vel = 0., nobad=False):
    pref = DIR_redux + (mjd) + '/'
    
    # read in the rss file
    rsshdu = fits.open(pref+'lvmCFrame-'+file_ID+'.fits')

    hdr = rsshdu[0].header

    r1 = rsshdu[1].data
    r1_hdr = rsshdu[1].header
    r1_err = rsshdu[2]

    wave=rsshdu[4].data 

    tab = Table(rsshdu[6].data)
    sci = (tab['targettype']=='science')
    if nobad:
        sci = (tab['targettype']=='science') & (tab['fibstatus'] == 0)

    rsshdu.close()

    # get ra/dec measured from coadd guiders?
    agcam_coadd = DIR_agcam+mjd+'/'+'lvm.sci.coadd_s'+file_ID+'.fits'
    if os.path.isfile(agcam_coadd):
        agcam_hdu = fits.open(agcam_coadd)
        agcam_hdr = agcam_hdu[1].header
        w = WCS(agcam_hdr)
        cen = w.pixel_to_world(2500,1000)
        racen = cen.ra.deg  #agcam_hdr['RAMEAS']
        deccen = cen.dec.deg #agcam_hdr['DECMEAS']
        pa = agcam_hdr['PAMEAS'] - 180.
        agcam_hdu.close()
    else:
        racen = hdr['POSCIRA']
        deccen = hdr['POSCIDE']
        try:
            pa = hdr['POSCIPA']
        except:
            pa = 0.0
            hdr['POSCIPA'] = pa

    if (pa == None):
        pa = 0.0
        hdr['POSCIPA'] = pa

    if 'fib_ra' in tab.colnames and 'fib_dec' in tab.colnames:
        ra_fib = tab['fib_ra'][sci]
        dec_fib = tab['fib_dec'][sci]
    else:
        ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], racen, deccen, pa)
    #ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], racen, deccen, pa)

    line_flux = make_line(wave, r1,sci, wl_shift_vel, whichone)

    return ra_fib.data, dec_fib.data, line_flux

def plotty(line_dict, vmin, vmax, title, filename, size=30):
#    size = 30

    fig = plt.figure(figsize=(8,8))
#    for dd in line_dict:
#        d = line_dict[dd]
#        print(dd,d)
    plt.scatter(line_dict['ra_fib'],line_dict['dec_fib'], c =line_dict['line'] ,s=size,vmin = vmin, vmax = vmax)

    plt.title(title)
    plt.xlabel('Ra [deg]')
    plt.ylabel('Dec [deg]')
    plt.colorbar()

    ax = plt.gca()

    xx = ax.get_xlim()
    plt.xlim(xx[1],xx[0])

    ax.set_aspect('equal', adjustable='box')
    plt.draw()
    plt.show()
    fig.savefig('figs/'+filename+'.png')

    plt.close()

def read_RSS_PT(fitsfile, nobad=False, ny_range=None, exposure=666):
    rsshdu = fits.open(fitsfile)
    hdr = rsshdu[0].header
    hdr['exposure']=exposure
    tab = Table(rsshdu['PT'].data)
    sci = np.full(len(tab), True)
    fiberid=tab['id']
    exp_fib=[]
    for IfibID,fibID in enumerate(fiberid):
        exp_fib.append(str(hdr['exposure'])+'.'+str(IfibID))
    try:
        tab['id'] = tab['fiberid']
    except:
        tab['id'] = np.array(exp_fib)
    tab['mask']=sci
    tab['fiberid']=fiberid
    tab['exposure']=hdr['exposure']*np.ones(len(tab),dtype=int)
    if (ny_range != None):
        tab=tab[ny_range[0]:ny_range[1]]
    return tab


def read_MaStar_PT(fitsfile, agcam_coadd, nobad=False, ny_range=None, exposure=666):
    rsshdu = fits.open(fitsfile)

    hdr = rsshdu[0].header
    hdr['exposure']=exposure
    tab = Table(rsshdu['CATALOGUE'].data)
    sci = np.full(len(tab), True)
    fiberid=tab['ID']
    exp_fib=[]
    for fibID in fiberid:
        exp_fib.append(str(hdr['exposure'])+'.'+str(fibID))
    tab['id']=np.array(exp_fib)
    tab['ra']=tab['RA']
    tab['dec']=tab['DEC']
    tab['mask']=sci
    tab['fiberid']=fiberid
    tab['exposure']=hdr['exposure']*np.ones(len(tab),dtype=int)
    if (ny_range != None):
        tab=tab[ny_range[0]:ny_range[1]]
    return tab
    

def read_LVMSIM_PT(fitsfile, agcam_coadd, nobad=False, ny_range=None, exposure=666):
    rsshdu = fits.open(fitsfile)

    hdr = rsshdu[0].header
    hdr['exposure']=exposure
    tab = Table(rsshdu['FIBERID'].data)
    sci = np.full(len(tab), True)
    racen = hdr['RA']
    deccen = hdr['DEC']
    pa = 0

    ra_fib = racen+tab['x']/3600./np.cos(deccen*np.pi/180.) 
    dec_fib = deccen- tab['y']/3600. 

    fiberid=tab['id']
    exp_fib=[]
    for fibID in fiberid:
        exp_fib.append(str(hdr['exposure'])+'.'+str(fibID))
    tab=Table()
    tab['id']=np.array(exp_fib)
    tab['ra']=ra_fib.data
    tab['dec']=dec_fib.data
    tab['mask']=sci
    tab['fiberid']=fiberid
    tab['exposure']=hdr['exposure']*np.ones(len(tab),dtype=int)
    if (ny_range != None):
        tab=tab[ny_range[0]:ny_range[1]]
    return tab
    

def read_PT(fitsfile, agcam_coadd, nobad=False, ny_range=None):
    rsshdu = fits.open(fitsfile)

    hdr = rsshdu[0].header
    tab = Table(rsshdu['SLITMAP'].data)
    sci = np.full(len(tab), True)
    mask_bad = (tab['targettype']=='science') & (tab['fibstatus'] == 0) 
    if nobad:
        sci = (tab['targettype']=='science') & (tab['fibstatus'] == 0)
    rsshdu.close()

    # get ra/dec measured from coadd guiders?
    # agcam_coadd = DIR_agcam+mjd+'/'+'lvm.sci.coadd_s'+file_ID+'.fits'
    if os.path.isfile(agcam_coadd):
        agcam_hdu = fits.open(agcam_coadd)
        agcam_hdr = agcam_hdu[1].header
        w = WCS(agcam_hdr)
        cen = w.pixel_to_world(2500,1000)
        racen = cen.ra.deg  #agcam_hdr['RAMEAS']
        deccen = cen.dec.deg #agcam_hdr['DECMEAS']
        pa = agcam_hdr['PAMEAS'] - 180.
        agcam_hdu.close()
    else:
        racen = hdr['POSCIRA']
        deccen = hdr['POSCIDE']
        try:
            pa = hdr['POSCIPA']
        except:
            pa = 0.0
            hdr['POSCIPA'] = pa
    if (pa == None):
        pa = 0.0
        hdr['POSCIPA'] = pa
    if (('ra' in tab.colnames) and ('dec' in tab.colnames)):
        ra_fib = tab['ra']
        dec_fib = tab['dec']
    else:
        if 'fib_ra' in tab.colnames and 'fib_dec' in tab.colnames:
            ra_fib = tab['fib_ra'][sci]
            dec_fib = tab['fib_dec'][sci]
        else:
            ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], racen, deccen, pa)
#        ra_fib, dec_fib = make_radec(tab['xpmm'][sci], tab['ypmm'][sci], racen, deccen, pa)
    fiberid=tab['fiberid'][sci]
    exp_fib=[]
    for fibID in fiberid:
        exp_fib.append(str(hdr['exposure'])+'.'+str(fibID))
    tab=Table()
    tab['id']=np.array(exp_fib)
    tab['ra']=ra_fib.data
    tab['dec']=dec_fib.data
    tab['mask']=mask_bad
    tab['fiberid']=fiberid
    tab['exposure']=hdr['exposure']*np.ones(len(tab),dtype=int)
    if (ny_range != None):
        tab=tab[ny_range[0]:ny_range[1]]
#    print(len(sci))
#    tab['mask']=sci
#    print(sci)
    return tab


def rsp_print_header(filename,wavenorm=None):
    """
        Writes the main output file header.
    
        Parameters
        ----------
        filename : str
            Output filename.
        """
    if isinstance(filename, io.TextIOWrapper):
        f_outfile = filename
    else:
        f_outfile = open(filename, 'a')
        
    print(f'# (1) MIN_CHISQ', file=f_outfile)
    print(f'# (2) LW Age (Gyr)', file=f_outfile)
    print(f'# (3) LW Age error', file=f_outfile)
    print(f'# (4) LW metallicity', file=f_outfile)
    print(f'# (5) LW metallicity error', file=f_outfile)
    print(f'# (6) Av', file=f_outfile)
    print(f'# (7) AV error', file=f_outfile)
    print(f'# (8) redshift', file=f_outfile)
    print(f'# (9) redshift error', file=f_outfile)
    print(f'# (10) velocity dispersion sigma, in AA', file=f_outfile)
    print(f'# (11) velocity dispersion error', file=f_outfile)
    print(f'# (12) median_FLUX', file=f_outfile)
    print(f'# (13) redshift_ssp', file=f_outfile)
    print(f'# (14) med_flux', file=f_outfile)
    print(f'# (15) StdDev_residual', file=f_outfile)
    print(f'# (16) MW Age (Gyr)', file=f_outfile)
    print(f'# (17) MW Age error', file=f_outfile)
    print(f'# (18) MW metallicity', file=f_outfile)
    print(f'# (19) MW metallicity error', file=f_outfile)
    print(f'# (20) Systemic Velocity km/s ', file=f_outfile)
    print(f'# (21) Log10 Average Mass-to-Light Ratio', file=f_outfile)
    print(f'# (22) Log10 Mass', file=f_outfile)
    print(f'# SSP_SFH {self.filename} ', file=f_outfile)
    print(f'# SSP_KIN {self.filename_nl_fit} ', file=f_outfile)
    print(f'# WAVE_NORM {wavenorm} AA', file=f_outfile)
    
    if not isinstance(filename, io.TextIOWrapper):
        f_outfile.close()


#
# Tool to plot a single spectrum in a simple way
#
def plot_spec_simple(dir='output/',file='output.m_lvmSCFrame-00006109.fits.gz',\
              name='none',cmap=None,\
              x_min=3600,x_max=9400,y_min=-0.2,y_max=2,text='',\
              file_ssp = 'output/m_lvmSCFrame-00006109',no_st=False,no_model=False,log=False,\
             id_lines=None,output='junk.pdf',figsize=(22,12)):
    tab_SSP=read_rsp(file_ssp)
    
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    file=dir+'/'+file
    hdu=fits.open(file)
    data=hdu[0].data
    (ny,nx)=data.shape
#    data=np.mean(data,axis=1)
    i0 = int(nx*0.45)
    i1 = int(nx*0.55)

    data = data / np.median(data[0,i0:i1])
    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    fig = plt.figure(figsize=figsize) 
    gs = fig.add_gridspec(nrows=5, ncols=7, left=0.075, right=0.97, hspace=0.0, wspace=0.05, bottom=0.12, top=0.99)
    ax0 = fig.add_subplot(gs[:-1, 0:6])
    ax1 = fig.add_subplot(gs[-1:, 0:6], sharex=ax0)
    ax2 = fig.add_subplot(gs[:, 6:7])
    ax2.axis('off')
    ax0.set_xlim(x_min,x_max)
    ax1.set_xlim(x_min,x_max)
    ax0.set_ylim(y_min,y_max)
    colors=color_cm_now(np.linspace(0,1,10))
    if (no_st==False):
        ax0.plot(wave,data[0,:],color=colors[7],alpha=1.0,label=r'Observed (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,data[1,:],color=colors[0],alpha=1.0,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,res,color=colors[4],alpha=1.0,\
             label=r'Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,gas_model,color=colors[9],alpha=1.0,\
                 label=r'EL model')
    
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])
    ax0.axvspan(5600, 5850, alpha=1.0, color='grey',zorder=10)
    ax0.axvspan(7400, 7800, alpha=1.0, color='grey',zorder=10)
    
    
#    print(len(res),std_res,flux,flux/std_res)
    
#    ax0.plot(wave,data[1,:],color=colors[9],alpha=1.0)
    #ticks = np.arange(crval,crval+nx*cdelt, 500*cdelt)
    #print(ticks)
    #ax0.set_xticks(ticks)
    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels,loc=(0.5,0.7),frameon=False,handlelength=1.5,\
               ncol=2,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    ax1.fill_between(x, y0, y1,color=colors[4],alpha=1.0)
#    ax1.plot(wave,100*res,color=colors[4],alpha=1.0,label=r'Residual (O$_\lambda$-M$_\lambda$)')    
    ax1.plot(wave,100*res,color='white',alpha=1.0,label=r'Residual (O$_\lambda$-M$_\lambda$)')   
    if (id_lines != None):
        # #fb6f92
        # #fb6f92
        colors=['black','#ffc09f','#79addc','#fb6f92','#adf7b6','#ffee93']
        #colors=['darkred','firebrick','indianred']
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax1.plot([line['wl'],line['wl']],[-1+2.5*even,6],color=colors[even])
            ax1.text(line['wl'],-4+2.5*even,line['eline'],color=colors[even],\
                     fontsize='16',horizontalalignment='center')
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
    ax1.set_ylim(-6.5,6.5)
    ax1.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax1.set_ylabel(r'$<$R$_\lambda$/O$_{5500}$$>$[\%]', fontsize=23)
    ax0.set_ylabel(r'$<$O$_\lambda$/O$_{5500}$$>$', fontsize=23)
    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)

    if (no_model==False):
        cols=('MIN_CHISQ','Teff','e_Teff','Log_g','e_Log_g','Fe','e_Fe','alpha','e_alpha','z','e_z','disp','e_disp','Av','e_Av')
#        cols=('MIN_CHISQ','Av','AV_error','vel_sys','Teff','e_Teff','log_g','e_log_g','Fe_H','e_Fe_H','a_Fe','e_a_Fe')
        d=0.95/len(cols)
        for i,key in enumerate(cols):
            val=np.round(tab_SSP[key].value[0],3)
            skey=key.replace('_','\_')
            ax2.text(0.05,0.95-d*i,f'{skey} : {val}');   
    if (log==True):
        ax0.set_yscale('log')
#    fig.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")


#
# Simple routine to plot the output models of the DAP
# 
    
def plot_spectra_simple(dir='output/',file='output.m_lvmSCFrame-00006109.fits.gz',n_sp=0,\
              name='none',cmap=None,\
              x_min=3600,x_max=9400,y_min=-0.2,y_max=2,text='',\
              file_ssp = 'output/m_lvmSCFrame-00006109',no_st=False,no_model=False,log=False,\
             id_lines=None,output='junk.pdf'):
    tab_SSP_RSS=read_rsp(file_ssp)
    tab_SSP=tab_SSP_RSS[n_sp]
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    file=dir+'/'+file
    hdu=fits.open(file)
    data=hdu[0].data
    data=data[n_sp,:,:]
    (ny,nx)=data.shape
#    data=np.mean(data,axis=1)
    i0 = int(nx*0.45)
    i1 = int(nx*0.55)

    data = data / np.median(data[0,i0:i1])
    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    fig = plt.figure(figsize=(22,12)) 
    gs = fig.add_gridspec(nrows=5, ncols=7, left=0.075, right=0.97, hspace=0.0, wspace=0.05, bottom=0.12, top=0.99)
    ax0 = fig.add_subplot(gs[:-1, 0:6])
    ax1 = fig.add_subplot(gs[-1:, 0:6], sharex=ax0)
    ax2 = fig.add_subplot(gs[:, 6:7])
    ax2.axis('off')
    ax0.set_xlim(x_min,x_max)
    ax1.set_xlim(x_min,x_max)
    ax0.set_ylim(y_min,y_max)
    colors=color_cm_now(np.linspace(0,1,10))
    if (no_st==False):
        ax0.plot(wave,data[0,:],color=colors[7],alpha=1.0,label=r'Observed (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,data[1,:],color=colors[0],alpha=1.0,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,res,color=colors[4],alpha=1.0,\
             label=r'Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,gas_model,color=colors[9],alpha=1.0,\
                 label=r'EL model')
    
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])
    ax0.axvspan(5600, 5850, alpha=1.0, color='grey',zorder=10)
    ax0.axvspan(7400, 7800, alpha=1.0, color='grey',zorder=10)
    
    
#    print(len(res),std_res,flux,flux/std_res)
    
#    ax0.plot(wave,data[1,:],color=colors[9],alpha=1.0)
    #ticks = np.arange(crval,crval+nx*cdelt, 500*cdelt)
    #print(ticks)
    #ax0.set_xticks(ticks)
    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles, labels,loc=(0.5,0.7),frameon=False,handlelength=1.5,\
               ncol=2,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    ax1.fill_between(x, y0, y1,color=colors[4],alpha=1.0)
#    ax1.plot(wave,100*res,color=colors[4],alpha=1.0,label=r'Residual (O$_\lambda$-M$_\lambda$)')    
    ax1.plot(wave,100*res,color='white',alpha=1.0,label=r'Residual (O$_\lambda$-M$_\lambda$)')   
    if (id_lines != None):
        colors=['black','#ffc09f','#79addc','#fb6f92','#adf7b6','#ffee93']
        #colors=['darkred','firebrick','indianred']
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax1.plot([line['wl'],line['wl']],[-1+2.5*even,6],color=colors[even])
            ax1.text(line['wl'],-4+2.5*even,line['eline'],color=colors[even],\
                     fontsize='16',horizontalalignment='center')
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
    ax1.set_ylim(-6.5,6.5)
    ax1.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax1.set_ylabel(r'$<$R$_\lambda$/O$_{5500}$$>$[\%]', fontsize=23)
    ax0.set_ylabel(r'$<$O$_\lambda$/O$_{5500}$$>$', fontsize=23)
    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)

    if (no_model==False):
        cols=('MIN_CHISQ','Teff','e_Teff','Log_g','e_Log_g','Fe','e_Fe','alpha','e_alpha','z','e_z','disp','e_disp','Av','e_Av')
#        cols=('MIN_CHISQ','Av','AV_error','vel_sys','Teff','e_Teff','log_g','e_log_g','Fe_H','e_Fe_H','a_Fe','e_a_Fe')
        d=0.95/len(cols)
        for i,key in enumerate(cols):
            try:
                val=np.round(tab_SSP[key].value,3)
            except:
                val=np.round(tab_SSP[key],3)
            skey=key.replace('_','\_')
            ax2.text(0.05,0.95-d*i,fr'{skey} : {val}');   
    if (log==True):
        ax0.set_yscale('log')
#    fig.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")

def read_rsp(file_ssp = 'output/m_lvmSCFrame-00006109'):
    col_SSP=header_columns_space(file_ssp,2)
    tab_SSP=ascii.read(file_ssp, delimiter=',', guess=True, comment='\s*#', names=col_SSP,\
                       fill_values=[('BAD', np.nan)])
    return tab_SSP

#tab_SSP=read_rsp()
#list_columns(tab_SSP.columns)

def header_columns_space(filename,column):
    COMMENT_CHAR = '#'
    col_NAME = []
    with open(filename, 'r') as td:
        for line in td:
            if ((line[0] == COMMENT_CHAR) and (line.find('HEADER')==-1) and (line.find(')')>-1)):
                line.replace("\)","")
                line.replace("\(","")
                line.replace("\/","")
                info = re.split(' +', line.rstrip('\n'))
                info_now=info[column]
                for i in np.arange(column+1,len(info)):
                    if ((info[i]) and not (")" in info[i])):
                        info_now=info_now+'_'+info[i]
                col_NAME.append(info_now)
    counts = {k:v for k,v in Counter(col_NAME).items() if v > 1}
    col_NAME_NEW = col_NAME[:]
    for i in reversed(range(len(col_NAME))):
        item = col_NAME[i]
        if item in counts and counts[item]:
            if (counts[item]>1):
                col_NAME_NEW[i] += str(counts[item]-1)
            counts[item]-=1                
    return col_NAME_NEW


def read_coeffs_RSP(coeffs_file='output_dap/dap-4-00006109.coeffs.txt'):
    COMMENT_CHAR = '#'
    tab=Table()
    nfb=0
    vals_max=-1
    a_vals=[]
    with open(coeffs_file, 'r') as td:
        for line in td:
            if line[0] == COMMENT_CHAR:
                cols=line.split()
                cols=cols[1:]
                cols[0]='rsp'
            else:
                vals=line.split()
                a_vals.append(vals)
    a_vals=np.array(a_vals, dtype=float)
    n_models=int(np.max(a_vals[:,0]))
    n_fib=int(a_vals.shape[0]/n_models)
    id_fib=[]
    i_fib=0
    for i in a_vals[:,0]:
        if (i==0):
            i_fib=i_fib+1
        I_fib=i_fib-1
        id_fib.append(I_fib)
    tab=Table()
    tab['id_fib']=id_fib
    for i,col_now in enumerate(cols):
        val_now=a_vals[:,i]
        if (col_now=='rsp'):
            val_now=val_now.astype(int)
        tab.add_column(val_now,name=col_now)
    return(tab)

def read_elines_RSP(elines_file='output_dap/dap-4-00006109.elines.txt'):
    COMMENT_CHAR = '#'
    tab=Table()
    nfb=0
    vals_max=-1
    a_vals=[]
    id_fib=0
    with open(elines_file, 'r') as td:
        for line in td:
            if line[0] == COMMENT_CHAR:
                cols=line.split()
                id_fib=int(cols[1])
            else:
                vals=line.split()
                vals=np.array(vals)
                if ((vals[0]=='eline') or (vals[0]=='poly1d')):
                    line_new=str(id_fib)+' '+line
                    vals=line_new.split()
                    a_vals.append(vals)
    a_vals=np.array(a_vals)
    a_vals=a_vals[:,0:10]
    tab=Table()
    tab.add_column(a_vals[:,0].astype(int),name='id_fib')
    tab.add_column(a_vals[:,1],name='model')
    tab.add_column(a_vals[:,2].astype(float),name='wl')
    tab.add_column(a_vals[:,4].astype(float),name='flux')
    tab.add_column(a_vals[:,5].astype(float),name='e_flux')
    tab.add_column(a_vals[:,6].astype(float),name='disp')
    tab.add_column(a_vals[:,7].astype(float),name='e_disp')
    tab.add_column(a_vals[:,8].astype(float),name='vel')
    tab.add_column(a_vals[:,9].astype(float),name='e_vel')    
    return tab



#
# routine to plot the spectra
#
def plot_spec(dir='output/',file='output.m_lvmSCFrame-00006109.fits.gz',\
              name='none',cmap=None,\
              x_min=3600,x_max=9600,y_min=-0.2,y_max=2,text='',\
              file_ssp = 'output/m_lvmSCFrame-00006109',no_st=False,no_model=False,log=False,\
              id_lines=None,output='junk.pdf',c_map='carlos',do_legend=True, insets=None,y0_d=0.5,y1_d=2.5,plot_el=False,tab_el=None,
              colors_in=['black','red','deepskyblue','blue','gold','forestgreen'],plot_res=False,show_scale=True,n_ord=11,figsize=(19,6)):

    tab_SSP=read_rsp(file_ssp)
    
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    file=dir+'/'+file
    hdu=fits.open(file)
    data=hdu[0].data
    (ny,nx)=data.shape
    
    
#    data=np.mean(data,axis=1)
    i0 = int(nx*0.45)
    i1 = int(nx*0.55)
    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    
    data = data / np.median(data[0,i0:i1])
    if (ny==7):
        ssp_model=data[6,:]
    else:
        gas = data[0,:] - data[5,:]
        smooth = data[3,:] - data[4,:] - gas        
        ssp_model = data[1,:]-smooth
        if (n_ord>2):
            try:
                f_rat = ssp_model/data[0,:]
                c_rat0 = np.polyfit(wave, f_rat, n_ord)
                p_rat0 = np.poly1d(c_rat0)(wave)
                ssp_model = ssp_model*(1+1/p_rat0)*0.5
                data[0,:]=(data[0,:]-smooth)*(1+1/p_rat0)*0.5
            except:
                show_scale=False
        if (show_scale==True):
            ssp_model = data[1,:]


    fig = plt.figure(figsize=figsize) 
    gs = fig.add_gridspec(nrows=5, ncols=6,  left=0.075, right=0.97, \
                          hspace=0.0, wspace=0.05, bottom=0.15, top=0.9)
    ax0 = fig.add_subplot(gs[:-1, :])
    ax1 = fig.add_subplot(gs[-1:, :], sharex=ax0)   
    ax0.set_xlim(x_min,x_max)
    ax1.set_xlim(x_min,x_max)
    ax0.set_ylim(y_min,y_max)        

        
 #   colors=['black','maroon','steelblue','darkorange','olive','grey']
#    colors=['black','palegreen','steelblue','darkorange','olive','grey']
#    colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
    colors=colors_in
    if (no_st==False):
        ax0.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3.0,label=r'Obs. Spectrum (O$_\lambda$)')
        ax1.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=2.0)#,label=r'Observed (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'St. Model (M$_\lambda$)',linestyle="solid")
            ax1.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,linestyle="solid")#,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]+smooth
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,res,color=colors[2],alpha=1.0,\
             label=r'St. Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                 label=r'Em. Lines model (E$_\lambda$)',linestyle="solid")
    if (plot_res==True):
        ax0.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                 label=r'Final residual (R$_\lambda$)',linestyle="solid")
    
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])
    ax1.axvspan(5650, 5850, alpha=0.3, color='grey',zorder=10)
    ax1.axvspan(7500, 7700, alpha=0.3, color='grey',zorder=10)
    

    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    #ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    if (do_legend):
        ax0.legend(handles, labels,loc=(0.03,1.01),frameon=True,handlelength=1.5,\
                   ncol=5,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    if (id_lines != None):
        #colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
        #        colors=['black','palegreen','steelblue','darkorange','olive','grey']
        #colors=['darkred','firebrick','indianred']
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
        
    if (insets != None):
        for inst in insets:
            #x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
            axins = ax0.inset_axes(
                [inst[0], inst[1], inst[2], inst[3]],
                xlim=(inst[4], inst[5]), ylim=(inst[6], inst[7]),yticklabels=[])#, xticklabels=[], yticklabels=[])
            if (no_st==False):
                axins.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3,label=r'Observed (O$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'Model (M$_\lambda$)',linestyle="solid")
            axins.plot(wave,res,color=colors[2],alpha=1.0,linewidth=3,\
                       label=r'Residual (O$_\lambda$-M$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                           label=r'E. lines model',linestyle="solid")
            if (plot_res==True):
                axins.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                           label=r'Em. lines model (E$_\lambda$)',linestyle="solid")
                
            if (plot_el == True):
                tab_el_now=tab_el[(tab_el['wl']>inst[4]) & (tab_el['wl']<inst[5])]
                for vals in tab_el_now:
                    axins.plot([vals['wl'],vals['wl']],\
                             [inst[7]-0.05*abs(inst[7]-inst[6]),\
                              inst[7]-0.15*abs(inst[7]-inst[6])],\
                             color='darkorange',linewidth=2)
                axins.text(inst[4]+0.65*np.abs(inst[5]-inst[4]),\
                           inst[7]-0.15*abs(inst[7]-inst[6]),f'{inst[8]}')#,\
                           #color='darkorange')
            ax0.indicate_inset_zoom(axins, edgecolor="black")

        
    ax0.set_ylabel(r'$<$O$_\lambda$/O$_{5500}$$>$', fontsize=23)
    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)
    ax1.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax1.set_ylim(y0_d,y1_d)
    if (plot_el == True):
        for vals in tab_el:
            ax0.plot([vals['wl'],vals['wl']],\
                     [y_min+0.01*abs(y_max-y_min),y_min+0.03*abs(y_max-y_min)],\
                     color='darkorange')
    if (log==True):
        ax0.set_yscale('log')
    plt.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")



#
# routine to plot the spectra
#
def plot_spec_no_sc(dir='output/',file='output.m_lvmSCFrame-00006109.fits.gz',\
              name='none',cmap=None,\
              x_min=3600,x_max=9600,y_min=-0.2,y_max=2,text='',\
              file_ssp = 'output/m_lvmSCFrame-00006109',no_st=False,no_model=False,log=False,\
              id_lines=None,output='junk.pdf',c_map='carlos',do_legend=True, insets=None,y0_d=0.5,y1_d=2.5,plot_el=False,tab_el=None,
              colors_in=['black','red','deepskyblue','blue','gold','forestgreen'],plot_res=False,show_scale=True,n_ord=11):

    tab_SSP=read_rsp(file_ssp)
    
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    file=dir+'/'+file
    hdu=fits.open(file)
    data=hdu[0].data
    (ny,nx)=data.shape
    
    
#    data=np.mean(data,axis=1)
    i0 = int(nx*0.45)
    i1 = int(nx*0.55)
    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    
    data = data / np.median(data[0,i0:i1])
    if (ny==7):
        ssp_model=data[6,:]
    else:
        gas = data[0,:] - data[5,:]
        smooth = data[3,:] - data[4,:] - gas        
        ssp_model = data[1,:]-smooth
        if (n_ord>2):
            try:
                f_rat = ssp_model/data[0,:]
                c_rat0 = np.polyfit(wave, f_rat, n_ord)
                p_rat0 = np.poly1d(c_rat0)(wave)
                ssp_model = ssp_model*(1+1/p_rat0)*0.5
            except:
                show_scale=False
        if (show_scale==True):
            ssp_model = data[1,:]


    fig = plt.figure(figsize=(19,6)) 
    gs = fig.add_gridspec(nrows=5, ncols=6,  left=0.075, right=0.97, \
                          hspace=0.0, wspace=0.05, bottom=0.15, top=0.9)
    ax0 = fig.add_subplot(gs[:-1, :])
    ax1 = fig.add_subplot(gs[-1:, :], sharex=ax0)   
    ax0.set_xlim(x_min,x_max)
    ax1.set_xlim(x_min,x_max)
    ax0.set_ylim(y_min,y_max)        

        
 #   colors=['black','maroon','steelblue','darkorange','olive','grey']
#    colors=['black','palegreen','steelblue','darkorange','olive','grey']
#    colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
    colors=colors_in
    if (no_st==False):
        ax0.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3.0,label=r'Obs. Spectrum (O$_\lambda$)')
        ax1.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=2.0)#,label=r'Observed (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'St. Model (M$_\lambda$)',linestyle="solid")
            ax1.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,linestyle="solid")#,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,res,color=colors[2],alpha=1.0,\
             label=r'St. Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                 label=r'Em. Lines model (E$_\lambda$)',linestyle="solid")
    if (plot_res==True):
        ax0.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                 label=r'Final residual (R$_\lambda$)',linestyle="solid")
    
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])
    ax1.axvspan(5650, 5850, alpha=0.3, color='grey',zorder=10)
    ax1.axvspan(7500, 7700, alpha=0.3, color='grey',zorder=10)
    

    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    #ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    if (do_legend):
        ax0.legend(handles, labels,loc=(0.03,1.01),frameon=True,handlelength=1.5,\
                   ncol=5,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    if (id_lines != None):
        #colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
        #        colors=['black','palegreen','steelblue','darkorange','olive','grey']
        #colors=['darkred','firebrick','indianred']
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
        
    if (insets != None):
        for inst in insets:
            #x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
            axins = ax0.inset_axes(
                [inst[0], inst[1], inst[2], inst[3]],
                xlim=(inst[4], inst[5]), ylim=(inst[6], inst[7]),yticklabels=[])#, xticklabels=[], yticklabels=[])
            if (no_st==False):
                axins.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3,label=r'Observed (O$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'Model (M$_\lambda$)',linestyle="solid")
            axins.plot(wave,res,color=colors[2],alpha=1.0,linewidth=3,\
                       label=r'Residual (O$_\lambda$-M$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                           label=r'E. lines model',linestyle="solid")
            if (plot_res==True):
                axins.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                           label=r'Em. lines model (E$_\lambda$)',linestyle="solid")
                
            if (plot_el == True):
                tab_el_now=tab_el[(tab_el['wl']>inst[4]) & (tab_el['wl']<inst[5])]
                for vals in tab_el_now:
                    axins.plot([vals['wl'],vals['wl']],\
                             [inst[7]-0.05*abs(inst[7]-inst[6]),\
                              inst[7]-0.15*abs(inst[7]-inst[6])],\
                             color='darkorange',linewidth=2)
                axins.text(inst[4]+0.65*np.abs(inst[5]-inst[4]),\
                           inst[7]-0.15*abs(inst[7]-inst[6]),f'{inst[8]}')#,\
                           #color='darkorange')
            ax0.indicate_inset_zoom(axins, edgecolor="black")

        
    ax0.set_ylabel(r'$<$O$_\lambda$/O$_{5500}$$>$', fontsize=23)
    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)
    ax1.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax1.set_ylim(y0_d,y1_d)
    if (plot_el == True):
        for vals in tab_el:
            ax0.plot([vals['wl'],vals['wl']],\
                     [y_min+0.01*abs(y_max-y_min),y_min+0.03*abs(y_max-y_min)],\
                     color='darkorange')
    if (log==True):
        ax0.set_yscale('log')
    plt.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")

def plot_spectra(dir='output/',file='output.m_lvmSCFrame-00006109.fits.gz',n_sp=0,\
                 name='none',cmap=None,\
                 x_min=3600,x_max=9400,y_min=-0.2,y_max=2,text='',\
                 file_ssp = 'output/m_lvmSCFrame-00006109',no_st=False,no_model=False,log=False,\
                 id_lines=None,output='junk.pdf',\
                 c_map='carlos',do_legend=True, insets=None,y0_d=0.5,y1_d=2.5,plot_el=False,tab_el=None,\
                 colors_in=['black','red','deepskyblue','blue','gold','forestgreen'],plot_res=False,show_scale=True,\
                 n_ord=11):
                 #colors_in=['black','#ffc09f','#79addc','#fb6f92','#adf7b6','#ffee93']):
#    tab_SSP_RSS=read_rsp(file_ssp)
#    tab_SSP=tab_SSP_RSS[n_sp]
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    file=dir+'/'+file
    hdu=fits.open(file)
    data=hdu[0].data
#    print(f'SHAPE output ={data.shape}')
    data=data[:,n_sp,:]
    (ny,nx)=data.shape
    
#    print(f'SHAPE output  ={data.shape} {n_sp}')
#    data=np.mean(data,axis=1)
    i0 = int(nx*0.45)
    i1 = int(nx*0.55)

    data = data / np.median(data[0,i0:i1])

    if (ny==7):
        ssp_model=data[6,:]
    else:
        gas = data[0,:] - data[5,:]
        smooth = data[3,:] - data[4,:] - gas        
        ssp_model = data[1,:]-smooth
        if (n_ord>2):
            try:
                f_rat = ssp_model/data[0,:]
                c_rat0 = np.polyfit(wave, f_rat, n_ord)
                p_rat0 = np.poly1d(c_rat0)(wave)
                ssp_model = ssp_model*(1+1/p_rat0)*0.5
            except:
                show_scale=False
        if (show_scale==True):
            ssp_model = data[1,:]
    


    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    fig = plt.figure(figsize=(19,6)) 
    gs = fig.add_gridspec(nrows=5, ncols=6,  left=0.075, right=0.97, \
                          hspace=0.0, wspace=0.05, bottom=0.15, top=0.9)
    ax0 = fig.add_subplot(gs[:-1, :])
    ax1 = fig.add_subplot(gs[-1:, :], sharex=ax0)   
    ax0.set_xlim(x_min,x_max)
    ax1.set_xlim(x_min,x_max)
    ax0.set_ylim(y_min,y_max)        


        
    colors=colors_in
    #['black','maroon','steelblue','darkorange','olive','grey']
    if (no_st==False):
        ax0.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3,label=r'Obs. Spectrum (O$_\lambda$)')
        ax1.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3)#,label=r'Observed (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'St. Model (M$_\lambda$)')
            ax1.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5)#,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,res,color=colors[2],alpha=1.0,\
             label=r'St. Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                 label=r'Em. Lines model (E$_\lambda$)')
    if (plot_res==True):
        ax0.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                 label=r'Final residual (R$_\lambda$)',linestyle="solid")


        
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])
    ax1.axvspan(5650, 5850, alpha=0.3, color='grey',zorder=10)
    ax1.axvspan(7500, 7700, alpha=0.3, color='grey',zorder=10)
    

    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    #ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    if (do_legend):
        ax0.legend(handles, labels,loc=(0.03,1.01),frameon=True,handlelength=1.5,\
                   ncol=5,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    if (id_lines != None):
        colors=colors_in
        #['darkred','firebrick','indianred']
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
        
    if (insets != None):
        for inst in insets:
            #x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
            axins = ax0.inset_axes(
                [inst[0], inst[1], inst[2], inst[3]],
                xlim=(inst[4], inst[5]), ylim=(inst[6], inst[7]),yticklabels=[])#, xticklabels=[], yticklabels=[])
            if (no_st==False):
                axins.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3,label=r'Observed (O$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'Model (M$_\lambda$)')
            axins.plot(wave,res,color=colors[2],alpha=1.0,\
                       label=r'Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
            if (no_model==False):
                axins.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                           label=r'E. lines model')
            if (plot_res==True):
                axins.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                           label=r'Em. Lines residual (R$_\lambda$)',linestyle="solid")
            if (plot_el == True):
                tab_el_now=tab_el[(tab_el['wl']>inst[4]) & (tab_el['wl']<inst[5])]
                for vals in tab_el_now:
                    axins.plot([vals['wl'],vals['wl']],\
                             [inst[7]-0.05*abs(inst[7]-inst[6]),\
                              inst[7]-0.15*abs(inst[7]-inst[6])],\
                             color='darkorange',linewidth=2)
                axins.text(inst[4]+0.65*np.abs(inst[5]-inst[4]),\
                           inst[7]-0.15*abs(inst[7]-inst[6]),f'{inst[8]}',\
                           color='darkorange')
            ax0.indicate_inset_zoom(axins, edgecolor="black")

        
    ax0.set_ylabel(r'$<$O$_\lambda$/O$_{5500}$$>$', fontsize=23)
    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)
    ax1.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax1.set_ylim(y0_d,y1_d)
    if (plot_el == True):
        for vals in tab_el:
            ax0.plot([vals['wl'],vals['wl']],\
                     [y_min+0.01*abs(y_max-y_min),y_min+0.03*abs(y_max-y_min)],\
                     color='darkorange')
    if (log==True):
        ax0.set_yscale('log')
    plt.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")


#    
# Routine used plot the integrated fitted spectra for the presentation article
#
#
# routine to plot the spectra
#
def plot_spec_art_working(dir='output/',file='output.m_lvmSCFrame-00006109.fits.gz',\
              name='none',cmap=None,\
              x_min=3600,x_max=9600,y_min=-0.2,y_max=2,text='',\
              file_ssp = 'output/m_lvmSCFrame-00006109',no_st=False,no_model=False,log=False,\
              id_lines=None,output='junk.pdf',c_map='carlos',do_legend=True, insets=None,y0_d=0.5,y1_d=2.5,plot_el=False,tab_el=None,
              colors_in=['black','red','deepskyblue','blue','gold','forestgreen'],plot_res=False,\
                  show_scale=True,n_ord=11,gamma=0.3, props=dict(facecolor='lightgrey', alpha=0.5, edgecolor='darkgrey')):

    tab_SSP=read_rsp(file_ssp)
    
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    file=dir+'/'+file
    hdu=fits.open(file)
    data=hdu[0].data
    (ny,nx)=data.shape
    
    
#    data=np.mean(data,axis=1)
    i0 = int(nx*0.45)
    i1 = int(nx*0.55)
    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    
    data = data / np.median(data[0,i0:i1])
    if (ny==7):
        ssp_model=data[6,:]
    else:
        gas = data[0,:] - data[5,:]
        smooth = data[3,:] - data[4,:] - gas        
        ssp_model = data[1,:]-smooth
        if (n_ord>2):
            try:
                f_rat = ssp_model/data[0,:]
                c_rat0 = np.polyfit(wave, f_rat, n_ord)
                p_rat0 = np.poly1d(c_rat0)(wave)
                ssp_model = ssp_model*(1+1/p_rat0)*0.5
                data[0,:]=(data[0,:]-smooth)*(1+1/p_rat0)*0.5
            except:
                show_scale=False
        if (show_scale==True):
            ssp_model = data[1,:]


    fig = plt.figure(figsize=(19,11)) 
    gs = fig.add_gridspec(nrows=10, ncols=6,  left=0.075, right=0.97, \
                          hspace=0.0, wspace=0.05, bottom=0.15, top=0.9)

    ax2 = fig.add_subplot(gs[:, :])
    ax0 = fig.add_subplot(gs[0:3, :])#, sharex=ax2)   
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.setp(ax2.spines.values(), color=None)
    
    
    ax0.set_xlim(x_min,x_max)

    if (y_min>=0):
        sign=1
    else:
        sign=-1
    ax0.set_ylim(sign*np.abs(y_min**gamma),y_max**gamma)        

        
 #   colors=['black','maroon','steelblue','darkorange','olive','grey']
#    colors=['black','palegreen','steelblue','darkorange','olive','grey']
#    colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
    colors=colors_in
    if (no_st==False):
        ax0.plot(wave,data[0,:]**gamma,color=colors[0],alpha=1.0,linewidth=3.0,label=r'Obs. Spectrum (O$_\lambda$)')
        #ax1.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=2.0)#,label=r'Observed (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,ssp_model**gamma,color=colors[1],alpha=1.0,linewidth=1.5,label=r'St. Model (M$_\lambda$)',linestyle="solid")
            #ax1.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,linestyle="solid")#,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]+smooth
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,(data[0,:]+smooth)**gamma-data[1,:]**gamma,color=colors[2],alpha=1.0,\
             label=r'St. Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,data[2,:]**gamma-data[1,:]**gamma,color=colors[3],alpha=1.0,linewidth=1.5,\
                 label=r'Em. Lines model (E$_\lambda$)',linestyle="solid")
    if (plot_res==True):
#        ax0.plot(wave,data[4,:]**gamma,color=colors[4],alpha=1.0,linewidth=1.5,\
        ax0.plot(wave,(data[0,:]+smooth)**gamma-data[1,:]**gamma-(data[2,:]**gamma-data[1,:]**gamma),color=colors[4],alpha=1.0,linewidth=1.5,\
                 label=r'Final residual (R$_\lambda$)',linestyle="solid")
    
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])
    #ax1.axvspan(5650, 5850, alpha=0.3, color='grey',zorder=10)
    #ax1.axvspan(7500, 7700, alpha=0.3, color='grey',zorder=10)
    

    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    #ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    if (do_legend):
        ax0.legend(handles, labels,loc=(0.03,1.01),frameon=True,handlelength=1.5,\
                   ncol=5,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    if (id_lines != None):
        #colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
        #        colors=['black','palegreen','steelblue','darkorange','olive','grey']
        #colors=['darkred','firebrick','indianred']
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
        
    if (insets != None):
        for Iinst,inst in enumerate(insets):
            #x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
            axins = ax0.inset_axes(
                [inst[0], inst[1], inst[2], inst[3]],
                xlim=(inst[4], inst[5]), ylim=(inst[6], inst[7]),yticklabels=[])#, xticklabels=[], yticklabels=[])
            if ((Iinst==0)|(Iinst==9)):
                axins.set_ylabel(r'Flux$_\lambda$', fontsize=23)

            if (no_st==False):
                axins.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3,label=r'Observed (O$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'Model (M$_\lambda$)',linestyle="solid")
            axins.plot(wave,res,color=colors[2],alpha=1.0,linewidth=3,\
                       label=r'Residual (O$_\lambda$-M$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                           label=r'E. lines model',linestyle="solid")
            if (plot_res==True):
                axins.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                           label=r'Em. lines model (E$_\lambda$)',linestyle="solid")
                
            if (plot_el == True):
                tab_el_now=tab_el[(tab_el['wl']>inst[4]) & (tab_el['wl']<inst[5])]
                for vals in tab_el_now:
                    axins.plot([vals['wl'],vals['wl']],\
                             [inst[7]-0.05*abs(inst[7]-inst[6]),\
                              inst[7]-0.15*abs(inst[7]-inst[6])],\
                             color='darkorange',linewidth=2)
                axins.text(inst[4]+0.1*np.abs(inst[5]-inst[4]),\
                           inst[7]-0.15*abs(inst[7]-inst[6]),fr'\bf {inst[8]}',bbox=props)#,\
                           #color='darkorange')
            #ax0.indicate_inset_zoom(axins, edgecolor="black")

        
    #ax0.set_ylabel(r'$<$O$_\lambda$/O$_{5500}$$>$', fontsize=23)
    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)
    ax2.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax2.set_ylim(y0_d,y1_d)
    if (plot_el == True):
        for vals in tab_el:
            ax0.plot([vals['wl'],vals['wl']],\
                     [y_max**gamma-0.02*abs(y_max**gamma-y_min**gamma),y_max**gamma-0.10*abs(y_max**gamma-y_min**gamma)],\
                     color='darkorange')
#            ax0.plot([vals['wl'],vals['wl']],\
#                     [y_max-0.05*abs(y_max-y_min),y_max-0.15*abs(y_max-y_min)],\
#                     color='darkorange')
    if (log==True):
        ax0.set_yscale('log')
    ax0.axvspan(5600, 5850, alpha=0.7, color='grey',zorder=10)
    ax0.axvspan(7400, 7800, alpha=0.7, color='grey',zorder=10)        
    sgamma=str(gamma)
#    ax0.set_ylabel(r'Flux$^{0.5}_\lambda$ [10$^{-16}$ erg/cm$^2$/s]', fontsize=23)  
    ax0.set_ylabel(r'Flux$^{0.5}_\lambda$', fontsize=23)  
    
    plt.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")

def plot_spec_art(dir='output/',file='output.m_lvmSCFrame-00006109.fits.gz',\
              name='none',cmap=None,\
              x_min=3600,x_max=9600,y_min=-0.2,y_max=2,text='',\
              file_ssp = 'output/m_lvmSCFrame-00006109',no_st=False,no_model=False,log=False,\
              id_lines=None,output='junk.pdf',c_map='carlos',do_legend=True, insets=None,y0_d=0.5,y1_d=2.5,plot_el=False,tab_el=None,
              colors_in=['black','red','deepskyblue','blue','gold','forestgreen'],plot_res=False,\
                  show_scale=True,n_ord=11,gamma=0.3, props=dict(facecolor='lightgrey', alpha=0.5, edgecolor='darkgrey'),\
                  alpha=1.0):

#    tab_SSP=read_rsp(file_ssp)
    
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    file=dir+'/'+file
    hdu=fits.open(file)
    
#    hdu[0].data=gamma_scale(hdu[0].data,0.5)
#    gamma=1.0
    
    data=hdu[0].data
    (ny,nx)=data.shape
    
    
#    data=np.mean(data,axis=1)
#    i0 = int(nx*0.45)
#    i1 = int(nx*0.55)
    i0 = int(nx*0.15)
    i1 = int(nx*0.25)
    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    
    med_sc = np.abs(np.nanmedian(data[0,i0:i1]))
    std_sc = np.nanstd(data[0,i0:i1])
    SN_sc = med_sc / std_sc
    print(f'# S/N: {SN_sc} {med_sc} {std_sc}')
    #     data = data / np.abs(np.nanmedian(data[0,i0:i1]))
    data = data / (med_sc+0.05*std_sc)
    if (ny==7):
        ssp_model=data[6,:]
    else:
        gas = data[0,:] - data[5,:]
        smooth = data[3,:] - data[4,:] - gas        
        ssp_model = data[1,:]-smooth

        mean_val_ssp = np.nanmean(ssp_model)
        print(f'# mean_val_ssp : {mean_val_ssp}')
        if (np.mean(ssp_model)<1e-5):
            ssp_model = data[1,:]
        
        if ((n_ord>2) and (SN_sc>3)):
            base = np.abs(data[0,:])
            f_rat = ssp_model/base
            #data[0,:]
            c_rat0 = np.polyfit(wave, f_rat, n_ord)
            p_rat0 = np.poly1d(c_rat0)(wave)
            ssp_model = ssp_model*(1+1/p_rat0)*0.5
            data[0,:]=(data[0,:]-smooth)*(1+1/p_rat0)*0.5
#            print(data[1,:])
            #except:
            #    show_scale=False  
            #    print('# Not scaling')
        if (show_scale==True):
            ssp_model = data[1,:]


    fig = plt.figure(figsize=(19,11)) 
    gs = fig.add_gridspec(nrows=10, ncols=6,  left=0.075, right=0.97, \
                          hspace=0.0, wspace=0.05, bottom=0.15, top=0.9)

    ax2 = fig.add_subplot(gs[:, :])
    ax0 = fig.add_subplot(gs[0:3, :])#, sharex=ax2)   
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.setp(ax2.spines.values(), color=None)
    
    
    ax0.set_xlim(x_min,x_max)

    if (y_min>=0):
        sign=1
    else:
        sign=-1
    ax0.set_ylim(sign*np.abs(y_min**gamma),y_max**gamma)        

        
 #   colors=['black','maroon','steelblue','darkorange','olive','grey']
#    colors=['black','palegreen','steelblue','darkorange','olive','grey']
#    colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
    colors=colors_in
            #ax1.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,linestyle="solid")#,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]+smooth
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,gamma_scale((data[0,:]+smooth),gamma)-gamma_scale(data[1,:],gamma),color=colors[2],alpha=alpha,\
             label=r'St. Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,gamma_scale(data[2,:],gamma)-gamma_scale(data[1,:],gamma),color=colors[3],alpha=alpha,linewidth=1.5,\
                 label=r'Em. Lines model (E$_\lambda$)',linestyle="solid")
    if (plot_res==True):
#        ax0.plot(wave,data[4,:]**gamma,color=colors[4],alpha=1.0,linewidth=1.5,\
        ax0.plot(wave,gamma_scale((data[0,:]+smooth),gamma)-gamma_scale(data[1,:],gamma)-(gamma_scale(data[2,:],gamma)-gamma_scale(data[1,:],gamma)),color=colors[4],alpha=alpha,linewidth=1.5,\
                 label=r'Final residual (R$_\lambda$)',linestyle="solid")
    
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])

    if (no_st==False):
        ax0.plot(wave,gamma_scale(data[0,:],gamma),\
                 color=colors[0],alpha=1.0,linewidth=3.0,\
                 label=r'Obs. Spectrum (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,gamma_scale(ssp_model,gamma),color=colors[1],\
                     alpha=alpha,linewidth=1.5,\
                     label=r'St. Model (M$_\lambda$)',linestyle="solid")
    

    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    #ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    if (do_legend):
        ax0.legend(handles, labels,loc=(0.03,1.01),frameon=True,handlelength=1.5,\
                   ncol=5,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    if (id_lines != None):
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
        
    if (insets != None):
        for Iinst,inst in enumerate(insets):
            #x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
            axins = ax0.inset_axes(
                [inst[0], inst[1], inst[2], inst[3]],
                xlim=(inst[4], inst[5]), ylim=(inst[6], inst[7]),yticklabels=[])#, xticklabels=[], yticklabels=[])
            if ((Iinst==0)|(Iinst==9)):
                axins.set_ylabel(r'Flux$_\lambda$', fontsize=23)

            if (no_st==False):
                axins.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3,label=r'Observed (O$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'Model (M$_\lambda$)',linestyle="solid")
            axins.plot(wave,res,color=colors[2],alpha=1.0,linewidth=3,\
                       label=r'Residual (O$_\lambda$-M$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                           label=r'E. lines model',linestyle="solid")
            if (plot_res==True):
                axins.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                           label=r'Em. lines model (E$_\lambda$)',linestyle="solid")
                
            if (plot_el == True):
                tab_el_now=tab_el[(tab_el['wl']>inst[4]) & (tab_el['wl']<inst[5])]
                for vals in tab_el_now:
                    axins.plot([vals['wl'],vals['wl']],\
                             [inst[7]-0.05*abs(inst[7]-inst[6]),\
                              inst[7]-0.15*abs(inst[7]-inst[6])],\
                             color='darkorange',linewidth=2)
                axins.text(inst[4]+0.1*np.abs(inst[5]-inst[4]),\
                           inst[7]-0.15*abs(inst[7]-inst[6]),fr'\bf {inst[8]}',bbox=props)#,\



        

    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)
    ax2.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax2.set_ylim(y0_d,y1_d)
    if (plot_el == True):
        for vals in tab_el:
            ax0.plot([vals['wl'],vals['wl']],\
                     [y_max**gamma-0.02*abs(y_max**gamma-y_min**gamma),y_max**gamma-0.10*abs(y_max**gamma-y_min**gamma)],\
                     color='darkorange')
    if (log==True):
        ax0.set_yscale('log')
    ax0.axvspan(5600, 5850, alpha=0.7, color='grey',zorder=10)
    ax0.axvspan(7400, 7800, alpha=0.7, color='grey',zorder=10)        
    sgamma=str(gamma)

    ax0.set_ylabel(r'Norm. Flux$^{0.5}_\lambda$', fontsize=23)  
    
    plt.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")

def plot_spec_art_hdu(dir='output/', hdu = None,\
              name='none',cmap=None,\
              x_min=3600,x_max=9600,y_min=-0.2,y_max=2,text='',\
              no_st=False,no_model=False,log=False,\
              id_lines=None,output='junk.pdf',c_map='carlos',do_legend=True, insets=None,y0_d=0.5,y1_d=2.5,plot_el=False,tab_el=None,
              colors_in=['black','red','deepskyblue','blue','gold','forestgreen'],plot_res=False,\
                  show_scale=True,n_ord=11,gamma=0.3, props=dict(facecolor='lightgrey', alpha=0.5, edgecolor='darkgrey'),\
                  alpha=1.0):

    
    if (cmap==None):
        color_cm_now = vel_map_CLC()
        cmap=color_cm_now
    if (hdu==None):
        return 0
    
    
    data=hdu[0].data
    (ny,nx)=data.shape
    
    
#    data=np.mean(data,axis=1)
#    i0 = int(nx*0.45)
#    i1 = int(nx*0.55)
    i0 = int(nx*0.15)
    i1 = int(nx*0.25)
    hdr=hdu[0].header
    crval = hdr['crval1']
    cdelt = hdr['cdelt1']
    crpix = hdr['crpix1']
    #print(data.shape)
    (ny,nx) = data.shape
    wave = crval+cdelt*(np.arange(0,nx)-(crpix-1))
    
    med_sc = np.abs(np.nanmedian(data[0,i0:i1]))
    std_sc = np.nanstd(data[0,i0:i1])
    SN_sc = med_sc / std_sc
    print(f'# S/N: {SN_sc} {med_sc} {std_sc}')
    #     data = data / np.abs(np.nanmedian(data[0,i0:i1]))
    data = data / (med_sc+0.05*std_sc)
    if (ny==7):
        ssp_model=data[6,:]
    else:
        gas = data[0,:] - data[5,:]
        smooth = data[3,:] - data[4,:] - gas        
        ssp_model = data[1,:]-smooth

        mean_val_ssp = np.nanmean(ssp_model)
        print(f'# mean_val_ssp : {mean_val_ssp}')
        if (np.mean(ssp_model)<1e-5):
            ssp_model = data[1,:]
        
        if ((n_ord>2) and (SN_sc>3)):
            base = np.abs(data[0,:])
            f_rat = ssp_model/base
            #data[0,:]
            c_rat0 = np.polyfit(wave, f_rat, n_ord)
            p_rat0 = np.poly1d(c_rat0)(wave)
            ssp_model = ssp_model*(1+1/p_rat0)*0.5
            data[0,:]=(data[0,:]-smooth)*(1+1/p_rat0)*0.5
#            print(data[1,:])
            #except:
            #    show_scale=False  
            #    print('# Not scaling')
        if (show_scale==True):
            ssp_model = data[1,:]


    fig = plt.figure(figsize=(19,11)) 
    gs = fig.add_gridspec(nrows=10, ncols=6,  left=0.075, right=0.97, \
                          hspace=0.0, wspace=0.05, bottom=0.15, top=0.9)

    ax2 = fig.add_subplot(gs[:, :])
    ax0 = fig.add_subplot(gs[0:3, :])#, sharex=ax2)   
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.setp(ax2.spines.values(), color=None)
    
    
    ax0.set_xlim(x_min,x_max)

    if (y_min>=0):
        sign=1
    else:
        sign=-1
    ax0.set_ylim(sign*np.abs(y_min**gamma),y_max**gamma)        

        
 #   colors=['black','maroon','steelblue','darkorange','olive','grey']
#    colors=['black','palegreen','steelblue','darkorange','olive','grey']
#    colors=['black','lightsalmon','steelblue','darkorange','olive','grey']
    colors=colors_in
            #ax1.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,linestyle="solid")#,label=r'Model (M$_\lambda$)')
    res = data[0,:]-data[1,:]+smooth
    gas_model = data[2,:]-data[1,:]
    ax0.plot(wave,gamma_scale((data[0,:]+smooth),gamma)-gamma_scale(data[1,:],gamma),color=colors[2],alpha=alpha,\
             label=r'St. Residual (O$_\lambda$-M$_\lambda$)',linewidth=3)
    if (no_model==False):
        ax0.plot(wave,gamma_scale(data[2,:],gamma)-gamma_scale(data[1,:],gamma),color=colors[3],alpha=alpha,linewidth=1.5,\
                 label=r'Em. Lines model (E$_\lambda$)',linestyle="solid")
    if (plot_res==True):
#        ax0.plot(wave,data[4,:]**gamma,color=colors[4],alpha=1.0,linewidth=1.5,\
        ax0.plot(wave,gamma_scale((data[0,:]+smooth),gamma)-gamma_scale(data[1,:],gamma)-(gamma_scale(data[2,:],gamma)-gamma_scale(data[1,:],gamma)),color=colors[4],alpha=alpha,linewidth=1.5,\
                 label=r'Final residual (R$_\lambda$)',linestyle="solid")
    
    std_res = np.std(res[2000:2500])
    flux = np.median(data[1,2000:2500])

    if (no_st==False):
        ax0.plot(wave,gamma_scale(data[0,:],gamma),\
                 color=colors[0],alpha=1.0,linewidth=3.0,\
                 label=r'Obs. Spectrum (O$_\lambda$)')
        if (no_model==False):
            ax0.plot(wave,gamma_scale(ssp_model,gamma),color=colors[1],\
                     alpha=alpha,linewidth=1.5,\
                     label=r'St. Model (M$_\lambda$)',linestyle="solid")
    

    ax0.xaxis.set_minor_locator(MultipleLocator(100))
    #ax0.axes.get_xaxis().set_visible(False)
    handles, labels = ax0.get_legend_handles_labels()
    if (do_legend):
        ax0.legend(handles, labels,loc=(0.03,1.01),frameon=True,handlelength=1.5,\
                   ncol=5,columnspacing=0.15, prop={'size': 22})
    x = np.arange(x_min,1.1*x_max,100)
    y1 = 5+0*x
    y0 = -5+0*x

    if (id_lines != None):
        mask_lines = (id_lines['wl'].value>=x_min) & (id_lines['wl'].value<x_max)
        tab_elines = id_lines[mask_lines]
        even=0
        for line in tab_elines:
            ax0.text(line['wl'],y_max-0.16*(even+1)*(y_max-y_min),\
                     f"{line['wl']}",color=colors[even],\
                     fontsize='16',horizontalalignment='center',rotation='vertical')
            even=even+1
            if (even>2):
                even=0
        #print(tab_elines)
        
    if (insets != None):
        for Iinst,inst in enumerate(insets):
            #x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original image
            axins = ax0.inset_axes(
                [inst[0], inst[1], inst[2], inst[3]],
                xlim=(inst[4], inst[5]), ylim=(inst[6], inst[7]),yticklabels=[])#, xticklabels=[], yticklabels=[])
            if ((Iinst==0)|(Iinst==9)):
                axins.set_ylabel(r'Flux$_\lambda$', fontsize=23)

            if (no_st==False):
                axins.plot(wave,data[0,:],color=colors[0],alpha=1.0,linewidth=3,label=r'Observed (O$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,ssp_model,color=colors[1],alpha=1.0,linewidth=1.5,label=r'Model (M$_\lambda$)',linestyle="solid")
            axins.plot(wave,res,color=colors[2],alpha=1.0,linewidth=3,\
                       label=r'Residual (O$_\lambda$-M$_\lambda$)')
            if (no_model==False):
                axins.plot(wave,gas_model,color=colors[3],alpha=1.0,linewidth=1.5,\
                           label=r'E. lines model',linestyle="solid")
            if (plot_res==True):
                axins.plot(wave,data[4,:],color=colors[4],alpha=1.0,linewidth=1.5,\
                           label=r'Em. lines model (E$_\lambda$)',linestyle="solid")
                
            if (plot_el == True):
                tab_el_now=tab_el[(tab_el['wl']>inst[4]) & (tab_el['wl']<inst[5])]
                for vals in tab_el_now:
                    axins.plot([vals['wl'],vals['wl']],\
                             [inst[7]-0.05*abs(inst[7]-inst[6]),\
                              inst[7]-0.15*abs(inst[7]-inst[6])],\
                             color='darkorange',linewidth=2)
                axins.text(inst[4]+0.1*np.abs(inst[5]-inst[4]),\
                           inst[7]-0.15*abs(inst[7]-inst[6]),fr'\bf {inst[8]}',bbox=props)#,\



        

    ax0.text(x_min+0.05*(x_max-x_min),y_min+0.85*(y_max-y_min),text,fontsize=21)
    ax2.set_xlabel(r'rest-frame wavelength [\AA]', fontsize=23)
    ax2.set_ylim(y0_d,y1_d)
    if (plot_el == True):
        for vals in tab_el:
            ax0.plot([vals['wl'],vals['wl']],\
                     [y_max**gamma-0.02*abs(y_max**gamma-y_min**gamma),y_max**gamma-0.10*abs(y_max**gamma-y_min**gamma)],\
                     color='darkorange')
    if (log==True):
        ax0.set_yscale('log')
    ax0.axvspan(5600, 5850, alpha=0.7, color='grey',zorder=10)
    ax0.axvspan(7400, 7800, alpha=0.7, color='grey',zorder=10)        
    sgamma=str(gamma)

    ax0.set_ylabel(r'Norm. Flux$^{0.5}_\lambda$', fontsize=23)  
    
    plt.tight_layout()
    fig.savefig(output, transparent=False, facecolor='white', edgecolor='white')#.pdf")


def adaptive_smooth(data, start_width, end_width):
    """
    Smooth an array with a filter that adapts in size from start_width to end_width.
    
    Parameters:
    - data (np.array): The input array to be smoothed.
    - start_width (int): The width of the smoothing kernel at the beginning of the array.
    - end_width (int): The width of the smoothing kernel at the end of the array.
    
    Returns:
    - np.array: The smoothed array.
    """
    # Create an array of kernel sizes changing linearly from start_width to end_width
    n_points = len(data)
    kernel_sizes = np.linspace(start_width, end_width, n_points).astype(int)
    # Output array initialization
    smoothed_data = np.zeros_like(data)
    
    # Apply varying filter
    for i in range(n_points):
        # Handle boundary effects by determining effective kernel size
        half_width = kernel_sizes[i] // 2
        start_index = max(0, i - half_width)
        end_index = min(n_points, i + half_width + 1)
        
        # Apply uniform filter to the local segment of the data
        smoothed_data[i] = np.median(data[start_index:end_index])
    
    return smoothed_data




    
def find_continuum(spec_s,niter=15,thresh=0.8,median_box_max=100,median_box_min=1):
    """
    find the continuum from a spectrum by smoothing and masking the values above
    the smoothed version in an iterative way.
    
    Parameters:
    - data (np.array): The input array from which we would like to find the continuum.
    - niter  (int): Maximum number of iterations
    - thresh (float): Threshold to compare the smoothed an unsmoother version
    - median_box_max (float): Maximum size of the smoothing box
    - median_box_min (float): Minumum size of the smoothing box    

    Returns:
    - np.array: continuum spectrum.
    """
    median_box=median_box_max
    spec_s_org = spec_s.copy()
    lim_spec = (-1)*np.abs(np.nanmin(spec_s))
    mask = (spec_s>lim_spec)
    #print(f'lim: {lim_spec}')
    #print(mask)
    #m_spec_s = adaptive_smooth(spec_s, median_box, int(median_box_max*0.5))
    m_spec_s = st_median_filter(median_box, spec_s)
    pixels = np.arange(0,spec_s.shape[0])
    i_len_in = len(spec_s_org[mask])
    for i in range(niter):
        mask = mask & (np.divide(m_spec_s, spec_s, where=spec_s != 0, out=np.zeros_like(spec_s)) > thresh)
        i_len = len(spec_s_org[mask])
        if (i_len==i_len_in):
            break
        else:
            i_len_in=i_len
        spec_s = np.interp(pixels, pixels[mask], spec_s[mask])
        m_spec_s = adaptive_smooth(spec_s, median_box, median_box_max)
#        m_spec_s = median_filter(median_box, spec_s)
        median_box = int(median_box*0.5)
        if (median_box<median_box_min):
            median_box=median_box_min
        

    spec_s = np.interp(pixels, pixels[mask], spec_s_org[mask])
    if (median_box_min>3):
        spec_s_out = median_filter(spec_s, size=median_box_min, mode='reflect')
    else:
        spec_s_out = spec_s
    return spec_s_out



    
def find_redshift(w_peak=np.array((6548.07,6562.84)),f_peak=np.array((1,1,1)),\
                  w_ref=(6548.05,6562.85,6583.45,6678.15,6716.44,6730.82),\
                  z_min=0,z_max=0.1,d_z=0.01,ds=0.5):
    W_z=[]
    for z in np.arange(z_min,z_max,d_z):
        W=0
        for w_p in w_peak:
            for w_r in w_ref:
                w_r_z=w_r*(1+z)
                W=W+np.exp(-0.5*((w_p-w_r_z)/ds)**2)/len(w_ref)
        W_z.append(W)
    W_z=np.array(W_z)
    z_z=np.arange(z_min,z_max,d_z)
    i_max=np.argmax(W_z)
    z=z_z[i_max]
    return z

def find_redshift_spec(wave,spec,w_min=6500,w_max=6800,\
                       w_min_ne=6350,w_max_ne=6500,\
                       w_ref=(6548.05,6562.85,6583.45,6678.15,6716.44,6730.82),do_plot=0,\
                       z_min=0,z_max=0.1,d_z=0.01,ds=0.5):

    mask_w = (wave>w_min) & (wave<w_max)
    mask_w_b = (wave>(w_min-60)) & (wave<(w_min-30))
    mask_ne = (wave>w_min_ne) & (wave<w_max_ne)
    std_dev = np.nanstd(spec[mask_w_b])
#    print(f'std_dev: {std_dev}')
#    print(f'vals = {spec[mask_w_b]}')
#    peak_threshold = 50*std_dev/np.nanmax(spec[mask_w])
#    plt.plot(spec[mask_w])
#    plt.show()
    nanmax = np.nanmax(spec[mask_w])
    peak_threshold = 50*std_dev/nanmax

    
    if (peak_threshold<0.1):
        peak_threshold=0.1

    if (peak_threshold>0.8):
        peak_threshold=0.8

        
    
#    print(f'peak_threshold: {peak_threshold} , {std_dev}, {np.nanmax(spec[mask_w])}')
    
#    peak_threshold=(np.nanmedian(spec[mask_ne])+3*np.nanstd(spec[mask_ne]))/np.nanmax(spec[mask_w])
#    if (peak_threshold>0.8):
#       peak_threshold=0.8
#    if (peak_threshold<0.2):
#       peak_threshold=0.2
    i_peak,peaks,w_peaks,f_peaks=peak_finder(wave[mask_w], spec[mask_w]/nanmax, 1,\
        peak_threshold=peak_threshold, dmin=2, plot=do_plot, verbose=0)

    mask_thrs = f_peaks>peak_threshold
    i_peak = i_peak[mask_thrs]
    f_peaks = f_peaks[mask_thrs]
    w_peaks = w_peaks[mask_thrs]

    print(f'# f_peaks auto_redshift : {f_peaks} using threshold: {peak_threshold}')
    if ((len(f_peaks)<1.5*len(w_ref))):

        if (len(f_peaks)>1):
            z=find_redshift(w_peak=w_peaks,f_peak=f_peaks, w_ref=w_ref, z_min=z_min,z_max=z_max,d_z=d_z)
        else:
            z=find_redshift(w_peak=w_peaks,f_peak=f_peaks, w_ref=(6562.85,6562.85), z_min=z_min,z_max=z_max,d_z=d_z)
    else:
        z=z_min
    return z


def binArray(data, axis, binstep, binsize, func=np.nanmean):
    data = np.array(data)
    dims = np.array(data.shape)
    argdims = np.arange(data.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    data = data.transpose(argdims)
    data = [func(np.take(data,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) for i in np.arange(dims[axis]//binstep)]
    data = np.array(data).transpose(argdims)
    return data

def bin1D(data,width):
   bindata = data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)
   return bindata


def read_DAP_file(dap_file,verbose=False):
    dap_hdu=fits.open(dap_file)
    tab_PT=Table(dap_hdu['PT'].data)
    tab_RSP=Table(dap_hdu['RSP'].data)
    tab_COEFFS=Table(dap_hdu['COEFFS'].data)
    tab_PE=Table(dap_hdu['PM_ELINES'].data)
    tab_NPE_B=Table(dap_hdu['NP_ELINES_B'].data)
    tab_NPE_R=Table(dap_hdu['NP_ELINES_R'].data)
    tab_NPE_I=Table(dap_hdu['NP_ELINES_I'].data)
    kel_ext = 0
    try:
        kel_ext = 1
        tab_KEL=Table(dap_hdu['PM_KEL'].data)
    except:
        kel_ext = 0
        print('No PM_KEL extension')

    sig_ext = 0
    try:
        sig_ext = 1
        tab_SIGMA = Table(dap_hdu['ELINES_SIGMA_CHI'].data)
    except:
        sig_ext = 0
        print('No SIGMA_CHI extension')

    #
    # Rename some entries!
    #
    tab_RSP.rename_column('Av','Av_st')
    tab_RSP.rename_column('e_Av','e_Av_st')
    tab_RSP.rename_column('z','z_st')
    tab_RSP.rename_column('e_z','e_z_st')
    tab_RSP.rename_column('disp','disp_st')
    tab_RSP.rename_column('e_disp','e_disp_st')
    tab_RSP.rename_column('flux','flux_st')
    tab_RSP.rename_column('redshift','redshift_st')
    tab_RSP.rename_column('med_flux','med_flux_st')
    tab_RSP.rename_column('e_med_flux','e_med_flux_st')
    tab_RSP.rename_column('sys_vel','vel_st')
    #
    # Parametric elines
    #
    tab_PE.rename_column('flux','flux_pe')
    tab_PE.rename_column('e_flux','e_flux_pe')
    tab_PE.rename_column('disp','disp_pe')
    tab_PE.rename_column('e_disp','e_disp_pe')
    tab_PE.rename_column('vel','vel_pe')
    tab_PE.rename_column('e_vel','e_vel_pe')

    if (kel_ext == 1):
        tab_KEL.rename_column('flux','flux_pek')
        tab_KEL.rename_column('e_flux','e_flux_pek')
        tab_KEL.rename_column('disp','disp_pek')
        tab_KEL.rename_column('e_disp','e_disp_pek')
        tab_KEL.rename_column('vel','vel_pek')
        tab_KEL.rename_column('e_vel','e_vel_pek')
    #
    # id	id_fib	rsp	TEFF	LOGG	META	ALPHAM	COEFF	Min.Coeff	log(M/L)	AV	N.Coeff	Err.Coeff
    #
    tab_COEFFS.rename_column('rsp','id_rsp')
    tab_COEFFS.rename_column('TEFF','Teff_rsp')
    tab_COEFFS.rename_column('LOGG','Log_g_rsp')
    tab_COEFFS.rename_column('META','Fe_rsp')
    tab_COEFFS.rename_column('ALPHAM','alpha_rsp')
    tab_COEFFS.rename_column('COEFF','W_rsp')
    tab_COEFFS.rename_column('Min.Coeff','min_W_rsp')
    tab_COEFFS.rename_column('log(M/L)','log_ML_rsp')
    tab_COEFFS.rename_column('AV','Av_rsp')
    tab_COEFFS.rename_column('N.Coeff','n_W_rsp')
    tab_COEFFS.rename_column('Err.Coeff','e_W_rsp')
    
    tab_DAP=Table(dap_hdu['PT'].data)
    tab_DAP=join(tab_DAP,tab_RSP,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_NPE_B,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_NPE_R,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_NPE_I,keys=['id'],join_type='left')

    #
    # order parametric emission line table
    #
    mask_elines = (tab_PE['model']=='eline')
    tab_PE = tab_PE[mask_elines]
    
    a_wl = np.unique(tab_PE['wl'])
    I=0
    for wl_now in a_wl:
        if (wl_now>0.0):
            tab_PE_now=tab_PE[tab_PE['wl']==wl_now]
            tab_PE_tmp=tab_PE_now['id','flux_pe','e_flux_pe','disp_pe','e_disp_pe','vel_pe','e_vel_pe']
            for cols in tab_PE_tmp.colnames:        
                if (cols != 'id'):
                    tab_PE_tmp.rename_column(cols,f'{cols}_{wl_now}')
            if (I==0):
                tab_PE_ord=tab_PE_tmp
            else:
                tab_PE_ord=join(tab_PE_ord,tab_PE_tmp,keys=['id'],join_type='left')
            I=I+1        
    tab_DAP=join(tab_DAP,tab_PE_ord,keys=['id'],join_type='left')


    #
    # order parametric emission line table with fixed kinematics
    #
    if (kel_ext == 1):
        mask_elines = (tab_KEL['model']=='eline')
        tab_KEL = tab_KEL[mask_elines]
    
        a_wl = np.unique(tab_KEL['wl'])
        I=0
        for wl_now in a_wl:
            if (wl_now>0.0):
                tab_KEL_now=tab_KEL[tab_KEL['wl']==wl_now]
                tab_KEL_tmp=tab_KEL_now['id','flux_pek','e_flux_pek','disp_pek','e_disp_pek','vel_pek','e_vel_pek']
                for cols in tab_KEL_tmp.colnames:        
                    if (cols != 'id'):
                        tab_KEL_tmp.rename_column(cols,f'{cols}_{wl_now}')
                if (I==0):
                    tab_KEL_ord=tab_KEL_tmp
                else:
                    tab_KEL_ord=join(tab_KEL_ord,tab_KEL_tmp,keys=['id'],join_type='left')
                I=I+1        
        tab_DAP=join(tab_DAP,tab_KEL_ord,keys=['id'],join_type='left')

    #
    # Order COEFFS table
    #
    a_rsp=np.unique(tab_COEFFS['id_rsp'])
    for I,rsp_now in enumerate(a_rsp):
        tab_C_now=tab_COEFFS[tab_COEFFS['id_rsp']==rsp_now]
        tab_C_tmp=tab_C_now['id','Teff_rsp', 'Log_g_rsp', 'Fe_rsp',\
                            'alpha_rsp', 'W_rsp', 'min_W_rsp',\
                            'log_ML_rsp', 'Av_rsp', 'n_W_rsp', 'e_W_rsp']
        for cols in tab_C_tmp.colnames:        
            if (cols != 'id'):
                tab_C_tmp.rename_column(cols,f'{cols}_{rsp_now}')
        if (I==0):
            tab_C_ord=tab_C_tmp
        else:
            tab_C_ord=join(tab_C_ord,tab_C_tmp,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_C_ord,keys=['id'],join_type='left')

    if (sig_ext == 1):
        tab_DAP=join(tab_DAP,tab_SIGMA,keys=['id'],join_type='left')
     
    if (verbose==True):
        print('---- ALL Table Columns -----')
        print('-------------------------------')
        print('|        PT                   |')
        print('-------------------------------')
        list_columns(tab_PT.colnames)
        print('----------------------------------')
        print('|        RSP                      |')
        print('----------------------------------')
        list_columns(tab_RSP.colnames)
        print('----------------------------------')
        print('|        PE_ord                   |')
        print('----------------------------------')
        list_columns(tab_PE_ord.colnames)
        if (kel_ext == 1):
            print('----------------------------------')
            print('|        PEK_ord                   |')
            print('----------------------------------')
            list_columns(tab_KEL_ord.colnames)
        print('----------------------------------')
        print('|        NPE_B                    |')
        print('----------------------------------')
        list_columns(tab_NPE_B.colnames,3)
        print('----------------------------------')
        print('|        NPE_R                    |')
        print('----------------------------------')
        list_columns(tab_NPE_R.colnames,3)
        print('----------------------------------')
        print('|        NPE_I                    |')
        print('----------------------------------')
        list_columns(tab_NPE_I.colnames,3)
        print('----------------------------------')
        print('|        C_ord                    |')
        print('----------------------------------')
        list_columns(tab_C_ord.colnames,4)
        if (sig_ext == 1):
            print('----------------------------------')
            print('|        SIGMA_CHI                   |')
            print('----------------------------------')
            list_columns(tab_SIGMA.colnames)
        
    return tab_DAP


def get_DAP_tab(tab_PT,tab_RSP,tab_COEFFS,tab_PE,tab_NPE_B,tab_NPE_R,tab_NPE_I,tab_KEL = None, kel_ext = 0, verbose=False):
    #
    # Rename some entries!
    #
     

    tab_RSP.rename_column('Av','Av_st')
    tab_RSP.rename_column('e_Av','e_Av_st')
    tab_RSP.rename_column('z','z_st')
    tab_RSP.rename_column('e_z','e_z_st')
    tab_RSP.rename_column('disp','disp_st')
    tab_RSP.rename_column('e_disp','e_disp_st')
    tab_RSP.rename_column('flux','flux_st')
    tab_RSP.rename_column('redshift','redshift_st')
    tab_RSP.rename_column('med_flux','med_flux_st')
    tab_RSP.rename_column('e_med_flux','e_med_flux_st')
    tab_RSP.rename_column('sys_vel','vel_st')
    #
    # Parametric elines
    #
    tab_PE.rename_column('flux','flux_pe')
    tab_PE.rename_column('e_flux','e_flux_pe')
    tab_PE.rename_column('disp','disp_pe')
    tab_PE.rename_column('e_disp','e_disp_pe')
    tab_PE.rename_column('vel','vel_pe')
    tab_PE.rename_column('e_vel','e_vel_pe')

    if (kel_ext == 1):
        tab_KEL.rename_column('flux','flux_pek')
        tab_KEL.rename_column('e_flux','e_flux_pek')
        tab_KEL.rename_column('disp','disp_pek')
        tab_KEL.rename_column('e_disp','e_disp_pek')
        tab_KEL.rename_column('vel','vel_pek')
        tab_KEL.rename_column('e_vel','e_vel_pek')
    #
    # id	id_fib	rsp	TEFF	LOGG	META	ALPHAM	COEFF	Min.Coeff	log(M/L)	AV	N.Coeff	Err.Coeff
    #
    tab_COEFFS.rename_column('rsp','id_rsp')
    tab_COEFFS.rename_column('TEFF','Teff_rsp')
    tab_COEFFS.rename_column('LOGG','Log_g_rsp')
    tab_COEFFS.rename_column('META','Fe_rsp')
    tab_COEFFS.rename_column('ALPHAM','alpha_rsp')
    tab_COEFFS.rename_column('COEFF','W_rsp')
    tab_COEFFS.rename_column('Min.Coeff','min_W_rsp')
    tab_COEFFS.rename_column('log(M/L)','log_ML_rsp')
    tab_COEFFS.rename_column('AV','Av_rsp')
    tab_COEFFS.rename_column('N.Coeff','n_W_rsp')
    tab_COEFFS.rename_column('Err.Coeff','e_W_rsp')
    
    tab_DAP=tab_PT
    tab_DAP=join(tab_DAP,tab_RSP,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_NPE_B,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_NPE_R,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_NPE_I,keys=['id'],join_type='left')

    #
    # order parametric emission line table
    #
    mask_elines = (tab_PE['model']=='eline')
    tab_PE = tab_PE[mask_elines]
    
    a_wl = np.unique(tab_PE['wl'])
    I=0
    for wl_now in a_wl:
        if (wl_now>0.0):
            tab_PE_now=tab_PE[tab_PE['wl']==wl_now]
            tab_PE_tmp=tab_PE_now['id','flux_pe','e_flux_pe','disp_pe','e_disp_pe','vel_pe','e_vel_pe']
            for cols in tab_PE_tmp.colnames:        
                if (cols != 'id'):
                    tab_PE_tmp.rename_column(cols,f'{cols}_{wl_now}')
            if (I==0):
                tab_PE_ord=tab_PE_tmp
            else:
                tab_PE_ord=join(tab_PE_ord,tab_PE_tmp,keys=['id'],join_type='left')
            I=I+1        
    tab_DAP=join(tab_DAP,tab_PE_ord,keys=['id'],join_type='left')


    #
    # order parametric emission line table with fixed kinematics
    #
    if (kel_ext == 1):
        mask_elines = (tab_KEL['model']=='eline')
        tab_KEL = tab_KEL[mask_elines]
    
        a_wl = np.unique(tab_KEL['wl'])
        I=0
        for wl_now in a_wl:
            if (wl_now>0.0):
                tab_KEL_now=tab_KEL[tab_KEL['wl']==wl_now]
                tab_KEL_tmp=tab_KEL_now['id','flux_pek','e_flux_pek','disp_pek','e_disp_pek','vel_pek','e_vel_pek']
                for cols in tab_KEL_tmp.colnames:        
                    if (cols != 'id'):
                        tab_KEL_tmp.rename_column(cols,f'{cols}_{wl_now}')
                if (I==0):
                    tab_KEL_ord=tab_KEL_tmp
                else:
                    tab_KEL_ord=join(tab_KEL_ord,tab_KEL_tmp,keys=['id'],join_type='left')
                I=I+1        
        tab_DAP=join(tab_DAP,tab_KEL_ord,keys=['id'],join_type='left')

    #
    # Order COEFFS table
    #
    a_rsp=np.unique(tab_COEFFS['id_rsp'])
    for I,rsp_now in enumerate(a_rsp):
        tab_C_now=tab_COEFFS[tab_COEFFS['id_rsp']==rsp_now]
        tab_C_tmp=tab_C_now['id','Teff_rsp', 'Log_g_rsp', 'Fe_rsp',\
                            'alpha_rsp', 'W_rsp', 'min_W_rsp',\
                            'log_ML_rsp', 'Av_rsp', 'n_W_rsp', 'e_W_rsp']
        for cols in tab_C_tmp.colnames:        
            if (cols != 'id'):
                tab_C_tmp.rename_column(cols,f'{cols}_{rsp_now}')
        if (I==0):
            tab_C_ord=tab_C_tmp
        else:
            tab_C_ord=join(tab_C_ord,tab_C_tmp,keys=['id'],join_type='left')
    tab_DAP=join(tab_DAP,tab_C_ord,keys=['id'],join_type='left')
     
    if (verbose==True):
        print('---- ALL Table Columns -----')
        print('-------------------------------')
        print('|        PT                   |')
        print('-------------------------------')
        list_columns(tab_PT.colnames)
        print('----------------------------------')
        print('|        RSP                      |')
        print('----------------------------------')
        list_columns(tab_RSP.colnames)
        print('----------------------------------')
        print('|        PE_ord                   |')
        print('----------------------------------')
        list_columns(tab_PE_ord.colnames)
        if (kel_ext == 1):
            print('----------------------------------')
            print('|        PEK_ord                   |')
            print('----------------------------------')
            list_columns(tab_KEL_ord.colnames)
        print('----------------------------------')
        print('|        NPE_B                    |')
        print('----------------------------------')
        list_columns(tab_NPE_B.colnames,3)
        print('----------------------------------')
        print('|        NPE_R                    |')
        print('----------------------------------')
        list_columns(tab_NPE_R.colnames,3)
        print('----------------------------------')
        print('|        NPE_I                    |')
        print('----------------------------------')
        list_columns(tab_NPE_I.colnames,3)
        print('----------------------------------')
        print('|        C_ord                    |')
        print('----------------------------------')
        list_columns(tab_C_ord.colnames,4)
        
    return tab_DAP
    

def fit_legendre_polynomial(x, y, degree):
    """
    Fits a Legendre polynomial to the given data points.

    Parameters:
    - x (array-like): The x-coordinates of the data points.
    - y (array-like): The y-coordinates of the data points.
    - degree (int): The degree of the Legendre polynomial to fit.

    Returns:
    - Legendre: The fitted Legendre polynomial object.
    - callable: A callable function representing the fitted polynomial.
    """
    # Normalize x to the interval [-1, 1] for Legendre polynomial fitting
    x_min, x_max = np.min(x), np.max(x)
    x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
    
    # Fit the Legendre polynomial
    coeffs = Legendre.fit(x_normalized, y, degree).convert().coef
    legendre_poly = Legendre(coeffs)
    vals = legendre_poly(2 * (x - x_min) / (x_max - x_min) - 1) 
    return vals


def sort_table_by_id(table_to_sort,reference_table):
    """
    Sorts `table_to_sort` based on the order of 'id' values in `reference_table`.

    Parameters
    ----------
    table_to_sort : astropy.table.Table
        The table you want to reorder.
    
    reference_table : astropy.table.Table
        The reference table containing 'id' values in the desired order.

    Returns
    -------
    sorted_table : astropy.table.Table
        A new table with the content of `table_to_sort` sorted to match the order in `reference_table`.
    """
    # Build a dictionary of {id: index} from the reference table
    order_dict = {id_val: i for i, id_val in enumerate(reference_table['id'])}

    # Add a temporary column that maps each row to its position in the reference
    table_to_sort['__order__'] = [order_dict.get(id_val, -1) for id_val in table_to_sort['id']]

    # Filter out rows whose id is not in the reference table
    filtered_table = table_to_sort[table_to_sort['__order__'] >= 0]

    # Sort by the temporary order column
    sorted_table = filtered_table.copy()
    sorted_table.sort('__order__')

    # Remove the temporary column
    sorted_table.remove_column('__order__')

    return sorted_table


def find_closest_indices(array, target_value):
    """
    Returns the indices of the elements in 'array' that are closest to 'target_value'.
    """
    array = np.asarray(array)
    diff = np.abs(array - target_value)
    min_diff = np.min(diff)
    indices = np.where(diff == min_diff)[0]
    return indices

def find_closest_indices_different(array, target_value):
    """
    Returns indices of elements in 'array' that are closest to 'target_value'
    but not equal to 'target_value'.
    """
    array = np.asarray(array)

    # Mask to ignore elements equal to the target value
    mask = array != target_value
    if not np.any(mask):
        return np.array([], dtype=int)  # No valid alternatives

    # Filtered array and corresponding indices
    filtered_array = array[mask]
    original_indices = np.where(mask)[0]

    # Compute distances to the target
    diff = np.abs(filtered_array - target_value)
    min_diff = np.min(diff)

    # Get indices within the filtered array
    closest_mask = diff == min_diff
    return original_indices[closest_mask]


def DAP_extract_coeffs(tab_C_ord): 
    '''
    This routine extract the original coeffs table from a DAP table.
    To be used for plotting the PDFs of the stellar population parameters.
    '''
    # Extract unique RSP IDs from column names
    rsp_ids = [col.split('_')[-1] for col in tab_C_ord.colnames if col.startswith('Teff_rsp_')]
    rsp_ids = list(set(rsp_ids))  # Get unique RSP IDs

    # Initialize an empty list to store individual tables
    tables = []

    for I,rsp_id in enumerate(rsp_ids):
        # Extract columns for the current RSP ID
        cols_to_extract = ['id'] + [col for col in tab_C_ord.colnames if col.endswith(f'_{rsp_id}')]
        tab_tmp = tab_C_ord[cols_to_extract]

        # Rename columns to remove the RSP ID suffix
        for col in tab_tmp.colnames:
            if col != 'id':
                tab_tmp.rename_column(col, col.replace(f'_{rsp_id}', ''))

        # Add the RSP ID as a column
        tab_tmp['id_rsp'] = rsp_id
        #tab_tmp['id_fib'] = I
        # Append the table to the list
        tables.append(tab_tmp)

        # Combine all tables into a single table
    tab_COEFFS = vstack_table(tables)    
    tab_COEFFS['id_fib'] = [int(rsp.split(".")[1]) for rsp in tab_COEFFS['id']]
    tab_COEFFS['id_rsp'] = tab_COEFFS['id_rsp'].astype(int)
    tab_COEFFS.rename_column('id_rsp', 'rsp')
    tab_COEFFS.rename_column('Teff_rsp', 'TEFF')
    tab_COEFFS.rename_column('Log_g_rsp', 'LOGG')
    tab_COEFFS.rename_column('Fe_rsp', 'META')
    tab_COEFFS.rename_column('alpha_rsp', 'ALPHAM')
    tab_COEFFS.rename_column('W_rsp', 'COEFF')
    tab_COEFFS.rename_column('min_W_rsp', 'Min.Coeff')
    tab_COEFFS.rename_column('log_ML_rsp', 'log(M/L)')
    tab_COEFFS.rename_column('Av_rsp', 'AV')
    tab_COEFFS.rename_column('n_W_rsp', 'N.Coeff')
    tab_COEFFS.rename_column('e_W_rsp', 'Err.Coeff')

    tab_COEFFS.sort('rsp', reverse=False)
    #tab_COEFFS.sort('id_fib', reverse=False)
    return tab_COEFFS





def get_den_levels(den_map=None,conts=[0.99,0.65,0.40,0.0]):    
    vals=[]
    levels=[]
    den_map_p = den_map/np.max(den_map)
    for idx,cuts in enumerate(np.arange(0.00,1.0,0.01)):
        mask_now= den_map_p>cuts
        levels.append(cuts)
        vals.append(den_map_p[mask_now].sum()/den_map_p.sum())
    vals_cont=np.array(conts)
    levels_cont=np.interp(vals_cont,np.array(levels),np.array(vals))
    return levels_cont,den_map_p


def PDF_plot(ax=None,hdu_now=None,xlabel=None,ylabel=None,\
             n_fib=None,tab_coeffs=None,\
             conts=[0.99,0.65,0.40,0.0],color_PDF='tomato',\
             pXhist = False, pYhist = False):
    (nz,ny,nx)=hdu_now.data.shape
    xLims=(hdu_now.header['CRVAL1']-0.5*hdu_now.header['CDELT1'],\
           hdu_now.header['CRVAL1']-0.5*hdu_now.header['CDELT1']+nx*hdu_now.header['CDELT1'])
    yLims=(hdu_now.header['CRVAL2']-0.5*hdu_now.header['CDELT2'],\
           hdu_now.header['CRVAL2']-0.5*hdu_now.header['CDELT2']+ny*hdu_now.header['CDELT2'])
    extent=(xLims[0], xLims[1], yLims[0], yLims[1])
    mask_fib = (tab_coeffs['id_fib']==n_fib)
    tab_now=tab_coeffs[mask_fib]
    den_map=np.average(hdu_now.data,axis=0)
    den_map_few=np.average(hdu_now.data,weights=tab_now['COEFF'],axis=0)

    ax.imshow(den_map,extent=extent,cmap='Greys',origin='lower', aspect='auto')
    levels_cont,den_map_p=get_den_levels(den_map=den_map) 
    p_cont=ax.contour(den_map_p,levels_cont,colors='k',origin='lower', extent=extent)

    levels_cont,den_map_p=get_den_levels(den_map=den_map_few) 
    p_cont_few=ax.contour(den_map_p,levels_cont, colors=color_PDF,origin='lower', extent=extent)
    p_cont_few=ax.contourf(den_map_p,levels_cont, colors=color_PDF,origin='lower', extent=extent,alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xhist = den_map_few.sum(axis=0)
    yhist = den_map_few.sum(axis=1)
    divider = make_axes_locatable(ax)
    if (pXhist):
        axLSX = divider.append_axes("top", 0.35, pad=0.0, sharex=ax)
        axLSX.plot(np.linspace(xLims[0], xLims[1], len(xhist)), xhist, color=color_PDF)
        axLSX.axis('off')  # Remove the labels and the borders of the axis           
        axLSX.set_xlim(ax.get_xlim())
        axLSX.tick_params(labelbottom=False)
        axLSX.set_ylim(0.001*np.max(xhist),1.01*np.max(xhist))
    if (pYhist):
        axLSY = divider.append_axes("right", 0.35, pad=0.0, sharey=ax)
        axLSY.plot(yhist,np.linspace(yLims[0],yLims[1],len(yhist)),color=color_PDF)
        axLSY.set_ylim(ax.get_ylim())  
        axLSY.axis('off') 
        axLSY.set_xlim(0.001*np.max(yhist),1.01*np.max(yhist))


    return den_map_few


def PDF_get(ax=None,hdu_now=None,xlabel=None,ylabel=None,\
             n_fib=None,tab_coeffs=None,\
             conts=[0.99,0.65,0.40,0.0],color_PDF='tomato'):
    (nz,ny,nx)=hdu_now.data.shape
    xLims=(hdu_now.header['CRVAL1'],hdu_now.header['CRVAL1']+nx*hdu_now.header['CDELT1'])
    yLims=(hdu_now.header['CRVAL2'],hdu_now.header['CRVAL2']+ny*hdu_now.header['CDELT2'])
    extent=(xLims[0], xLims[1], yLims[0], yLims[1])
    mask_fib = (tab_coeffs['id_fib']==n_fib)
    tab_now=tab_coeffs[mask_fib]
    den_map=np.average(hdu_now.data,axis=0)
    den_map_few=np.average(hdu_now.data,weights=tab_now['COEFF'],axis=0)
    return den_map_few



def PDF_plot_cl(ax=None,hdu_now=None,xlabel=None,ylabel=None,\
             n_fibs=None,\
             conts=[0.99,0.65,0.40,0.0],cmap='Spectral',c_cut=0.8):
    (nz,ny,nx)=hdu_now.data.shape
    xLims=(hdu_now.header['CRVAL1'],hdu_now.header['CRVAL1']+nx*hdu_now.header['CDELT1'])
    yLims=(hdu_now.header['CRVAL2'],hdu_now.header['CRVAL2']+ny*hdu_now.header['CDELT2'])
    extent=(xLims[0], xLims[1], yLims[0], yLims[1])
    

    den_map=np.average(hdu_now.data,axis=0)

    
    ax.imshow(den_map,extent=extent,cmap='Greys',origin='lower', aspect='auto')
    levels_cont,den_map_p=get_den_levels(den_map=den_map) 
    p_cont=ax.contour(den_map_p,levels_cont,colors='k',origin='lower', extent=extent)


    cmap_class = mpl.colormaps[cmap]
    colors = cmap_class(np.linspace(0, c_cut, len(n_fibs)))
    for I,n_fib in enumerate(n_fibs):
        den_map_few=hdu_now.data[n_fib,:,:]
        levels_cont,den_map_p=get_den_levels(den_map=den_map_few) 
        p_cont_few=ax.contour(den_map_p,levels_cont, colors=colors[I],origin='lower', extent=extent, label=f'\#{n_fib}')
        color_now = mpl.colors.to_hex(colors[I], keep_alpha=True)
#        print(I,colors[I])
        p_cont_few=ax.contourf(den_map_p,levels_cont, colors=color_now,origin='lower', extent=extent, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def get_X_sq_out_mod(wave,data,wmin,wmax):
    wave_mask = (wave>wmin) & (wave<wmax)
    n_p = len(data[4,:][wave_mask])
    chi_sq_org = np.sum(((data[0,:])[wave_mask])**2/np.abs((data[0,:])[wave_mask]))/(n_p-3-1)
    chi_sq = np.sum(((data[0,:]-data[8,:])[wave_mask])**2/np.abs((data[0,:])[wave_mask]))/(n_p-3-1)
    chi_sq_np = np.sum(((data[0,:]-(data[6,:]+data[1,:]))[wave_mask])**2/np.abs((data[0,:])[wave_mask]))/(n_p-3-1)

    return np.array((chi_sq_org, chi_sq, chi_sq_np))


def median_fit_spectrum(wave, spec, dw, method="legendre", degree=11, spline_k=3, spline_s=0.0, min_points_bin=1, return_nodes=False):
    """
    Median-bin a spectrum in wavelength intervals of width `dw`, then fit:
      - method='legendre' -> Legendre polynomial
      - method='spline'   -> spline (UnivariateSpline)

    Returns the fitted trend evaluated on the original `wave`.
    """
    wave = np.asarray(wave, dtype=float)
    spec = np.asarray(spec, dtype=float)

    if wave.ndim != 1 or spec.ndim != 1 or wave.size != spec.size:
        raise ValueError("`wave` and `spec` must be 1D arrays with the same length.")
    if dw <= 0:
        raise ValueError("`dw` must be > 0.")

    finite = np.isfinite(wave) & np.isfinite(spec)
    if finite.sum() < 2:
        out = np.full_like(wave, np.nan, dtype=float)
        return (out, np.array([]), np.array([])) if return_nodes else out

    w = wave[finite]
    s = spec[finite]

    # Sort by wavelength
    idx = np.argsort(w)
    w = w[idx]
    s = s[idx]

    # Build bins and compute median points
    bins = np.arange(w.min(), w.max() + dw, dw)
    if bins.size < 2:
        bins = np.array([w.min(), w.max() + 1e-6])

    x_nodes, y_nodes = [], []
    for i in range(len(bins) - 1):
        m = (w >= bins[i]) & (w < bins[i + 1])
        if np.count_nonzero(m) >= min_points_bin:
            x_nodes.append(np.nanmedian(w[m]))
            y_nodes.append(np.nanmedian(s[m]))

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.size < 2:
        out = np.full_like(wave, np.nan, dtype=float)
        return (out, x_nodes, y_nodes) if return_nodes else out

    # Fit on median nodes and evaluate on original wave
    if method.lower() == "legendre":
        deg = int(min(degree, x_nodes.size - 1))
        x0, x1 = x_nodes.min(), x_nodes.max()
        x_nodes_n = 2.0 * (x_nodes - x0) / (x1 - x0) - 1.0
        wave_n = 2.0 * (wave - x0) / (x1 - x0) - 1.0
        poly = Legendre.fit(x_nodes_n, y_nodes, deg).convert()
        trend = poly(wave_n)

    elif method.lower() == "spline":
        k = int(min(spline_k, x_nodes.size - 1))
        if k < 1:
            k = 1
        spl = UnivariateSpline(x_nodes, y_nodes, k=k, s=spline_s)
        trend = spl(wave)

    else:
        raise ValueError("`method` must be 'legendre' or 'spline'.")

    return (trend, x_nodes, y_nodes) if return_nodes else trend


def stddev_fit_spectrum(wave, spec, dw, method="legendre", degree=11, spline_k=3, spline_s=0.0, min_points_bin=1, return_nodes=False):
    """
    Median-bin a spectrum in wavelength intervals of width `dw`, then fit:
      - method='legendre' -> Legendre polynomial
      - method='spline'   -> spline (UnivariateSpline)

    Returns the fitted trend evaluated on the original `wave`.
    """
    wave = np.asarray(wave, dtype=float)
    spec = np.asarray(spec, dtype=float)

    if wave.ndim != 1 or spec.ndim != 1 or wave.size != spec.size:
        raise ValueError("`wave` and `spec` must be 1D arrays with the same length.")
    if dw <= 0:
        raise ValueError("`dw` must be > 0.")

    finite = np.isfinite(wave) & np.isfinite(spec)
    if finite.sum() < 2:
        out = np.full_like(wave, np.nan, dtype=float)
        return (out, np.array([]), np.array([])) if return_nodes else out

    w = wave[finite]
    s = spec[finite]

    # Sort by wavelength
    idx = np.argsort(w)
    w = w[idx]
    s = s[idx]

    # Build bins and compute median points
    bins = np.arange(w.min(), w.max() + dw, dw)
    if bins.size < 2:
        bins = np.array([w.min(), w.max() + 1e-6])

    x_nodes, y_nodes = [], []
    for i in range(len(bins) - 1):
        m = (w >= bins[i]) & (w < bins[i + 1])
        if np.count_nonzero(m) >= min_points_bin:
            x_nodes.append(np.nanstd(w[m]))
            y_nodes.append(np.nanstd(s[m]))

    x_nodes = np.asarray(x_nodes, dtype=float)
    y_nodes = np.asarray(y_nodes, dtype=float)

    if x_nodes.size < 2:
        out = np.full_like(wave, np.nan, dtype=float)
        return (out, x_nodes, y_nodes) if return_nodes else out

    # Fit on median nodes and evaluate on original wave
    if method.lower() == "legendre":
        deg = int(min(degree, x_nodes.size - 1))
        x0, x1 = x_nodes.min(), x_nodes.max()
        x_nodes_n = 2.0 * (x_nodes - x0) / (x1 - x0) - 1.0
        wave_n = 2.0 * (wave - x0) / (x1 - x0) - 1.0
        poly = Legendre.fit(x_nodes_n, y_nodes, deg).convert()
        trend = poly(wave_n)

    elif method.lower() == "spline":
        k = int(min(spline_k, x_nodes.size - 1))
        if k < 1:
            k = 1
        spl = UnivariateSpline(x_nodes, y_nodes, k=k, s=spline_s)
        trend = spl(wave)

    else:
        raise ValueError("`method` must be 'legendre' or 'spline'.")

    return (trend, x_nodes, y_nodes) if return_nodes else trend


def decompose_spectrum(
    wl,
    int_flux_cw,
    n_narrow=3,
    n_medium=5,
    n_broad=11,
    *,
    config=None,
    n_iter=11,
    method="spline",
    degree=11,
    spline_k=3,
    spline_s=0.0,
    min_points_bin=1,
    fit_func=None,
):
    """Split a spectrum into continuum, emission lines and background.

    Parameters
    ----------
    wl : array
        Wavelength array.
    int_flux_cw : array
        Integrated flux to decompose.
    n_narrow, n_medium, n_broad : int
        Smoothing sizes, in the units ``median_fit_spectrum`` expects.
        ``n_medium`` sets the trend subtracted to isolate the emission
        lines, ``n_broad`` the upper envelope that caps the background,
        and ``n_narrow`` the scale of the iterative background erosion.
        Typical values: 3, 5, 11.
    config : dict, optional
        Kept for drop-in compatibility with the original inline code,
        which called ``config.get("smooth_size", n_x)``.  **If this dict
        contains "smooth_size", that single value replaces all three
        scales** and n_narrow/n_medium/n_broad have no effect; a warning
        is issued when that happens.  Leave it as None to use the three
        scales as intended.
    n_iter : int
        Number of erosion iterations for the background.  Defaults to 11,
        as in the original code.  This is unrelated to ``degree``, which
        also happened to be 11 there.
    method, degree, spline_k, spline_s, min_points_bin
        Passed straight through to ``median_fit_spectrum``.  Note
        ``return_nodes`` is not exposed: every call here consumes the
        fitted spectrum arithmetically, so it is always False.
    fit_func : callable, optional
        Override for ``median_fit_spectrum``, mainly for testing.

    Returns
    -------
    dict
        ``trend_cw`` : medium-scale trend, the pseudo-continuum.
        ``trend_cw_broad`` : broad-scale trend, used only as an upper
        bound on the background.
        ``eline_only`` : emission-line component.  Strictly positive:
        where the raw residual was negative it is set to the smallest
        positive residual, not to zero (see Notes).
        ``st_flux_cw`` : ``int_flux_cw - eline_only``, the line-free
        spectrum.
        ``pos_abs_flux_cw`` : ``trend_cw - st_flux_cw``.  This is zero
        wherever the raw residual was already positive, and positive
        only where it was clipped, so it isolates the absorption
        component.
        ``bg_spec`` : the lower-envelope background.
        ``res_bg_spec`` : ``int_flux_cw - bg_spec``.

    Notes
    -----
    The negative values of ``eline_only`` are replaced by the *minimum
    positive* value rather than by zero (the commented-out alternative in
    the original code).  That leaves a small positive pedestal across
    every absorption region, which then propagates into ``st_flux_cw``.
    It is preserved here because it is what the original did, but it is
    worth being deliberate about: if a spectrum has no positive residual
    at all the original would raise on an empty ``nanmin``; this version
    falls back to 0.0 and warns instead.
    """
    fit = fit_func if fit_func is not None else median_fit_spectrum
    if fit is None:
        raise ImportError(
            "median_fit_spectrum could not be imported from lvmdap.dap_tools; "
            "pass fit_func= explicitly."
        )

    wl = np.asarray(wl)
    int_flux_cw = np.asarray(int_flux_cw, dtype=float)
    if wl.shape != int_flux_cw.shape:
        raise ValueError(
            f"wl and int_flux_cw must have the same shape, got "
            f"{wl.shape} and {int_flux_cw.shape}."
        )

    # Reproduce config.get("smooth_size", n_x) exactly: a present key
    # overrides all three scales at once.
    s_narrow, s_medium, s_broad = n_narrow, n_medium, n_broad
    if config is not None and "smooth_size" in config:
        s = config["smooth_size"]
        warnings.warn(
            f"config['smooth_size']={s} overrides n_narrow={n_narrow}, "
            f"n_medium={n_medium} and n_broad={n_broad}: all three stages "
            f"will use {s}. Drop 'smooth_size' from config to use the three "
            f"separate scales.",
            stacklevel=2,
        )
        s_narrow = s_medium = s_broad = s

    def _fit(y, smooth_size):
        return fit(
            wl, y, smooth_size,
            method=method, degree=degree,
            spline_k=spline_k, spline_s=spline_s,
            min_points_bin=min_points_bin, return_nodes=False,
        )

    # --- trends -----------------------------------------------------------
    trend_cw = _fit(int_flux_cw, s_medium)
    trend_cw_broad = _fit(int_flux_cw, s_broad)

    # --- emission lines ---------------------------------------------------
    eline_only = int_flux_cw - trend_cw
    neg = eline_only < 0
    if neg.any():
        pos = eline_only[eline_only > 0]
        if pos.size:
            eline_only[neg] = np.nanmin(pos)
        else:
            warnings.warn(
                "No positive residual in int_flux_cw - trend_cw; clipping "
                "eline_only at 0 instead of the minimum positive value.",
                stacklevel=2,
            )
            eline_only[neg] = 0.0

    st_flux_cw = int_flux_cw - eline_only
    pos_abs_flux_cw = trend_cw - st_flux_cw

    # --- background: iterative erosion towards the lower envelope ---------
    bg_spec_0 = np.minimum(trend_cw, int_flux_cw)
    bg_spec = bg_spec_0                      # guards n_iter == 0
    for _ in range(int(n_iter)):
        bg_spec = _fit(bg_spec_0, s_narrow)
        bg_spec_0 = np.minimum(bg_spec, int_flux_cw)

    bg_spec = np.minimum(bg_spec, trend_cw_broad)
    bg_spec = np.minimum(bg_spec, int_flux_cw)
    bg_spec = _fit(bg_spec, s_narrow)
    res_bg_spec = int_flux_cw - bg_spec

    return {
        "trend_cw": trend_cw,
        "trend_cw_broad": trend_cw_broad,
        "eline_only": eline_only,
        "st_flux_cw": st_flux_cw,
        "pos_abs_flux_cw": pos_abs_flux_cw,
        "bg_spec": bg_spec,
        "res_bg_spec": res_bg_spec,
    }


#
# Redshift match emission lines
#


C_KMS = 299792.458


@dataclass
class RedshiftSolution:
    z: float
    velocity_kms: float
    n_matches: int
    score: float
    rms_velocity_kms: float
    matches: list


def redshift_to_velocity(z, relativistic=True):
    """
    Convert redshift to radial velocity.

    For small z:
        v = c*z

    Relativistic Doppler interpretation:
        beta = ((1+z)^2 - 1) / ((1+z)^2 + 1)

    Note: for cosmological redshifts this velocity should not generally be
    interpreted as the recession velocity.
    """
    if relativistic:
        x = (1.0 + z) ** 2
        beta = (x - 1.0) / (x + 1.0)
        return beta * C_KMS

    return z * C_KMS


def velocity_residual(obs_wave, predicted_wave):
    """
    Velocity difference between an observed line and the predicted
    wavelength at a given redshift.
    """
    return C_KMS * (obs_wave / predicted_wave - 1.0)


def match_lines(
    z,
    rest_waves,
    obs_waves,
    rest_names=None,
    velocity_tolerance=200.0,
):
    """
    Find the optimal one-to-one matching between reference and observed
    emission lines for a given redshift.

    Missing reference lines and spurious observed lines are allowed.
    """

    rest_waves = np.asarray(rest_waves, dtype=float)
    obs_waves = np.asarray(obs_waves, dtype=float)

    if rest_names is None:
        rest_names = [f"line_{i}" for i in range(len(rest_waves))]

    predicted = rest_waves * (1.0 + z)

    nobs = len(obs_waves)
    nref = len(rest_waves)

    # -----------------------------------------------------------
    # Cost matrix
    #
    # Columns 0:nref correspond to real reference lines.
    # Additional columns allow every observed line to remain unmatched.
    # -----------------------------------------------------------

    UNMATCHED_COST = 1.0
    INVALID_COST = 1000.0

    cost = np.full((nobs, nref + nobs), INVALID_COST)

    velocity_matrix = np.zeros((nobs, nref))

    for i, obs in enumerate(obs_waves):
        for j, pred in enumerate(predicted):

            dv = velocity_residual(obs, pred)
            velocity_matrix[i, j] = dv

            if abs(dv) <= velocity_tolerance:

                # Keep valid matches much cheaper than leaving
                # a detection unmatched.
                #
                # Number of matches therefore dominates the score,
                # while residual velocity breaks ties.
                cost[i, j] = 0.1 * (
                    abs(dv) / velocity_tolerance
                ) ** 2

        # Dummy column corresponding to this observation being false/spurious
        cost[i, nref + i] = UNMATCHED_COST

    rows, cols = linear_sum_assignment(cost)

    matches = []

    for i, j in zip(rows, cols):

        if j < nref and cost[i, j] < UNMATCHED_COST:

            obs = obs_waves[i]
            rest = rest_waves[j]
            pred = predicted[j]

            dv = velocity_matrix[i, j]

            matches.append(
                {
                    "name": rest_names[j],
                    "rest_wave": rest,
                    "observed_wave": obs,
                    "predicted_wave": pred,
                    "dv_kms": dv,
                    "z_line": obs / rest - 1.0,
                }
            )

    return matches


def refine_redshift(
    z,
    rest_waves,
    obs_waves,
    rest_names=None,
    velocity_tolerance=200.0,
    max_iterations=10,
):
    """
    Iteratively refine a redshift solution using the median redshift
    of matched lines.
    """

    current_z = z

    for _ in range(max_iterations):

        matches = match_lines(
            current_z,
            rest_waves,
            obs_waves,
            rest_names,
            velocity_tolerance,
        )

        if len(matches) == 0:
            break

        individual_z = np.array(
            [m["z_line"] for m in matches]
        )

        new_z = np.median(individual_z)

        if abs(new_z - current_z) < 1e-10:
            current_z = new_z
            break

        current_z = new_z

    final_matches = match_lines(
        current_z,
        rest_waves,
        obs_waves,
        rest_names,
        velocity_tolerance,
    )

    return current_z, final_matches


def find_redshift_elines(
    rest_waves,
    obs_waves,
    rest_names=None,
    zmin=-0.01,
    zmax=2.0,
    velocity_tolerance=200.0,
    min_matches=2,
    relativistic_velocity=True,
    merge_velocity=50.0,
):
    """
    Determine the most likely redshift(s).

    Parameters
    ----------
    rest_waves : array
        Rest-frame wavelengths.

    obs_waves : array
        Detected observed-frame wavelengths.
        The list may be incomplete and may contain false detections.

    rest_names : list, optional
        Names of reference lines.

    zmin, zmax : float
        Allowed redshift interval.

    velocity_tolerance : float
        Maximum residual velocity for accepting a line match.

    min_matches : int
        Minimum number of matched lines required.

    merge_velocity : float
        Solutions closer than this velocity are considered duplicates.

    Returns
    -------
    solutions : list of RedshiftSolution
        Sorted best-first.
    """

    rest_waves = np.asarray(rest_waves, dtype=float)
    obs_waves = np.asarray(obs_waves, dtype=float)

    if rest_names is None:
        rest_names = [
            f"{w:.2f}" for w in rest_waves
        ]

    # ============================================================
    # STEP 1
    #
    # Every observed/reference pair defines a possible redshift.
    # ============================================================

    candidates = []

    for obs in obs_waves:
        z_values = obs / rest_waves - 1.0

        valid = (z_values >= zmin) & (z_values <= zmax)

        candidates.extend(z_values[valid])

    candidates = np.asarray(candidates)

    # ============================================================
    # STEP 2
    #
    # Evaluate and refine every hypothesis.
    # ============================================================

    raw_solutions = []

    for z0 in candidates:

        z, matches = refine_redshift(
            z0,
            rest_waves,
            obs_waves,
            rest_names,
            velocity_tolerance,
        )

        nmatch = len(matches)

        if nmatch < min_matches:
            continue

        dvs = np.array([m["dv_kms"] for m in matches])

        rms = np.sqrt(np.mean(dvs**2))

        # -----------------------------------------------
        # Ranking score
        #
        # Matching another line is much more important
        # than slightly improving the residual.
        # -----------------------------------------------

        residual_penalty = np.mean(
            (dvs / velocity_tolerance) ** 2
        )

        score = nmatch - 0.25 * residual_penalty

        velocity = redshift_to_velocity(
            z,
            relativistic=relativistic_velocity,
        )

        raw_solutions.append(
            RedshiftSolution(
                z=z,
                velocity_kms=velocity,
                n_matches=nmatch,
                score=score,
                rms_velocity_kms=rms,
                matches=matches,
            )
        )

    # ============================================================
    # STEP 3
    #
    # Sort and eliminate duplicate solutions.
    # ============================================================

    raw_solutions.sort(
        key=lambda x: (
            -x.n_matches,
            x.rms_velocity_kms,
            -x.score,
        )
    )

    unique = []

    for sol in raw_solutions:

        duplicate = False

        for existing in unique:

            # Approximate redshift separation expressed as velocity
            dz = abs(sol.z - existing.z)

            dv = C_KMS * dz / (1.0 + existing.z)

            if dv < merge_velocity:
                duplicate = True
                break

        if not duplicate:
            unique.append(sol)

    return unique


def print_solution(solution, rank=1):

    print()
    print("=" * 75)
    print(f"SOLUTION {rank}")
    print("=" * 75)

    print(f"z                  = {solution.z:.8f}")
    print(
        f"velocity           = {solution.velocity_kms:.2f} km/s"
    )
    print(f"matched lines      = {solution.n_matches}")
    print(
        f"velocity RMS       = "
        f"{solution.rms_velocity_kms:.2f} km/s"
    )
    print(f"score              = {solution.score:.3f}")

    print()
    print(
        f"{'Line':12s} "
        f"{'Rest':>10s} "
        f"{'Observed':>12s} "
        f"{'Predicted':>12s} "
        f"{'dv km/s':>10s}"
    )

    print("-" * 75)

    for m in solution.matches:

        print(
            f"{m['name']:12s} "
            f"{m['rest_wave']:10.2f} "
            f"{m['observed_wave']:12.2f} "
            f"{m['predicted_wave']:12.2f} "
            f"{m['dv_kms']:10.2f}"
        )



def find_redshift_weights(
    rest_waves,
    obs_waves,
    rest_names=None,
    line_weights=None,
    line_groups=None,
    zmin=-0.01,
    zmax=2.0,
    velocity_tolerance=200.0,
    min_matches=2,
    relativistic_velocity=True,
    merge_velocity=50.0,
    residual_weight=0.5,
):
    """
    Determine the most likely redshift(s) from an incomplete and potentially
    contaminated list of observed emission lines.

    This version includes:

      1. Individual weights for each reference emission line.
      2. Extra bonuses when characteristic line complexes are identified.
      3. A penalty based on the velocity residuals of the matched lines.
      4. Missing reference lines are NOT penalized.
      5. Spurious observed lines are allowed and can remain unmatched.
      6. Each observed/reference line can be matched at most once.
      7. Candidate redshifts are refined from the matched lines.
      8. Duplicate redshift solutions are merged.

    Parameters
    ----------
    rest_waves : array-like
        Rest-frame wavelengths of the bona-fide reference emission lines.

    obs_waves : array-like
        Observed wavelengths of detected emission-line candidates. This
        list may be incomplete and may contain false detections.

    rest_names : list of str, optional
        Names corresponding to rest_waves.

    line_weights : dict, optional
        Dictionary assigning a weight to each emission line.
        Example:
            {"Halpha": 3.0, "[OIII]5007": 3.0}

        Lines absent from the dictionary receive weight=1.0.

    line_groups : list, optional
        List of physically meaningful line groups. Each element should be:

            (group_name, [member1, member2, ...], bonus)

        Example:
            ("OIII_doublet",
             ["[OIII]4959", "[OIII]5007"],
             3.0)

    zmin, zmax : float
        Allowed redshift interval.

    velocity_tolerance : float
        Maximum allowed velocity residual, in km/s, for considering an
        observed/reference pair a valid match.

    min_matches : int
        Minimum number of matched reference lines required for accepting
        a candidate solution.

    relativistic_velocity : bool
        If True, convert z to velocity using the relativistic Doppler
        formula. Otherwise use v = c*z.

    merge_velocity : float
        Solutions separated by less than this velocity, in km/s, are
        considered duplicates.

    residual_weight : float
        Strength of the penalty associated with velocity residuals.

    Returns
    -------
    solutions : list of RedshiftSolution
        Solutions sorted from best to worst.
    """

    rest_waves = np.asarray(rest_waves, dtype=float)
    obs_waves = np.asarray(obs_waves, dtype=float)

    # ------------------------------------------------------------
    # Line names
    # ------------------------------------------------------------

    if rest_names is None:
        rest_names = [
            f"{w:.2f}" for w in rest_waves
        ]

    if len(rest_names) != len(rest_waves):
        raise ValueError(
            "rest_names and rest_waves must have the same length."
        )

    # ------------------------------------------------------------
    # Default individual weights
    # ------------------------------------------------------------

    if line_weights is None:
        line_weights = {
            name: 1.0
            for name in rest_names
        }

    # ------------------------------------------------------------
    # Default line groups
    # ------------------------------------------------------------

    if line_groups is None:
        line_groups = []

    # ============================================================
    # STEP 1
    #
    # Generate all possible candidate redshifts.
    #
    # Every observed/reference pair defines:
    #
    #     z = lambda_obs/lambda_rest - 1
    #
    # False lines generate random candidates. The correct redshift
    # should generate several mutually consistent matches.
    # ============================================================

    candidates = []

    for obs in obs_waves:

        z_values = obs / rest_waves - 1.0

        valid = (
            (z_values >= zmin)
            & (z_values <= zmax)
        )

        candidates.extend(z_values[valid])

    candidates = np.asarray(candidates, dtype=float)

    if len(candidates) == 0:
        return []

    # ============================================================
    # STEP 2
    #
    # Evaluate every candidate.
    # ============================================================

    raw_solutions = []

    for z0 in candidates:

        # --------------------------------------------------------
        # Refine z iteratively using all lines that match this
        # candidate.
        # --------------------------------------------------------

        z, matches = refine_redshift(
            z0,
            rest_waves,
            obs_waves,
            rest_names,
            velocity_tolerance,
        )

        nmatch = len(matches)

        if nmatch < min_matches:
            continue

        # --------------------------------------------------------
        # Names of uniquely matched reference lines
        # --------------------------------------------------------

        matched_names = {
            m["name"]
            for m in matches
        }

        # ========================================================
        # INDIVIDUAL LINE WEIGHTS
        # ========================================================

        matched_weight = sum(
            line_weights.get(name, 1.0)
            for name in matched_names
        )

        # ========================================================
        # LINE COMPLEX / DOUBLET / TRIPLET BONUSES
        # ========================================================

        group_bonus = 0.0

        detected_groups = []

        for group_name, members, bonus in line_groups:

            if all(
                member in matched_names
                for member in members
            ):
                group_bonus += bonus

                detected_groups.append(
                    group_name
                )

        # ========================================================
        # VELOCITY RESIDUAL PENALTY
        # ========================================================

        dvs = np.array(
            [
                m["dv_kms"]
                for m in matches
            ],
            dtype=float,
        )

        rms = np.sqrt(
            np.mean(dvs**2)
        )

        residual_penalty = np.mean(
            (
                dvs /
                velocity_tolerance
            ) ** 2
        )

        # ========================================================
        # FINAL SCORE
        #
        # Positive contributions:
        #
        #     individual lines
        #     characteristic groups
        #
        # Negative contribution:
        #
        #     wavelength/velocity mismatch
        #
        # Missing expected lines have NO penalty.
        #
        # False observed lines have NO penalty unless they are
        # incorrectly matched.
        # ========================================================

        score = (
            matched_weight
            + group_bonus
            - residual_weight * residual_penalty
        )

        # --------------------------------------------------------
        # Convert z to velocity
        # --------------------------------------------------------

        velocity = redshift_to_velocity(
            z,
            relativistic=relativistic_velocity,
        )

        solution = RedshiftSolution(
            z=z,
            velocity_kms=velocity,
            n_matches=nmatch,
            score=score,
            rms_velocity_kms=rms,
            matches=matches,
        )

        # Optional extra information
        #
        # Dataclasses without slots allow us to attach these
        # attributes dynamically.
        solution.matched_weight = matched_weight
        solution.group_bonus = group_bonus
        solution.detected_groups = detected_groups
        solution.residual_penalty = residual_penalty

        raw_solutions.append(solution)

    # ============================================================
    # STEP 3
    #
    # Sort solutions.
    #
    # The weighted score is now the primary ranking criterion.
    #
    # For equal scores:
    #     - prefer more matched lines
    #     - prefer smaller velocity RMS
    # ============================================================

    raw_solutions.sort(
        key=lambda x: (
            -x.score,
            -x.n_matches,
            x.rms_velocity_kms,
        )
    )

    # ============================================================
    # STEP 4
    #
    # Remove duplicate solutions corresponding to essentially
    # the same redshift.
    # ============================================================

    unique = []

    for sol in raw_solutions:

        duplicate = False

        for existing in unique:

            # Redshift difference converted approximately to
            # rest-frame velocity difference.
            dv = (
                C_KMS
                * abs(sol.z - existing.z)
                / (1.0 + existing.z)
            )

            if dv < merge_velocity:
                duplicate = True
                break

        if not duplicate:
            unique.append(sol)

    return unique



def running_std_spec(wave, flux, width, mode="wavelength", ddof=1):
    """
    Calculate the running standard deviation of a spectrum.

    Parameters
    ----------
    wave : array-like
        Wavelength array.

    flux : array-like
        Flux array.

    width : float or int
        Width of the running window.

        If mode="wavelength":
            width is expressed in the same units as wave
            (e.g. Angstrom).

        If mode="pixels":
            width is the number of spectral pixels.

    mode : {"wavelength", "pixels"}
        Defines how the running window is specified.

    ddof : int
        Delta degrees of freedom for np.nanstd.
        ddof=1 gives the sample standard deviation.
        ddof=0 gives the population standard deviation.

    Returns
    -------
    std : ndarray
        Running standard deviation evaluated at each wavelength.
    """

    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if wave.shape != flux.shape:
        raise ValueError("wave and flux must have the same shape")

    std = np.full_like(flux, np.nan)

    # ----------------------------------------------------------
    # Fixed wavelength interval
    # ----------------------------------------------------------

    if mode == "wavelength":

        half_width = width / 2.0

        for i, w in enumerate(wave):

            mask = (
                (wave >= w - half_width)
                & (wave <= w + half_width)
                & np.isfinite(flux)
            )

            if np.sum(mask) > ddof:
                std[i] = np.nanstd(
                    flux[mask],
                    ddof=ddof
                )

    # ----------------------------------------------------------
    # Fixed number of pixels
    # ----------------------------------------------------------

    elif mode == "pixels":

        width = int(width)

        if width < 2:
            raise ValueError("width must be >= 2 pixels")

        half = width // 2

        for i in range(len(flux)):

            i0 = max(0, i - half)
            i1 = min(len(flux), i + half + 1)

            values = flux[i0:i1]

            if np.sum(np.isfinite(values)) > ddof:
                std[i] = np.nanstd(
                    values,
                    ddof=ddof
                )

    else:
        raise ValueError(
            "mode must be 'wavelength' or 'pixels'"
        )

    return std







def interp2d_scattered_linear(
    X, Y, val,
    X_new, Y_new,
    method="linear"
):
    """
    Interpolate scattered values val(X,Y) onto new points (X_new,Y_new).

    Parameters
    ----------
    X, Y : array-like
        Coordinates of the input points.

    val : array-like
        Values at the input points.

    X_new, Y_new : array-like
        Coordinates where the values are required.

    method : {"linear", "nearest"}
        Interpolation method.

        "linear":
            Linear interpolation inside the convex hull.
            Nearest-neighbour extrapolation outside the hull.

        "nearest":
            Nearest-neighbour interpolation/extrapolation everywhere.

    Returns
    -------
    val_new : ndarray
        Interpolated values, with the same shape as X_new/Y_new.
    """

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    val = np.asarray(val, dtype=float)

    X_new = np.asarray(X_new, dtype=float)
    Y_new = np.asarray(Y_new, dtype=float)

    if not (X.shape == Y.shape == val.shape):
        raise ValueError("X, Y and val must have the same shape")

    if X_new.shape != Y_new.shape:
        raise ValueError("X_new and Y_new must have the same shape")

    # Remove invalid input data
    good = (
        np.isfinite(X)
        & np.isfinite(Y)
        & np.isfinite(val)
    )

    points = np.column_stack((X[good], Y[good]))
    new_points = np.column_stack(
        (X_new.ravel(), Y_new.ravel())
    )

    # ---------------------------------------------------------
    # Nearest neighbour
    # ---------------------------------------------------------

    nearest = NearestNDInterpolator(
        points,
        val[good]
    )

    if method == "nearest":

        result = nearest(new_points)

    # ---------------------------------------------------------
    # Linear interpolation + nearest extrapolation
    # ---------------------------------------------------------

    elif method == "linear":

        linear = LinearNDInterpolator(
            points,
            val[good],
            fill_value=np.nan
        )

        result = linear(new_points)

        # Outside the convex hull LinearNDInterpolator returns NaN.
        # Replace these values with nearest-neighbour extrapolation.
        outside = ~np.isfinite(result)

        if np.any(outside):
            result[outside] = nearest(
                new_points[outside]
            )

    else:
        raise ValueError(
            "method must be 'linear' or 'nearest'"
        )

    return np.asarray(result).reshape(X_new.shape)



def interp2d_scattered_array(
    X, Y, val,
    X_new, Y_new,
    method="linear"
):
    """
    Interpolate scalar or vector values defined at scattered (X,Y) points.

    Parameters
    ----------
    X, Y : array-like, shape (N,)
        Coordinates of the input points.

    val : array-like, shape (N,) or (N, M) or (N, ...)
        Value(s) associated with each input point.

    X_new, Y_new : array-like
        Coordinates of the target points.

    method : {"linear", "nearest"}

    Returns
    -------
    result : ndarray
        If val.shape == (N,), result.shape == X_new.shape

        If val.shape == (N, M), result.shape == X_new.shape + (M,)
    """

    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    val = np.asarray(val)

    X_new = np.asarray(X_new, dtype=float)
    Y_new = np.asarray(Y_new, dtype=float)

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same length")

    if val.shape[0] != len(X):
        raise ValueError(
            "The first dimension of val must match len(X) and len(Y)"
        )

    if X_new.shape != Y_new.shape:
        raise ValueError(
            "X_new and Y_new must have the same shape"
        )

    # Valid spatial coordinates
    good = np.isfinite(X) & np.isfinite(Y)

    X = X[good]
    Y = Y[good]
    val = val[good]

    points = np.column_stack((X, Y))

    new_points = np.column_stack(
        (X_new.ravel(), Y_new.ravel())
    )

    nearest = NearestNDInterpolator(
        points,
        val
    )

    if method == "nearest":

        result = nearest(new_points)

    elif method == "linear":

        linear = LinearNDInterpolator(
            points,
            val,
            fill_value=np.nan
        )

        result = linear(new_points)

        # For vector-valued val, result may have shape (Nnew, M)
        # so determine which TARGET POINTS failed.
        if result.ndim == 1:
            outside = ~np.isfinite(result)
        else:
            outside = ~np.all(np.isfinite(result), axis=tuple(range(1, result.ndim)))

        if np.any(outside):
            result[outside] = nearest(
                new_points[outside]
            )

    else:
        raise ValueError(
            "method must be 'linear' or 'nearest'"
        )

    output_shape = X_new.shape + val.shape[1:]

    return np.asarray(result).reshape(output_shape)





def interp2d_scattered(
    X, Y, val,
    X_new, Y_new,
    method="linear",
    smoothing=0.0,
):
    """
    Interpolate/extrapolate scattered 2D data.

    Parameters
    ----------
    X, Y : array-like
        Coordinates of original scattered points.

    val : array-like
        Values associated with the original points.

    X_new, Y_new : array-like
        Coordinates of target points.

    method : {"nearest", "linear", "spline", "rbf"}
        nearest :
            Nearest-neighbour interpolation/extrapolation.

        linear :
            Linear interpolation inside the convex hull and
            nearest-neighbour extrapolation outside.

        spline / rbf :
            Smooth RBF interpolation with natural extrapolation.

    smoothing : float
        RBF smoothing parameter.
        0 gives exact interpolation.

    Returns
    -------
    result : ndarray
        Interpolated/extrapolated values with shape X_new.shape.
    """

    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    val = np.asarray(val, dtype=float).ravel()

    X_new = np.asarray(X_new, dtype=float)
    Y_new = np.asarray(Y_new, dtype=float)

    if not (len(X) == len(Y) == len(val)):
        raise ValueError("X, Y and val must have the same length")

    if X_new.shape != Y_new.shape:
        raise ValueError(
            "X_new and Y_new must have the same shape"
        )

    # Remove invalid points
    good = (
        np.isfinite(X)
        & np.isfinite(Y)
        & np.isfinite(val)
    )

    points = np.column_stack(
        (X[good], Y[good])
    )

    values = val[good]

    new_points = np.column_stack(
        (
            X_new.ravel(),
            Y_new.ravel()
        )
    )

    # ---------------------------------------------------------
    # Nearest neighbour
    # ---------------------------------------------------------

    if method == "nearest":

        interp = NearestNDInterpolator(
            points,
            values
        )

        result = interp(new_points)

    # ---------------------------------------------------------
    # Linear + nearest extrapolation
    # ---------------------------------------------------------

    elif method == "linear":

        linear = LinearNDInterpolator(
            points,
            values,
            fill_value=np.nan
        )

        result = linear(new_points)

        outside = ~np.isfinite(result)

        if np.any(outside):

            nearest = NearestNDInterpolator(
                points,
                values
            )

            result[outside] = nearest(
                new_points[outside]
            )

    # ---------------------------------------------------------
    # Smooth spline-like interpolation/extrapolation
    # ---------------------------------------------------------

    elif method in ("spline", "rbf"):

        interp = RBFInterpolator(
            points,
            values,
            kernel="thin_plate_spline",
            smoothing=smoothing,
        )

        result = interp(new_points)

    else:

        raise ValueError(
            "method must be "
            "'nearest', 'linear', 'spline', or 'rbf'"
        )

    return np.asarray(result).reshape(
        X_new.shape
    )



####################################################################################

def process_elines_spec(
    wl,
    flux,
    eflux,
    eline_spec,
    lsf,
    tab_el,
    z_guess=0.0,
    peak_threshold=1.0,
    n_cut_peaks=2.0,
    smooth_size=5,
    velocity_tolerance=150.0,
    do_plot=0,
    verbose=0
):
    """
    Identify, model, and subtract emission lines from one spectrum.

    Only peaks associated with the reference wavelengths in tab_el
    are considered.

    Parameters
    ----------
    wl : ndarray
        Observed wavelength array in Angstrom.
    flux : ndarray
        Original observed spectrum.
    eflux : ndarray
        Flux error spectrum.
    eline_spec : ndarray
        Emission-line component obtained from spectral decomposition.
    lsf : ndarray
        Instrumental Gaussian sigma in Angstrom.
    tab_el : astropy.table.Table
        Reference emission-line table containing 'wl' and 'id'.
    z_guess : float
        Adopted redshift, used to shift reference wavelengths.
    peak_threshold : float
        Detection threshold passed to peak_finder.
    n_cut_peaks : float
        Minimum peak amplitude in units of local noise.
    smooth_size : int
        Smoothing width for estimating the noise.
    velocity_tolerance : float
        Maximum velocity separation (km/s) between a detected
        peak and the expected wavelength of a reference line.
    do_plot : int
        Plotting mode.
    verbose : int
        Verbosity level.

    Returns
    -------
    dict
        emission_model : modeled emission-line spectrum.
        stellar_spectrum : observed spectrum minus emission model.
        w_peaks : wavelengths of detected peaks.
        f_peaks : amplitudes of detected peaks.
        mask_good_peaks : significant positive peak mask.
        mask_matched_peaks : peaks matched to tab_el.
        noise : estimated local noise spectrum.
    """

    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    eflux = np.asarray(eflux, dtype=float)
    eline_spec = np.asarray(eline_spec, dtype=float)
    lsf = np.asarray(lsf, dtype=float)

    # ---------------------------------------------------------
    # 1. Detect peaks in the emission-line spectrum
    # ---------------------------------------------------------
    #     i_peak,peaks,w_peaks,f_peaks=peak_finder(wave[mask_w], spec[mask_w]/nanmax, 1,\
    #    peak_threshold=peak_threshold, dmin=2, plot=do_plot, verbose=0)
    i_peak, peaks, w_peaks, f_peaks = peak_finder(
        wl,
        eline_spec,
        2,
        peak_threshold=peak_threshold,
        dmin=2,
        plot=do_plot,
        verbose=verbose
    )

    w_peaks = np.asarray(w_peaks, dtype=float)
    f_peaks = np.asarray(f_peaks, dtype=float)

    # ---------------------------------------------------------
    # 2. Estimate local noise
    # ---------------------------------------------------------

    std_spec = running_std_spec(
        wl,
        eline_spec,
        width=smooth_size,
        mode="pixels",
        ddof=1
    )

    noise_input = 0.5 * (2.0 * std_spec + eflux)

    abs_noise = median_fit_spectrum(
        wl,
        noise_input,
        smooth_size,
        method="spline",
        degree=11,
        spline_k=3,
        spline_s=0.0,
        min_points_bin=1,
        return_nodes=False
    )

    abs_noise = np.asarray(abs_noise, dtype=float)

    # ---------------------------------------------------------
    # 3. Select significant positive peaks
    # ---------------------------------------------------------

    mask_good_peaks = np.zeros(len(w_peaks), dtype=bool)

    if len(w_peaks) > 0:

        peak_indices = np.array([
            np.argmin(np.abs(wl - w))
            for w in w_peaks
        ], dtype=int)

        mask_good_peaks = (
            np.isfinite(w_peaks) &
            np.isfinite(f_peaks) &
            (f_peaks > 0) &
            (f_peaks > n_cut_peaks * abs_noise[peak_indices])
        )

    # ---------------------------------------------------------
    # 4. Match detected peaks with tab_el
    # ---------------------------------------------------------

    c_kms = 299792.458

    rest_waves = np.asarray(tab_el['wl'], dtype=float)
    observed_reference = rest_waves * (1.0 + z_guess)

    mask_matched_peaks = np.zeros(len(w_peaks), dtype=bool)

    for j, w_peak in enumerate(w_peaks):

        if not mask_good_peaks[j]:
            continue

        for w_ref in observed_reference:

            delta_v = (
                c_kms * abs(w_peak - w_ref) / w_ref
            )

            if delta_v <= velocity_tolerance:
                mask_matched_peaks[j] = True
                break

    # ---------------------------------------------------------
    # 5. Construct the emission-line model
    # ---------------------------------------------------------

    emission_model = np.zeros_like(flux)

    for w_now, f_now in zip(
        w_peaks[mask_matched_peaks],
        f_peaks[mask_matched_peaks]
    ):

        idx = np.argmin(np.abs(wl - w_now))

        sigma_inst = lsf[idx]

        if not np.isfinite(sigma_inst) or sigma_inst <= 0:
            continue

        # Gaussian integrated flux from peak amplitude
        line_flux = (
            np.sqrt(2.0 * np.pi)
            * f_now
            * sigma_inst
        )

        emission_model += eline(
            wl,
            w_now,
            line_flux,
            sigma_inst,
            0.0,
            dap_fwhm=1.0
        )

    # ---------------------------------------------------------
    # 6. Subtract emission-line model
    # ---------------------------------------------------------

    stellar_spectrum = flux - emission_model

    # ---------------------------------------------------------
    # 7. Diagnostics
    # ---------------------------------------------------------

    if verbose:

        print(f'# Detected peaks: {len(w_peaks)}')
        print(f'# Significant peaks: {np.count_nonzero(mask_good_peaks)}')
        print(f'# Matched emission lines: {np.count_nonzero(mask_matched_peaks)}')

    # ---------------------------------------------------------
    # 8. Return results
    # ---------------------------------------------------------

    return {
        'emission_model': emission_model,
        'stellar_spectrum': stellar_spectrum,
        'w_peaks': w_peaks,
        'f_peaks': f_peaks,
        'mask_good_peaks': mask_good_peaks,
        'mask_matched_peaks': mask_matched_peaks,
        'noise': abs_noise
    }



def find_and_subtract_elines_spec(
    wl,
    flux,
    eflux,
    eline_spec,
    lsf,
    tab_el,
    z_min,
    z_max,
    peak_threshold=1.0,
    n_cut_peaks=2.0,
    smooth_size=5,
    velocity_tolerance=15.0,
    min_matches=3,
    line_groups=None,
    subtract_mode="matched",
    do_plot=0,
    verbose=0
):
    """
    Identify and subtract emission lines from one observed spectrum.

    Parameters
    ----------
    wl : ndarray
        Wavelength array in Angstrom.
    flux : ndarray
        Original observed spectrum.
    eflux : ndarray
        Flux uncertainty spectrum.
    eline_spec : ndarray
        Emission-line component obtained from spectral decomposition.
    lsf : ndarray
        Instrumental Gaussian sigma in Angstrom.
    tab_el : astropy.table.Table
        Emission-line reference table. Required columns:
        'wl', 'id'.

        Optional columns:
        'weight'   : relative weight for redshift determination.
        'use_for_z': whether to use the line for redshift matching.
        'subtract' : whether the line should be subtracted.

    z_min, z_max : float
        Allowed redshift interval.
    peak_threshold : float
        Threshold passed to peak_finder.
    n_cut_peaks : float
        Minimum peak amplitude relative to local noise.
    smooth_size : int
        Smoothing width for the noise estimate.
    velocity_tolerance : float
        Redshift matching tolerance in km/s.
    min_matches : int
        Minimum number of matched emission lines.
    line_groups : list, optional
        Groups and bonuses passed to find_redshift_weights.
    subtract_mode : str
        'matched': subtract only peaks matching tab_el.
        'all'    : subtract all significant positive peaks.
    do_plot : int
        Plotting mode.
    verbose : int
        Verbosity level.

    Returns
    -------
    dict
        Dictionary containing the stellar spectrum, emission-line
        model, kinematic parameters and peak diagnostics.
    """

    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    eflux = np.asarray(eflux, dtype=float)
    eline_spec = np.asarray(eline_spec, dtype=float)
    lsf = np.asarray(lsf, dtype=float)

    if not all(
        arr.shape == wl.shape
        for arr in (flux, eflux, eline_spec, lsf)
    ):
        raise ValueError("All spectral arrays must have the same shape")

    if subtract_mode not in ("matched", "all"):
        raise ValueError("subtract_mode must be 'matched' or 'all'")

    # ---------------------------------------------------------
    # 1. Read emission-line configuration
    # ---------------------------------------------------------

    rest_waves = np.asarray(tab_el["wl"], dtype=float)
    rest_names = [str(name) for name in tab_el["id"]]

    if "weight" in tab_el.colnames:
        weights = np.asarray(tab_el["weight"], dtype=float)
    else:
        weights = np.ones(len(tab_el))

    if "use_for_z" in tab_el.colnames:
        use_for_z = np.asarray(tab_el["use_for_z"], dtype=bool)
    else:
        use_for_z = np.ones(len(tab_el), dtype=bool)

    if "subtract" in tab_el.colnames:
        use_for_subtraction = np.asarray(
            tab_el["subtract"], dtype=bool
        )
    else:
        use_for_subtraction = np.ones(len(tab_el), dtype=bool)

    line_weights = {
        name: float(weight)
        for name, weight, use in zip(
            rest_names, weights, use_for_z
        )
        if use
    }

    rest_waves_z = rest_waves[use_for_z]
    rest_names_z = [
        name for name, use in zip(rest_names, use_for_z) if use
    ]

    # Only retain line groups compatible with selected lines
    if line_groups is None:
        line_groups = []

    selected_names = set(rest_names_z)

    active_groups = [
        (group_name, members, bonus)
        for group_name, members, bonus in line_groups
        if set(members).issubset(selected_names)
    ]

    # ---------------------------------------------------------
    # 2. Detect emission-line peaks
    # ---------------------------------------------------------

    i_peak, peaks, w_peaks, f_peaks = peak_finder(
        wl,
        eline_spec,
        2,
        peak_threshold=peak_threshold,
        dmin=0.3,
        plot=do_plot,
        verbose=verbose
    )

    w_peaks = np.asarray(w_peaks, dtype=float)
    f_peaks = np.asarray(f_peaks, dtype=float)

    # ---------------------------------------------------------
    # 3. Estimate local noise
    # ---------------------------------------------------------

    std_spec = running_std_spec(
        wl,
        eline_spec,
        width=smooth_size,
        mode="pixels",
        ddof=1
    )

    noise_input = 0.5 * (2.0 * std_spec + eflux)

    abs_noise = median_fit_spectrum(
        wl,
        noise_input,
        smooth_size,
        method="spline",
        degree=11,
        spline_k=3,
        spline_s=0.0,
        min_points_bin=1,
        return_nodes=False
    )

    abs_noise = np.asarray(abs_noise, dtype=float)

    # ---------------------------------------------------------
    # 4. Select significant positive peaks
    # ---------------------------------------------------------

    if len(w_peaks) > 0:

        peak_indices = np.array([
            np.argmin(np.abs(wl - w))
            for w in w_peaks
        ], dtype=int)

        mask_good_peaks = (
            np.isfinite(w_peaks) &
            np.isfinite(f_peaks) &
            (f_peaks > 0) &
            (f_peaks > abs_noise[peak_indices] * n_cut_peaks)
        )

    else:
        mask_good_peaks = np.zeros(0, dtype=bool)

    good_waves = w_peaks[mask_good_peaks]
    good_fluxes = f_peaks[mask_good_peaks]

    if verbose:
        print(
            f'# Found {len(w_peaks)} peaks, '
            f'{len(good_waves)} significant'
        )

    # ---------------------------------------------------------
    # 5. Determine redshift using selected reference lines
    # ---------------------------------------------------------

    find_z_out = []

    if (len(good_waves) >= min_matches and
            len(rest_waves_z) >= min_matches):

        find_z_out = find_redshift_weights(
            rest_waves=rest_waves_z,
            obs_waves=good_waves,
            rest_names=rest_names_z,
            line_weights=line_weights,
            line_groups=active_groups,
            zmin=z_min,
            zmax=z_max,
            velocity_tolerance=velocity_tolerance,
            min_matches=min_matches
        )

    velocity = np.nan
    velocity_error = np.nan
    velocity_score = 0.0
    velocity_valid = False
    z_best = np.nan

    if len(find_z_out) > 0:

        best = find_z_out[0]

        z_best = float(best.z)
        velocity = float(best.velocity_kms)
        velocity_error = float(best.rms_velocity_kms)
        velocity_score = float(best.score)
        velocity_valid = True

    # ---------------------------------------------------------
    # 6. Select peaks for emission-line subtraction
    # ---------------------------------------------------------

    subtraction_mask = np.zeros(
        len(good_waves), dtype=bool
    )

    if subtract_mode == "all":

        subtraction_mask[:] = True

    else:

        # Use the best measured z when available.
        # Otherwise use the midpoint of the search interval.
        if np.isfinite(z_best):
            z_reference = z_best
        else:
            z_reference = 0.5 * (z_min + z_max)

        for j, w_obs in enumerate(good_waves):

            for w_rest, selected in zip(
                rest_waves, use_for_subtraction
            ):

                if not selected:
                    continue

                w_expected = w_rest * (1.0 + z_reference)

                delta_v = (
                    299792.458 *
                    abs(w_obs - w_expected) / w_expected
                )

                if delta_v <= velocity_tolerance:
                    subtraction_mask[j] = True
                    break

    # ---------------------------------------------------------
    # 7. Construct Gaussian emission-line model
    # ---------------------------------------------------------

    emission_model = np.zeros_like(flux)

    for w_now, f_now in zip(
        good_waves[subtraction_mask],
        good_fluxes[subtraction_mask]
    ):

        idx = np.argmin(np.abs(wl - w_now))

        sigma_inst = lsf[idx]

        if not np.isfinite(sigma_inst) or sigma_inst <= 0:
            continue

        # Convert Gaussian peak amplitude to integrated flux
        line_flux = (
            np.sqrt(2.0 * np.pi)
            * f_now
            * sigma_inst
        )

        emission_model += eline(
            wl,
            w_now,
            line_flux,
            sigma_inst,
            0.0,
            dap_fwhm=1.0
        )

    # ---------------------------------------------------------
    # 8. Subtract emission-line model
    # ---------------------------------------------------------

    stellar_spectrum = flux - emission_model

    # ---------------------------------------------------------
    # 9. Return results
    # ---------------------------------------------------------

    return {
        "stellar_spectrum": stellar_spectrum,
        "emission_model": emission_model,
        "velocity": velocity,
        "velocity_error": velocity_error,
        "velocity_score": velocity_score,
        "velocity_valid": velocity_valid,
        "redshift": z_best,
        "redshift_solutions": find_z_out,
        "w_peaks": w_peaks,
        "f_peaks": f_peaks,
        "mask_good_peaks": mask_good_peaks,
        "subtraction_mask": subtraction_mask,
        "noise": abs_noise
    }


def cluster_emission_lines(
    tab_el,
    max_cluster_span=100.0,
    max_adjacent_distance=30.0,
    wl_col="wl",
):
    """
    Cluster emission lines by wavelength using two criteria:cluster_emission_lines

    1) Maximum total cluster span:
           wl_current - wl_first <= max_cluster_span

    2) Maximum separation between adjacent lines:
           wl_current - wl_previous <= max_adjacent_distance

    Every line belongs to exactly one cluster.

    Parameters
    ----------
    tab_el : astropy.table.Table
        Input emission-line table.

    max_cluster_span : float
        Maximum wavelength span of a cluster [Angstrom].

    max_adjacent_distance : float
        Maximum wavelength separation between adjacent lines in
        the same cluster [Angstrom].

    wl_col : str
        Name of wavelength column.

    Returns
    -------
    clusters : list
        List of astropy Tables, one per cluster.

    cluster_id : ndarray
        Cluster ID for every row in the original table.
    """

    wl = np.asarray(tab_el[wl_col], dtype=float)

    # Sort by wavelength
    order = np.argsort(wl)
    tab = tab_el[order]
    wl_sorted = np.asarray(tab[wl_col], dtype=float)

    n = len(tab)

    if n == 0:
        return [], np.array([], dtype=int)

    cid_sorted = np.zeros(n, dtype=int)

    cluster_id = 0
    cluster_start = wl_sorted[0]

    for i in range(1, n):

        dist_from_start = wl_sorted[i] - cluster_start
        dist_from_previous = wl_sorted[i] - wl_sorted[i - 1]

        # Start a new cluster if EITHER criterion fails
        if (
            dist_from_start > max_cluster_span
            or dist_from_previous > max_adjacent_distance
        ):
            cluster_id += 1
            cluster_start = wl_sorted[i]

        cid_sorted[i] = cluster_id

    # Build cluster tables
    clusters = [
        tab[cid_sorted == k]
        for k in range(cluster_id + 1)
    ]

    # Restore cluster IDs to original row order
    ids_original = np.empty(n, dtype=int)
    ids_original[order] = cid_sorted

    return clusters, ids_original