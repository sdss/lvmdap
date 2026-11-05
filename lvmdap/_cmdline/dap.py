#!/usr/bin/env python3

'''
##############################################################################
#                                                                            #
#                        DAP Command Line Interface                          #
#                                                                            #
##############################################################################
'''


'''
##############################################################################
#                                                                            #
# START: Import required packages                                            #
##############################################################################
'''

#################################################################
# Defining the number of cores to to use in paralelization      #
#################################################################

from email import parser
from logging import config
import sys,os
from tabnanny import verbose
from unittest import result

from lvmdap import config
from lvmdap.pyFIT3D.common.stats import shift_convolve
from lvmdap.pyFIT3D.common.tools import peak_finder 

import wave

from lvmdap.pyFIT3D.modelling.dust import spec_apply_dust
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


##################################################################
# Import required general packages                               #
##################################################################
import time
import numpy as np
import argparse
from copy import deepcopy as copy
from pprint import pprint

from scipy.ndimage import median_filter
from astropy.table import Table
from astropy.table import join as tab_join
from astropy.io import fits
import yaml

###################################################################
# Import required lvmdap packages                                 #
###################################################################
from lvmdap.pyFIT3D.common.io import clean_preview_results_files, print_time
from lvmdap.modelling.auto_rsp_tools import auto_rsp_elines_single_main
from lvmdap.pyFIT3D.common.io import trim_waves, write_img_header
from lvmdap.pyFIT3D.common.gas_tools import detect_create_ConfigEmissionModel, kin_rss_elines_main_return
from lvmdap.pyFIT3D.common.io import create_ConfigAutoSSP_from_lists
from lvmdap.pyFIT3D.common.io import create_emission_lines_file_from_list
from lvmdap.pyFIT3D.common.io import create_emission_lines_mask_file_from_list
from lvmdap.pyFIT3D.common.gas_tools import kin_rss_elines_main
from lvmdap.pyFIT3D.common.constants import __c__, __Ha_central_wl__, _MODELS_ELINE_PAR
from lvmdap.modelling.synthesis import StellarSynthesis
from lvmdap.dap_tools import find_redshift_elines, load_LVM_rss, print_solution, read_PT, plot_spec, read_rsp
from lvmdap.dap_tools import plot_spec_art, plot_spec_art_hdu, Table_mean_rows
from lvmdap.dap_tools import load_LVMSIM_rss, read_LVMSIM_PT
from lvmdap.dap_tools import load_in_rss, read_MaStar_PT
from lvmdap.dap_tools import plot_spectra, read_coeffs_RSP, read_elines_RSP, read_tab_EL
from lvmdap.dap_tools import find_redshift_spec, repair_masked_rss_spectra
from lvmdap.dap_tools import find_continuum
from lvmdap.dap_tools import fit_legendre_polynomial
from lvmdap.dap_tools import read_RSS_PT
from lvmdap.dap_tools import sort_table_by_id
from lvmdap.dap_tools import find_closest_indices_different 
from lvmdap.dap_tools import get_DAP_tab
from lvmdap.flux_elines_tools import flux_elines_RSS_EW_cl
from lvmdap.pek_tools import fit_pek_rss, interpolate_ha_kinematics
from lvmdap.dap_tools import read_DAP_file, map_plot_DAP, map_plot_DAP_rgb, nanaverage, find_closest_indices
from lvmdap.pyFIT3D.common.io import array_to_fits, trim_waves, write_img_header, array_to_hdu
from lvmdap.dap_tools import eline, get_X_sq_out_mod
from lvmdap.modelling.auto_rsp_tools import auto_rsp_elines_rnd
from lvmdap.dap_tools import decompose_spectrum, decompose_spectrum_masked_iterative
from lvmdap.dap_tools import find_redshift_elines,print_solution
from lvmdap.dap_tools import find_redshift_weights,running_std_spec
from lvmdap.dap_tools import median_fit_spectrum,fit_emission_line_system
from lvmdap.pyFIT3D.common.io import print_verbose
from lvmdap.dap_tools import process_elines_spec, cluster_emission_lines
from lvmdap.dap_tools import interp2d_scattered
from lvmdap.pyFIT3D.common.stats import shift_convolve
from lvmdap.pyFIT3D.modelling.dust import spec_apply_dust
from lvmdap.pek_tools import print_progress_bar
from lvmdap.linked_line_fitter import fit_linked_lines
from astropy.table import vstack as tab_vstack
from scipy.spatial import cKDTree, Delaunay

import re
####################################################################
# Define the default values for the fitting using pyFIT3D          #
####################################################################
CWD = os.path.abspath(".")
EXT_CHOICES = ["CCM", "CAL"]
EXT_CURVE = EXT_CHOICES[0]
EXT_RV = 3.1
N_MC = 20



##############################################################################
# START: LVM-DAP main functions
##############################################################################

def select_tab_el_strong(tab_el, tab_m_PM, sn_min=10.0,
                         max_lines=10, min_separation=5.0):
    """Select strong FIT3D lines from integrated PM measurements.

    The long line list ``tab_el`` supplies the candidate IDs/wavelengths.
    Use signed integrated ``flux_<id>_<wl:.2f>_pm / e_flux_..._pm`` as S/N;
    retain only finite values strictly greater than ``sn_min``. Exclude
    *both* members of every adjacent pair in the FULL rest-wavelength list
    separated by less than ``min_separation`` Angstrom. Among the survivors,
    take up to ``max_lines`` with the highest S/N. Return the subset in its
    original ``tab_el`` order (same IDs, wavelengths, columns).

    The initial integrated FIT3D fit must run before this selection, because
    it produces ``tab_m_PM``; the returned subset is used for RSS fibers.
    This does not add a flux-sign constraint to any fitting operation.
    """
    sn_min = float(sn_min)
    max_lines = int(max_lines)
    min_separation = float(min_separation)
    if not np.isfinite(sn_min):
        raise ValueError('pm_strong_sn_min must be finite')
    if max_lines < 1:
        raise ValueError('pm_strong_max_lines must be at least 1')
    if not np.isfinite(min_separation) or min_separation < 0:
        raise ValueError('pm_strong_min_separation must be nonnegative and finite')
    if len(tab_m_PM) != 1:
        raise ValueError('Integrated tab_m_PM must have exactly one row')
    if not len(tab_el):
        return tab_el[:0].copy()

    waves = np.asarray(tab_el['wl'], dtype=float)
    if not np.all(np.isfinite(waves)):
        raise ValueError('tab_el contains nonfinite line wavelengths')
    order = np.argsort(waves, kind='stable')
    too_close = np.zeros(len(tab_el), dtype=bool)
    if len(order) > 1:
        close_pairs = np.diff(waves[order]) < min_separation
        too_close[order[:-1][close_pairs]] = True
        too_close[order[1:][close_pairs]] = True

    candidates = []
    for i, line in enumerate(tab_el):
        if too_close[i]:
            continue
        key = f"{line['id']}_{waves[i]:.2f}"
        f_col = f'flux_{key}_pm'
        ef_col = f'e_flux_{key}_pm'
        if f_col not in tab_m_PM.colnames or ef_col not in tab_m_PM.colnames:
            continue
        f, ef = tab_m_PM[f_col][0], tab_m_PM[ef_col][0]
        if np.ma.is_masked(f) or np.ma.is_masked(ef):
            continue
        try:
            flux = float(f)
            e_flux = float(ef)
        except (ValueError, TypeError):
            continue
        if not np.isfinite(e_flux) or e_flux <= 0:
            continue
        sn = flux / e_flux
        if np.isfinite(sn) and sn > sn_min:
            candidates.append((i, sn))

    # Rank by decreasing S/N, then increasing wavelength for exact ties.
    candidates.sort(key=lambda item: (-item[1], waves[item[0]], item[0]))
    selected = sorted(i for i, _ in candidates[:max_lines])
    return tab_el[selected].copy()



def select_best_emission_lines(tab_el, tab_fe, tab_PM, tab_PEK=None):
    """Choose the lowest finite per-line X_sq independently for every fiber.

    A candidate needs its existing finite X_sq, flux, velocity and observed
    Gaussian dispersion (>0). No additional S/N, sign, quality, or flux priors
    are introduced. All fitted parameters, errors, and local fit diagnostics
    come from the SAME selected NP, PM or PEK row, never from mixed methods.
    Exact ties prefer NP, then PM, then PEK. If no candidate is measurable,
    the derived entries remain NaN and best_method remains empty.

    Selected local X_sq values describe the *original complete method model*;
    they are not reinterpreted as local chi-square of the hybrid BEST model.
    Its separately measured whole-spectrum quality is X_sq_BEST/RMS_BEST.
    """
    from astropy.table import Table

    methods = [('NP', tab_fe, ''), ('PM', tab_PM, '_pm'),
               ('PEK', tab_PEK, '_pek')]
    available = [(name, tab, suffix) for name, tab, suffix in methods
                 if tab is not None]
    if not available:
        raise ValueError('No emission-line tables supplied for BEST selection')
    nf = len(available[0][1])
    for method, tab, _ in available:
        if len(tab) != nf:
            raise ValueError(f'{method} table fiber count does not agree')
    best = Table()
    for col in ('id', 'id_fib'):
        original = next((tab[col] for _, tab, _ in available
                         if col in tab.colnames), None)
        if original is not None:
            best[col] = original.copy()
    if 'id_fib' not in best.colnames:
        best['id_fib'] = np.arange(nf, dtype=np.int64)

    # Resolve legacy wavelength formatting without changing source tables.
    # All output BEST line identifiers use exactly two decimals.
    def fetch(tab, field, key, suffix):
        name = f'{field}_{key}{suffix}'
        if name in tab.colnames:
            return np.ma.asarray(tab[name], dtype=float).filled(np.nan)
        prefix, number = key.rsplit('_', 1)
        pattern = re.compile(r'^' + re.escape(field+'_'+prefix+'_') +
                             r'(\d+(?:\.\d+)?)' + re.escape(suffix) + r'$')
        alternatives = [col for col in tab.colnames
                        if (m := pattern.match(col)) is not None
                        and abs(float(m.group(1)) - float(number)) < 0.004]
        if len(alternatives) > 1:
            raise ValueError(f'Ambiguous {field} columns for {key}{suffix}')
        if alternatives:
            return np.ma.asarray(tab[alternatives[0]], dtype=float).filled(np.nan)
        return np.full(nf, np.nan)

    parameters = ('flux', 'e_flux', 'vel', 'e_vel', 'disp', 'e_disp',
                  'EW', 'e_EW', 'sigma_kms', 'X_sq', 'RMS', 'fr')
    for line in tab_el:
        line_id = line['id']
        if isinstance(line_id, bytes):
            line_id = line_id.decode('utf-8')
        key = f'{str(line_id).strip()}_{float(line["wl"]):.2f}'
        stack = []
        for method, tab, suffix in available:
            cols = {field: fetch(tab, field, key, suffix)
                    for field in parameters}
            valid = (np.isfinite(cols['X_sq']) & np.isfinite(cols['flux']) &
                     np.isfinite(cols['vel']) & np.isfinite(cols['disp']) &
                     (cols['disp'] > 0))
            stack.append((method, cols, valid))
        selected_score = np.full(nf, np.inf)
        selected = np.full(nf, -1, dtype=int)
        for k, (_, cols, valid) in enumerate(stack):
            better = valid & (cols['X_sq'] < selected_score)
            selected_score[better] = cols['X_sq'][better]
            selected[better] = k
        for field in parameters:
            data = np.full(nf, np.nan)
            for k, (_, cols, _) in enumerate(stack):
                choose = selected == k
                data[choose] = cols[field][choose]
            best[f'{field}_{key}_best'] = data
        names = np.full(nf, '', dtype='U3')
        for k, (method, _, _) in enumerate(stack):
            names[selected == k] = method
        best[f'best_method_{key}_best'] = names
    return best


def create_best_emission_model(wl, tab_best):
    """Reconstruct selected signed Gaussian lines without changing any fit.

    Same eline() profile as NP and PM (observed sigma in Angstrom;
    dap_fwhm=1). Build as float32 with only a per-fiber temporary double
    precision Gaussian, not a new double-precision RSS cube.
    """
    wave = np.asarray(wl, dtype=float)
    model = np.zeros((len(tab_best), len(wave)), dtype=np.float32)
    keys = [c[len('flux_'):-len('_best')]
            for c in tab_best.colnames
            if c.startswith('flux_') and c.endswith('_best')]
    for key in keys:
        rest = float(key.rsplit('_', 1)[-1])
        flux = np.asarray(tab_best[f'flux_{key}_best'], float)
        vel = np.asarray(tab_best[f'vel_{key}_best'], float)
        disp = np.asarray(tab_best[f'disp_{key}_best'], float)
        valid = (np.isfinite(flux) & np.isfinite(vel) &
                 np.isfinite(disp) & (disp > 0))
        for i in np.flatnonzero(valid):
            model[i] += eline(wave, rest, flux[i], disp[i], vel[i],
                              dap_fwhm=1.0)
    return model


def add_best_global_quality(tab_best, wl, gas, model, errors, bad_pixels):
    """Global nominal reduced chi-square and RMS of the composite BEST model.

    Degrees of freedom match add_emission_fit_quality(): N_valid_pixels minus
    3 times the number of selected lines with measured kinematics. Local
    X_sq_{line}_best remains the original source's selection score.
    """
    observed = np.asarray(gas, dtype=float)
    reconstructed = np.asarray(model, dtype=float)
    uncertainty = np.asarray(errors, dtype=float)
    mask = np.asarray(bad_pixels, bool)
    if observed.ndim == 1:
        observed = observed[None, :]
        reconstructed = reconstructed[None, :]
        uncertainty = uncertainty[None, :]
        mask = mask[None, :]
    if (observed.shape != reconstructed.shape or observed.shape != uncertainty.shape
            or observed.shape != mask.shape or len(tab_best) != observed.shape[0]):
        raise ValueError('BEST quality spectrum/model/error/mask dimensions disagree')
    names = [name for name in tab_best.colnames
             if name.startswith('best_method_') and name.endswith('_best')]
    n_measured = np.zeros(len(tab_best), dtype=int)
    for name in names:
        n_measured += (np.asarray(tab_best[name]) != '').astype(int)
    chi = np.full(len(tab_best), np.nan)
    rms = np.full(len(tab_best), np.nan)
    for i in range(len(tab_best)):
        valid = (~mask[i] & np.isfinite(observed[i]) &
                 np.isfinite(reconstructed[i]) & np.isfinite(uncertainty[i]) &
                 (uncertainty[i] > 0))
        n = int(valid.sum())
        if not n or not n_measured[i]:
            continue
        residual = observed[i, valid] - reconstructed[i, valid]
        rms[i] = np.sqrt(np.mean(residual**2))
        dof = n - 3*n_measured[i]
        if dof > 0:
            chi[i] = np.sum((residual/uncertainty[i,valid])**2) / dof
    tab_best['X_sq_BEST'] = chi
    tab_best['RMS_BEST'] = rms
    return tab_best


def adopt_best_model_planes(planes, model):
    """Replace joint, residual and gas-subtracted planes using BEST gas.

    MODEL order: 0=observed, 1=stellar, 2=joint, 3=gas input,
    4=joint residual, 5=gas-subtracted, 6=NP gas, 7=PEK/PM gas,
    8=BEST gas. Preserve every other original plane unchanged.
    """
    data = np.asarray(planes)
    if data.ndim not in (2, 3) or data.shape[0] not in (8, 9):
        raise ValueError(f'Unexpected MODEL shape {data.shape}')
    if data.ndim == 2:
        gas_best = np.asarray(model).reshape(-1)
    else:
        gas_best = np.asarray(model)
    if gas_best.shape != data.shape[1:]:
        raise ValueError(f'BEST gas shape {gas_best.shape} != MODEL {data.shape[1:]}')
    if data.shape[0] == 8:
        data = np.concatenate((data, np.empty_like(data[2:3])), axis=0)
    data[8] = gas_best
    data[2] = data[1] + data[8]
    data[4] = data[0] - data[2]
    data[5] = data[0] - data[8]
    return data


# Output-only diagnostics.  These routines never change fitted parameters,
# spectral models, original errors/masks, or which lines are measured.
FIT_QUALITY_W_BR = 4940.0  # 71 of 215 long-list lines in B
FIT_QUALITY_W_RI = 7350.0  # 71 in R, 73 in I


def add_emission_fit_quality(tab, wl, gas_spectrum, model_spectrum,
                             error_spectrum, bad_pixel_mask=None, suffix=''):
    """Add global and per-emission-line X_sq (reduced chi-square) and RMS.

    Each row is compared against that method's *complete gas-emission model*
    and the common continuum-subtracted gas spectrum. A pixel is included only
    if the original mask is clear, observed/model/error are finite, and its
    original error is positive. RMS is unweighted, in flux-density units.

    Global metrics use the full spectral wavelength range. Local metrics use
    +/- 3 measured Gaussian sigmas around each line's observed centroid; they
    compare against the FULL gas model, not an isolated line. This avoids
    treating adjacent/blended emission as residuals of the selected line.

    For consistency across parametric PM/PEK and moment-based NP, report a
    *nominal* reduced chi-square using three effective parameters (integrated
    flux, velocity, width) per measured line: nu_global=N-3*N_lines and
    nu_local=N_window-3. This is a diagnostic approximation, not the exact
    degrees of freedom of tied clusters or non-parametric measurements.
    Undefined chi-square (nu<=0) and unmeasurable lines remain NaN.

    Global columns are method-specific: NP uses X_sq/RMS, PM uses
    X_sq_pm/RMS_pm, and PEK uses X_sq_PEK/RMS_PEK. Per-line columns use
    X_sq_<id>_<wave>[_pm|_pek], RMS_..., and fr_... .  fr is the sum of
    absolute full-model residuals / sum of absolute gas flux in the same
    unmasked local window; a zero denominator yields NaN.
    No numerical values of existing columns are modified.
    """
    wl = np.asarray(wl, dtype=float)
    observed = np.asarray(gas_spectrum, dtype=float)
    model = np.asarray(model_spectrum, dtype=float)
    errors = np.asarray(error_spectrum, dtype=float)
    single = observed.ndim == 1
    if single:
        observed = observed[None, :]
        model = model[None, :]
        errors = errors[None, :]
    if observed.ndim != 2 or model.shape != observed.shape or errors.shape != observed.shape:
        raise ValueError('Gas data, emission model and errors must share a spectrum/RSS shape')
    if wl.ndim != 1 or wl.size != observed.shape[1]:
        raise ValueError('Wavelength grid must match the spectrum pixel dimension')
    if not np.all(np.diff(wl) > 0):
        raise ValueError('Fit-quality wavelength grid must be strictly increasing')
    if len(tab) != observed.shape[0]:
        raise ValueError('Table must contain one row per model spectrum')
    if suffix not in ('', '_pm', '_pek'):
        raise ValueError('Only NP, PM and PEK suffix conventions are supported')
    if bad_pixel_mask is None:
        bad = np.zeros(observed.shape, dtype=bool)
    else:
        bad = np.asarray(bad_pixel_mask, dtype=bool)
        if single and bad.ndim == 1:
            bad = bad[None, :]
        if bad.shape != observed.shape:
            raise ValueError('Original bad-pixel mask must match the spectral data')

    valid = (~bad & np.isfinite(wl)[None, :] & np.isfinite(observed)
             & np.isfinite(model) & np.isfinite(errors) & (errors > 0))
    delta = observed - model
    line_keys = [c[len('flux_'):-len(suffix)] if suffix else c[len('flux_'):]
                 for c in tab.colnames if c.startswith('flux_')
                 and (c.endswith(suffix) if suffix else
                      not c.endswith(('_pm', '_pek')))]
    # Only complete kinematic line entries count toward effective degrees of
    # freedom. Missing or masked line results do not create fake measurements.
    measured_counts = np.zeros(len(tab), dtype=int)
    all_metrics = {}
    for key in line_keys:
        columns = [f'{field}_{key}{suffix}' for field in ('flux', 'vel', 'disp')]
        if not all(c in tab.colnames for c in columns):
            raise KeyError(f'Missing fit-parameter columns for {key}{suffix}')
        fluxes, velocities, widths = (np.asarray(tab[c], dtype=float) for c in columns)
        measurable = (np.isfinite(fluxes) & np.isfinite(velocities)
                      & np.isfinite(widths) & (widths > 0))
        measured_counts += measurable.astype(int)
        rest_wave = float(key.rsplit('_', 1)[-1])
        local_chi = np.full(len(tab), np.nan, dtype=float)
        local_rms = np.full(len(tab), np.nan, dtype=float)
        local_fr = np.full(len(tab), np.nan, dtype=float)
        for i in np.flatnonzero(measurable):
            center = rest_wave * (1.0 + velocities[i] / 299792.458)
            # Search only the local pixels; avoid scanning the entire RSS
            # wavelength grid for every one of the 215 catalogue lines.
            low = np.searchsorted(wl, center - 3.0 * widths[i], side='left')
            high = np.searchsorted(wl, center + 3.0 * widths[i], side='right')
            local_valid = valid[i, low:high]
            n = int(np.count_nonzero(local_valid))
            if n:
                r = delta[i, low:high][local_valid]
                local_rms[i] = np.sqrt(np.mean(r**2))
                observed_local = observed[i, low:high][local_valid]
                denominator = np.sum(np.abs(observed_local))
                if np.isfinite(denominator) and denominator > 0:
                    local_fr[i] = np.sum(np.abs(r)) / denominator
                if n > 3:
                    e = errors[i, low:high][local_valid]
                    local_chi[i] = np.sum((r / e)**2) / (n - 3)
        all_metrics[f'X_sq_{key}{suffix}'] = local_chi
        all_metrics[f'RMS_{key}{suffix}'] = local_rms
        all_metrics[f'fr_{key}{suffix}'] = local_fr

    global_chi = np.full(len(tab), np.nan, dtype=float)
    global_rms = np.full(len(tab), np.nan, dtype=float)
    for i in range(len(tab)):
        n = int(np.count_nonzero(valid[i]))
        if n and measured_counts[i]:
            residual = delta[i, valid[i]]
            global_rms[i] = np.sqrt(np.mean(residual**2))
            dof = n - 3 * measured_counts[i]
            if dof > 0:
                global_chi[i] = np.sum((residual/errors[i, valid[i]])**2)/dof
    # Preserve a predictable order: global fields, then paired per-line fields.
    # Global names distinguish the three measurement methods directly in
    # their source FITS tables; local line suffixes remain unchanged.
    global_suffix = {'': '', '_pm': '_pm', '_pek': '_PEK'}[suffix]
    tab[f'X_sq{global_suffix}'] = global_chi
    tab[f'RMS{global_suffix}'] = global_rms
    for key, values in all_metrics.items():
        tab[key] = values
    return tab



def add_gas_sigma_kms(tab, wl, lsf_spectrum, suffix=''):
    """Store the LSF-deconvolved intrinsic Gaussian velocity sigma per line.

    Parameters
    ----------
    tab : astropy.table.Table
        Per-fiber NP (suffix=''), PM ('_pm'), or PEK ('_pek') line table.
        Its disp_* columns contain *observed* Gaussian sigma in Angstrom.
    wl : 1D array
        Wavelength sampling of the spectral LSF, in Angstrom.
    lsf_spectrum : 1D or 2D array
        Instrumental Gaussian sigma in Angstrom, on ``wl``. A single 1D
        array is shared by all fibers; 2D is (n_fibers, n_wavelengths).
    suffix : str
        FITS line-column suffix, unchanged from the measurement method.

    Notes
    -----
    sigma_intrinsic_kms = c * sqrt(sigma_observed_A**2 - LSF_A**2)
                          / lambda_observed_A.
    The LSF is interpolated at that line's measured observed wavelength.
    A line with sigma_observed_A < LSF_A is unresolved: NaN is stored;
    neither the LSF nor the measured width is modified. This corrects
    the legacy sqrt(abs(...)) and global LSF-rescaling prescriptions.
    This function only adds output columns; it does not alter fit results.
    """
    if suffix not in ('', '_pm', '_pek'):
        raise ValueError('Unknown line suffix: ' + str(suffix))
    wave = np.asarray(wl, dtype=float)
    if wave.ndim != 1 or not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        raise ValueError('wl must be a finite, strictly increasing 1D grid')
    n_fib = len(tab)
    lsf = np.asarray(lsf_spectrum, dtype=float)
    if lsf.shape == (len(wave),):
        lsf = np.broadcast_to(lsf, (n_fib, len(wave)))
    if lsf.shape != (n_fib, len(wave)):
        raise ValueError('lsf_spectrum shape must be (n_wl,) or (n_fiber, n_wl)')
    c_kms = 299792.458
    disp_names = [name for name in tab.colnames
                  if name.startswith('disp_') and
                  (name.endswith(suffix) if suffix else
                   not name.endswith(('_pm', '_pek')))]
    for disp_name in disp_names:
        key = disp_name[len('disp_'):]
        # The spectral line identity ends in its rest wavelength and optional suffix.
        wavelength_key = key[:-len(suffix)] if suffix else key
        try:
            wave_rest = float(wavelength_key.rsplit('_', 1)[-1])
        except (ValueError, IndexError):
            continue
        vel_name = 'vel_' + key
        if vel_name not in tab.colnames:
            continue
        sigma_obs = np.asarray(tab[disp_name], dtype=float)
        velocity = np.asarray(tab[vel_name], dtype=float)
        centers = wave_rest * (1.0 + velocity / c_kms)
        result = np.full(n_fib, np.nan, dtype=np.float32)
        for i in range(n_fib):
            if not (np.isfinite(centers[i]) and centers[i] > 0 and
                    np.isfinite(sigma_obs[i]) and sigma_obs[i] > 0):
                continue
            lsf_here = np.interp(centers[i], wave, lsf[i], left=np.nan, right=np.nan)
            if not (np.isfinite(lsf_here) and lsf_here > 0):
                continue
            intrinsic_squared = sigma_obs[i]**2 - lsf_here**2
            if intrinsic_squared >= 0:
                result[i] = c_kms * np.sqrt(intrinsic_squared) / centers[i]
        tab['sigma_kms_' + key] = result
    return tab




def build_average_emission_quality_table(wl, gas_spectrum, error_spectrum,
                                         bad_pixel_mask, models, tab_el,
                                         reference_pm):
    """Return per-line diagnostics of the average RSS gas spectrum as a table.

    For each method independently, mean the gas and full emission models over
    the same unmasked/finite fibers at each pixel.  Propagate independent
    original pixel errors for the mean in quadrature.  Measure each full-model
    residual within +/-3 integrated PM Gaussian sigmas, without new S/N or
    positivity selections.  These statistics are not spatially resolved fits:
    velocity mixing can broaden the averaged emission lines.
    """
    from astropy.table import Table
    wave = np.asarray(wl, float)
    gas = np.asarray(gas_spectrum, float)
    err = np.asarray(error_spectrum, float)
    bad = np.asarray(bad_pixel_mask, bool)
    if gas.ndim != 2 or err.shape != gas.shape or bad.shape != gas.shape:
        raise ValueError('Average-quality spectrum, error and mask shapes must match')
    if wave.shape != (gas.shape[1],):
        raise ValueError('Average-quality wavelength grid size mismatch')
    rows = []
    results = {}
    for suffix, mod in models.items():
        model = np.asarray(mod, float)
        if model.shape != gas.shape:
            raise ValueError(f'Average-quality {suffix} model shape mismatch')
        ok = (~bad & np.isfinite(gas) & np.isfinite(model)
              & np.isfinite(err) & (err > 0))
        n = np.sum(ok, axis=0)
        good = n > 0
        data_sum = np.sum(np.where(ok, gas, 0.0), axis=0)
        model_sum = np.sum(np.where(ok, model, 0.0), axis=0)
        err_sq_sum = np.sum(np.where(ok, err**2, 0.0), axis=0)
        gmean = np.full(wave.shape, np.nan)
        mmean = np.full(wave.shape, np.nan)
        emean = np.full(wave.shape, np.nan)
        gmean[good] = data_sum[good] / n[good]
        mmean[good] = model_sum[good] / n[good]
        emean[good] = np.sqrt(err_sq_sum[good]) / n[good]
        results[suffix] = (gmean, mmean, emean)
    for line in tab_el:
        key = f"{str(line['id']).strip()}_{float(line['wl']):.2f}"
        vel_key = f'vel_{key}_pm'
        disp_key = f'disp_{key}_pm'
        if (vel_key in reference_pm.colnames and disp_key in reference_pm.colnames
                and len(reference_pm) == 1):
            v = float(reference_pm[vel_key][0])
            sigma = float(reference_pm[disp_key][0])
        else:
            v = np.nan
            sigma = np.nan
        if np.isfinite(v) and np.isfinite(sigma) and sigma > 0:
            center = float(line['wl']) * (1.0 + v / 299792.458)
            lo = np.searchsorted(wave, center - 3.0*sigma, side='left')
            hi = np.searchsorted(wave, center + 3.0*sigma, side='right')
        else:
            lo = hi = 0
        row = {'line': key, 'wl': float(line['wl'])}
        for suffix in ('np', 'pm', 'pek', 'best'):
            chi = rms = fr = np.nan
            if suffix in results and hi > lo:
                g, m, e = results[suffix]
                valid = (np.isfinite(g[lo:hi]) & np.isfinite(m[lo:hi])
                         & np.isfinite(e[lo:hi]) & (e[lo:hi] > 0))
                nn = int(np.count_nonzero(valid))
                if nn:
                    resid = (g[lo:hi]-m[lo:hi])[valid]
                    rms = float(np.sqrt(np.mean(resid**2)))
                    norm = np.sum(np.abs(g[lo:hi][valid]))
                    if norm > 0 and np.isfinite(norm):
                        fr = float(np.sum(np.abs(resid)) / norm)
                    if nn > 3:
                        chi = float(np.sum((resid/e[lo:hi][valid])**2) / (nn-3))
            row[f'X_sq_avg_{suffix}'] = chi
            row[f'RMS_avg_{suffix}'] = rms
            row[f'fr_avg_{suffix}'] = fr
        rows.append(row)
    # Store these measurements in the DAP FITS product, not a separate ECSV.
    return Table(rows=rows)



def build_average_measurement_statistics_table(tab_pt, measurement_tables, expnum):
    """Summarize fiber measurements with paired error columns and field metadata.

    Only scalar numeric measurement columns having an ``e_<name>`` partner are
    included. The sample count ``N_fibers`` refers to finite measurements,
    *independently of whether their reported errors are usable*. ``N_error``
    counts those measurements with finite positive errors. Signed measurements
    are accepted; no S/N cuts or positivity constraints are added to values.

    Mean and sample standard deviation use all finite measurement values.
    For N>=2, the error of the mean combines sample scatter and available
    reported uncertainties:

      sqrt(std**2 / N + sum(e_i**2 for valid errors) / N**2).

    This retains the previous conservative convention; intrinsic spatial
    scatter and measurement noise may overlap. If N_error < N_fibers, the
    reported-error contribution is incomplete (possibly an underestimate),
    indicated explicitly by N_error. If no errors are reported, ``error`` is
    the scatter-only standard error when N>=2; when N=1 it is NaN.
    ``std`` is NaN for N=1. No missing errors are imputed as exact zeros.

    COEFFS are grouped by their rsp template identifier, which is encoded in
    the parameter name (e.g. 'weight[rsp=3]'). EXPNUM, RA_MEAN and DEC_MEAN
    are metadata parameter rows; their error columns and uncertainty fields
    are empty/NaN. RA is a circular mean, DEC an arithmetic mean, both using
    PT fibers with finite RA and Dec.
    """
    names = ('source', 'parameter', 'N_fibers', 'N_error',
             'mean', 'std', 'error')
    dtypes = ('U16', 'U160', 'i8', 'i8', 'f8', 'f8', 'f8')
    result = Table(names=names, dtype=dtypes)

    ra = np.ma.asarray(tab_pt['ra'], dtype=float).filled(np.nan)
    dec = np.ma.asarray(tab_pt['dec'], dtype=float).filled(np.nan)
    if ra.ndim != 1 or dec.ndim != 1 or ra.shape != dec.shape:
        raise ValueError('PT ra and dec must be aligned one-dimensional columns')
    valid_coords = np.isfinite(ra) & np.isfinite(dec)
    n_coords = int(np.count_nonzero(valid_coords))
    ra_mean = dec_mean = np.nan
    if n_coords:
        theta = np.deg2rad(np.mod(ra[valid_coords], 360.0))
        sine = np.mean(np.sin(theta))
        cosine = np.mean(np.cos(theta))
        if np.hypot(sine, cosine) > 1.e-12:
            ra_mean = float(np.rad2deg(np.arctan2(sine, cosine)) % 360.0)
        dec_mean = float(np.mean(dec[valid_coords]))

    exp_mean = float(expnum) if expnum is not None and int(expnum) >= 0 else np.nan
    for param, mean, n in (('EXPNUM', exp_mean, len(tab_pt)),
                           ('RA_MEAN', ra_mean, n_coords),
                           ('DEC_MEAN', dec_mean, n_coords)):
        result.add_row(('META', param, n, 0, mean, np.nan, np.nan))

    for source, table in measurement_tables.items():
        if table is None:
            continue
        if source == 'COEFFS' and 'rsp' in table.colnames:
            rsp_ids = np.asarray(table['rsp'], dtype=int)
            subgroups = [(int(k), rsp_ids == k) for k in np.unique(rsp_ids)]
        else:
            subgroups = [(None, np.ones(len(table), dtype=bool))]

        for column in table.colnames:
            if column.startswith('e_') or 'e_' + column not in table.colnames:
                continue
            error_column = 'e_' + column
            try:
                values = np.ma.asarray(table[column], dtype=float).filled(np.nan)
                errors = np.ma.asarray(table[error_column], dtype=float).filled(np.nan)
            except (TypeError, ValueError):
                continue
            if values.ndim != 1 or errors.shape != values.shape:
                continue

            for rsp, subgroup in subgroups:
                has_measurement = subgroup & np.isfinite(values)
                n = int(np.count_nonzero(has_measurement))
                if n == 0:
                    continue
                has_error = has_measurement & np.isfinite(errors) & (errors > 0)
                ne = int(np.count_nonzero(has_error))
                x = values[has_measurement]
                mean = float(np.mean(x))
                std = float(np.std(x, ddof=1)) if n > 1 else np.nan
                scatter_term = std**2 / n if n > 1 else 0.0
                measurement_term = float(np.sum(errors[has_error]**2)) / n**2
                error = (float(np.sqrt(scatter_term + measurement_term))
                         if n > 1 or ne > 0 else np.nan)
                parameter = column if rsp is None else f'{column}[rsp={rsp}]'
                result.add_row((str(source), parameter, n, ne,
                                mean, std, error))
    return result


def add_average_xsq_statistics(tab_avg_stats, measurement_tables):
    """Append every finite global and per-line X_sq measurement to the summary.

    These are fit-quality statistics, not measurements with an ``e_`` column.
    Their means and sample standard deviations are computed across fibers;
    ``error`` is the scatter standard error (std/sqrt(N)) when N>1.
    ``N_error`` is zero because no independent formal X_sq uncertainties exist.
    The original NP/PM/PEK quality columns are never modified.
    """
    for source, table in measurement_tables.items():
        if table is None:
            continue
        for name in table.colnames:
            if not (name == 'X_sq' or name.startswith('X_sq_')):
                continue
            try:
                data = np.ma.asarray(table[name], dtype=float).filled(np.nan)
            except (TypeError, ValueError):
                continue
            if data.ndim != 1:
                continue
            values = data[np.isfinite(data)]
            n = len(values)
            if n == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else np.nan
            error = float(std / np.sqrt(n)) if n > 1 else np.nan
            tab_avg_stats.add_row((str(source), str(name), n, 0, mean, std, error))
    return tab_avg_stats



def average_model_channels(model_cube):
    """Masked mean of every MODEL plane, with no full-cube temporary arrays.

    Input: (n_plane, n_fiber, n_wave). Nonfinite entries do not contribute.
    Each channel is averaged independently over valid fibers for each pixel.
    Output: (n_plane, n_wave), float32, with NaN at wholly masked pixels.
    """
    planes = np.asarray(model_cube)
    if planes.ndim != 3:
        raise ValueError('MODEL must have (n_plane, n_fiber, n_wavelength) axes')
    n_plane, n_fiber, n_wave = planes.shape
    sums = np.zeros((n_plane, n_wave), dtype=np.float64)
    counts = np.zeros((n_plane, n_wave), dtype=np.int32)
    for i in range(n_fiber):
        spec = planes[:, i, :]
        valid = np.isfinite(spec)
        sums += np.where(valid, spec, 0.0)
        counts += valid
    average = np.full((n_plane, n_wave), np.nan, dtype=np.float32)
    ok = counts > 0
    average[ok] = sums[ok] / counts[ok]
    return average

def get_input_expnum(hdr_primary, hdr_flux, tab_pt):
    """Find EXPNUM in an input FITS header, then in PT if unambiguous."""
    for header in (hdr_primary, hdr_flux):
        if header is None:
            continue
        value = header.get('EXPNUM')
        if value is not None:
            try:
                number = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            return number
    for name in ('EXPNUM', 'exposure', 'expnum'):
        if name not in tab_pt.colnames:
            continue
        try:
            data = np.ma.asarray(tab_pt[name], dtype=float).filled(np.nan)
            unique = np.unique(data[np.isfinite(data)])
        except (TypeError, ValueError):
            continue
        if len(unique) == 1 and unique[0].is_integer():
            return int(unique[0])
    print('# WARNING: EXPNUM unavailable in input headers/PT; using -1 in AVG_PARAM_STATS')
    return -1


def plot_first_last_rss_diagnostics(wl, rss_flux, rss_eflux, bad_pixel_mask,
                                    output_model, pek_model, output_path, label,
                                    fig_type='png', best_model=None):
    """Save first/last RSS spectra and three zooms per fiber, without plt.show.

    Use existing output-model planes: 1=stellar, 6=NP gas, 7=PM gas.
    Plot PEK directly from its separate model array when available.  Does not
    change a single fit, mask, error or model plane.
    """
    import matplotlib.pyplot as plt
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(rss_flux, dtype=float)
    err = np.asarray(rss_eflux, dtype=float)
    bad = np.asarray(bad_pixel_mask, dtype=bool)
    # Plot only two fibers: do not double the entire float32 MODEL cube.
    planes = np.asarray(output_model)
    if (flux.ndim != 2 or err.shape != flux.shape or bad.shape != flux.shape
            or planes.ndim != 3 or planes.shape[1:] != flux.shape
            or planes.shape[0] < 8 or wl.shape != (flux.shape[1],)):
        raise ValueError('Unexpected RSS/model shape for diagnostic figures')
    pek = None if pek_model is None else np.asarray(pek_model, dtype=float)
    best = None if best_model is None else np.asarray(best_model)
    if pek is not None and pek.shape != flux.shape:
        raise ValueError('Unexpected PEK model shape for diagnostic figures')
    if best is not None and best.shape != flux.shape:
        raise ValueError('Unexpected BEST model shape for diagnostic figures')
    views = [('full', None), ('5000', (4800, 5030)),
             ('6500', (6500, 6600)), ('6700', (6700, 6750))]
    saved = []
    with plt.rc_context({'text.usetex': False}):
        for name, idx in (('first', 0), ('last', len(flux)-1)):
            usable = ~bad[idx] & np.isfinite(wl)
            observed = np.where(usable, flux[idx], np.nan)
            errors = np.where(usable, err[idx], np.nan)
            stellar = np.where(usable, planes[1, idx], np.nan)
            pm = np.where(usable, planes[7, idx], np.nan)
            np_model = np.where(usable, planes[6, idx], np.nan)
            pek_now = None if pek is None else np.where(usable, pek[idx], np.nan)
            best_now = None if best is None else np.where(usable, best[idx], np.nan)
            for part, limits in views:
                keep = np.ones(wl.shape, dtype=bool) if limits is None else (
                    (wl >= limits[0]) & (wl <= limits[1]))
                if not np.any(keep):
                    continue
                fig, ax = plt.subplots(figsize=(12, 4.5))
                x = wl[keep]
                ax.plot(x, observed[keep], color='black', lw=1, label='Observed')
                ax.plot(x, stellar[keep], color='tab:orange', lw=1, label='Stellar model')
                ax.plot(x, (stellar+np_model)[keep], color='tab:blue', lw=1,
                        label='Stellar + NP')
                ax.plot(x, (stellar+pm)[keep], color='tab:green', lw=1,
                        label='Stellar + PM')
                if pek_now is not None:
                    ax.plot(x, (stellar+pek_now)[keep], color='tab:red', lw=1,
                            label='Stellar + PEK')
                if best_now is not None:
                    ax.plot(x, (stellar+best_now)[keep], color='tab:purple', lw=1.2,
                            label='Stellar + BEST')
                ax.plot(x, errors[keep], color='gray', lw=0.8,
                        alpha=0.6, label='Pixel error')
                ax.set(xlabel='Wavelength [Angstrom]', ylabel='Flux density',
                       title=f'{label}: {name} fiber #{idx} ({part})')
                ax.legend(fontsize=8, ncol=3)
                fig.tight_layout()
                filename = os.path.join(output_path,
                                        f'output_{name}{"" if part == "full" else "_"+part}.{label}.{fig_type}')
                fig.savefig(filename)
                plt.close(fig)
                saved.append(filename)
    return saved

def _no_traceback(type, value, traceback):
  print(value)



def empirical_eline_errors(
    flux, disp, wl_obs, noise, dw,
    c_flux=0.67, c_vel=1.46, c_disp=0.70
):
    """
    Empirical Gaussian-line uncertainties (Lenz & Ayres type).

    Parameters
    ----------
    flux : float
        Integrated emission-line flux.
    disp : float
        Observed Gaussian sigma in Angstrom.
    wl_obs : float
        Observed central wavelength in Angstrom.
    noise : float
        Local error-spectrum or residual RMS (flux density).
    dw : float
        Wavelength sampling in Angstrom/pixel.

    Returns
    -------
    e_flux : float
        Integrated-flux uncertainty.
    e_vel : float
        Centroid velocity uncertainty, km/s.
    e_disp : float
        Gaussian velocity-dispersion uncertainty, km/s.
    """
    c = 299792.458

    if not all(np.isfinite(x) for x in
               (flux, disp, wl_obs, noise, dw)):
        return np.nan, np.nan, np.nan

    if disp <= 0 or wl_obs <= 0 or noise <= 0 or dw <= 0:
        return np.nan, np.nan, np.nan

    fwhm = 2.354820045 * disp
    amplitude = abs(flux) / (np.sqrt(2*np.pi) * disp)

    Q = (amplitude / noise) * np.sqrt(fwhm / dw)

    if Q <= 0:
        return np.nan, np.nan, np.nan

    e_flux = abs(flux) / (c_flux * Q)
    e_vel = (c / wl_obs) * fwhm / (c_vel * Q)
    e_disp = (c / wl_obs) * disp / (c_disp * Q)

    return e_flux, e_vel, e_disp


def pm_cluster_weighted_errors(wl, eflux, rest_waves, fluxes, vel, sigma,
                               bad_pixel_mask=None, fit_padding=30.0,
                               fit_background=True):
    """Return (e_flux per component, e_vel km/s, e_sigma Angstrom).

    Args:
        wl, eflux: observed Angstrom grid and its per-pixel 1-sigma flux-density errors.
        rest_waves: reference vacuum/air wavelengths for the cluster, as used by fit.
        fluxes: fitted integrated flux for each line (same units as flux density * Angstrom).
        vel: fitted common v0 in km/s; sigma: fitted common Gaussian wavelength sigma in Angstrom.
        bad_pixel_mask: optional FITS integer/boolean mask; all nonzero entries excluded.
        fit_padding: same window as fit_emission_line_system.
        fit_background: if True, marginalize over locally fitted constant background.

    Assumptions: independent spectral errors, a common Gaussian sigma and common
    velocity for the fitted cluster, no prior constraints or parameter bounds.
    The returned uncertainty of sigma is in Angstrom, matching PM 'disp'.
    """
    wl = np.asarray(wl, dtype=float)
    eflux = np.asarray(eflux, dtype=float)
    rest_waves = np.atleast_1d(np.asarray(rest_waves, dtype=float))
    fluxes = np.atleast_1d(np.asarray(fluxes, dtype=float))
    if bad_pixel_mask is None:
        bad = np.zeros(wl.shape, dtype=bool)
    else:
        bad = np.asarray(bad_pixel_mask) != 0
    if wl.ndim != 1 or wl.shape != eflux.shape or wl.shape != bad.shape:
        raise ValueError('wl, eflux, and mask must have the same one-dimensional shape')
    if rest_waves.shape != fluxes.shape or rest_waves.size == 0:
        raise ValueError('rest_waves and fluxes must have identical non-empty shapes')

    n = len(fluxes)
    fallback = (np.full(n, np.nan), np.nan, np.nan)
    c = 299792.458
    if (not np.all(np.isfinite(rest_waves)) or
        not np.all(np.isfinite(fluxes)) or
        not np.isfinite(vel) or not np.isfinite(sigma) or sigma <= 0):
        return fallback

    centers = rest_waves * (1.0 + vel / c)
    # LVM's wavelength array is monotonically increasing.  Select the
    # cluster interval with binary search, rather than scanning the full RSS.
    lo = np.searchsorted(wl, np.min(centers) - fit_padding, side='left')
    hi = np.searchsorted(wl, np.max(centers) + fit_padding, side='right')
    window_wl = wl[lo:hi]
    window_e = eflux[lo:hi]
    valid = (~bad[lo:hi] & np.isfinite(window_wl) &
             np.isfinite(window_e) & (window_e > 0))
    npar = n + 2 + int(fit_background)
    if np.count_nonzero(valid) <= npar + 2:
        return fallback

    x = window_wl[valid]
    e = window_e[valid]
    delta = (x[:, None] - centers[None, :]) / sigma
    profiles = np.exp(-0.5 * delta**2) / (np.sqrt(2*np.pi) * sigma)
    # model = sum_i F_i * profiles[:,i] + B
    d_flux = profiles
    d_velocity = np.sum(fluxes[None, :] * profiles * (delta / sigma) *
                        (rest_waves[None, :] / c), axis=1)
    d_sigma = np.sum(fluxes[None, :] * profiles *
                     (delta**2 - 1.0) / sigma, axis=1)
    columns = [d_flux, d_velocity[:, None], d_sigma[:, None]]
    if fit_background:
        columns.append(np.ones((len(x), 1)))
    jac = np.column_stack(columns)
    whitened = jac / e[:, None]
    try:
        _, s, vt = np.linalg.svd(whitened, full_matrices=False)
        if s.size != npar or s[-1] <= s[0] * 1e-10:
            return fallback  # Unconstrained or degenerate, do not invent precise errors.
        # (J^T W J)^-1 = V diag(1/s^2) V^T; U is not needed.
        cov = (vt.T * (1.0 / s**2)) @ vt
        errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        return fallback
    return errors[:n], float(errors[n]), float(errors[n+1])


##############################################################################
# END: LVM-DAP main functions
##############################################################################


def load_input_data(args, ny_range=None, nx_range=None,
                    sky_hack=False, mask_to_val=True, flux_scale=1e16):
    """
    Load spectral data, associated errors, LSF, and pointing table.

    Supported input formats:
      - LVM RSS / LVMCFrame (default)
      - Generic RSS (args.in_rss)
      - LVM Simulator (args.lvmsim)

    Parameters
    ----------
    args : argparse.Namespace
        Command-line and YAML configuration parameters.
    ny_range, nx_range : optional
        Spectral selection ranges passed to the loading functions.
    sky_hack : bool
        Apply the sky_hack option when loading LVM data.
    mask_to_val : bool
        Replace non-finite flux and error values with interpolated values.

    Returns
    -------
    wl__w : ndarray
        Wavelength array.
    rss_flux_org : ndarray
        Original RSS flux spectra, after optional cleaning.
    rss_eflux_org : ndarray
        Original RSS error spectra, after optional cleaning.
    hdr_flux_org : fits.Header
        Flux FITS header.
    hdr_0 : fits.Header
        Primary FITS header.
    LSF_rss_org : ndarray
        LSF array for each spectrum.
    tab_PT_org : astropy.table.Table
        Pointing table.
    LSF_mean : ndarray
        Mean instrumental LSF (Gaussian sigma).
    input_mask : ndarray
        Original DRP MASK flags, aligned with the returned RSS.
    bad_pixels : bool ndarray
        True for any DRP-flagged or invalid flux/error pixel.
    """

    # ---------------------------------------------------------
    # 1. Load spectra according to the input format
    # ---------------------------------------------------------

    if not args.lvmsim:

        if not args.in_rss:

            print('# Reading data in the LVMCFrame format...')

            (wl__w, rss_flux_org, rss_eflux_org,
             hdr_flux_org, hdr_0, LSF_rss_org, input_mask) = load_LVM_rss(
                args.lvm_file,
                ny_range=ny_range,
                nx_range=nx_range,
                sky_hack=sky_hack,
                m2a=1,
                get_lsf=True,
                get_mask=True,  flux_scale=flux_scale
            )

            # Convert wavelengths when expressed in meters
            if np.median(wl__w) < 1:
                wl__w = wl__w * 10e9

            tab_PT_org = read_PT(
                args.lvm_file, 'none', ny_range=ny_range
            )

            print(f'# Mean wavelength {np.median(wl__w)}')

        else:

            print('# Reading data in a RSS format...')

            (wl__w, rss_flux_org, rss_eflux_org,
             hdr_flux_org, hdr_0) = load_in_rss(
                args.lvm_file,
                ny_range=ny_range,
                nx_range=nx_range, flux_scale = flux_scale
            )

            LSF_rss_org = np.zeros_like(rss_flux_org)
            input_mask = np.zeros(rss_flux_org.shape, dtype=np.uint8)

            try:
                tab_PT_org = read_MaStar_PT(
                    args.lvm_file, 'none', ny_range=ny_range
                )

            except Exception:
                try:
                    tab_PT_org = read_RSS_PT(
                        args.lvm_file, ny_range=ny_range
                    )

                except Exception:
                    NL = rss_flux_org.shape[0]

                    tab_PT_org = Table()
                    tab_PT_org['id'] = np.arange(NL)
                    tab_PT_org['ID'] = np.arange(NL)
                    tab_PT_org['ra'] = 1.0 * tab_PT_org['id']
                    tab_PT_org['dec'] = 1.0 * tab_PT_org['id']
                    tab_PT_org['mask'] = np.full(NL, True)
                    tab_PT_org['fiberid'] = tab_PT_org['id']
                    tab_PT_org['exposure'] = np.ones(NL, dtype=int)

    else:

        print('# Reading data in the LVM Simulator format...')

        (wl__w, rss_flux_org, rss_eflux_org,
         hdr_flux_org, hdr_0) = load_LVMSIM_rss(
            args.lvm_file,
            ny_range=ny_range,
            nx_range=nx_range,
            flux_scale=flux_scale
        )

        LSF_rss_org = np.zeros_like(rss_flux_org)
        input_mask = np.zeros(rss_flux_org.shape, dtype=np.uint8)

        tab_PT_org = read_LVMSIM_PT(
            args.lvm_file, 'none', ny_range=ny_range
        )

    # ---------------------------------------------------------
    # 2. Determine the mean instrumental LSF
    # ---------------------------------------------------------

    try:
        with fits.open(args.lvm_file) as hdu_LSF:
            LSF_mean = np.nanmedian(
                hdu_LSF['LSF'].data, axis=0
            ) / 2.354

    except Exception:
        LSF_mean = args.sigma_gas * np.ones(
            rss_flux_org.shape[1]
        )

    # ---------------------------------------------------------
    # 3. Preserve DRP bitmask, plus unusable flux/error pixels.
    # ---------------------------------------------------------
    input_mask = np.asarray(input_mask)
    if input_mask.shape != rss_flux_org.shape:
        raise ValueError(
            f'MASK shape {input_mask.shape} != FLUX shape {rss_flux_org.shape}'
        )

    bad_pixels = (
        (input_mask != 0) |
        ~np.isfinite(rss_flux_org) |
        ~np.isfinite(rss_eflux_org) |
        (rss_eflux_org <= 0)
    )
    print(f'# DRP MASK / invalid data: {bad_pixels.sum()} / '
          f'{bad_pixels.size} pixels ({100*bad_pixels.mean():.3f}%)')

    if mask_to_val:
        # Values are synthetic; eflux at bad positions is inflated so that
        # such pixels have effectively zero weight in subsequent fitting.
        rss_flux_org, rss_eflux_org = repair_masked_rss_spectra(
            rss_flux_org, rss_eflux_org, bad_pixels, error_inflation=1e6
        )
    else:
        # Explicitly hide flagged samples from fitting if interpolation disabled.
        rss_flux_org = np.asarray(rss_flux_org, dtype=float).copy()
        rss_eflux_org = np.asarray(rss_eflux_org, dtype=float).copy()
        rss_flux_org[bad_pixels] = np.nan
        rss_eflux_org[bad_pixels] = np.nan

    print('# Reading input fits file finished...')

    # ---------------------------------------------------------
    # 4. Return loaded data
    # ---------------------------------------------------------

    return (
        wl__w,
        rss_flux_org,
        rss_eflux_org,
        hdr_flux_org,
        hdr_0,
        LSF_rss_org,
        tab_PT_org,
        LSF_mean,
        input_mask,
        bad_pixels
    )
#---------------------------------------------------------------
def str_to_bool(value):
    if isinstance(value, bool):
        return value

    if str(value).lower() in ('true', 'yes', '1', 'on'):
        return True

    if str(value).lower() in ('false', 'no', '0', 'off'):
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid Boolean value: {value}"
    )

def parse_dap_arguments(cmd_args=None):
    """
    Parse command-line arguments and load the YAML configuration.

    Parameters
    ----------
    cmd_args : list, optional
        Command-line arguments. If None, use sys.argv[1:].

    Returns
    -------
    args : argparse.Namespace
        Complete set of command-line and YAML parameters.
    """

    if cmd_args is None:
        cmd_args = sys.argv[1:]

    # ---------------------------------------------------------
    # 1. Define the command-line arguments
    # ---------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="lvm-dap-yaml LVM_FILE OUTPUT_LABEL CONFIG.YAML"
    )

    parser.add_argument(
        "lvm_file",
        help="Input LVM spectrum to fit"
    )

    parser.add_argument(
        "label",
        help="String identifying the current run"
    )

    parser.add_argument(
        "config_yaml",
        help="YAML file with fitting parameters"
    )

    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debugging mode"
    )

    parser.add_argument(
        "--lvmsim",
        action="store_true",
        help="Input spectra are in LVM Simulator format"
    )

    parser.add_argument(
        "--in_rss",
        action="store_true",
        help="Input spectra are in generic RSS format"
    )

    parser.add_argument(
        "--plot",
        type=np.int64,
        default=0,
        help="Plot mode: 0=no plot, 1=interactive, 2=save to file"
    )

    parser.add_argument(
        "--verbose",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable verbose output (True/False)"
    )

    parser.add_argument(
        "--full_verbose",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable full verbose output (True/False)"
    )

    # ---------------------------------------------------------
    # 2. First pass: obtain the YAML filename
    # ---------------------------------------------------------

    args_initial, _ = parser.parse_known_args(cmd_args)

    print(f'# Reading configuration: {args_initial.config_yaml}')

    with open(args_initial.config_yaml, 'r') as yaml_file:
        dap_config_args = yaml.safe_load(yaml_file)

    if not isinstance(dap_config_args, dict):
        raise ValueError(
            "DAP YAML configuration must contain a dictionary"
        )

    # ---------------------------------------------------------
    # 3. Add YAML configuration parameters
    # ---------------------------------------------------------

    lvmdap_dir = dap_config_args.get('lvmdap_dir', '')

    existing_options = set(parser._option_string_actions)

    for key, value in dap_config_args.items():

        if isinstance(value, str):
            value = value.replace("..", lvmdap_dir)

        option = '--' + key

        if option in existing_options:
            parser.set_defaults(**{key: value})
        else:
            parser.add_argument(option, default=value)

    # ---------------------------------------------------------
    # 4. Additional command-line parameters
    # ---------------------------------------------------------

    parser.add_argument(
        "--flux-scale",
        metavar=("min", "max"),
        type=np.float64,
        nargs=2,
        default=[-1, 1],
        help="Flux scale of the input spectrum"
    )

    # ---------------------------------------------------------
    # 5. Final parsing
    # ---------------------------------------------------------

    args = parser.parse_args(cmd_args)

    print(cmd_args)
    print(args)

    if not args.debug:
        sys.excepthook = _no_traceback
    else:
        pprint("COMMAND LINE ARGUMENTS")
        pprint(f"{args}\n")

    # ---------------------------------------------------------
    # 6. Configure interactive plotting
    # ---------------------------------------------------------

    if args.plot == 1:
        print("# Visualize fitting...")

        from matplotlib import use as mpl_use
        mpl_use('TkAgg')

    return args

#---------------------------------------------------------


def initialize_dap_parameters(args):
    """
    Initialize optional DAP parameters using default values
    when they are not provided in the YAML configuration.

    The input Namespace is modified and returned.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line and YAML configuration parameters.

    Returns
    -------
    args : argparse.Namespace
        Initialized configuration parameters.
    """

    # ---------------------------------------------------------
    # 1. Default values
    # ---------------------------------------------------------

    defaults = {
        'ny_range': None,
        'nx_range': None,
        'out_plot_format': 'png',
        'only_integrated': False,
        'sky_hack': False,
        'mask_to_val': True,
        'auto_redshift': False,
        'auto_z_min': -0.003,
        'auto_z_max': 0.005,
        'auto_z_d': 0.00001,
        'dump_model': False,
        'SN_CUT': 3,
        'SN_CUT_INT': 3,
        'auto_z_bg': True,
        'auto_z_sc': True,
        'kin_fixed': 2,
        'N_MC': 20,
        'n_loops': 5,
        'w_range_FE': 30,
        'smooth_size': 21,
        'n_leg': 11,
        # Fit both PM engines and retain the better strong-line cluster.
        'pm_fit_engine': 'both',
        # Optional second-stage constrained Gaussian emission-line fit.
        'fit_PEK': True,
        'pek_velocity_tolerance': 50.0,       # km/s around interpolated Halpha v
        'pek_dispersion_tolerance': 2.5,     # Angstrom around observed Halpha sigma
        'pek_fit_padding': 20.0,            # Angstrom outside each cluster
        'pek_max_cluster_span': 75.0,       # Angstrom
        'pek_max_adjacent_distance': 25.0,  # Angstrom
        "verbose": False,
    }

    # ---------------------------------------------------------
    # 2. Assign missing parameters
    # ---------------------------------------------------------

    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # ---------------------------------------------------------
    # 3. Initialize flux scale
    # ---------------------------------------------------------

    if (args.flux_scale[0] == -1 and
            args.flux_scale[1] == 1):

        args.flux_scale = list(args.flux_scale_org)

    print('# Input parameters initialized ...')

    return args
#------------------------------------------------------------------

def create_dap_info_hdu(args):
    """
    Create a FITS binary table containing the DAP configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Final DAP configuration parameters.

    Returns
    -------
    hdu_info : fits.BinTableHDU
        FITS extension named INFO with columns 'param' and 'value'.
    """

    param_names = []
    param_values = []

    for name, value in vars(args).items():
        param_names.append(name)
        param_values.append(str(value))

    tab_info = Table(
        [param_names, param_values],
        names=('param', 'value')
    )

    hdu_info = fits.BinTableHDU(
        tab_info,
        name='INFO'
    )

    return hdu_info

# -----------------------------------------------------------------------


def create_integrated_spectra(
    wl,
    rss_flux,
    rss_eflux,
    LSF_rss,
    auto_z_min=-0.003,
    auto_z_max=0.005,
    SN_threshold=3,
    we_range=(6500, 6600),
    cw_range=(6400, 6500),
    bad_pixels=None
):
    """
    Generate integrated spectra using several weighting schemes.

    Parameters
    ----------
    wl : ndarray
        Wavelength array in Angstrom.
    rss_flux, rss_eflux : ndarray
        RSS flux and error arrays (n_fibers, n_wave).
        These should already have the desired fiber masks applied.
    LSF_rss : ndarray
        Instrumental LSF for each fiber and wavelength.
    auto_z_min, auto_z_max : float
        Redshift limits for S/N selection.
    SN_threshold : float
        Minimum peak S/N for selecting fibers.
    we_range : tuple
        Emission-line weighting wavelength interval.
    cw_range : tuple
        Continuum weighting wavelength interval.

    Returns
    -------
    dict
        Integrated spectra, error estimate, weighted LSF,
        ratio, difference, and S/N selection information.
    """

    if bad_pixels is None:
        bad_pixels = np.zeros(rss_flux.shape, dtype=bool)
    bad_pixels = np.asarray(bad_pixels, dtype=bool)
    if bad_pixels.shape != rss_flux.shape:
        raise ValueError('bad_pixels shape must match rss_flux')
    valid = (~bad_pixels) & np.isfinite(rss_flux) & np.isfinite(rss_eflux) & (rss_eflux > 0)
    masked_flux = np.where(valid, rss_flux, np.nan)

    # ---------------------------------------------------------
    # 1. Integrated spectrum selected by emission-line S/N
    # ---------------------------------------------------------

    w0 = 6530 * (1 + auto_z_min)
    w1 = 6650 * (1 + auto_z_max)

    mask_line = (wl > w0) & (wl < w1)

    sn = np.full_like(rss_flux, -np.inf, dtype=float)
    np.divide(rss_flux, rss_eflux, out=sn, where=valid)
    SN_map = np.max(sn[:, mask_line], axis=1)

    high_SN_mask = SN_map > SN_threshold
    n_high_SN = np.count_nonzero(high_SN_mask)

    print(
        f'# Fibers with peak S/N > {SN_threshold}: '
        f'{n_high_SN}'
    )

    if n_high_SN < 3:
        print('# Fewer than 3 fibers; using all fibers')
        selected_flux = rss_flux
        selected_eflux = rss_eflux
        selected_bad = bad_pixels
    else:
        selected_flux = rss_flux[high_SN_mask]
        selected_eflux = rss_eflux[high_SN_mask]
        selected_bad = bad_pixels[high_SN_mask]



    # ---------------------------------------------------------
    # Inverse-variance weighted mean and its propagated error
    # ---------------------------------------------------------

    valid = (
        ~selected_bad &
        np.isfinite(selected_flux) &
        np.isfinite(selected_eflux) &
        (selected_eflux > 0)
    )

    # Inverse-variance weights; invalid pixels receive zero weight
    weights = np.zeros_like(selected_eflux, dtype=float)

    np.divide(
        1.0,
        selected_eflux**2,
        out=weights,
        where=valid
    )

    sum_weights = np.sum(weights, axis=0)

    # Weighted mean flux
    weighted_flux = np.sum(
        weights * np.where(valid, selected_flux, 0.0),
        axis=0
    )

    int_flux = np.full(wl.shape, np.nan, dtype=float)

    np.divide(
        weighted_flux,
        sum_weights,
        out=int_flux,
        where=sum_weights > 0
    )

    # Properly propagated uncertainty
    int_eflux = np.full(wl.shape, np.nan, dtype=float)

    np.sqrt(
        np.divide(
            1.0,
            sum_weights,
            out=int_eflux,
            where=sum_weights > 0
        ),
        out=int_eflux
    )



    # ---------------------------------------------------------
    # 2. Emission-line-weighted integrated spectrum
    # ---------------------------------------------------------

    we_min, we_max = we_range
    mask_we = (wl > we_min) & (wl < we_max)

    with np.errstate(invalid='ignore'):
        we = np.nanmean(masked_flux[:, mask_we], axis=1)
    we = np.where(np.isfinite(we), we, 0.0)
    if not np.isfinite(we.sum()) or we.sum() == 0:
        we = np.ones_like(we)
    we_spec = we[:, np.newaxis] * np.ones_like(rss_flux) / we.sum()
    int_flux_we = nanaverage(masked_flux, we_spec, axis=0)

    # ---------------------------------------------------------
    # 3. Continuum-weighted integrated spectrum
    # ---------------------------------------------------------

    wc_min, wc_max = cw_range
    mask_cw = (wl > wc_min) & (wl < wc_max)

    with np.errstate(invalid='ignore'):
        cw = np.nanmean(masked_flux[:, mask_cw], axis=1)
    cw = np.where(np.isfinite(cw), cw, 0.0)
    if not np.isfinite(cw.sum()) or cw.sum() == 0:
        cw = np.ones_like(cw)
    cw_spec = cw[:, np.newaxis] * np.ones_like(rss_flux) / cw.sum()
    int_flux_cw = nanaverage(masked_flux, cw_spec, axis=0)

    # ---------------------------------------------------------
    # 4. Continuum-weighted instrumental LSF
    # ---------------------------------------------------------

    int_lsf = nanaverage(
        np.where(~bad_pixels & np.isfinite(LSF_rss), LSF_rss, np.nan),
        cw_spec,
        axis=0
    )

    # ---------------------------------------------------------
    # 5. Ratio and difference spectra
    # ---------------------------------------------------------

    with np.errstate(divide='ignore', invalid='ignore'):
        rat_flux_cw = int_flux_we / int_flux_cw

    int_flux_w_ce = int_flux_cw - int_flux_we

    # ---------------------------------------------------------
    # 6. Return products
    # ---------------------------------------------------------

    return {
        'int_flux': int_flux,
        'int_eflux': int_eflux,
        'int_flux_we': int_flux_we,
        'int_flux_cw': int_flux_cw,
        'int_flux_w_ce': int_flux_w_ce,
        'rat_flux_cw': rat_flux_cw,
        'int_lsf': int_lsf,
        'SN_map': SN_map,
        'high_SN_mask': high_SN_mask,
        'we': we,
        'cw': cw
    }


def prepare_integrated_spectrum(m_flux, e_flux, smooth_box=51):
    """
    Prepare the integrated spectrum for RSP fitting.

    Parameters
    ----------
    m_flux : ndarray
        Integrated flux spectrum.
    e_flux : ndarray
        Integrated error spectrum.
    smooth_box : int
        Size of the median filter (default=51).

    Returns
    -------
    m_flux : ndarray
        Cleaned integrated spectrum.
    e_flux : ndarray
        Cleaned error spectrum.
    s_flux : ndarray
        Median-filtered integrated spectrum.
    m_flux_SN : float
        Estimated continuum signal-to-noise ratio.
    m_flux_rss : ndarray
        Integrated spectrum as a 2D RSS array (1, n_wave).
    """

    m_flux = np.array(m_flux, copy=True)
    e_flux = np.array(e_flux, copy=True)

    # Replace non-finite values
    m_flux[~np.isfinite(m_flux)] = np.nanmedian(m_flux)
    e_flux[~np.isfinite(e_flux)] = np.nanmedian(e_flux)

    # Replace negative fluxes with small positive values
    mask_negative = m_flux < 0
    m_flux[mask_negative] = e_flux[mask_negative] / 1e6

    # Calculate continuum S/N
    i0 = int(0.25 * m_flux.shape[0])
    i1 = int(0.35 * m_flux.shape[0])

    m_flux_median = np.nanmedian(m_flux[i0:i1])
    m_eflux_median = np.nanmedian(e_flux[i0:i1])

    m_flux_SN = m_flux_median / m_eflux_median

    print(
        f'# m_flux: {m_flux_median}, '
        f'm_eflux: {m_eflux_median} => {m_flux_SN}'
    )

    # Median-filtered spectrum
    s_flux = median_filter(m_flux, size=smooth_box)

    # Convert to RSS format (one integrated spectrum)
    m_flux_rss = m_flux[np.newaxis, :].copy()

    print(f'# Number of spectral pixels : {m_flux.shape[0]}')

    return m_flux, e_flux, s_flux, m_flux_SN, m_flux_rss


#-------------------------------------------------------------------------
# Detection of redshift
#-------------------------------------------------------------------------


REFERENCE_LINES = {
    "[OII]3726": 3726.03,
    "[OII]3729": 3728.82,
    "Hepsilon": 3970.07,
    "Hdelta": 4101.77,
    "Hgamma": 4340.49,
    "HeII4686": 4685.68,
    "Hbeta": 4861.36,
    "[OIII]4959": 4958.91,
    "[OIII]5007": 5006.84,
    "[OI]6300": 6300.30,
    "[NII]6548": 6548.05,
    "Halpha": 6562.85,
    "[NII]6583": 6583.45,
    "HeI6678": 6678.15,
    "[SII]6716": 6716.44,
    "[SII]6731": 6730.82,
    "[ArV]7006": 7005.67,
    "[SIII]9069": 9068.62,
    "[SIII]9531": 9530.62,
}

LINE_WEIGHTS = {
    "[OII]3726": 1.5,
    "[OII]3729": 1.5,
    "Hepsilon": 0.8,
    "Hdelta": 1.0,
    "Hgamma": 1.2,
    "HeII4686": 1.0,
    "Hbeta": 2.5,
    "[OIII]4959": 2.0,
    "[OIII]5007": 3.0,
    "[OI]6300": 1.2,
    "[NII]6548": 1.5,
    "Halpha": 3.0,
    "[NII]6583": 2.5,
    "HeI6678": 0.8,
    "[SII]6716": 1.8,
    "[SII]6731": 1.8,
    "[ArV]7006": 0.6,
    "[SIII]9069": 1.8,
    "[SIII]9531": 2.3,
}

LINE_GROUPS = [
    (
        "OII_doublet",
        ["[OII]3726", "[OII]3729"],
        2.0,
    ),
    (
        "OIII_doublet",
        ["[OIII]4959", "[OIII]5007"],
        3.0,
    ),
    (
        "Halpha_NII_complex",
        ["[NII]6548", "Halpha", "[NII]6583"],
        4.0,
    ),
    (
        "SII_doublet",
        ["[SII]6716", "[SII]6731"],
        2.5,
    ),
    (
        "SIII_doublet",
        ["[SIII]9069", "[SIII]9531"],
        2.5,
    ),
]


def estimate_integrated_redshift(
    wl__w,
    int_flux_cw,
    e_flux,
    args,
    m_flux_SN=None
):
    """
    Estimate redshift from the continuum-weighted integrated spectrum.

    Uses spectral decomposition, emission-line peak detection,
    and weighted emission-line matching with group bonuses.

    Parameters
    ----------
    wl__w : ndarray
        Observed wavelength array in Angstrom.
    int_flux_cw : ndarray
        Continuum-weighted integrated spectrum.
    e_flux : ndarray
        Error spectrum associated with the integrated spectrum.
    args : argparse.Namespace
        DAP configuration parameters.
    m_flux_SN : float, optional
        Integrated spectrum S/N diagnostic.

    Returns
    -------
    args : argparse.Namespace
        Updated DAP configuration.
    out_decompose_flux_rss : dict
        Results of spectral decomposition.
    find_z_out_w : sequence
        Ranked weighted-redshift solutions.
    z_guess : float
        Best estimated redshift, or configured fallback.
    """

    print('# Searching for the guess redshift using the integrated spectrum...')

    # ---------------------------------------------------------
    # 1. Initialize configuration
    # ---------------------------------------------------------

    n_narrow = int(getattr(args, "n_narrow", 3))
    n_medium = int(getattr(args, "n_medium", 5))
    n_broad = int(getattr(args, "n_broad", 21))
    n_iter = int(getattr(args, "n_iter", 5))

    peak_threshold = float(getattr(args, "peak_threshold", 1))
    n_cut_peaks = float(getattr(args, "n_cut_peaks", 2))

    velocity_tolerance = float(
        getattr(args, "velocity_tolerance", 15.0)
    )
    min_matches = int(getattr(args, "min_matches", 3))

    auto_redshift = getattr(args, "auto_redshift", False)

    z_min = args.auto_z_min
    z_max = args.auto_z_max

    print(
        f'# Using n_narrow={n_narrow}, '
        f'n_medium={n_medium}, '
        f'n_broad={n_broad}, '
        f'n_iter={n_iter}'
    )

    # ---------------------------------------------------------
    # 2. Decompose integrated spectrum
    # ---------------------------------------------------------

    print(
        '# Decomposing the integrated spectrum '
        'background and emission lines...'
    )

    out_decompose_flux_rss = decompose_spectrum(
        wl__w,
        int_flux_cw,
        n_narrow=n_narrow,
        n_medium=n_medium,
        n_broad=n_broad,
        n_iter=n_iter
    )

    # ---------------------------------------------------------
    # 3. Find emission-line peaks
    # ---------------------------------------------------------

    print(
        '# Decomposition finished. Searching for peaks '
        'in the emission-line spectrum...'
    )

    i_peak, peaks, w_peaks, f_peaks = peak_finder(
        wl__w,
        out_decompose_flux_rss["eline_only"],
        2,
        peak_threshold=peak_threshold,
        dmin=0.3,
        plot=args.plot,
        verbose=args.verbose
    )

    w_peaks = np.asarray(w_peaks)
    f_peaks = np.asarray(f_peaks)

    # ---------------------------------------------------------
    # 4. Select significant emission-line peaks
    # ---------------------------------------------------------

    peak_indices = np.array([
        np.argmin(np.abs(wl__w - w))
        for w in w_peaks
    ], dtype=int)

    mask_good_peaks = (
        np.isfinite(w_peaks) &
        np.isfinite(f_peaks) &
        (f_peaks > 0) &
        (f_peaks > e_flux[peak_indices] * n_cut_peaks)
    )

    n_good_peaks = np.count_nonzero(mask_good_peaks)

    print(
        f'# Found {len(w_peaks)} peaks, '
        f'{n_good_peaks} are above the threshold '
        f'of {n_cut_peaks} times the error spectrum'
    )

    obs_waves = w_peaks[mask_good_peaks]

    # ---------------------------------------------------------
    # 5. Prepare reference emission lines
    # ---------------------------------------------------------

    rest_names = list(REFERENCE_LINES.keys())

    rest_waves = np.array(
        list(REFERENCE_LINES.values()),
        dtype=float
    )
    #print(f'# Observed peaks wavelengths: {obs_waves}')
    #print(f'# Reference lines: {rest_waves}')

    # ---------------------------------------------------------
    # 6. Weighted redshift search
    # ---------------------------------------------------------

    print('# Searching for weighted redshift solutions...')
    print(f'# Redshift interval: {z_min}, {z_max}')
    print(f'# Velocity tolerance: {velocity_tolerance} km/s')
    print(f'# Minimum line matches: {min_matches}')

    print(f'{z_min} < z < {z_max}, {velocity_tolerance} km/s, {min_matches} matches')
    if len(obs_waves) >= min_matches:

        find_z_out_w = find_redshift_weights(
            rest_waves=rest_waves,
            obs_waves=obs_waves,
            rest_names=rest_names,
            line_weights=LINE_WEIGHTS,
            line_groups=LINE_GROUPS,
            zmin=z_min,
            zmax=z_max,
            velocity_tolerance=velocity_tolerance,
            min_matches=min_matches
        )

    else:

        print(
            '# Insufficient significant peaks for '
            'weighted redshift determination'
        )

        find_z_out_w = []

    # ---------------------------------------------------------
    # 7. Report the best solutions
    # ---------------------------------------------------------

    if bool(getattr(args, 'verbose', False)):
        print(f'# Number of redshift solutions: {len(find_z_out_w)}')

        for rank, solution in enumerate(find_z_out_w[:5], start=1):
            print_solution(solution, rank)

    # ---------------------------------------------------------
    # 8. Select best redshift
    # ---------------------------------------------------------

    if len(find_z_out_w) > 0:

        best_solution = find_z_out_w[0]
        z_guess = best_solution.z

        print(f'# Best weighted solution:')
        print(f'# Redshift = {best_solution.z}')
        print(f'# Velocity = {best_solution.velocity_kms} km/s')
        print(f'# RMS = {best_solution.rms_velocity_kms} km/s')

    else:
 
        z_guess = args.redshift[0]

        print(
            '# No weighted redshift solution found. '
            f'Using input redshift: {z_guess}'
        )

    print(f'# Guess redshift = {z_guess}')

    # ---------------------------------------------------------
    # 9. Update DAP parameters if requested
    # ---------------------------------------------------------

    if auto_redshift:

        auto_z = z_guess

        args.redshift[0] = auto_z
        delta_redshift = args.redshift[3] - args.redshift[2]
        args.redshift[2] = auto_z - delta_redshift / 2
        args.redshift[3] = auto_z + delta_redshift / 2

        args.w_range_nl[0] *= (1 + auto_z)
        args.w_range_nl[1] *= (1 + auto_z)

        print(f'# Adopted automatic redshift: {auto_z}')
        print(f'# Updated redshift limits: {args.redshift[2:4]}')
        print(f'# Updated NL wavelength range: {args.w_range_nl}')

    else:

        print(
            '# Automatic redshift disabled. '
            f'Keeping input redshift: {args.redshift[0]}'
        )

    return args, out_decompose_flux_rss, find_z_out_w, z_guess







def estimate_integrated_redshift(
    wl__w,
    int_flux_cw,
    e_flux,
    args,
    m_flux_SN=None
):
    """
    Estimate redshift from the continuum-weighted integrated spectrum.

    Uses spectral decomposition, emission-line peak detection,
    and weighted emission-line matching with group bonuses.

    Parameters
    ----------
    wl__w : ndarray
        Observed wavelength array in Angstrom.
    int_flux_cw : ndarray
        Continuum-weighted integrated spectrum.
    e_flux : ndarray
        Error spectrum associated with the integrated spectrum.
    args : argparse.Namespace
        DAP configuration parameters.
    m_flux_SN : float, optional
        Integrated spectrum S/N diagnostic.

    Returns
    -------
    args : argparse.Namespace
        Updated DAP configuration.
    out_decompose_flux_rss : dict
        Results of spectral decomposition.
    find_z_out_w : sequence
        Ranked weighted-redshift solutions.
    z_guess : float
        Best estimated redshift, or configured fallback.
    """

    print('# Searching for the guess redshift using the integrated spectrum...')

    # ---------------------------------------------------------
    # 1. Initialize configuration
    # ---------------------------------------------------------

    n_narrow = int(getattr(args, "n_narrow", 3))
    n_medium = int(getattr(args, "n_medium", 5))
    n_broad = int(getattr(args, "n_broad", 21))
    n_iter = int(getattr(args, "n_iter", 5))

    peak_threshold = float(getattr(args, "peak_threshold", 1))
    n_cut_peaks = float(getattr(args, "n_cut_peaks", 2))

    velocity_tolerance = float(
        getattr(args, "velocity_tolerance", 15.0)
    )
    min_matches = int(getattr(args, "min_matches", 3))

    auto_redshift = getattr(args, "auto_redshift", False)

    z_min = args.auto_z_min
    z_max = args.auto_z_max

    print(
        f'# Using n_narrow={n_narrow}, '
        f'n_medium={n_medium}, '
        f'n_broad={n_broad}, '
        f'n_iter={n_iter}'
    )

    # ---------------------------------------------------------
    # 2. Decompose integrated spectrum
    # ---------------------------------------------------------

    print(
        '# Decomposing the integrated spectrum '
        'background and emission lines...'
    )

    # The DRP pixel mask can leave entire wavelength columns with no valid
    # fibers. Their integrated weighted mean is NaN. Interpolate those gaps
    # BEFORE decompose_spectrum, because peak_finder uses the standard
    # deviation of the resulting emission-line spectrum (which must be finite).
    # This is a DETECTION-ONLY copy: never change the input or its DRP mask.
    wave_z = np.asarray(wl__w, dtype=float)
    flux_z = np.asarray(int_flux_cw, dtype=float).copy()
    if wave_z.ndim != 1 or flux_z.shape != wave_z.shape:
        raise ValueError('wl__w and int_flux_cw must be matching 1D arrays')
    if not np.all(np.isfinite(wave_z)) or not np.all(np.diff(wave_z) > 0):
        raise ValueError('Peak detection requires finite increasing wavelengths')
    valid_input_z = np.isfinite(flux_z)
    if np.count_nonzero(valid_input_z) < 3:
        raise ValueError('Too few valid samples to estimate integrated redshift')
    n_invalid_input_z = np.count_nonzero(~valid_input_z)
    if n_invalid_input_z:
        flux_z[~valid_input_z] = np.interp(
            wave_z[~valid_input_z], wave_z[valid_input_z],
            flux_z[valid_input_z]
        )
        print(f'# Redshift detection: interpolated {n_invalid_input_z} '
              'invalid integrated-spectrum pixels (detection only)')

    out_decompose_flux_rss = decompose_spectrum(
        wl__w,
        flux_z,
        n_narrow=n_narrow,
        n_medium=n_medium,
        n_broad=n_broad,
        n_iter=n_iter
    )

    # ---------------------------------------------------------
    # 3. Find emission-line peaks
    # ---------------------------------------------------------

    print(
        '# Decomposition finished. Searching for peaks '
        'in the emission-line spectrum...'
    )

    eline_z = np.asarray(
        out_decompose_flux_rss['eline_only'], dtype=float
    ).copy()
    valid_eline_z = np.isfinite(eline_z)
    if np.count_nonzero(valid_eline_z) < 3:
        raise ValueError('Decomposed emission-line spectrum is not finite')
    if not np.all(valid_eline_z):
        eline_z[~valid_eline_z] = np.interp(
            wave_z[~valid_eline_z], wave_z[valid_eline_z],
            eline_z[valid_eline_z]
        )
        print(f'# Redshift detection: interpolated '
              f'{np.count_nonzero(~valid_eline_z)} non-finite residual pixels')
    # Do not use any repaired center as a measured line detection.
    valid_centers_z = valid_input_z & valid_eline_z
    print(f'# Redshift detection: finite emission spectrum='
          f'{np.isfinite(eline_z).sum()}/{eline_z.size}, '
          f'std={np.std(eline_z):.5g}, '
          f'max={np.max(eline_z):.5g}')

    i_peak, peaks, w_peaks, f_peaks = peak_finder(
        wl__w,
        eline_z,
        2,
        peak_threshold=peak_threshold,
        dmin=0.3,
        plot=args.plot,
        verbose=0
    )

    w_peaks = np.atleast_1d(np.asarray(w_peaks, dtype=float))
    f_peaks = np.atleast_1d(np.asarray(f_peaks, dtype=float))

    # Backup for cases where pyFIT3D's median-filter peak finder suppresses
    # narrow emission lines (or returns fewer than the required matches).
    # Only activate on failure; the standard peak detector remains primary.
    if len(w_peaks) < min_matches:
        from scipy.signal import find_peaks as scipy_find_peaks
        from scipy.ndimage import median_filter as scipy_median_filter

        smooth_z = scipy_median_filter(eline_z, size=3, mode='nearest')
        background_z = np.median(smooth_z)
        scatter_z = 1.4826 * np.median(np.abs(smooth_z - background_z))
        good_noise_z = (np.isfinite(e_flux) & (np.asarray(e_flux) > 0))
        noise_z = (float(np.median(np.asarray(e_flux)[good_noise_z]))
                   if np.any(good_noise_z) else 0.0)
        threshold_z = max(peak_threshold * scatter_z,
                          n_cut_peaks * noise_z, 0.0)
        candidate_idx_z, _ = scipy_find_peaks(
            smooth_z,
            height=background_z + threshold_z,
            prominence=max(scatter_z, 0.25 * noise_z, 1e-12),
            distance=2
        )

        # The redshift interval is known; retain candidates compatible
        # with at least one catalog line, allowing pixel and velocity margins.
        if candidate_idx_z.size:
            rest_z = np.asarray(list(REFERENCE_LINES.values()), dtype=float)
            min_wave_z = rest_z * (1.0 + z_min)
            max_wave_z = rest_z * (1.0 + z_max)
            margin_z = (2 * np.median(np.diff(wave_z)) +
                        max_wave_z * velocity_tolerance / 299792.458)
            candidate_wave_z = wave_z[candidate_idx_z]
            plausible_z = np.any(
                (candidate_wave_z[:, None] >=
                 min_wave_z[None, :] - margin_z[None, :]) &
                (candidate_wave_z[:, None] <=
                 max_wave_z[None, :] + margin_z[None, :]), axis=1
            )
            candidate_idx_z = candidate_idx_z[
                plausible_z & valid_centers_z[candidate_idx_z]
            ]

        if candidate_idx_z.size:
            # Combine detector results and backup, without duplicate pixels.
            old_idx_z = np.searchsorted(wave_z, w_peaks)
            old_idx_z = np.clip(old_idx_z, 0, len(wave_z) - 1)
            all_idx_z = np.unique(np.r_[old_idx_z, candidate_idx_z])
            w_peaks = wave_z[all_idx_z]
            f_peaks = eline_z[all_idx_z]
        print(f'# Backup peak detection: {len(candidate_idx_z)} '
              'additional candidate wavelengths')

    # ---------------------------------------------------------
    # 4. Select significant emission-line peaks
    # ---------------------------------------------------------

    peak_indices = np.array([
        np.argmin(np.abs(wl__w - w))
        for w in w_peaks
    ], dtype=int)

    mask_good_peaks = (
        np.isfinite(w_peaks) &
        np.isfinite(f_peaks) &
        (f_peaks > 0) &
        valid_centers_z[peak_indices] &
        np.isfinite(e_flux[peak_indices]) &
        (e_flux[peak_indices] > 0) &
        (f_peaks > e_flux[peak_indices] * n_cut_peaks)
    )

    n_good_peaks = np.count_nonzero(mask_good_peaks)

    print(
        f'# Found {len(w_peaks)} peaks, '
        f'{n_good_peaks} are above the threshold '
        f'of {n_cut_peaks} times the error spectrum'
    )

    obs_waves = w_peaks[mask_good_peaks]

    # ---------------------------------------------------------
    # 5. Prepare reference emission lines
    # ---------------------------------------------------------

    rest_names = list(REFERENCE_LINES.keys())

    rest_waves = np.array(
        list(REFERENCE_LINES.values()),
        dtype=float
    )

    # ---------------------------------------------------------
    # 6. Weighted redshift search
    # ---------------------------------------------------------

    print('# Searching for weighted redshift solutions...')
    print(f'# Redshift interval: {z_min}, {z_max}')
    print(f'# Velocity tolerance: {velocity_tolerance} km/s')
    print(f'# Minimum line matches: {min_matches}')

#    print(f'# Reference line wavelengths: {rest_waves}')
#    print(f'# Observed peak wavelengths: {obs_waves}')


    if len(obs_waves) >= min_matches:

        find_z_out_w = find_redshift_weights(
            rest_waves=rest_waves,
            obs_waves=obs_waves,
            rest_names=rest_names,
            line_weights=LINE_WEIGHTS,
            line_groups=LINE_GROUPS,
            zmin=z_min,
            zmax=z_max,
            velocity_tolerance=velocity_tolerance,
            min_matches=min_matches
        )

    else:

        print(
            '# Insufficient significant peaks for '
            'weighted redshift determination'
        )

        find_z_out_w = []

    # ---------------------------------------------------------
    # 7. Report the best solutions
    # ---------------------------------------------------------

    if bool(getattr(args, 'verbose', False)):
        print(f'# Number of redshift solutions: {len(find_z_out_w)}')

        for rank, solution in enumerate(find_z_out_w[:5], start=1):
            print_solution(solution, rank)

    # ---------------------------------------------------------
    # 8. Select best redshift
    # ---------------------------------------------------------

    if len(find_z_out_w) > 0:

        best_solution = find_z_out_w[0]
        z_guess = best_solution.z

        print(f'# Best weighted solution:')
        print(f'# Redshift = {best_solution.z}')
        print(f'# Velocity = {best_solution.velocity_kms} km/s')
        print(f'# RMS = {best_solution.rms_velocity_kms} km/s')

    else:

        z_guess = args.redshift[0]

        print(
            '# No weighted redshift solution found. '
            f'Using input redshift: {z_guess}'
        )

    print(f'# Guess redshift = {z_guess}')

    # ---------------------------------------------------------
    # 9. Update DAP parameters if requested
    # ---------------------------------------------------------

    if auto_redshift:

        auto_z = z_guess

        args.redshift[0] = auto_z
        delta_redshift = args.redshift[3] - args.redshift[2]
        args.redshift[2] = auto_z - delta_redshift / 2
        args.redshift[3] = auto_z + delta_redshift / 2

#        args.w_range_nl[0] *= (1 + auto_z)
#        args.w_range_nl[1] *= (1 + auto_z)

        print(f'# Adopted automatic redshift: {auto_z}')
        print(f'# Updated redshift limits: {args.redshift[2:4]}')
        print(f'# Updated NL wavelength range: {args.w_range_nl}')

    else:

        print(
            '# Automatic redshift disabled. '
            f'Keeping input redshift: {args.redshift[0]}'
        )

    return args, out_decompose_flux_rss, find_z_out_w, z_guess


def estimate_integrated_redshift_NW(
    wl__w,
    int_flux_cw,
    e_flux,
    tab_el_strong,
    args,
    m_flux_SN=None
):
    """
    Estimate the redshift from emission-line peaks in the
    continuum-weighted integrated spectrum.

    The function:
      1. Decomposes the integrated spectrum.
      2. Detects peaks in the emission-line component.
      3. Selects peaks above the error threshold.
      4. Matches observed peaks to known emission lines.
      5. Updates the fitting parameters if auto_redshift is enabled.

    Parameters
    ----------
    wl__w : ndarray
        Observed wavelength array in Angstrom.
    int_flux_cw : ndarray
        Continuum-weighted integrated spectrum.
    e_flux : ndarray
        Error spectrum associated with the integrated spectrum.
    tab_el_strong : astropy.table.Table
        Table containing the rest wavelengths ('wl') and
        identifiers ('id') of strong emission lines.
    args : argparse.Namespace
        DAP configuration parameters.
    m_flux_SN : float, optional
        Integrated spectrum S/N diagnostic.

    Returns
    -------
    args : argparse.Namespace
        Configuration with potentially updated redshift
        and nonlinear fitting wavelength ranges.
    out_decompose_flux_rss : dict
        Spectral decomposition results.
    find_z_out : sequence
        Candidate redshift solutions.
    auto_z : float
        Estimated or adopted redshift.
    """

    # ---------------------------------------------------------
    # 1. Initialize parameters
    # ---------------------------------------------------------

    print('# Searching for the guess redshift using the integrated spectrum...')

    n_narrow = int(getattr(args, "n_narrow", 3))
    n_medium = int(getattr(args, "n_medium", 5))
    n_broad = int(getattr(args, "n_broad", 21))
    n_iter = int(getattr(args, "n_iter", 5))

    peak_threshold = float(getattr(args, "peak_threshold", 1))
    n_cut_peaks = float(getattr(args, "n_cut_peaks", 2))

    auto_redshift = getattr(args, "auto_redshift", False)

    print(
        f'# Using n_narrow={n_narrow}, '
        f'n_medium={n_medium}, '
        f'n_broad={n_broad}, '
        f'n_iter={n_iter}'
    )

    # ---------------------------------------------------------
    # 2. Decompose the integrated spectrum
    # ---------------------------------------------------------

    print(
        '# Decomposing the integrated spectrum '
        'background and emission lines...'
    )

    out_decompose_flux_rss = decompose_spectrum(
        wl__w,
        int_flux_cw,
        n_narrow=n_narrow,
        n_medium=n_medium,
        n_broad=n_broad,
        n_iter=n_iter
    )

    # ---------------------------------------------------------
    # 3. Find emission-line peaks
    # ---------------------------------------------------------

    print(
        '# Decomposition finished. Searching for peaks '
        'in the emission line spectrum...'
    )

    i_peak, peaks, w_peaks, f_peaks = peak_finder(
        wl__w,
        out_decompose_flux_rss["eline_only"],
        2,
        peak_threshold=peak_threshold,
        dmin=0.3,
        plot=args.plot,
        verbose=0
    )

    # ---------------------------------------------------------
    # 4. Select statistically significant peaks
    # ---------------------------------------------------------

    peak_indices = np.array([
        np.argmin(np.abs(wl__w - w))
        for w in w_peaks
    ], dtype=int)

    mask_good_peaks = (
        (f_peaks > 0) &
        (f_peaks > e_flux[peak_indices] * n_cut_peaks)
    )

    n_good_peaks = np.count_nonzero(mask_good_peaks)

    print(
        f'# Found {len(w_peaks)} peaks, '
        f'{n_good_peaks} are above the threshold '
        f'of {n_cut_peaks} times the error spectrum'
    )

    # ---------------------------------------------------------
    # 5. Identify emission lines and estimate redshift
    # ---------------------------------------------------------

    find_z_out = find_redshift_elines(
        rest_waves=tab_el_strong['wl'],
        obs_waves=w_peaks[mask_good_peaks],
        rest_names=tab_el_strong['id'],
        zmin=args.auto_z_min,
        zmax=args.auto_z_max,
        velocity_tolerance=15,
        min_matches=1
    )

    print(
        f'# Redshift search range: '
        f'{args.auto_z_min}, {args.auto_z_max}'
    )

    if len(find_z_out) > 0:

        best_solution = find_z_out[0]

        if bool(getattr(args, 'verbose', False)):
            print_solution(best_solution, 1)

        print("Best solution, z  =", best_solution.z)
        print("Best solution, vel=", best_solution.velocity_kms)
        print("Best solution, rms=", best_solution.rms_velocity_kms)

    # ---------------------------------------------------------
    # 6. Adopt the redshift
    # ---------------------------------------------------------

    if auto_redshift:

        if len(find_z_out) > 0:

            auto_z = find_z_out[0].z

            print(
                f'# Automatic redshift adopted: {auto_z}'
            )

        else:

            auto_z = args.redshift[0]

            print(
                '# No redshift solution found, '
                f'using input redshift {auto_z}'
            )

            print(
                f'# No auto_z peaks found, '
                f'SN={m_flux_SN}; using input parameters'
            )

        # -----------------------------------------------------
        # 7. Update redshift and fitting wavelength ranges
        # -----------------------------------------------------

        args.redshift[0] = auto_z

        args.redshift[2] = (
            args.redshift[2] * (1 + auto_z) + auto_z
        )

        args.redshift[3] = (
            args.redshift[3] * (1 + auto_z) + auto_z
        )

        args.w_range_nl[0] *= (1 + auto_z)
        args.w_range_nl[1] *= (1 + auto_z)

        print(
            f'# Auto_z derivation '
            f'({args.auto_z_min}, {args.auto_z_max}, '
            f'{getattr(args, "auto_z_d", 0.00001)}): {auto_z}'
        )

    else:

        auto_z = args.redshift[0]

        print(
            '# Auto_z derivation disabled, '
            f'using input redshift {auto_z}'
        )

    return args, out_decompose_flux_rss, find_z_out, auto_z

#----------------------------------------------------------------
def create_rsp_table(id_fib, SPS):
    """
    Create an Astropy table with the RSP fitting results.

    Parameters
    ----------
    id_fib : int
        Fiber identifier.
    SPS : StellarSynthesis
        Stellar population synthesis fitting object.

    Returns
    -------
    tab_rsp : astropy.table.Table
        Single-row table containing the fiber ID and
        the RSP fitting parameters.
    """

    # Extract RSP fitting results
    rsp_cols, a_rsp = SPS.output_return()

    # Define column names
    a_rsp_cols = ['id_fib'] + rsp_cols.split(',')

    # Combine fiber ID and fitting results
    a_rsp = np.concatenate(([id_fib], a_rsp.flatten()))

    # Create a single-row Astropy table
    tab_rsp = Table(
        rows=[a_rsp],
        names=a_rsp_cols,
        dtype=['int64'] + ['float64'] * (len(a_rsp_cols) - 1)
    )

    return tab_rsp


def create_coeffs_table(id_fib, SPS):
    """
    Create an Astropy table with the RSP Monte Carlo coefficients.

    Parameters
    ----------
    id_fib : int
        Fiber identifier.
    SPS : StellarSynthesis
        Stellar population synthesis fitting object.

    Returns
    -------
    tab_coeffs_MC : astropy.table.Table
        Table containing the RSP coefficients and their
        associated fiber identifier.
    """

    # Extract Monte Carlo coefficients
    coeffs_cols, a_coeffs_MC = SPS.output_coeffs_MC_return()
    # Define column names
    a_coeffs_cols = coeffs_cols.split(',')
    a_coeffs_cols[0] = 'rsp'

    # Create coefficients table
    tab_coeffs_MC = Table(
        a_coeffs_MC,
        names=a_coeffs_cols
    )

    # Ensure RSP index is an integer
    tab_coeffs_MC['rsp'] = tab_coeffs_MC['rsp'].astype('int64')

    # Add fiber identifier as the first column
    tab_coeffs_MC.add_column(
        np.full(len(tab_coeffs_MC), id_fib, dtype='int64'),
        name='id_fib',
        index=0
    )

    return tab_coeffs_MC


def create_elines_table(id_fib, SPS):
    """
    Create an Astropy table containing the emission-line fitting
    results from the SPS configuration.

    Parameters
    ----------
    id_fib : int
        Fiber identifier.
    SPS : StellarSynthesis
        Stellar population synthesis fitting object.

    Returns
    -------
    tab_elines : astropy.table.Table
        Table containing emission-line and polynomial model
        parameters with columns:

        id_fib, model, wl, flux, e_flux,
        disp, e_disp, vel, e_vel
    """

    rows = []

    # ---------------------------------------------------------
    # Extract all emission-line fitting systems
    # ---------------------------------------------------------

    for system, elcf in zip(
        SPS.config.systems,
        SPS.config.systems_config
    ):

        EL = system['EL']

        model_types = EL.config.model_types
        fit_params = EL.final_fit_params_mean
        fit_errors = EL.final_fit_params_std

        # -----------------------------------------------------
        # Extract individual models within each system
        # -----------------------------------------------------

        for model_type, params, errors in zip(
            model_types,
            fit_params,
            fit_errors
        ):

            params = np.asarray(params)
            errors = np.asarray(errors)

            if model_type == 'eline':

                wl = float(params[0])
                flux = float(params[1])
                e_flux = float(errors[1])
                disp = float(params[2])
                e_disp = float(errors[2])
                vel = float(params[3])
                e_vel = float(errors[3])

            elif model_type == 'poly1d':

                wl = float(params[0])

                flux = 0.0
                e_flux = 0.0
                disp = 0.0
                e_disp = 0.0
                vel = 0.0
                e_vel = 0.0

            else:

                # Preserve other model types if present
                wl = float(params[0])

                flux = 0.0
                e_flux = 0.0
                disp = 0.0
                e_disp = 0.0
                vel = 0.0
                e_vel = 0.0

            rows.append((
                int(id_fib),
                str(model_type),
                wl,
                flux,
                e_flux,
                disp,
                e_disp,
                vel,
                e_vel
            ))

    # ---------------------------------------------------------
    # Create Astropy table
    # ---------------------------------------------------------

    tab_elines = Table(
        rows=rows,
        names=[
            'id_fib',
            'model',
            'wl',
            'flux',
            'e_flux',
            'disp',
            'e_disp',
            'vel',
            'e_vel'
        ],
        dtype=[
            'int64',
            'U9',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64'
        ]
    )

    return tab_elines


def create_elines_table_EL(id_fib, EL):
    """
    Create an Astropy table containing the emission-line fitting
    results from the SPS configuration.

    Parameters
    ----------
    id_fib : int
        Fiber identifier.
    SPS : StellarSynthesis
        Stellar population synthesis fitting object.

    Returns
    -------
    tab_elines : astropy.table.Table
        Table containing emission-line and polynomial model
        parameters with columns:

        id_fib, model, wl, flux, e_flux,
        disp, e_disp, vel, e_vel
    """

    rows = []

    # ---------------------------------------------------------
    # Extract all emission-line fitting systems
    # ---------------------------------------------------------

   

    model_types = EL.config.model_types
    fit_params = EL.final_fit_params_mean
    fit_errors = EL.final_fit_params_std

    # -----------------------------------------------------
    # Extract individual models within each system
    # -----------------------------------------------------

    for model_type, params, errors in zip(
        model_types,
        fit_params,
        fit_errors
    ):

        params = np.asarray(params)
        errors = np.asarray(errors)

        wl = float(params[0])
        flux = float(params[1])
        e_flux = float(errors[1])
        disp = float(params[2])
        e_disp = float(errors[2])
        vel = float(params[3])
        e_vel = float(errors[3])


        rows.append((
            int(id_fib),
            str(model_type),
            wl,
            flux,
            e_flux,
            disp,
            e_disp,
            vel,
            e_vel
        ))

    # ---------------------------------------------------------
    # Create Astropy table
    # ---------------------------------------------------------

    tab_elines = Table(
        rows=rows,
        names=[
            'id_fib',
            'model',
            'wl',
            'flux',
            'e_flux',
            'disp',
            'e_disp',
            'vel',
            'e_vel'
        ],
        dtype=[
            'int64',
            'U9',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64',
            'float64'
        ]
    )

    return tab_elines


def create_elines_wide_table_EL(id_fib, EL):
    """
    Create a wide-format emission-line table directly from an EL object.

    Each fiber occupies one row, with six columns per
    emission line: flux, e_flux, disp, e_disp, vel, e_vel.

    Parameters
    ----------
    id_fib : int
        Fiber identifier.
    EL : ConfigEmissionModel
        Fitted emission-line model object.

    Returns
    -------
    tab_PE_ord : astropy.table.Table
        Single-row table with dynamically generated columns:

        id, flux_WL, e_flux_WL, disp_WL, e_disp_WL,
        vel_WL, e_vel_WL

        where WL is the rest wavelength of each emission line.
    """

    # ---------------------------------------------------------
    # 1. Extract emission-line parameters
    # ---------------------------------------------------------

    elines = {}

    model_types = EL.config.model_types
    fit_params = EL.final_fit_params_mean
    fit_errors = EL.final_fit_params_std

    for model_type, params, errors in zip(
        model_types, fit_params, fit_errors
    ):

        # Only emission-line models
        if model_type != 'eline':
            continue

        params = np.asarray(params)
        errors = np.asarray(errors)

        wl = float(params[0])

        elines[wl] = {
            'flux': float(params[1]),
            'e_flux': float(errors[1]),
            'disp': float(params[2]),
            'e_disp': float(errors[2]),
            'vel': float(params[3]),
            'e_vel': float(errors[3])
        }

    # ---------------------------------------------------------
    # 2. Generate columns dynamically
    # ---------------------------------------------------------

    columns = {'id': [int(id_fib)]}

    for wl in sorted(elines):

        for param in (
            'flux', 'e_flux',
            'disp', 'e_disp',
            'vel', 'e_vel'
        ):

            columns[f'{param}_{wl}'] = [
                elines[wl][param]
            ]

    # ---------------------------------------------------------
    # 3. Create and return Astropy table
    # ---------------------------------------------------------

    tab_PE_ord = Table(columns)

    return tab_PE_ord


def create_elines_wide_table(id_fib, SPS):
    """
    Create a wide-format emission-line table directly from SPS.

    Emission lines and their wavelengths are obtained dynamically
    from SPS.config.systems.

    Parameters
    ----------
    id_fib : int
        Original fiber identifier.
    SPS : StellarSynthesis
        Stellar population synthesis fitting object.

    Returns
    -------
    tab_PE_ord : astropy.table.Table
        Single-row table containing six parameters per
        emission line, with dynamically generated column names.
    """

    # ---------------------------------------------------------
    # 1. Extract emission-line parameters
    # ---------------------------------------------------------

    elines = {}

    for system in SPS.config.systems:

        EL = system['EL']

        model_types = EL.config.model_types
        fit_params = EL.final_fit_params_mean
        fit_errors = EL.final_fit_params_std

        for model_type, params, errors in zip(
            model_types, fit_params, fit_errors
        ):

            if model_type != 'eline':
                continue

            params = np.asarray(params)
            errors = np.asarray(errors)

            wl = float(params[0])

            elines[wl] = {
                'flux': float(params[1]),
                'e_flux': float(errors[1]),
                'disp': float(params[2]),
                'e_disp': float(errors[2]),
                'vel': float(params[3]),
                'e_vel': float(errors[3])
            }

    # ---------------------------------------------------------
    # 2. Construct columns dynamically
    # ---------------------------------------------------------

    columns = {'id': [int(id_fib)]}

    for wl in sorted(elines):

        for param in (
            'flux', 'e_flux',
            'disp', 'e_disp',
            'vel', 'e_vel'
        ):

            columns[f'{param}_{wl}'] = [
                elines[wl][param]
            ]

    # ---------------------------------------------------------
    # 3. Create Astropy table
    # ---------------------------------------------------------

    tab_PE_ord = Table(columns)

    return tab_PE_ord



def combine_elines_wide_tables(tables):
    """
    Combine single-fiber wide emission-line tables.

    Different sets of emission-line columns are supported.
    Missing measurements are filled with NaN.

    Parameters
    ----------
    tables : list of astropy.table.Table
        Single-row emission-line tables.

    Returns
    -------
    tab_PE_ord : astropy.table.Table
        Combined table, one row per fiber.
    """

    if not tables:
        return Table()

    tab_PE_ord = tab_vstack(
        tables,
        join_type='outer',
        metadata_conflicts='silent'
    )

    # Convert missing entries to NaN
    for col in tab_PE_ord.colnames:

        if col == 'id':
            continue

        if np.ma.isMaskedArray(tab_PE_ord[col]):
            tab_PE_ord[col] = tab_PE_ord[col].filled(np.nan)

    return tab_PE_ord



def create_elines_model_pm(wl, tab_pm, dap_fwhm=1.0, verbose=False):
    """
    Generate emission-line spectra from the parameters stored in tab_pm.

    Parameters
    ----------
    wl : ndarray
        Wavelength array in Angstrom.
    tab_pm : astropy.table.Table
        Table containing fitted emission-line parameters with columns:
        flux_{id}_{wl}_pm
        disp_{id}_{wl}_pm
        vel_{id}_{wl}_pm

        Each row corresponds to one spectrum.
    dap_fwhm : float
    lvmdap_masked_iterative_decomposition.patch    Parameter passed to eline().
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    spec2D_elines_pm : ndarray
        Emission-line model with shape (n_spectra, n_wave).
    """

    wl = np.asarray(wl, dtype=float)

    # ---------------------------------------------------------
    # 1. Identify emission lines dynamically from tab_pm
    # ---------------------------------------------------------

    line_names = [
        col[5:-3]
        for col in tab_pm.colnames
        if col.startswith('flux_') and col.endswith('_pm')
    ]


    # ---------------------------------------------------------
    # 2. Initialize emission-line model
    # ---------------------------------------------------------

    spec2D_elines_pm = np.zeros(
        (len(tab_pm), len(wl)), dtype=float
    )

    if verbose:
        print('##############################################')
        print('# Creating emission-line models from tab_pm')
        print(f'# Number of spectra: {len(tab_pm)}')
        print(f'# Number of lines: {len(line_names)}')

    # ---------------------------------------------------------
    # 3. Generate models
    # ---------------------------------------------------------

    for line_name in line_names:

        # Extract rest wavelength from the column name
        w_now = float(line_name.rsplit('_', 1)[-1])

        col_flux = f'flux_{line_name}_pm'
        col_disp = f'disp_{line_name}_pm'
        col_vel = f'vel_{line_name}_pm'

        for col in (col_disp, col_vel):
            if col not in tab_pm.colnames:
                raise KeyError(f'Missing column: {col}')

        for i in range(len(tab_pm)):

            F = float(tab_pm[col_flux][i])
            D = float(tab_pm[col_disp][i])
            V = float(tab_pm[col_vel][i])

            if not (
                np.isfinite(F) and
                np.isfinite(D) and
                np.isfinite(V) and
                D > 0
            ):
                continue

            spec2D_elines_pm[i] += eline(
                wl,
                w_now,
                F,
                D,
                V,
                dap_fwhm=dap_fwhm
            )

    if verbose:
        print('# Emission-line models completed')

    return spec2D_elines_pm

def create_elines_model_el(wl, tab_pm, dap_fwhm=1.0, verbose=False):
    """
    Generate emission-line spectra from the parameters stored in tab_pm.

    Parameters
    ----------
    wl : ndarray
        Wavelength array in Angstrom.
    tab_pm : astropy.table.Table
        Table containing fitted emission-line parameters with columns:
        flux_{id}_{wl}_pm
        disp_{id}_{wl}_pm
        vel_{id}_{wl}_pm

        Each row corresponds to one spectrum.
    dap_fwhm : float
        Parameter passed to eline().
    verbose : bool
        Print diagnostic information.

    Returns
    -------
    spec2D_elines_pm : ndarray
        Emission-line model with shape (n_spectra, n_wave).
    """

    wl = np.asarray(wl, dtype=float)

    # ---------------------------------------------------------
    # 1. Identify emission lines dynamically from tab_pm
    # ---------------------------------------------------------

    line_names = [
        col[5:]
        for col in tab_pm.colnames
        if col.startswith('flux_')
    ]


    # ---------------------------------------------------------
    # 2. Initialize emission-line model
    # ---------------------------------------------------------

    spec2D_elines_pm = np.zeros(
        (len(tab_pm), len(wl)), dtype=float
    )

    if verbose:
        print('##############################################')
        print('# Creating emission-line models from tab_pm')
        print(f'# Number of spectra: {len(tab_pm)}')
        print(f'# Number of lines: {len(line_names)}')

    # ---------------------------------------------------------
    # 3. Generate models
    # ---------------------------------------------------------

    for line_name in line_names:

        # Extract rest wavelength from the column name
        w_now = float(line_name.rsplit('_', 1)[-1])

        col_flux = f'flux_{line_name}'
        col_disp = f'disp_{line_name}'
        col_vel = f'vel_{line_name}'

        for col in (col_disp, col_vel):
            if col not in tab_pm.colnames:
                raise KeyError(f'Missing column: {col}')

        for i in range(len(tab_pm)):

            F = float(tab_pm[col_flux][i])
            D = float(tab_pm[col_disp][i])
            V = float(tab_pm[col_vel][i])

            if not (
                np.isfinite(F) and
                np.isfinite(D) and
                np.isfinite(V) and
                D > 0
            ):
                continue

            spec2D_elines_pm[i] += eline(
                wl,
                w_now,
                F,
                D,
                V,
                dap_fwhm=dap_fwhm
            )

    if verbose:
        print('# Emission-line models completed')

    return spec2D_elines_pm




def fit_individual_rss_spectrum(
    wl,
    flux,
    eflux,
    lsf,
    tab_PT,
    i_fib,
    tab_el,
    el_clusters,
    hdr_flux,
    args_copy,
    eline_spec=None,
    z_guess=None,
    dap_fwhm=1.0,
    plot=False,
    SPS_master=None,
    list_abs_lines=[3968.469, 3933.663],   
    adopt_st_z = True,
    SN_cut_st_z = 10, 
    verbose = False
):
    """Fit one RSS spectrum and return its DAP products.

    Parameters
    ----------
    wl, flux, eflux, lsf : (n_wave,) arrays
        Observed wavelength, flux, 1-sigma error and fiber-specific
        instrumental Gaussian sigma, respectively (wavelengths in Angstrom).
    tab_PT : astropy.table.Table
        Pointing table, ordered like the *masked* input RSS.
        Column 'id' preserves the original fiber identity.
    i_fib : int
        Zero-based row index within the masked RSS / tab_PT.
    tab_el : astropy.table.Table
        Emission-line reference table for process_elines_spec and EW fits.
    el_clusters : sequence of astropy tables
        Emission-line clusters, each with 'wl' and 'id' columns.
        Sigma guesses are measured from this fiber's lsf, not cluster means.
    hdr_flux : astropy.io.fits.Header
        Header describing the input wavelength sampling.
    args : argparse.Namespace
        Same configuration used for integrated-spectrum processing.
    eline_spec : (n_wave,) array, optional
        Already-decomposed emission-line component. If None, decompose
        this individual spectrum using decompose_spectrum().
   lvmdap_masked_iterative_decomposition.patch z_guess : float, optional
        Adopted redshift for line matching/fitting. Defaults to args.redshift[0].
        This function does NOT estimate redshift.
    dap_fwhm : float
        FWHM parameter forwarded to create_elines_model_pm().
    plot : bool
        Plot a diagnostic graph using matplotlib, if requested.

    Returns
    -------
    tab_rsp, tab_coeffs, tab_elines, tab_PM, tab_fe, hdu_sps
        Per-fiber tables and the SPS FITS HDU output.

    Notes
    -----
    - id_fib is the integer RSS row, while id retains tab_PT['id'][i_fib]
      without coercing IDs such as '6112.3' to integers.
    - Assumes process_elines_spec accepts velocity_tolerance and returns
      'stellar_spectrum' and 'emission_model', as in the current DAP version.
    - Does not refit redshift. The redshift can be supplied per fiber.
    """
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    eflux = np.asarray(eflux, dtype=float)
    lsf = np.asarray(lsf, dtype=float)
    i_fib = int(i_fib)
    args = copy(args_copy)

    if wl.ndim != 1 or any(
        a.ndim != 1 or a.shape != wl.shape for a in (flux, eflux, lsf)
    ):
        raise ValueError("wl, flux, eflux, and lsf must be 1D arrays of equal length")
    if not (0 <= i_fib < len(tab_PT)):
        raise IndexError(f"i_fib={i_fib} is outside tab_PT (n={len(tab_PT)})")
    if 'id' not in tab_PT.colnames:
        raise KeyError("tab_PT must contain an 'id' column")

    pt_id = tab_PT['id'][i_fib]
    z = float(args.redshift[0] if z_guess is None else z_guess)
    # Fitting bounds are absolute redshifts: recenter their old offsets
    # around the adopted redshift without double-counting the old center.
    z_fit_min = z + (float(args.redshift[2]) - float(args.redshift[0]))
    z_fit_max = z + (float(args.redshift[3]) - float(args.redshift[0]))
    verbose = bool(getattr(args, 'verbose', False))

    if verbose:
        print(f"# START single-spectrum fit: i_fib={i_fib}, id={pt_id}, z={z}")

    # ------------------------------------------------------------
    # 1. Decompose this fiber (or reuse a supplied decomposition).
    # ------------------------------------------------------------
    if eline_spec is None:
        decomp = decompose_spectrum(
            wl,
            flux,
            n_narrow=int(getattr(args, 'n_narrow', 3)),
            n_medium=int(getattr(args, 'n_medium', 5)),
            n_broad=int(getattr(args, 'n_broad', 21)),
            n_iter=int(getattr(args, 'n_iter', 5)),
        )
        eline_spec = decomp['eline_only']
    else:
        eline_spec = np.asarray(eline_spec, dtype=float)
        if eline_spec.shape != wl.shape:
            raise ValueError("eline_spec and wl must have equal shapes")

    # ------------------------------------------------------------
    # 2. Subtract preliminary emission-line estimates.
    # ------------------------------------------------------------
    initial = process_elines_spec(
        wl=wl,
        flux=flux,
        eflux=eflux,
        eline_spec=eline_spec,
        lsf=lsf,
        tab_el=tab_el,
        z_guess=z,
        peak_threshold=float(getattr(args, 'peak_threshold', 1.0)),
        n_cut_peaks=float(getattr(args, 'n_cut_peaks', 2.0)),
        smooth_size=int(getattr(args, 'n_medium', 5)),
        velocity_tolerance=float(getattr(args, 'velocity_tolerance', 100.0)),
        verbose=verbose,
    )
    spec_stellar = np.asarray(initial['stellar_spectrum'], dtype=float)

    # ------------------------------------------------------------
    # 2.1  Fit the stellar absorptions to derive the velocity
    # ------------------------------------------------------------


    out_fit_abs_lines = fit_emission_line_system(
        wl=wl,
        flux=spec_stellar,
        eflux=eflux,
        line_wavelengths=list_abs_lines,
        redshift=z,
        fit_padding = 30,
        velocity_range=1.5*float(getattr(args, 'velocity_tolerance', 50.0)),
        sigma_guess=1.2,
        sigma_bounds=(0.5, 2.5),
        error_scale=1,
        tie_kinematics=True,
        add_background=True,
        plot=args.plot,
        bg_order=1, 
        run_mode = 'BOTH',
        verbose=verbose)

    args.redshift[0] = z
    abs_out = out_fit_abs_lines["fit"]
    cf_Ca_HK = out_fit_abs_lines["config"]
    sel_wl_Ca_HK = out_fit_abs_lines["mask"]
    vel_mask_st = True
    vel_map_st = abs_out.final_fit_params_mean[0][3]
    vel_e_map_st = abs_out.final_fit_params_std[0][3]
    z_st = vel_map_st/__c__
    print(f'# Velocity derived from abs. lines: {vel_map_st}')
    print(f'# Redshift derived from abs. lines: {z_st}')
    w_min_st_SN = np.min(list_abs_lines)*(1+z_st)-20
    w_max_st_SN = np.max(list_abs_lines)*(1+z_st)+20
    SN_obs = np.median(spec_stellar[(wl>w_min_st_SN)&(wl<w_max_st_SN)]/eflux[(wl>w_min_st_SN)&(wl<w_max_st_SN)])
    print(f'# S/N in the region of the absorption lines: {SN_obs}')
    if (SN_obs>SN_cut_st_z) & (adopt_st_z):
        print(f'# Adopting the redshift derived from the absorption lines: {z_st}')
        z = z_st
        args.redshift[0] = z_st
        z_fit_min = z + (float(args.redshift[2]) - float(args.redshift[0]))
        z_fit_max = z + (float(args.redshift[3]) - float(args.redshift[0]))
    
    # ------------------------------------------------------------
    # 3. Fit the stellar population model on line-subtracted flux.
    # ------------------------------------------------------------
    _, SPS = auto_rsp_elines_rnd(
        wl, spec_stellar, eflux,
        ssp_file=args.rsp_file,
        ssp_nl_fit_file=args.rsp_nl_file,
        config_file=args.config_file,
        w_min=args.w_range[0],
        w_max=args.w_range[1],
        nl_w_min=args.w_range_nl[0],
        nl_w_max=args.w_range_nl[1],
        mask_list=args.mask_file,
        min=args.flux_scale[0],
        max=args.flux_scale[1],
        elines_mask_file=args.emission_lines_file,
        fit_gas=not args.ignore_gas,
        refine_gas=not args.single_gas_fit,
        sigma_gas=args.sigma_gas,
        input_redshift=z,
        delta_redshift=args.redshift[1],
        min_redshift=z_fit_min,
        max_redshift=z_fit_max,
        input_sigma=args.sigma[0],
        delta_sigma=args.sigma[1],
        min_sigma=args.sigma[2],
        max_sigma=args.sigma[3],
        input_AV=args.AV[0],
        delta_AV=args.AV[1],
        min_AV=args.AV[2],
        max_AV=args.AV[3],
        sigma_inst=args.sigma_inst,
        spaxel_id=f'{args.label}_{i_fib}',
        out_path=args.output_path,
        plot=int(bool(plot)),
        SN_CUT=getattr(args, 'SN_CUT', 3),
        verbose=verbose,
        full_verbose=bool(getattr(args, 'full_verbose', False)),
        SPS_master=SPS_master
    )


    # ------------------------------------------------------------
    # 4. Correct low-frequency stellar continuum residuals.
    # ------------------------------------------------------------
    model_uncorr = np.asarray(SPS.spectra['model_ssp_min_uncorr'], dtype=float)
    res_smooth = median_fit_spectrum(
        wl,
        spec_stellar - model_uncorr,
        11,
        method='legendre',
        degree=11,
        spline_k=3,
        spline_s=0.0,
        min_points_bin=3,
        return_nodes=False,
    )



    SPS.spectra['orig_flux'] = flux.copy()
    SPS.spectra['orig_flux_ratio'] = flux.copy()
    SPS.spectra['model_ssp_min'] = model_uncorr + res_smooth
    SPS.spectra['res_ssp'] = flux - SPS.spectra['model_ssp_min']
    SPS.spectra['res_ssp_no_corr'] = flux - model_uncorr
    # ------------------------------------------------------------
    # 5. Fit parametric emission-line complexes; build tab_PM.
    # ------------------------------------------------------------
    tab_PM = Table({'id_fib': np.array([i_fib], dtype=np.int64)})
    tab_PM['id'] = [pt_id]

    gas_spec = np.asarray(SPS.spectra['res_ssp'], dtype=float)
    stellar_model = np.asarray(SPS.spectra['model_ssp_min'], dtype=float)

    for i_cl, cluster in enumerate(el_clusters):
        if len(cluster) == 0:
            continue

        rest_waves = np.asarray(cluster['wl'], dtype=float)
        # Do not reuse a mean LSF from the integrated spectrum.
        local_lsf = np.interp(rest_waves * (1.0 + z), wl, lsf)
        good_lsf = local_lsf[np.isfinite(local_lsf) & (local_lsf > 0)]
        if not len(good_lsf):
            raise ValueError(f'Cluster {i_cl}: no positive finite fiber LSF')

        sigma_guess = float(np.mean(good_lsf))
        sigma_scatter = float(np.std(good_lsf))
        sigma_min = max(1e-3, 0.8 * sigma_guess - sigma_scatter)
        sigma_max = max(sigma_min * 1.01, 1.2 * sigma_guess + sigma_scatter)

        if verbose:
            print(f'# Fiber {i_fib}, cluster {i_cl}: '
                  f'{len(cluster)} lines; sigma={sigma_guess:.3f} A')

        fit_result = fit_emission_line_system(
            wl=wl,
            flux=gas_spec,
            eflux=eflux,
            line_wavelengths=rest_waves,
            redshift=z,
            fit_padding=30,
            velocity_range=1,
            sigma_guess=sigma_guess,
            sigma_bounds=(sigma_min, sigma_max),
            error_scale=1,
            n_mc=30,
            n_loops=5,
            tie_kinematics=True,
            add_background=True,
            plot=int(bool(plot)),
            bg_order=1,
            verbose=verbose,
        )
        EL = fit_result['fit']

        for i_line, line in enumerate(cluster):
            pars = np.asarray(EL.final_fit_params_mean[i_line], dtype=float)
            errs = np.asarray(EL.final_fit_params_std[i_line], dtype=float)
            line_name = f"{line['id']}_{float(line['wl']):.2f}"

            tab_PM[f'flux_{line_name}_pm'] = float(pars[1])
            tab_PM[f'e_flux_{line_name}_pm'] = float(errs[1])
            tab_PM[f'vel_{line_name}_pm'] = float(pars[3])
            tab_PM[f'e_vel_{line_name}_pm'] = float(errs[3])
            tab_PM[f'disp_{line_name}_pm'] = float(pars[2])
            tab_PM[f'e_disp_{line_name}_pm'] = float(errs[2])

            

            center = float(line['wl']) * (1 + z)
            window = (wl > center - 20) & (wl < center + 20)
            cont = stellar_model[window]
            cont = cont[np.isfinite(cont)]
            if len(cont) == 0:
                cont_flux = np.nan
                e_cont_flux = np.nan
            else:
                cont_flux = float(np.median(cont))
                e_cont_flux = float(np.std(cont))

            if not np.isfinite(cont_flux) or abs(cont_flux) < 1e-20:
                ew, e_ew = np.nan, np.nan
            else:
                F, e_F = float(pars[1]), float(errs[1])
                ew = -F / cont_flux
                e_ew = np.sqrt(
                    (e_F / cont_flux) ** 2 +
                    (e_cont_flux * F / cont_flux ** 2) ** 2
                )

            tab_PM[f'EW_{line_name}_pm'] = ew
            tab_PM[f'e_EW_{line_name}_pm'] = e_ew

    # ------------------------------------------------------------
    # 6. Build final fitted emission model and joint residual.
    # ------------------------------------------------------------

    SPS.spectra['orig_flux'] = flux.copy()
    SPS.spectra['orig_flux_ratio'] = flux.copy()
    SPS.spectra['raw_model_elines'] = create_elines_model_pm(
        wl, tab_PM, dap_fwhm=dap_fwhm, verbose=verbose
    )[0]
    SPS.spectra['model_joint'] = (
        SPS.spectra['model_ssp_min'] + SPS.spectra['raw_model_elines']
    )
    SPS.spectra['res_joint'] = SPS.spectra['orig_flux'] - SPS.spectra['model_joint']
    SPS.spectra['res_joint_no_corr'] = SPS.spectra['orig_flux'] - (SPS.spectra['model_ssp_min_uncorr'] + SPS.spectra['raw_model_elines'])


    # ------------------------------------------------------------
    # 7. Generate RSP, coefficients, RSP emission-line outputs, HDU.
    # ------------------------------------------------------------
    tab_rsp = create_rsp_table(i_fib, SPS)
    tab_coeffs = create_coeffs_table(i_fib, SPS)
    tab_elines = create_elines_table(i_fib, SPS)


    # ------------------------------------------------------------
    # 8. Nonparametric flux / equivalent-width measurements.
    #    flux_elines_RSS_EW_cl expects arrays of shape (n_fib,n_wave).
    # ------------------------------------------------------------
    hdr_single = hdr_flux.copy()
    hdr_single['NAXIS2'] = 1

    vel_yx = np.array([z * 299792.458], dtype=float)
    fe_data, fe_hdr = flux_elines_RSS_EW_cl(
        gas_spec[np.newaxis, :],
        hdr_single,
        5,
        tab_el,
        vel_yx,
        0.8,
        eflux__wyx=eflux[np.newaxis, :],
        flux_ssp__wyx=stellar_model[np.newaxis, :],
        w_range=getattr(args, 'w_range_FE', 30),
        plot=int(bool(plot)),
        sigma_we=1.0,
        verbose=verbose,
    )

    fe_colnames = [
        (str(fe_hdr[f'NAME{k}']) + '_' + str(fe_hdr[f'WAVE{k}']))
        .replace(' ', '_')
        for k in range(fe_data.shape[0])
    ]
    tab_fe = Table(np.asarray(fe_data).T, names=fe_colnames)
    if len(tab_fe) != 1:
        raise ValueError(f'Expected one flux/EW row, got {len(tab_fe)}')
    tab_fe.add_column(np.array([i_fib], dtype=np.int64), name='id_fib', index=0)
    SPS.spectra['gas_model_NP'] = create_elines_model_el(
        wl, tab_fe, dap_fwhm=dap_fwhm, verbose=verbose
    )[0]
    hdu_sps = SPS.output_fits_hdu(gas_models=True)


    # Add original PT ID to other tables without changing integer id_fib.
    for tab in (tab_rsp, tab_coeffs, tab_elines, tab_fe):
        if 'id' not in tab.colnames:
            tab.add_column([pt_id] * len(tab), name='id', index=0)

    # ------------------------------------------------------------
    # 9. Optional diagnostic plot; no file output requested.
    # SPS.spectra['model_ssp_min_uncorr']
    # ------------------------------------------------------------
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(wl, spec_stellar, color='grey', lw=3.5, label='Decomposed st. spec.')
        ax.plot(wl, flux, color='black', lw=1.5, label='Obs. spec.')
        ax.plot(wl, SPS.spectra['model_joint'], lw=1.5, alpha=0.7, color='yellow', label='Joint model')
        ax.plot(wl, stellar_model, lw=2.0, alpha=0.8, color='tab:red',label='Stellar model')
        ax.fill_between(wl, (-1)*eflux, eflux, color='grey', alpha=0.6)
        ax.plot(wl, SPS.spectra['res_joint'], lw=2.0, color='tab:orange', label='Residual')
        ax.set(xlabel='Wavelength (Angstrom)', ylabel='Flux',
               title=f'RSS fiber {i_fib}, PT id={pt_id}')
        ax.legend()
        fig.tight_layout()
        plt.show()

    if verbose:
        print(f'# END single-spectrum fit: i_fib={i_fib}, id={pt_id}')

    return tab_rsp, tab_coeffs, tab_elines, tab_PM, tab_fe, hdu_sps, SPS



def estimate_gas_redshift(
    wl, flux, eflux,
    z=0.0,
    z_min=-0.003,
    z_max=0.005,
    peak_threshold=1.0,
    n_cut_peaks=2.0,
    velocity_tolerance=15.0,
    min_matches=3,
    plot=False,
    verbose=False,
    bad_pixel_mask=None,
):
    """
    Detect redshift from a gas-only spectrum, respecting the DRP pixel mask.

    No stellar decomposition or args required. Return just the best z, or
    the input z if no sufficiently supported matching solution is found.

    bad_pixel_mask: True = bad (shape of wl). Optional. The input arrays
    are NEVER modified. Interpolation is for peak detection only, and
    interpolated pixels may not be accepted as observed peak centers.

    Requires the existing globals peak_finder(), find_redshift_weights(),
    REFERENCE_LINES, LINE_WEIGHTS, and LINE_GROUPS in dap.py.
    """
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.ndimage import median_filter

    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    eflux = np.asarray(eflux, dtype=float)
    if wl.ndim != 1 or flux.shape != wl.shape or eflux.shape != wl.shape:
        raise ValueError('wl, flux, eflux must be matching 1D arrays')
    if len(wl) < 5 or not np.all(np.isfinite(wl)) or not np.all(np.diff(wl) > 0):
        raise ValueError('wl must have at least five finite increasing pixels')
    if z_min > z_max or min_matches < 1:
        raise ValueError('Invalid redshift range or min_matches')

    if bad_pixel_mask is None:
        bad = np.zeros(wl.shape, dtype=bool)
    else:
        bad = np.asarray(bad_pixel_mask, dtype=bool)
        if bad.shape != wl.shape:
            raise ValueError('bad_pixel_mask must have the same shape as wl')

    # A DRP-flagged pixel with finite flux is STILL invalid data.
    measured = (~bad & np.isfinite(flux) &
                np.isfinite(eflux) & (eflux > 0))
    if measured.sum() < max(5, min_matches):
        if verbose:
            print('# Gas redshift: too few unmasked measurements; using input z')
        return float(z)

    detection_flux = flux.copy()
    if (~measured).any():
        detection_flux[~measured] = np.interp(
            wl[~measured], wl[measured], flux[measured]
        )

    # Subtract a broad local baseline: gas-only residuals can still have
    # imperfect stellar-continuum subtraction or low-frequency structure.
    # A small smoothing is used ONLY by the secondary peak finder.
    continuum = median_filter(detection_flux, size=51, mode='nearest')
    signal = detection_flux - continuum
    good_signal = signal[measured]
    median_signal = np.median(good_signal)
    scatter = 1.4826 * np.median(np.abs(good_signal - median_signal))
    pixel_noise = float(np.median(eflux[measured]))
    dw = float(np.median(np.diff(wl)))

    rest_names = list(REFERENCE_LINES.keys())
    rest_waves = np.asarray(list(REFERENCE_LINES.values()), dtype=float)

    # Restrict candidates to catalog lines in the supplied redshift interval.
    # This substantially reduces false detections from sky-line residuals.
    margin = np.maximum(
        1.5 * dw,
        rest_waves * (1.0 + z_max) * velocity_tolerance / 299792.458,
    )
    allowed = np.any(
        (wl[:, None] >= rest_waves[None, :] * (1 + z_min) - margin) &
        (wl[:, None] <= rest_waves[None, :] * (1 + z_max) + margin),
        axis=1,
    )

    def clean_candidates(indices):
        """Measured, catalog-plausible positive line peaks above local noise."""
        indices = np.asarray(indices, dtype=int).ravel()
        indices = indices[(indices >= 0) & (indices < wl.size)]
        indices = np.unique(indices)
        if indices.size == 0:
            return indices
        good = (
            measured[indices] & allowed[indices] &
            (signal[indices] > 0) &
            (signal[indices] > n_cut_peaks * eflux[indices])
        )
        return indices[good]

    # Primary: preserve the original pyFIT3D peak_finder algorithm.
    try:
        _, _, w_peaks, _ = peak_finder(
            wl, detection_flux, 2,
            peak_threshold=peak_threshold, dmin=0.3,
            plot=int(bool(plot)), verbose=verbose,
        )
        w_peaks = np.atleast_1d(np.asarray(w_peaks, dtype=float))
        w_peaks = w_peaks[np.isfinite(w_peaks)]
        # Nearest grid pixel to each reported peak wavelength.
        j = np.searchsorted(wl, w_peaks)
        j = np.clip(j, 0, len(wl) - 1)
        j_left = np.maximum(j - 1, 0)
        j = np.where(abs(wl[j_left] - w_peaks) < abs(wl[j] - w_peaks),
                     j_left, j)
        primary = clean_candidates(j)
    except (ValueError, FloatingPointError, TypeError, IndexError) as exc:
        if verbose:
            print(f'# Gas redshift: peak_finder failed ({exc}); using fallback')
        primary = np.empty(0, dtype=int)

    def refined_centers(indices):
        # Parabolic subpixel centroid around each measured peak.
        # This avoids grid-quantization biases in the redshift solution.
        centers = wl[indices].astype(float).copy()
        for k, i in enumerate(indices):
            if (i < 1 or i + 1 >= len(wl) or
                    not (measured[i - 1] and measured[i] and measured[i + 1])):
                continue
            yl, ym, yr = signal[i - 1:i + 2]
            curvature = yl - 2.0 * ym + yr
            if curvature < -1e-12:
                dx = 0.5 * (yl - yr) / curvature
                if np.isfinite(dx) and abs(dx) <= 0.75:
                    centers[k] += dx * 0.5 * (wl[i + 1] - wl[i - 1])
        return centers

    def match(indices):
        if indices.size < min_matches:
            return []
        return find_redshift_weights(
            rest_waves=rest_waves,
            obs_waves=refined_centers(indices),
            rest_names=rest_names,
            line_weights=LINE_WEIGHTS,
            line_groups=LINE_GROUPS,
            zmin=z_min, zmax=z_max,
            velocity_tolerance=velocity_tolerance,
            min_matches=min_matches,
        )

    solutions = match(primary)
    candidate_count = len(primary)

    # Secondary: if no valid weighted match emerged, use scipy peaks on a
    # lightly smoothed gas-only spectrum. This catches cases in which
    # peak_finder misses narrow/strong lines, even when it returns other peaks.
    if not solutions:
        smooth_signal = median_filter(signal, size=3, mode='nearest')
        candidate_idx, _ = find_peaks(
            smooth_signal,
            height=max(peak_threshold * scatter,
                       n_cut_peaks * pixel_noise, 0.0),
            prominence=max(0.5 * scatter, 0.5 * pixel_noise, 1e-12),
            distance=2,
        )
        secondary = clean_candidates(candidate_idx)
        all_candidates = np.union1d(primary, secondary)
        candidate_count = len(all_candidates)
        solutions = match(all_candidates)
        if verbose:
            print(f'# Gas redshift: {len(primary)} primary peaks, '
                  f'{len(secondary)} secondary peaks, '
                  f'{candidate_count} unique significant peaks')

    if not solutions:
        if verbose:
            print(f'# Gas redshift: no weighted match; using input z={z:.8f}')
        return float(z)

    z_best = float(solutions[0].z)
    if verbose:
        print(f'# Gas redshift: z={z_best:.8f}; '
              f'{candidate_count} significant catalog-compatible peaks')
    return z_best




def fit_pm_cluster_scipy(wl, gas_spec, eflux, lsf, cluster, z, args,
                         bad_pixel_mask=None):
    """Optional FIT3D-independent PM replacement for one emission-line group.

    Returns line parameters in the original FIT3D array convention:
    rest wavelength, *integrated flux*, observed Gaussian sigma [Angstrom],
    line velocity [km/s].  This enables reuse of the existing PM table,
    continuum, equivalent-width, and weak-line reconstruction paths.

    A single intrinsic gas velocity dispersion and velocity are tied within
    the cluster. The empirical instrumental LSF is interpolated separately
    at the observed position of every line. No additional flux-sign or S/N
    restrictions, no rescaled pixel noise, and no invented precision.
    """
    rest = np.asarray(cluster['wl'], dtype=float)
    ans = fit_linked_lines(
        wl, gas_spec, eflux, rest, lsf,
        initial_velocity=float(z) * 299792.458,
        initial_sigma_kms=float(getattr(args, 'pm_scipy_initial_sigma_kms', 30.0)),
        velocity_bounds=getattr(args, 'pm_scipy_velocity_bounds', None),
        sigma_kms_bounds=getattr(args, 'pm_scipy_sigma_kms_bounds', None),
        bad_pixel_mask=bad_pixel_mask,
        fit_padding=30.0,
        max_nfev=int(getattr(args, 'pm_scipy_max_nfev', 150)),
        background_degree=0,
    )
    if not ans['success'] or not np.all(np.isfinite(ans['flux'])):
        raise RuntimeError('SciPy PM cluster fit unsuccessful: ' + ans['message'])
    pars = np.column_stack((rest, ans['flux'], ans['disp'], ans['vel']))
    errs = np.column_stack((np.full(len(rest), np.nan),
                            ans['e_flux'], ans['e_disp'], ans['e_vel']))
    return pars, errs


def score_pm_cluster_common_pixels(wl, gas_spec, eflux, rest_waves,
                                   fit_params, z, bad_pixel_mask=None,
                                   fit_padding=30.0):
    """Compare PM engines on identical original pixels using fitted line profiles.

    Both candidates are evaluated over one fixed cluster interval, centered
    at the *same input redshift*, with original error weights and bad-pixel
    mask. Each line uses its fitted integrated flux, velocity and observed
    Gaussian sigma (Angstrom); no instrumental-resolution approximation is
    introduced here. A constant background is re-estimated in each candidate
    *only for scoring*, because both engines fit a nuisance background that
    is absent from the saved gas-only PM table/model.

    The nominal degrees of freedom are N - (N_lines + 2 kinematics + 1
    background), equal for the two single-component linked-kinematic models.
    Returns NaN if the model or number of valid pixels is undefined.
    """
    wave = np.asarray(wl, dtype=float)
    obs = np.asarray(gas_spec, dtype=float)
    errors = np.asarray(eflux, dtype=float)
    rest = np.asarray(rest_waves, dtype=float)
    pars = np.asarray(fit_params, dtype=float)
    if not (wave.ndim == obs.ndim == errors.ndim == 1
            and wave.shape == obs.shape == errors.shape):
        raise ValueError('PM quality arrays must be matching 1D spectra')
    if pars.ndim != 2 or pars.shape[0] < len(rest) or pars.shape[1] < 4:
        return np.nan
    pars = pars[:len(rest)]
    if not (np.all(np.isfinite(pars[:, [1, 2, 3]]))
            and np.all(pars[:, 2] > 0)):
        return np.nan
    bad = (np.zeros_like(wave, dtype=bool) if bad_pixel_mask is None
           else np.asarray(bad_pixel_mask, dtype=bool))
    if bad.shape != wave.shape:
        raise ValueError('PM quality mask must match wavelength array')
    centers_0 = rest * (1.0 + float(z))
    use = ((wave >= np.min(centers_0) - fit_padding)
           & (wave <= np.max(centers_0) + fit_padding)
           & ~bad & np.isfinite(wave) & np.isfinite(obs)
           & np.isfinite(errors) & (errors > 0))
    n = int(np.count_nonzero(use))
    dof = n - (len(rest) + 3)
    if dof <= 0:
        return np.nan
    x = wave[use]
    sigma = pars[:, 2]
    centers = rest * (1.0 + pars[:, 3] / 299792.458)
    profiles = np.exp(-0.5 * ((x[:, None] - centers[None, :])
                              / sigma[None, :])**2) / (
                                  np.sqrt(2.0 * np.pi) * sigma[None, :])
    model = profiles @ pars[:, 1]
    # Weighted nuisance constant is evaluated identically for both engines.
    invvar = 1.0 / errors[use]**2
    residual = obs[use] - model
    background = np.sum(invvar * residual) / np.sum(invvar)
    return float(np.sum(invvar * (residual - background)**2) / dof)


def select_pm_strong_clusters(tab_el, tab_el_strong, max_cluster_span,
                              max_adjacent_distance):
    """Select PM/FIT3D anchors from the configured short emission-line list.

    Keep the IDs and exact wavelengths of the *long* list so that PM and NP
    output column names agree. Matching is wavelength-based, not a new S/N cut.
    No strong line is silently dropped; the short list must be a subset of the
    long one to the 0.05-A accuracy of the emission-line definitions.
    """
    if not len(tab_el_strong):
        raise ValueError('PM strong emission-line list is empty')
    all_waves = np.asarray(tab_el['wl'], dtype=float)
    strong_waves = np.asarray(tab_el_strong['wl'], dtype=float)
    selected = np.zeros(len(tab_el), dtype=bool)
    unmatched = []
    for wave in strong_waves:
        if not len(all_waves):
            unmatched.append(wave)
            continue
        j = int(np.argmin(np.abs(all_waves - wave)))
        if abs(all_waves[j] - wave) > 0.05:
            unmatched.append(wave)
        else:
            selected[j] = True
    if unmatched:
        raise ValueError('PM strong lines missing from long line list: '
                         + ', '.join(f'{w:.3f}' for w in unmatched))
    tab_pm_strong = tab_el[selected]
    clusters, _ = cluster_emission_lines(
        tab_pm_strong,
        max_cluster_span=max_cluster_span,
        max_adjacent_distance=max_adjacent_distance,
    )
    return clusters


def reconstruct_pm_from_strong(wl, gas_spec, eflux, lsf, bad_pixel_mask,
                               tab_el, el_clusters, tab_PM_strong,
                               stellar_model, fit_padding=30.0):
    """Complete PM using fixed strong-line kinematics and linear weak-line fits.

    FIT3D fluxes/kinematics/errors for strong lines are retained unchanged.
    Each other line inherits velocity and *intrinsic velocity dispersion* from
    its nearest strong line in rest wavelength. The instrumental Gaussian sigma
    is removed at the anchor and recombined at the target wavelength using the
    actual fiber LSF; intrinsic dispersion is scaled with wavelength.

    Fit only the remaining integrated Gaussian fluxes, jointly for each blend,
    by weighted linear least squares on the FINAL gas spectrum. Subtract fixed
    strong-line templates before solving weak amplitudes, and use the original
    pixel errors and mask. Fluxes may be negative; no new S/N/positivity cuts.
    Formal flux errors are conditional on the inherited kinematics and fixed
    strong-line profiles; they do NOT include strong-line covariance.
    """
    wl = np.asarray(wl, float)
    gas_spec = np.asarray(gas_spec, float)
    eflux = np.asarray(eflux, float)
    lsf = np.asarray(lsf, float)
    mask = np.asarray(bad_pixel_mask, bool)
    stellar_model = np.asarray(stellar_model, float)
    if any(arr.shape != wl.shape for arr in
           (gas_spec, eflux, lsf, mask, stellar_model)):
        raise ValueError('PM reconstruction requires same-length spectral arrays')
    if not np.all(np.diff(wl) > 0):
        raise ValueError('PM reconstruction requires increasing wavelength')

    result = tab_PM_strong.copy()
    strong_anchors = []
    for line in tab_el:
        key = f"{line['id']}_{float(line['wl']):.2f}"
        if f'flux_{key}_pm' not in tab_PM_strong.colnames:
            continue
        v = float(tab_PM_strong[f'vel_{key}_pm'][0])
        sigma_obs = float(tab_PM_strong[f'disp_{key}_pm'][0])
        if not (np.isfinite(v) and np.isfinite(sigma_obs) and sigma_obs > 0):
            continue
        rest = float(line['wl'])
        wave_obs = rest * (1.0 + v / __c__)
        sigma_inst = float(np.interp(wave_obs, wl, lsf))
        if not (np.isfinite(sigma_inst) and sigma_inst > 0):
            continue
        strong_anchors.append((rest, v, sigma_obs, sigma_inst,
                               float(tab_PM_strong[f'e_vel_{key}_pm'][0]),
                               float(tab_PM_strong[f'e_disp_{key}_pm'][0])))
    if not strong_anchors:
        raise ValueError('No finite strong-line PM velocity/dispersion anchors')
    anchors = np.asarray(strong_anchors, dtype=float)

    # Preserve the measured FIT3D strong columns; populate only missing lines.
    weak = {}
    for line in tab_el:
        rest = float(line['wl'])
        key = f"{line['id']}_{rest:.2f}"
        if f'flux_{key}_pm' in tab_PM_strong.colnames:
            continue
        anchor = anchors[np.argmin(np.abs(anchors[:, 0] - rest))]
        rest_a, vel, sigma_a, lsf_a, e_vel, e_sigma_a = anchor
        center = rest * (1.0 + vel / __c__)
        lsf_here = float(np.interp(center, wl, lsf))
        if not (np.isfinite(lsf_here) and lsf_here > 0):
            raise ValueError(f'Invalid fiber LSF at {center:.2f} A')
        intrinsic_sigma2_a = max(0.0, sigma_a**2 - lsf_a**2)
        wave_a = rest_a * (1.0 + vel / __c__)
        scale = center / wave_a
        sigma = float(np.sqrt(intrinsic_sigma2_a * scale**2 + lsf_here**2))
        # Linear propagation of the FIT3D width error with known LSF.
        e_sigma = (abs(scale**2 * sigma_a / sigma) * e_sigma_a
                   if np.isfinite(e_sigma_a) else np.nan)
        weak[key] = (center, sigma)
        for field, val in (('flux', np.nan), ('e_flux', np.nan),
                           ('vel', vel), ('e_vel', e_vel),
                           ('disp', sigma), ('e_disp', e_sigma),
                           ('EW', np.nan), ('e_EW', np.nan)):
            result[f'{field}_{key}_pm'] = val

    valid_pixel = (~mask & np.isfinite(wl) & np.isfinite(gas_spec)
                   & np.isfinite(eflux) & (eflux > 0))

    # Solve weak/blended lines together for each cluster. Strong-line fluxes
    # remain fixed and are removed from the data before fitting the weak ones.
    for cluster in el_clusters:
        if not len(cluster):
            continue
        cluster_keys = [f"{line['id']}_{float(line['wl']):.2f}" for line in cluster]
        pending = [key for key in cluster_keys if key in weak]
        if not pending:
            continue
        # Include measured strong-line centers (not just the redshift guess).
        centers = []
        for key in cluster_keys:
            if key in weak:
                centers.append(weak[key][0])
            else:
                rest = float(key.rsplit('_', 1)[1])
                vel = float(result[f'vel_{key}_pm'][0])
                if np.isfinite(vel):
                    centers.append(rest * (1.0 + vel / __c__))
        if not centers:
            continue
        lo = np.searchsorted(wl, min(centers) - fit_padding, side='left')
        hi = np.searchsorted(wl, max(centers) + fit_padding, side='right')
        indices = np.flatnonzero(valid_pixel[lo:hi]) + lo
        if len(indices) < len(pending):
            continue  # Undefined amplitude, do not replace with NP flux.
        x = wl[indices]
        residual = gas_spec[indices].copy()
        for key in cluster_keys:
            if key in weak:
                continue
            f = float(result[f'flux_{key}_pm'][0])
            vel = float(result[f'vel_{key}_pm'][0])
            sigma = float(result[f'disp_{key}_pm'][0])
            if not (np.isfinite(f) and np.isfinite(vel)
                    and np.isfinite(sigma) and sigma > 0):
                continue
            rest = float(key.rsplit('_', 1)[1])
            center = rest * (1 + vel / __c__)
            template = np.exp(-0.5 * ((x - center) / sigma)**2) / (np.sqrt(2*np.pi) * sigma)
            residual -= f * template
        profiles = np.column_stack([
            np.exp(-0.5 * ((x - weak[key][0]) / weak[key][1])**2)
            / (np.sqrt(2*np.pi) * weak[key][1])
            for key in pending
        ])
        a = profiles / eflux[indices, None]
        b = residual / eflux[indices]
        # Compact SVD: non-iterative; detect under-constrained blends.
        try:
            u, sval, vt = np.linalg.svd(a, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        if (len(sval) != len(pending) or not len(sval)
                or sval[-1] <= np.finfo(float).eps * max(a.shape) * sval[0]):
            continue
        fluxes = vt.T @ ((u.T @ b) / sval)
        errors = np.sqrt(np.sum((vt.T / sval[None, :])**2, axis=1))
        for key, f, e_f in zip(pending, fluxes, errors):
            result[f'flux_{key}_pm'][0] = float(f)
            result[f'e_flux_{key}_pm'][0] = float(e_f)

    # Recompute PM equivalent widths for the reconstructed lines only; retain
    # FIT3D EWs for the strong lines, and leave NP measurements untouched.
    for key, (center, sigma) in weak.items():
        f = float(result[f'flux_{key}_pm'][0])
        e_f = float(result[f'e_flux_{key}_pm'][0])
        continuum = stellar_model[(wl > center - 20) & (wl < center + 20)]
        continuum = continuum[np.isfinite(continuum)]
        if not len(continuum):
            continue
        cont = float(np.median(continuum))
        if not np.isfinite(cont) or abs(cont) < 1e-20:
            continue
        e_cont = float(np.std(continuum))
        result[f'EW_{key}_pm'][0] = -f / cont
        result[f'e_EW_{key}_pm'][0] = np.sqrt(
            (e_f / cont)**2 + (e_cont * f / cont**2)**2)
    return result


def fit_individual_rss_spectrum_or_adopt_result(
    wl,
    flux,
    eflux,
    lsf,
    tab_PT,
    i_fib,
    tab_el,
    el_clusters,
    hdr_flux,
    args_copy,
    eline_spec=None,
    z_guess=None,
    dap_fwhm=1.0,
    plot=False,
    SPS_master=None,
    list_abs_lines=[3968.469, 3933.663],   
    adopt_st_z = True,
    SN_cut_st_z = 10, 
    tab_rsp_adopt=None,
    tab_coeffs_adopt=None,
    fit_PM = True,
    fit_NP = True,
    empirical_errors = True,  # True -> per-pixel-noise joint Fisher errors
    pm_strong_clusters = None,  # Optional short-list FIT3D, complete PM by linear amplitudes
    verbose = False,
    bad_pixel_mask=None
):
    """Fit one RSS spectrum and return its DAP products.

    Parameters
    ----------
    wl, flux, eflux, lsf : (n_wave,) arrays
        Observed wavelength, flux, 1-sigma error and fiber-specific
        instrumental Gaussian sigma, respectively (wavelengths in Angstrom).
    tab_PT : astropy.table.Table
        Pointing table, ordered like the *masked* input RSS.
        Column 'id' preserves the original fiber identity.
    i_fib : int
        Zero-based row index within the masked RSS / tab_PT.
    tab_el : astropy.table.Table
        Emission-line reference table for process_elines_spec and EW fits.
    el_clusters : sequence of astropy tables
        Emission-line clusters, each with 'wl' and 'id' columns.
        Sigma guesses are measured from this fiber's lsf, not cluster means.
    hdr_flux : astropy.io.fits.Header
        Header describing the input wavelength sampling.
    args : argparse.Namespace
        Same configuration used for integrated-spectrum processing.
    eline_spec : (n_wave,) array, optional
        Already-decomposed emission-line component. If None, decompose
        this individual spectrum using decompose_spectrum().
    z_guess : float, optional
        Adopted redshift for line matching/fitting. Defaults to args.redshift[0].
        This function does NOT estimate redshift.
    dap_fwhm : float
        FWHM parameter forwarded to create_elines_model_pm().
    plot : bool
        Plot a diagnostic graph using matplotlib, if requested.

    Returns
    -------
    tab_rsp, tab_coeffs, tab_elines, tab_PM, tab_fe, hdu_sps
        Per-fiber tables and the SPS FITS HDU output.

    Notes
    -----
    - id_fib is the integer RSS row, while id retains tab_PT['id'][i_fib]
      without coercing IDs such as '6112.3' to integers.
    - Assumes process_elines_spec accepts velocity_tolerance and returns
      'stellar_spectrum' and 'emission_model', as in the current DAP version.
    - Does not refit redshift. The redshift can be supplied per fiber.
    """
    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    eflux = np.asarray(eflux, dtype=float)
    lsf = np.asarray(lsf, dtype=float)

    # Integrated-spectrum calls do not provide a per-fiber DRP mask.
    # A missing mask therefore means that no extra pixels are excluded.
    if bad_pixel_mask is None:
        bad_pixel_mask = np.zeros(wl.shape, dtype=bool)
    else:
        bad_pixel_mask = np.asarray(bad_pixel_mask, dtype=bool)
        if bad_pixel_mask.shape != wl.shape:
            raise ValueError('bad_pixel_mask must have the same shape as wl')

    i_fib = int(i_fib)
    args = copy(args_copy)
    if str(getattr(args, 'pm_fit_engine', 'both')).lower() not in ('fit3d', 'scipy', 'both'):
        raise ValueError('pm_fit_engine must be fit3d, scipy or both')

    if wl.ndim != 1 or any(
        a.ndim != 1 or a.shape != wl.shape for a in (flux, eflux, lsf)
    ):
        raise ValueError("wl, flux, eflux, and lsf must be 1D arrays of equal length")
    if not (0 <= i_fib < len(tab_PT)):
        raise IndexError(f"i_fib={i_fib} is outside tab_PT (n={len(tab_PT)})")
    if 'id' not in tab_PT.colnames:
        raise KeyError("tab_PT must contain an 'id' column")

    pt_id = tab_PT['id'][i_fib]
    z = float(args.redshift[0] if z_guess is None else z_guess)
    # Fitting bounds are absolute redshifts: recenter their old offsets
    # around the adopted redshift without double-counting the old center.
    z_fit_min = z + (float(args.redshift[2]) - float(args.redshift[0]))
    z_fit_max = z + (float(args.redshift[3]) - float(args.redshift[0]))
    verbose = bool(getattr(args, 'verbose', False))

    if verbose:
        print(f"# START single-spectrum fit: i_fib={i_fib}, id={pt_id}, z={z}")

    # ------------------------------------------------------------
    # 1. Decompose this fiber (or reuse a supplied decomposition).
    # ------------------------------------------------------------
    if (tab_rsp_adopt is not None) and (tab_coeffs_adopt is not None):
        eline_spec = np.zeros(wl.shape, dtype=float)
        spec_stellar = flux.copy()
        if verbose:
            print(f'# We do not create an eline_spec for {i_fib}')
    else:
        if verbose:
            print(f'# Decomposing the spectrum for {i_fib}')
        if eline_spec is None:
            decomp = decompose_spectrum_masked_iterative(
                wl,
                flux,
                tab_el = tab_el,
                redshift=z,
                mask_nsigma=2.0,
                n_narrow=int(getattr(args, 'n_narrow', 3)),
                n_medium=int(getattr(args, 'n_medium', 5)),
                n_broad=int(getattr(args, 'n_broad', 21)),
                n_iter=int(getattr(args, 'n_iter', 5)),
            )
            eline_spec = decomp['eline_only']
        else:
            if verbose:
                print(f'# Using the supplied eline_spec for {i_fib}')
            eline_spec = np.asarray(eline_spec, dtype=float)
            if eline_spec.shape != wl.shape:
                raise ValueError("eline_spec and wl must have equal shapes")

        # ------------------------------------------------------------
        # 2. Subtract preliminary emission-line estimates.
        # ------------------------------------------------------------
        if verbose:
            print(f'# Subtracting preliminary emission-line estimates for {i_fib}')
        initial = process_elines_spec(
            wl=wl,
            flux=flux,
            eflux=eflux,
            eline_spec=eline_spec,
            lsf=lsf,
            tab_el=tab_el,
            z_guess=z,
            peak_threshold=float(getattr(args, 'peak_threshold', 1.0)),
            n_cut_peaks=float(getattr(args, 'n_cut_peaks', 2.0)),
            smooth_size=int(getattr(args, 'n_medium', 5)),
            velocity_tolerance=float(getattr(args, 'velocity_tolerance', 100.0)),
            do_plot=args.plot,
            verbose=verbose,
        )
        spec_stellar = np.asarray(initial['stellar_spectrum'], dtype=float)

        # ------------------------------------------------------------
        # 2.1  Fit the stellar absorptions to derive the velocity
        # ------------------------------------------------------------
        # ... (fitting code here)

        if verbose:
            print(f'# Fitting the absorption lines for {i_fib}')
        out_fit_abs_lines = fit_emission_line_system(
            wl=wl,
            flux=spec_stellar,
            eflux=eflux,
            line_wavelengths=list_abs_lines,
            redshift=z,
            fit_padding = 30,
            velocity_range=1.5*args.redshift[1]*__c__,
            sigma_guess=1.2,
            sigma_bounds=(0.5, 2.0),
            error_scale=1,
            tie_kinematics=True,
            add_background=True,
            plot=args.plot,
            bg_order=1, 
            run_mode = 'BOTH',
            verbose=verbose)

        args.redshift[0] = z
        abs_out = out_fit_abs_lines["fit"]
        cf_Ca_HK = out_fit_abs_lines["config"]
        sel_wl_Ca_HK = out_fit_abs_lines["mask"]
        vel_mask_st = True
        vel_map_st = abs_out.final_fit_params_mean[0][3]
        vel_e_map_st = abs_out.final_fit_params_std[0][3]
        z_st = vel_map_st/__c__
        if (verbose == True):
            print(f'# Velocity derived from abs. lines: {vel_map_st}')
            print(f'# Redshift derived from abs. lines: {z_st}')
        w_min_st_SN = np.min(list_abs_lines)*(1+z_st)-20
        w_max_st_SN = np.max(list_abs_lines)*(1+z_st)+20
        SN_obs = np.median(spec_stellar[(wl>w_min_st_SN)&(wl<w_max_st_SN)]/eflux[(wl>w_min_st_SN)&(wl<w_max_st_SN)])
        if (verbose == True):
            print(f'# S/N in the region of the absorption lines: {SN_obs}')
        if (SN_obs>SN_cut_st_z) & (adopt_st_z):
            if (verbose == True):
                print(f'# Adopting the redshift derived from the absorption lines: {z_st}')
            z = z_st
            args.redshift[0] = z_st
            #
            # Fixing z!
            #
            z_fit_min = z + (float(args.redshift[2]) - float(args.redshift[0]))
            z_fit_max = z + (float(args.redshift[3]) - float(args.redshift[0]))
    
    # ------------------------------------------------------------
    # 3. Fit the stellar population model on line-subtracted flux.
    # ------------------------------------------------------------

    if (tab_rsp_adopt is not None) and (tab_coeffs_adopt is not None):
        if verbose:
            print(f'# Adopting the stellar population fit from the a previous fit for {i_fib}')
        SPS = SPS_master
        ssp = SPS.models
        n_rsp = ssp.flux_models.shape[0]
        i0 = i_fib*n_rsp
        i1 = (i_fib+1)*n_rsp
        coeffs_input = tab_coeffs_adopt["N.Coeff"][i0:i1]/np.sum(tab_coeffs_adopt["N.Coeff"][i0:i1])    
        redshift = tab_rsp_adopt["redshift"][i_fib]+np.min(np.diff(wl))/np.mean(wl)
        wavenorm_corr = ssp.wavenorm*(1 + redshift)
        half_norm_range = 45  # Angstrom
        w_norm_min = wavenorm_corr - half_norm_range
        w_norm_max = wavenorm_corr + half_norm_range
        sigma = tab_rsp_adopt["disp"][i_fib]
        flux_st=np.median(spec_stellar[(wl >= w_norm_min) & (wl <= w_norm_max)])
        A_V = tab_rsp_adopt['Av'][i_fib]
        model_rsp = np.nansum(ssp.flux_models.T*coeffs_input,axis=1)
        rsp_wave = ssp.wavelength
        model_rsp = shift_convolve(wl, rsp_wave, model_rsp, redshift, sigma,float(getattr(args, "sigma_inst", 0.001)))
        SPS.spectra['model_ssp_min_uncorr'] = spec_apply_dust(wl, model_rsp, A_V, R_V=3.1, extlaw='CCM')*flux_st
        tab_rsp_adopt["med_flux"][i_fib] = np.median(spec_stellar[(wl >= w_norm_min) & (wl <= w_norm_max)])
        tab_rsp_adopt["e_med_flux"][i_fib] = np.std(spec_stellar[(wl >= w_norm_min) & (wl <= w_norm_max)])
        tab_rsp_adopt["flux"][i_fib] = np.sum(spec_stellar[(wl >= args.w_range[0]) & (wl <= args.w_range[1])])

    else:
        if verbose:
            print(f'# Fitting the stellar population model for fiber {i_fib}')

        _, SPS = auto_rsp_elines_rnd(
            wl, spec_stellar, eflux,
            ssp_file=args.rsp_file,
            ssp_nl_fit_file=args.rsp_nl_file,
            config_file=args.config_file,
            w_min=args.w_range[0],
            w_max=args.w_range[1],
            nl_w_min=args.w_range_nl[0],
            nl_w_max=args.w_range_nl[1],
            mask_list=args.mask_file,
            min=args.flux_scale[0],
            max=args.flux_scale[1],
            elines_mask_file=args.emission_lines_file,
            fit_gas=not args.ignore_gas,
            refine_gas=not args.single_gas_fit,
            sigma_gas=args.sigma_gas,
            input_redshift=z,
            delta_redshift=0,
            min_redshift=z_fit_min,
            max_redshift=z_fit_max,
            input_sigma=args.sigma[0],
            delta_sigma=args.sigma[1],
            min_sigma=args.sigma[2],
            max_sigma=args.sigma[3],
            input_AV=args.AV[0],
            delta_AV=args.AV[1],
            min_AV=args.AV[2],
            max_AV=args.AV[3],
            sigma_inst=args.sigma_inst,
            spaxel_id=f'{args.label}_{i_fib}',
            out_path=args.output_path,
            plot=int(bool(plot)),
            verbose=verbose,
            full_verbose=bool(getattr(args, 'full_verbose', False)),
            bin_AV=args.bin_AV,
            SPS_master=SPS_master
        )

    # ------------------------------------------------------------
    # 4. Correct low-frequency stellar continuum residuals.
    # ------------------------------------------------------------
    model_uncorr = np.asarray(SPS.spectra['model_ssp_min_uncorr'], dtype=float)
    model_uncorr_joint = np.asarray(SPS.spectra['model_joint'], dtype=float)
    # wave, spec, dw, method="legendre", degree=11, spline_k=3, spline_s=0.0, min_points_bin=1, return_nodes=False
    res_smooth = median_fit_spectrum(
        wl,
        spec_stellar - model_uncorr_joint,
        11,
        method=str(getattr(args, 'smooth_method', 'spline')),
        degree=11,
        spline_k=3,
        spline_s=0.0,
        min_points_bin=3,
        return_nodes=False,
    )

    SPS.spectra['orig_flux'] = flux.copy()
    SPS.spectra['orig_flux_ratio'] = flux.copy()
    SPS.spectra['model_ssp_min'] = model_uncorr + res_smooth
    SPS.spectra['res_ssp'] = flux - SPS.spectra['model_ssp_min']
    SPS.spectra['res_ssp_no_corr'] = flux - model_uncorr

    # ------------------------------------------------------------
    # 5. Fit parametric emission-line complexes; build tab_PM.
    # ------------------------------------------------------------

    #############################################################################
    # We need to find the redshift using the full set of detected emission lines!
    #############################################################################


    tab_PM = Table({'id_fib': np.array([i_fib], dtype=np.int64)})
    # Only the selected strong clusters are sent to FIT3D when configured.
    # The full el_clusters and tab_el remain available for NP and PM recovery.
    pm_clusters = el_clusters if pm_strong_clusters is None else pm_strong_clusters
    if (fit_PM) & (len(pm_clusters)>0):
        tab_PM['id'] = [pt_id]

        gas_spec = np.asarray(SPS.spectra['res_ssp'], dtype=float)
        stellar_model = np.asarray(SPS.spectra['model_ssp_min'], dtype=float)

        ##############################################################

        z_gas = estimate_gas_redshift(
            wl=wl,
            flux=gas_spec,
            eflux=eflux,
            z=args.redshift[0],
            z_min=args.redshift[2],
            z_max=args.redshift[3],
            velocity_tolerance=30,
            min_matches=3,
            verbose=args.verbose,
            bad_pixel_mask=bad_pixel_mask
        )

        z = z_gas


        for i_cl, cluster in enumerate(pm_clusters):
            if len(cluster) == 0:
                continue

            rest_waves = np.asarray(cluster['wl'], dtype=float)
            # Do not reuse a mean LSF from the integrated spectrum.
            local_lsf = np.interp(rest_waves * (1.0 + z), wl, lsf)
            good_lsf = local_lsf[np.isfinite(local_lsf) & (local_lsf > 0)]
            if not len(good_lsf):
                raise ValueError(f'Cluster {i_cl}: no positive finite fiber LSF')

            sigma_guess = float(np.mean(good_lsf))
            sigma_scatter = float(np.std(good_lsf))
            sigma_min = max(1e-3, 0.8 * sigma_guess - sigma_scatter)
            sigma_max = max(sigma_min * 1.01, 1.2 * sigma_guess + sigma_scatter)

            if verbose:
                print(f'# Fiber {i_fib}, cluster {i_cl}: '
                      f'{len(cluster)} lines; sigma={sigma_guess:.3f} A')

            engine = str(getattr(args, 'pm_fit_engine', 'both')).lower()
            candidates = {}
            failures = {}
            if engine in ('fit3d', 'both'):
                try:
                    fit_result = fit_emission_line_system(
                        wl=wl,
                        flux=gas_spec,
                        eflux=eflux,
                        line_wavelengths=rest_waves,
                        redshift=z,
                        fit_padding=30,
                        velocity_range=30,
                        sigma_guess=sigma_guess,
                        sigma_bounds=(sigma_min, sigma_max),
                        error_scale=1,
                        n_mc=30,
                        n_loops=5,
                        tie_kinematics=True,
                        add_background=True,
                        plot=int(bool(plot)),
                        bg_order=1,
                        verbose=verbose,
                    )
                    EL = fit_result['fit']
                    candidates['fit3d'] = (
                        np.asarray(EL.final_fit_params_mean, dtype=float),
                        np.asarray(EL.final_fit_params_std, dtype=float))
                except Exception as exc:
                    if engine == 'fit3d':
                        raise
                    failures['fit3d'] = str(exc)
            if engine in ('scipy', 'both'):
                try:
                    candidates['scipy'] = fit_pm_cluster_scipy(
                        wl, gas_spec, eflux, lsf, cluster, z, args,
                        bad_pixel_mask=bad_pixel_mask)
                except Exception as exc:
                    if engine == 'scipy':
                        raise
                    failures['scipy'] = str(exc)

            if engine == 'both':
                scores = {name: score_pm_cluster_common_pixels(
                    wl, gas_spec, eflux, rest_waves, params, z,
                    bad_pixel_mask=bad_pixel_mask, fit_padding=30.0)
                    for name, (params, _) in candidates.items()}
                valid_scores = {name: score for name, score in scores.items()
                                if np.isfinite(score)}
                if not valid_scores:
                    raise RuntimeError(
                        f'Fiber {i_fib}, PM cluster {i_cl}: neither engine '
                        f'produced a valid common-pixel fit; '
                        f'scores={scores}, failures={failures}')
                # Deterministic tie: retain the FIT3D result.
                winner = min(valid_scores,
                             key=lambda name: (valid_scores[name],
                                               0 if name == 'fit3d' else 1))
                if verbose:
                    print(f'# Fiber {i_fib}, PM cluster {i_cl}: '
                          f'FIT3D={scores.get("fit3d", np.nan):.3f}, '
                          f'SciPy={scores.get("scipy", np.nan):.3f}; '
                          f'selected {winner}')
            else:
                winner = engine
            fit_params, fit_errors = candidates[winner]
            use_scipy_pm = winner == 'scipy'

            # Joint heteroscedastic covariance for this whole emission-line
            # cluster. Uses the SAME per-wavelength eflux as the NP MC.
            # Includes tied velocity/dispersion and the fitted constant background.
            if empirical_errors and not use_scipy_pm:
                fit_pars = np.asarray(fit_params, dtype=float)
                fit_pars = fit_pars[:len(cluster), :]
                pm_flux_errors, pm_vel_error, pm_disp_error = pm_cluster_weighted_errors(
                    wl=wl,
                    eflux=eflux,
                    rest_waves=rest_waves,
                    fluxes=fit_pars[:, 1],
                    vel=float(fit_pars[0, 3]),
                    sigma=float(fit_pars[0, 2]),
                    bad_pixel_mask=bad_pixel_mask,
                    fit_padding=30.0,
                    fit_background=True,
                )

            for i_line, line in enumerate(cluster):
                pars = np.asarray(fit_params[i_line], dtype=float)
                errs = np.asarray(fit_errors[i_line], dtype=float)
                line_name = f"{line['id']}_{float(line['wl']):.2f}"

                tab_PM[f'flux_{line_name}_pm'] = float(pars[1])
                tab_PM[f'e_flux_{line_name}_pm'] = float(errs[1])
                tab_PM[f'vel_{line_name}_pm'] = float(pars[3])
                tab_PM[f'e_vel_{line_name}_pm'] = float(errs[3])
                tab_PM[f'disp_{line_name}_pm'] = float(pars[2])
                tab_PM[f'e_disp_{line_name}_pm'] = float(errs[2])

                if empirical_errors and not use_scipy_pm:
                    # The covariance may be undefined for a degenerate blend.
                    # In that case retain the original pyFIT3D MC errors.
                    if np.isfinite(pm_flux_errors[i_line]):
                        tab_PM[f'e_flux_{line_name}_pm'] = pm_flux_errors[i_line]
                    if np.isfinite(pm_vel_error):
                        tab_PM[f'e_vel_{line_name}_pm'] = pm_vel_error
                    if np.isfinite(pm_disp_error):
                        # e_disp remains in Angstrom, like 'disp' itself.
                        tab_PM[f'e_disp_{line_name}_pm'] = pm_disp_error

                center = float(line['wl']) * (1 + z)
                window = (wl > center - 20) & (wl < center + 20)
                cont = stellar_model[window]
                cont = cont[np.isfinite(cont)]
                if len(cont) == 0:
                    cont_flux = np.nan
                    e_cont_flux = np.nan
                else:
                    cont_flux = float(np.median(cont))
                    e_cont_flux = float(np.std(cont))

                if not np.isfinite(cont_flux) or abs(cont_flux) < 1e-20:
                    ew, e_ew = np.nan, np.nan
                else:
                    F = float(pars[1])
                    e_F = float(tab_PM[f'e_flux_{line_name}_pm'][0])
                    ew = -F / cont_flux
                    e_ew = np.sqrt(
                        (e_F / cont_flux) ** 2 +
                        (e_cont_flux * F / cont_flux ** 2) ** 2
                    )

                tab_PM[f'EW_{line_name}_pm'] = ew
                tab_PM[f'e_EW_{line_name}_pm'] = e_ew

    # ------------------------------------------------------------
    # 6. Build final fitted emission model and joint residual.
    # ------------------------------------------------------------

    SPS.spectra['orig_flux'] = flux.copy()
    SPS.spectra['orig_flux_ratio'] = flux.copy()
    SPS.spectra['raw_model_elines'] = create_elines_model_pm(
        wl, tab_PM, dap_fwhm=dap_fwhm, verbose=verbose
    )[0]
    SPS.spectra['res_joint_no_corr'] = SPS.spectra['orig_flux'] - (SPS.spectra['model_ssp_min_uncorr'] + SPS.spectra['raw_model_elines'])
    res_smooth = median_fit_spectrum(
        wl,
        SPS.spectra['res_joint_no_corr'],
        11,
        method=str(getattr(args, 'smooth_method', 'spline')),
        degree=11,
        spline_k=3,
        spline_s=0.0,
        min_points_bin=3,
        return_nodes=False,
    )
    SPS.spectra['model_ssp_min'] = SPS.spectra['model_ssp_min_uncorr'] + res_smooth

    # Update gas and stellar spectra after the FINAL continuum correction.
    SPS.spectra['res_ssp'] = (
        SPS.spectra['orig_flux'] - SPS.spectra['model_ssp_min']
    )

    gas_spec = np.asarray(SPS.spectra['res_ssp'], dtype=float)
    stellar_model = np.asarray(SPS.spectra['model_ssp_min'], dtype=float)

    SPS.spectra['model_joint'] = (
        SPS.spectra['model_ssp_min'] + SPS.spectra['raw_model_elines']
    )

    SPS.spectra['res_joint'] = SPS.spectra['orig_flux'] - SPS.spectra['model_joint']

    # ------------------------------------------------------------
    # 7. Generate RSP, coefficients, RSP emission-line outputs, HDU.
    # ------------------------------------------------------------
    if (tab_rsp_adopt is not None) and (tab_coeffs_adopt is not None):
        tab_rsp = tab_rsp_adopt[i_fib:i_fib+1]
        tab_coeffs = tab_coeffs_adopt[i_fib:i_fib+1]
    else:
        tab_rsp = create_rsp_table(i_fib, SPS)
        tab_coeffs = create_coeffs_table(i_fib, SPS)
    tab_elines = create_elines_table(i_fib, SPS)    


    # ------------------------------------------------------------
    # 8. Nonparametric flux / equivalent-width measurements.
    #    flux_elines_RSS_EW_cl expects arrays of shape (n_fib,n_wave).
    # ------------------------------------------------------------
    hdr_single = hdr_flux.copy()
    hdr_single['NAXIS2'] = 1

    ################################################################################
    # Re-fine the redshift using th parametric fit weighted by the S/N of the lines
    ################################################################################

    vel_yx = np.array([z * 299792.458], dtype=float)
    if (fit_NP == True) and (len(tab_el)>0):
        fe_data, fe_hdr = flux_elines_RSS_EW_cl(
            gas_spec[np.newaxis, :],
            hdr_single,
            max(30, int(getattr(args, "N_MC", 30))),
            tab_el,
            vel_yx,
            0.8,
            eflux__wyx=eflux[np.newaxis, :],
            flux_ssp__wyx=stellar_model[np.newaxis, :],
            w_range=getattr(args, 'w_range_FE', 30),
            plot=int(bool(plot)),
            sigma_we=1.0,
            verbose=verbose,
            bad_pixel_mask=bad_pixel_mask[np.newaxis, :] if bad_pixel_mask is not None else None,
        )

        fe_colnames = [
            (str(fe_hdr[f'NAME{k}']) + '_' + str(fe_hdr[f'WAVE{k}']))
            .replace(' ', '_')
            for k in range(fe_data.shape[0])
        ]
        tab_fe = Table(np.asarray(fe_data).T, names=fe_colnames)
        if len(tab_fe) != 1:
            raise ValueError(f'Expected one flux/EW row, got {len(tab_fe)}')
        tab_fe.add_column(np.array([i_fib], dtype=np.int64), name='id_fib', index=0)
        if verbose:
            print(f'# Fiber {i_fib}: {len(fe_colnames)} non-parametric flux/EW measurements *******')
        SPS.spectra['gas_model_NP'] = create_elines_model_el(
            wl, tab_fe, dap_fwhm=dap_fwhm, verbose=verbose
        )[0]
    else:
        tab_fe = Table()
        tab_fe.add_column(np.array([i_fib], dtype=np.int64), name='id_fib', index=0)

    # Fast reconstruction AFTER the final stellar correction and NP run.
    # The FIT3D strong table is retained as tab_PM_strong. This deliberately
    # does not rescale or modify any values in tab_fe.
    if fit_PM and pm_strong_clusters is not None:
        tab_PM_strong = tab_PM
        tab_PM = reconstruct_pm_from_strong(
            wl=wl,
            gas_spec=gas_spec,
            eflux=eflux,
            lsf=lsf,
            bad_pixel_mask=bad_pixel_mask,
            tab_el=tab_el,
            el_clusters=el_clusters,
            tab_PM_strong=tab_PM_strong,
            stellar_model=stellar_model,
            fit_padding=30.0,
        )
        SPS.spectra['raw_model_elines'] = create_elines_model_pm(
            wl, tab_PM, dap_fwhm=dap_fwhm, verbose=verbose
        )[0]
        SPS.spectra['model_joint'] = (
            SPS.spectra['model_ssp_min'] + SPS.spectra['raw_model_elines']
        )
        SPS.spectra['res_joint'] = SPS.spectra['orig_flux'] - SPS.spectra['model_joint']
        SPS.spectra['res_joint_no_corr'] = (
            SPS.spectra['orig_flux'] -
            (SPS.spectra['model_ssp_min_uncorr'] + SPS.spectra['raw_model_elines'])
        )

    # Output-only fit statistics for final PM and NP gas models. Do not
    # confuse these with the stellar-population RSP's own fit statistics.
    if fit_PM and any(c.startswith('flux_') and c.endswith('_pm')
                      for c in tab_PM.colnames):
        add_emission_fit_quality(
            tab_PM, wl, gas_spec, SPS.spectra['raw_model_elines'], eflux,
            bad_pixel_mask=bad_pixel_mask, suffix='_pm')
    if fit_NP and any(c.startswith('flux_') for c in tab_fe.colnames):
        add_emission_fit_quality(
            tab_fe, wl, gas_spec, SPS.spectra['gas_model_NP'], eflux,
            bad_pixel_mask=bad_pixel_mask, suffix='')

    # Add original PT ID to other tables without changing integer id_fib.
    for tab in (tab_rsp, tab_coeffs, tab_elines, tab_fe):
        if 'id' not in tab.colnames:
            tab.add_column([pt_id] * len(tab), name='id', index=0)

    if verbose:
        print(f'# Fiber {i_fib}: Completed single-spectrum fit')
        print(f'#fit_NP: {fit_NP}, fit_PM: {fit_PM} ********')
    hdu_sps = SPS.output_fits_hdu(gas_model_NP=fit_NP, gas_model_PM=fit_PM)
    # ------------------------------------------------------------
    # 9. Optional diagnostic plot; no file output requested.
    # SPS.spectra['model_ssp_min_uncorr']
    # ------------------------------------------------------------
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(wl, spec_stellar, color='cyan', lw=4, label='Decomposed st. spec.')
        ax.plot(wl, flux, color='black', lw=1.5, label='Obs. spec.')
        ax.plot(wl, SPS.spectra['model_joint'], lw=1.5, alpha=0.7, color='yellow', label='Joint model')
        ax.plot(wl, stellar_model, lw=2.0, alpha=0.8, color='tab:red',label='Stellar model')
        ax.fill_between(wl, (-1)*eflux, eflux, color='grey', alpha=0.6)
        ax.plot(wl, SPS.spectra['res_joint'], lw=2.0, color='tab:orange', label='Residual')
        if (fit_NP == True) and (len(tab_el)>0):
            ax.plot(wl, SPS.spectra['gas_model_NP'], lw=2.0, color='tab:blue', label='Non-parametric gas model')
        ax.set(xlabel='Wavelength (Angstrom)', ylabel='Flux',
               title=f'RSS fiber {i_fib}, PT id={pt_id}')
        ax.legend()
        fig.tight_layout()
        plt.show()

    if verbose:
        print(f'# END single-spectrum fit: i_fib={i_fib}, id={pt_id}')

    return tab_rsp, tab_coeffs, tab_elines, tab_PM, tab_fe, hdu_sps, SPS


def split_fe_table_org(tab_fe, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI):
    """
    Split an emission-line flux table into B, R, and I tables.

    Parameters
    ----------
    tab_fe : astropy.table.Table
        Emission-line table with column names ending in
        the corresponding wavelength, e.g. flux_Halpha_6562.85.
    w_br : float
        Boundary between B and R channels.
    w_ri : float
        Boundary between R and I channels.

    Returns
    -------
    tab_fe_B, tab_fe_R, tab_fe_I : astropy.table.Table
        Tables containing the measurements in each wavelength
        range. Columns without a wavelength suffix (e.g. 'id')
        are preserved in all three tables.
    """

    colnames_B = []
    colnames_R = []
    colnames_I = []
    common_cols = []

    for colname in tab_fe.colnames:

        try:
            wave = float(colname.rsplit('_', 1)[-1])
        except ValueError:
            common_cols.append(colname)
            continue

        if wave < w_br:
            colnames_B.append(colname)
        elif wave < w_ri:
            colnames_R.append(colname)
        else:
            colnames_I.append(colname)

    tab_fe_B = tab_fe[common_cols + colnames_B]
    tab_fe_R = tab_fe[common_cols + colnames_R]
    tab_fe_I = tab_fe[common_cols + colnames_I]

    return tab_fe_B, tab_fe_R, tab_fe_I





def split_fe_table(tab_fe, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI):
    """
    Split an emission-line table into B, R, and I channels.

    Supported column formats:
        flux_Halpha_6562.85
        flux_Halpha_6562.85_pm
        e_flux_Halpha_6562.85_pm
        EW_Halpha_6562.85_pm

    Identifier columns are retained in all three tables.
    """

    id_columns = [
        c for c in tab_fe.colnames
        if c in ('id', 'id_fib', 'fiberid', 'ID',
                  'X_sq', 'RMS', 'X_sq_pm', 'RMS_pm', 'X_sq_PEK', 'RMS_PEK',
                  'X_sq_BEST', 'RMS_BEST')
    ]

    colnames_B = []
    colnames_R = []
    colnames_I = []
    unknown_columns = []

    for colname in tab_fe.colnames:

        if colname in id_columns:
            continue

        # Extract wavelength before optional '_pm' or '_pek' suffix.
        match = re.search(
            r'_(\d+(?:\.\d+)?)(?:_pm|_pek|_best)?$',
            colname
        )

        if match is None:
            unknown_columns.append(colname)
            continue

        wave = float(match.group(1))

        if wave < w_br:
            colnames_B.append(colname)
        elif wave < w_ri:
            colnames_R.append(colname)
        else:
            colnames_I.append(colname)

    if unknown_columns:
        raise ValueError(
            'Cannot classify columns without wavelength: '
            + str(unknown_columns[:20])
        )

    tab_fe_B = tab_fe[id_columns + colnames_B]
    tab_fe_R = tab_fe[id_columns + colnames_R]
    tab_fe_I = tab_fe[id_columns + colnames_I]

    # FITS binary tables support at most 999 columns
    for name, tab in (
        ('B', tab_fe_B),
        ('R', tab_fe_R),
        ('I', tab_fe_I)
    ):
        if len(tab.colnames) > 999:
            raise ValueError(
                f'{name} table has {len(tab.colnames)} columns '
                '(FITS maximum is 999)'
            )

    return tab_fe_B, tab_fe_R, tab_fe_I



def propagate_stellar_field_errors(
    ra_st, dec_st, values, errors, ra_target, dec_target,
    min_spline_points=10, k=4, power=2.0,
    method="spline", smoothing=0.0, verbose=False, weight_cache=None,
):
    """Propagate independent source-fit errors through stellar interpolation.

    The interpolation is linear in the measured *values*. For the existing
    IDW and spline branches construct the same interpolation weights W, then
    use Var(target) = sum_j W_ij**2 * e_j**2. The spline weights come from an
    RBFInterpolator with an identity-valued right-hand side; this uses the
    same kernel, smoothing, source positions and convex-hull selection as
    interpolate_stellar_field. Exact source locations keep their source errors.

    No noise floor, S/N cut or spatial-scatter prior is introduced. If a
    contributor has no finite positive error, the target uncertainty is NaN,
    not a spurious zero. This propagates formal errors only; spatial model
    uncertainty and correlations between interpolated fibers are distinct.
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import QhullError

    ra_st = np.asarray(ra_st, dtype=float)
    dec_st = np.asarray(dec_st, dtype=float)
    values = np.ma.asarray(values, dtype=float).filled(np.nan)
    errors = np.ma.asarray(errors, dtype=float).filled(np.nan)
    ra_target = np.asarray(ra_target, dtype=float)
    dec_target = np.asarray(dec_target, dtype=float)
    if (ra_st.ndim != 1 or not
            (ra_st.shape == dec_st.shape == values.shape == errors.shape)):
        raise ValueError('Source arrays must be aligned one-dimensional arrays')
    if ra_target.ndim != 1 or ra_target.shape != dec_target.shape:
        raise ValueError('Target coordinates must be aligned one-dimensional arrays')
    if k < 1 or power <= 0:
        raise ValueError('k must be positive and power must be > 0')

    result = np.full(len(ra_target), np.nan, dtype=float)
    valid = np.isfinite(ra_st) & np.isfinite(dec_st) & np.isfinite(values)
    target_valid = np.isfinite(ra_target) & np.isfinite(dec_target)
    if not np.any(valid) or not np.any(target_valid):
        return result

    ra0 = ra_st[valid][0]
    dec0 = float(np.median(dec_st[valid]))
    cosine = np.cos(np.deg2rad(dec0))

    def project(ra, dec):
        dra = (ra - ra0 + 180.0) % 360.0 - 180.0
        return np.column_stack((dra * cosine, dec - dec0))

    source_xy = project(ra_st[valid], dec_st[valid])
    source_err = errors[valid]
    unique_xy, inverse = np.unique(source_xy, axis=0, return_inverse=True)
    n_unique = len(unique_xy)
    # A repeated position is assigned its original arithmetic-mean weight,
    # consistent with interpolate_stellar_field's duplicate merging.
    counts = np.bincount(inverse)
    unique_variance = np.bincount(
        inverse, weights=np.where(np.isfinite(source_err) & (source_err > 0),
                                  source_err**2, 0.0), minlength=n_unique
    ) / counts**2
    missing_unique = np.bincount(
        inverse, weights=(~np.isfinite(source_err) | (source_err <= 0)).astype(float),
        minlength=n_unique
    ) > 0

    # Identical valid-source subsets reuse the same spatial interpolation
    # weights across different error columns, avoiding repeated RBF solves.
    cache_key = valid.tobytes()
    weights = None if weight_cache is None else weight_cache.get(cache_key)
    if weights is None:
        target_xy = project(ra_target[target_valid], dec_target[target_valid])
        n_near = min(k, n_unique)
        tree = cKDTree(unique_xy)
        distance, indices = tree.query(target_xy, k=n_near)
        distance = np.asarray(distance).reshape(-1, n_near)
        indices = np.asarray(indices).reshape(-1, n_near)
        idw = 1.0 / np.maximum(distance, 1e-12)**power
        idw /= np.sum(idw, axis=1, keepdims=True)
        weights = np.zeros((len(target_xy), n_unique), dtype=float)
        np.put_along_axis(weights, indices, idw, axis=1)
        exact = distance[:, 0] < 1e-10
        if np.any(exact):
            weights[exact] = 0.0
            weights[np.flatnonzero(exact), indices[exact, 0]] = 1.0

        if method == 'spline' and n_unique >= min_spline_points:
            centered = unique_xy - unique_xy.mean(axis=0)
            if np.linalg.matrix_rank(centered) == 2:
                try:
                    inside = Delaunay(unique_xy).find_simplex(target_xy) >= 0
                    if np.any(inside):
                        # RBF interpolation is linear in the supplied values.
                        # An identity right-hand side yields all weights with
                        # one solve, and is reused for all paired error columns.
                        basis = RBFInterpolator(
                            unique_xy, np.eye(n_unique),
                            kernel='thin_plate_spline', smoothing=smoothing
                        )
                        fitted_weights = basis(target_xy[inside])
                        finite = np.all(np.isfinite(fitted_weights), axis=1)
                        rows = np.flatnonzero(inside)[finite]
                        weights[rows] = fitted_weights[finite]
                except (ValueError, RuntimeError, np.linalg.LinAlgError, QhullError) as exc:
                    if verbose:
                        print(f'# RSP uncertainty RBF unavailable; IDW used: {exc}')

        # Exact source positions retain their original fitted errors.
        if np.any(exact):
            weights[exact] = 0.0
            weights[np.flatnonzero(exact), indices[exact, 0]] = 1.0
        if weight_cache is not None:
            if len(weight_cache) >= 3:
                weight_cache.pop(next(iter(weight_cache)))
            weight_cache[cache_key] = weights
    # Invalid source errors cannot be interpreted as precise measurements.
    # Ignore only numerically negligible interpolation weights.
    required_missing = np.any(
        (np.abs(weights) > 1e-12) & missing_unique[None, :], axis=1
    )
    propagated = np.sqrt(np.maximum((weights**2) @ unique_variance, 0.0))
    propagated[required_missing] = np.nan
    result[target_valid] = propagated
    return result


def interpolate_rsp_coeffs(
    tab_PT_st,
    tab_rsp_st,
    tab_coeffs_st,
    tab_PT,
    method="spline",
    smoothing=0,
    decimals=4,
    verbose=False,
    min_spline_points=10,
    k=4,
    power=2.0,
):
    """
    Interpolate stellar population results from bright stellar
    continuum fibers to all science fibers.

    Parameters
    ----------
    tab_PT_st : astropy.table.Table
        Pointing table for the bright stellar continuum fibers.
    tab_rsp_st : astropy.table.Table
        RSP results for those fibers, one row per fiber.
    tab_coeffs_st : astropy.table.Table
        Template coefficients for those fibers, containing
        'id_fib' and 'rsp'.
    pt_table : astropy.table.Table
        Complete pointing table. If a 'mask' column exists,
        only the selected fibers are interpolated.
    method : str
        Interpolation method for interp2d_scattered.
    smoothing : float
        Interpolation smoothing parameter.
    decimals : int
        Number of decimals for interpolated coefficients.
    verbose : bool
        Print progress information.

    Returns
    -------
    tab_rsp_interp : astropy.table.Table
        Interpolated RSP properties, one row per target fiber.
    tab_coeffs_interp : astropy.table.Table
        Interpolated coefficients, one row per fiber/template.
    """

    # ---------------------------------------------------------
    # 1. Select target fibers
    # ---------------------------------------------------------


    n_fibers = len(tab_PT)
    n_st = len(tab_rsp_st)

    if n_fibers == 0 or n_st == 0:
        raise ValueError("Source and target tables must not be empty")

    # ---------------------------------------------------------
    # 2. Align source pointing table with tab_rsp_st
    # ---------------------------------------------------------

    if len(tab_PT_st) != n_st:
        raise ValueError(
            "tab_PT_st and tab_rsp_st must describe the same fibers"
        )

    source_ids = np.asarray(tab_rsp_st['id_fib'], dtype=int)

    if len(np.unique(source_ids)) != n_st:
        raise ValueError("Duplicate id_fib values in tab_rsp_st")

    if 'id' in tab_PT_st.colnames and 'id' in tab_rsp_st.colnames:

        lookup = {
            str(v): i for i, v in enumerate(tab_PT_st['id'])
        }

        source_order = [
            lookup[str(v)] for v in tab_rsp_st['id']
        ]

    elif 'id_fib' in tab_PT_st.colnames:

        lookup = {
            int(v): i for i, v in enumerate(tab_PT_st['id_fib'])
        }

        source_order = [
            lookup[int(v)] for v in source_ids
        ]

    else:
        # Assume that the two tables have the same row order
        source_order = np.arange(n_st)

    ra_st = np.asarray(tab_PT_st['ra'][source_order], dtype=float)
    dec_st = np.asarray(tab_PT_st['dec'][source_order], dtype=float)

    ra_target = np.asarray(tab_PT['ra'], dtype=float)
    dec_target = np.asarray(tab_PT['dec'], dtype=float)

    # ---------------------------------------------------------
    # 3. Interpolation helper
    # ---------------------------------------------------------

    def interpolate(values):
        return interpolate_stellar_field(
            ra_st,
            dec_st,
            values,
            ra_target,
            dec_target,
            min_spline_points=min_spline_points,
            k=k,
            power=power,
            method=method,
            smoothing=smoothing,
            verbose=verbose
        )

    def interpolate_many(values):
        """
        Interpolate values from selected bright fibers to all fibers.

       Uses interp2d_scattered when enough valid spatial points
        are available, otherwise falls back to nearest neighbor.
        """



        values = np.ma.asarray(values, dtype=float).filled(np.nan)

        valid = (
            np.isfinite(ra_st) &
            np.isfinite(dec_st) &
            np.isfinite(values)
        )

        n_valid = np.count_nonzero(valid)

        if n_valid == 0:
            return np.full(n_fibers, np.nan)

        # Source coordinates and values
        points = np.column_stack((
            ra_st[valid],
            dec_st[valid]
        ))
        data = values[valid]

        # Combine duplicate coordinates by averaging their values
        points_unique, inverse = np.unique(
            points, axis=0, return_inverse=True
        )

        data_unique = (
            np.bincount(inverse, weights=data) /
            np.bincount(inverse)
        )

        n_unique = len(points_unique)

        # ---------------------------------------------------------
        # Standard 2D interpolation
        # ---------------------------------------------------------

        if n_unique >= 3:

            rank = np.linalg.matrix_rank(
                points_unique - points_unique.mean(axis=0)
            )

            if rank == 2:
                try:
                    return np.asarray(
                        interp2d_scattered(
                            points_unique[:, 0],
                            points_unique[:, 1],
                            data_unique,
                            ra_target,
                            dec_target,
                            method=method,
                            smoothing=smoothing
                        ),
                        dtype=float
                    ).reshape(-1)

                except (ValueError, np.linalg.LinAlgError) as exc:
                    if verbose:
                        print(f'# 2D interpolation failed: {exc}')

        # ---------------------------------------------------------
        # Fallback: nearest neighbor
        # ---------------------------------------------------------

        if verbose:
            print(
                f'# Using nearest neighbor: '
                f'{n_unique} distinct spatial points'
            )

        tree = cKDTree(points_unique)

        target_points = np.column_stack((
            ra_target, dec_target
        ))

        result = np.full(n_fibers, np.nan)

        valid_target = np.all(
            np.isfinite(target_points), axis=1
        )

        _, idx = tree.query(target_points[valid_target])

        result[valid_target] = data_unique[idx]

        return result



    def interpolate_only_good(values):

        values = np.ma.asarray(
            values, dtype=float
        ).filled(np.nan)

        valid = (
            np.isfinite(ra_st) &
            np.isfinite(dec_st) &
            np.isfinite(values)
        )

        if not np.any(valid):
            return np.full(n_fibers, np.nan)

        result = interp2d_scattered(
            ra_st[valid],
            dec_st[valid],
            values[valid],
            ra_target,
            dec_target,
            method=method,
            smoothing=smoothing
        )

        result = np.asarray(result, dtype=float).reshape(-1)

        if len(result) != n_fibers:
            raise ValueError("Unexpected interpolation output size")

        return result

    # ---------------------------------------------------------
    # 4. Interpolate RSP properties
    # ---------------------------------------------------------

    error_weight_cache = {}  # Same source geometry across RSP/COEFFS errors
    tab_rsp_interp = Table()

    tab_rsp_interp['id_fib'] = np.arange(
        n_fibers, dtype=int
    )

    if 'id' in tab_PT.colnames:
        tab_rsp_interp['id'] = tab_PT['id']

    skip_rsp = {'id_fib', 'id', 'ID', 'fiberid'}

    for col in tab_rsp_st.colnames:

        if col in skip_rsp:
            continue

        if verbose:
            print(f'# Interpolating RSP: {col}')

        # Formal uncertainties require variance propagation, not ordinary
        # interpolation of their numerical error values.
        if col.startswith('e_') and col[2:] in tab_rsp_st.colnames:
            tab_rsp_interp[col] = propagate_stellar_field_errors(
                ra_st, dec_st, tab_rsp_st[col[2:]], tab_rsp_st[col],
                ra_target, dec_target, min_spline_points=min_spline_points,
                k=k, power=power, method=method, smoothing=smoothing,
                verbose=verbose, weight_cache=error_weight_cache
            )
        else:
            tab_rsp_interp[col] = interpolate(tab_rsp_st[col])

    # ---------------------------------------------------------
    # 5. Identify available RSP templates
    # ---------------------------------------------------------

    templates = np.sort(
        np.unique(np.asarray(tab_coeffs_st['rsp'], dtype=int))
    )

    n_templates = len(templates)

    if verbose:
        print(f'# Stellar fibers: {n_st}')
        print(f'# Target fibers: {n_fibers}')
        print(f'# RSP templates: {n_templates}')

    # ---------------------------------------------------------
    # 6. Initialize coefficient table
    # ---------------------------------------------------------

    # Fiber-major ordering:
    # fiber 0: templates 0 ... N-1
    # fiber 1: templates 0 ... N-1
    # ...

    tab_coeffs_interp = Table()

    tab_coeffs_interp['id_fib'] = np.repeat(
        np.arange(n_fibers, dtype=int),
        n_templates
    )

    tab_coeffs_interp['rsp'] = np.tile(
        templates, n_fibers
    )

    if 'id' in tab_PT.colnames:
        tab_coeffs_interp['id'] = np.repeat(
            np.asarray(tab_PT['id']),
            n_templates
        )

    coeff_cols = [
        col for col in tab_coeffs_st.colnames
        if col not in ('id_fib', 'rsp', 'id', 'ID', 'fiberid')
    ]

    for col in coeff_cols:
        tab_coeffs_interp[col] = np.full(
            n_fibers * n_templates,
            np.nan,
            dtype=float
        )

    # ---------------------------------------------------------
    # 7. Interpolate coefficients template by template
    # ---------------------------------------------------------

    input_rsp = np.asarray(tab_coeffs_st['rsp'], dtype=int)
    input_fib = np.asarray(tab_coeffs_st['id_fib'], dtype=int)

    for j, i_rsp in enumerate(templates):

        mask_rsp = input_rsp == i_rsp
        indices = np.flatnonzero(mask_rsp)

        # Locate source fiber IDs for this template
        coeff_ids = input_fib[indices]

        if len(coeff_ids) != n_st or len(np.unique(coeff_ids)) != n_st:
            raise ValueError(
                f'Incomplete or duplicate coefficients for RSP {i_rsp}'
            )

        lookup = {
            int(fid): idx
            for fid, idx in zip(coeff_ids, indices)
        }

        # Align coefficients with the source RSP table
        ordered_indices = [
            lookup[int(fid)] for fid in source_ids
        ]

        # Positions of this template in the output table
        output_indices = (
            np.arange(n_fibers) * n_templates + j
        )

        for col in coeff_cols:

            if col.startswith('e_') and col[2:] in coeff_cols:
                # Keep precision of propagated errors: decimal rounding can
                # erase a physically meaningful small uncertainty.
                values = propagate_stellar_field_errors(
                    ra_st, dec_st,
                    tab_coeffs_st[col[2:]][ordered_indices],
                    tab_coeffs_st[col][ordered_indices],
                    ra_target, dec_target,
                    min_spline_points=min_spline_points,
                    k=k, power=power, method=method,
                    smoothing=smoothing, verbose=verbose,
                    weight_cache=error_weight_cache
                )
                tab_coeffs_interp[col][output_indices] = values
            else:
                values = interpolate(tab_coeffs_st[col][ordered_indices])
                tab_coeffs_interp[col][output_indices] = (
                    np.around(values, decimals=decimals)
                )

        if verbose:
            print(
                f'# Interpolated template {j + 1}/{n_templates}: '
                f'RSP {i_rsp}'
            )

    return tab_rsp_interp, tab_coeffs_interp



def preserve_rsp_coeffs_samples(
    tab_PT_st,
    tab_rsp_st,
    tab_coeffs_st,
    tab_PT,
    tab_rsp_interp,
    tab_coeffs_interp,
):
    """
    Preserve measured RSP and coefficient values at sampled fibers.

    Assumes tab_rsp_st and tab_PT_st have the same row order
    unless tab_rsp_st has an 'id' column.

    Matches target fibers through pt_table['id'], and matches
    coefficient rows through ('id_fib', 'rsp').

    Returns copies of both interpolated tables.
    """

    tab_rsp_interp = tab_rsp_interp.copy()
    tab_coeffs_interp = tab_coeffs_interp.copy()

    # Target fiber table in the same order as the interpolation
    tab_target = tab_PT.copy()

    if len(tab_target) != len(tab_rsp_interp):
        raise ValueError("Target table and interpolated RSP row counts differ")

    # Map original fiber IDs to target positions
    target_map = {
        str(fid): i
        for i, fid in enumerate(tab_target['id'])
    }

    if len(target_map) != len(tab_target):
        raise ValueError("Duplicate IDs in target pointing table")

    # Map fitted RSP rows to source pointing-table rows
    if 'id' in tab_rsp_st.colnames:
        rsp_map = {
            str(fid): i
            for i, fid in enumerate(tab_rsp_st['id'])
        }
    else:
        if len(tab_rsp_st) != len(tab_PT_st):
            raise ValueError(
                "Cannot align source tables by row order"
            )

        rsp_map = {
            str(fid): i
            for i, fid in enumerate(tab_PT_st['id'])
        }

    rsp_cols = [
        col for col in tab_rsp_st.colnames
        if col in tab_rsp_interp.colnames
        and col not in ('id', 'id_fib', 'ID', 'fiberid')
    ]

    coeff_cols = [
        col for col in tab_coeffs_st.colnames
        if col in tab_coeffs_interp.colnames
        and col not in ('id', 'id_fib', 'rsp', 'ID', 'fiberid')
    ]

    source_coeff_fib = np.asarray(
        tab_coeffs_st['id_fib'], dtype=int
    )
    source_coeff_rsp = np.asarray(
        tab_coeffs_st['rsp'], dtype=int
    )

    target_coeff_fib = np.asarray(
        tab_coeffs_interp['id_fib'], dtype=int
    )
    target_coeff_rsp = np.asarray(
        tab_coeffs_interp['rsp'], dtype=int
    )

    for fid in tab_PT_st['id']:

        key = str(fid)

        if key not in target_map or key not in rsp_map:
            raise ValueError(f"Cannot align sampled fiber {key}")

        i_target = target_map[key]
        i_source = rsp_map[key]

        # Restore measured RSP properties
        for col in rsp_cols:
            tab_rsp_interp[col][i_target] = (
                tab_rsp_st[col][i_source]
            )

        # Restore measured coefficients, template by template
        source_fib = int(tab_rsp_st['id_fib'][i_source])

        src_rows = np.flatnonzero(
            source_coeff_fib == source_fib
        )
        dst_rows = np.flatnonzero(
            target_coeff_fib == i_target
        )

        dst_by_rsp = {
            target_coeff_rsp[j]: j for j in dst_rows
        }

        for j_source in src_rows:

            template = source_coeff_rsp[j_source]

            if template not in dst_by_rsp:
                raise ValueError(
                    f"Missing target RSP template {template}"
                )

            j_target = dst_by_rsp[template]

            for col in coeff_cols:
                tab_coeffs_interp[col][j_target] = (
                    tab_coeffs_st[col][j_source]
                )

    return tab_rsp_interp, tab_coeffs_interp



def interpolate_stellar_field(
    ra_st, dec_st, values, ra_target, dec_target,
    min_spline_points=10, k=4, power=2.0,
    method="spline", smoothing=0.0, verbose=False,
):
    """Interpolate a stellar quantity with sparse-sample IDW fallback.

    Sparse sources (<min_spline_points), degenerate geometry, and
    points outside the source convex hull use inverse-distance weights.
    Within a well-sampled 2D convex hull, use interp2d_scattered().
    Values are exact at coincident valid source coordinates.
    """
    ra_st = np.asarray(ra_st, dtype=float)
    dec_st = np.asarray(dec_st, dtype=float)
    values = np.ma.asarray(values, dtype=float).filled(np.nan)
    ra_target = np.asarray(ra_target, dtype=float)
    dec_target = np.asarray(dec_target, dtype=float)

    if ra_st.ndim != 1 or not (ra_st.shape == dec_st.shape == values.shape):
        raise ValueError('Source coordinates and values must be 1D, equal-length')
    if ra_target.shape != dec_target.shape or ra_target.ndim != 1:
        raise ValueError('Target RA and Dec must be 1D, equal-length')
    if k < 1 or power <= 0:
        raise ValueError('k must be positive and power must be > 0')

    result = np.full(len(ra_target), np.nan, dtype=float)
    valid = np.isfinite(ra_st) & np.isfinite(dec_st) & np.isfinite(values)
    if not np.any(valid):
        return result

    # Locally project RA with cos(Dec) to use meaningful angular distances.
    ra0 = ra_st[valid][0]
    dec0 = float(np.median(dec_st[valid]))
    cos_dec0 = np.cos(np.deg2rad(dec0))

    def project(ra, dec):
        dra = (ra - ra0 + 180.0) % 360.0 - 180.0
        return np.column_stack((dra * cos_dec0, dec - dec0))

    xy_source = project(ra_st[valid], dec_st[valid])
    z = values[valid]

    # Repeated sky coordinates cannot be distinguished by interpolation.
    xy_source, inv = np.unique(xy_source, axis=0, return_inverse=True)
    z = np.bincount(inv, weights=z) / np.bincount(inv)
    n_source = len(z)

    valid_target = np.isfinite(ra_target) & np.isfinite(dec_target)
    if not np.any(valid_target):
        return result
    xy_target = project(ra_target[valid_target], dec_target[valid_target])

    # Robust IDW for any number of valid sampled fibers, including one.
    tree = cKDTree(xy_source)
    n_near = min(int(k), n_source)
    distance, indices = tree.query(xy_target, k=n_near)
    distance = np.asarray(distance).reshape(-1, n_near)
    indices = np.asarray(indices).reshape(-1, n_near)

    weights = 1.0 / np.maximum(distance, 1e-12)**power
    interpolated = np.sum(weights * z[indices], axis=1) / weights.sum(axis=1)
    exact = distance[:, 0] < 1e-10
    interpolated[exact] = z[indices[exact, 0]]

    # Use the existing spline only for well-sampled 2D geometry, inside hull.
    if method == 'spline' and n_source >= min_spline_points:
        centered = xy_source - xy_source.mean(axis=0)
        if np.linalg.matrix_rank(centered) == 2:
            try:
                hull = Delaunay(xy_source)
                inside = hull.find_simplex(xy_target) >= 0
                inside_indices = np.flatnonzero(inside)
                if len(inside_indices):
                    smooth_values = np.asarray(
                        interp2d_scattered(
                            xy_source[:, 0], xy_source[:, 1], z,
                            xy_target[inside, 0], xy_target[inside, 1],
                            method=method, smoothing=smoothing,
                        ), dtype=float
                    ).reshape(-1)
                    if len(smooth_values) != len(inside_indices):
                        raise ValueError('Unexpected spline output length')
                    good = np.isfinite(smooth_values)
                    interpolated[inside_indices[good]] = smooth_values[good]
            except (ValueError, RuntimeError, np.linalg.LinAlgError, QhullError) as exc:
                if verbose:
                    print(f'# 2D spline unavailable; using IDW: {exc}')

    # Enforce exact recovery at observed source coordinates.
    interpolated[exact] = z[indices[exact, 0]]
    result[valid_target] = interpolated
    return result





###############################################################################
# MAIN script. Uses the entries from commands lines: Depcreated since version #
###############################################################################
def _main(cmd_args=sys.argv[1:]):
    print("########################################")
    print("### Do nothing")


#########################################################################
# MAIN script. Uses the entries from commands lines and the YAML file
#########################################################################
def _dap_yaml(cmd_args=sys.argv[1:]):



    print(f'##############################################################')
    print(f'# LVM-DAP Started ...')
    print(f'##############################################################')
    PLATESCALE = 112.36748321030637
    empirical_errors = True
    #####################################################################
    # LVM DAP version and date
    #####################################################################
    dap_ver = '2.0.0'
    dap_date = '2026-10-09'


    #####################################################################
    # START: Initialize DAP configuration
    #####################################################################
    args = parse_dap_arguments(cmd_args)
    args = initialize_dap_parameters(args)
    hdu_info = create_dap_info_hdu(args)

    ny_range = args.ny_range
    nx_range = args.nx_range
    out_plot_format = args.out_plot_format
    only_integrated = args.only_integrated
    sky_hack = args.sky_hack
    mask_to_val = args.mask_to_val

    auto_redshift = args.auto_redshift
    auto_z_min = args.auto_z_min
    auto_z_max = args.auto_z_max
    auto_z_del = args.auto_z_d
    dump_model = args.dump_model

    SN_CUT = args.SN_CUT
    SN_CUT_INT = args.SN_CUT_INT
    auto_z_bg = args.auto_z_bg
    auto_z_sc = args.auto_z_sc

    kin_fixed = args.kin_fixed
    N_MC = args.N_MC
    n_loops = args.n_loops
    w_range_FE = args.w_range_FE
    smooth_size = args.smooth_size
    n_leg = args.n_leg
    flux_scaling = float(getattr(args, "flux_scaling", 1e16))
    print(f'# Flux scaling factor: {flux_scaling}')
    #####################################################################
    # END: Initialize DAP configuration
    #####################################################################
    print(f'# Input parameters initialized ...')


    #####################################################################
    # START: Reading the input data to be analyzed
    #####################################################################
    print(f'# Reading input data ...')
    (wl__w, rss_flux_org, rss_eflux_org,
     hdr_flux_org, hdr_0, LSF_rss_org,
     tab_PT_org, LSF_mean, input_mask_org, bad_pixels_org) = load_input_data(
         args,
         ny_range=ny_range,
         nx_range=nx_range,
         sky_hack=sky_hack,
         mask_to_val=mask_to_val,
         flux_scale=flux_scaling
     )
    ###############################################################################
    # This is the default tab_el_strong. We dynamically select later a better list
    ###############################################################################
    tab_el_strong = read_tab_EL(args.emission_lines_file)
    ######################################################################
    # We mask the non-science spectra and broken fibers ones
    ######################################################################
    rss_flux = rss_flux_org[tab_PT_org['mask']]
    rss_eflux = rss_eflux_org[tab_PT_org['mask']]
    tab_PT = tab_PT_org[tab_PT_org['mask']]
    LSF_rss = LSF_rss_org[tab_PT_org['mask']]
    input_mask = input_mask_org[tab_PT_org['mask']]
    bad_pixels = bad_pixels_org[tab_PT_org['mask']]
    hdr_flux=copy(hdr_flux_org)
    hdr_flux['NAXIS2']=len(tab_PT)
    print(f'# Number of spectra to analyze : {len(tab_PT)}')

    #####################################################################################
    # Reading the emission lines file and identify the instrumental LSF and blended lines
    #####################################################################################
    tab_el=read_tab_EL(args.emission_lines_file_long)
    lsf = []
    blended = []
    for wl_el in tab_el['wl']:
      i_wl_el = find_closest_indices(wl__w, wl_el)
      lsf_now = LSF_mean[i_wl_el[0]]
      lsf.append(lsf_now)
      i_wl_near = find_closest_indices_different(tab_el['wl'], wl_el)
      dist_abs = np.abs(wl_el-tab_el['wl'][i_wl_near[0]])
      if (dist_abs<4*args.sigma_gas):
        blended.append(1)
      else:
        blended.append(0)
    blended = np.array(blended)    
    tab_el['blended'] = blended
    lsf = np.array(lsf)
    tab_el['lsf'] = lsf


    ########################################################
    # Clusters of emission lines for future fitting
    ########################################################

    el_clusters, el_cluster_id = cluster_emission_lines(
        tab_el,
        max_cluster_span=float(getattr(args, "max_cluster_span", 300.0)),
        max_adjacent_distance=float(getattr(args, "max_adjacent_distance", 100.0)),
    )
    pm_strong_clusters = select_pm_strong_clusters(
        tab_el, tab_el_strong,
        max_cluster_span=float(getattr(args, "max_cluster_span", 300.0)),
        max_adjacent_distance=float(getattr(args, "max_adjacent_distance", 100.0)),
    )
    print(f'# PM strong lines: {sum(len(c) for c in pm_strong_clusters)} '
          f'/ {len(tab_el)} long-list lines')


    #####################################################################
    # END: Reading the input data to be analyzed
    #####################################################################


    #####################################################################
    # START: Generate integrated spectra
    #####################################################################

    integrated = create_integrated_spectra(
        wl__w,
        rss_flux,
        rss_eflux,
        LSF_rss,
        auto_z_min=args.auto_z_min,
        auto_z_max=args.auto_z_max,
        SN_threshold=args.SN_CUT,
        we_range=(6500, 6600),
        cw_range=(6400, 6500),
        bad_pixels=bad_pixels
    )

    int_flux = integrated['int_flux']
    int_eflux = integrated['int_eflux']
    int_flux_we = integrated['int_flux_we']
    int_flux_cw = integrated['int_flux_cw']
    int_flux_w_ce = integrated['int_flux_w_ce']
    rat_flux_cw = integrated['rat_flux_cw']
    int_lsf = integrated['int_lsf']
    SN_map = integrated['SN_map']
    high_SN_mask = integrated['high_SN_mask']
    #m_flux = int_flux_cw
    #e_flux = int_eflux
    m_flux, e_flux, s_flux, m_flux_SN, m_flux_rss = \
      prepare_integrated_spectrum(int_flux_cw, int_eflux, smooth_box=51)
    m_flux_rss = m_flux[np.newaxis, :].copy()

    if ((args.flux_scale[0]==-1) and (args.flux_scale[1]==1)):
      args.flux_scale[0]=-0.1*np.abs(np.median(m_flux_rss))
      args.flux_scale[1]=3*np.abs(np.median(m_flux_rss))+10*np.std(m_flux_rss)

    #####################################################################
    # END: Generate integrated spectra
    #####################################################################
    # Initialize kinematics
    vel__yx = np.zeros(1)
    sigma__yx = 0.8


    #####################################################################
    # START: Automatic search of the guess redshift
    #####################################################################

    (args,
     out_decompose_flux_rss,
     find_z_out_w,
     z_guess) = estimate_integrated_redshift(
         wl__w,
         int_flux_cw,
         e_flux,
         args,
         m_flux_SN=m_flux_SN
     )


    #####################################################################
    # END: Automatic redefine the guess redshift
    #####################################################################


    ###########################################################################
    #
    # START: Removing the emission lines and fitting just the stellar continuum 
    #
    ###########################################################################



    #########################################################
    # We fit the mean/integrated spectrum with RSPs 
    #
    # (1) We define and remove the output files.
    #     We try to minimize the reading/writing of files to speed up the process in version 2.0.0
    #
    try:
        os.makedirs(args.output_path)
        print(f'# dir {args.output_path} created')
    except:
        print(f'# dir {args.output_path} alrady exists')
    out_file_PT = os.path.join(args.output_path, f"m_{args.label}.PT.ecsv")
    out_file_fe = os.path.join(args.output_path, f"m_{args.label}.fe.ecsv")
    out_file_elines = os.path.join(args.output_path, f"m_{args.label}.elines.txt")
    out_file_single = os.path.join(args.output_path, f"m_{args.label}.single.txt")
    out_file_coeffs = os.path.join(args.output_path, f"m_{args.label}.coeffs.txt")
    out_file_fit = os.path.join(args.output_path, f"m_{args.label}.output.fits")
    out_file_ps = os.path.join(args.output_path, f"m_{args.label}.rsp.txt")

    if args.clear_outputs:
        clean_preview_results_files(out_file_ps, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
        clean_preview_results_files(out_file_fe, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
    # ----------------------------------------------------------------------------------------------

    #
    # (2) We set the random seed to ensure reproducibility of the fitting results
    #
    seed = print_time(print_seed=False, get_time_only=True)
    np.random.seed(seed)
    #
    # (3) We fit the integrated spectrum with RSPs
    #

    print(f'### START RSP fitting the integrated spectrum ###')
    lsf_mean = np.mean(LSF_rss, axis=0)
 
    n_narrow = int(getattr(args, "n_narrow", 3))
    n_medium = int(getattr(args, "n_medium", 5))
    n_broad = int(getattr(args, "n_broad", 21)) 
    n_iter = int(getattr(args, "n_iter", 5))

    i_fib = 0 

    tab_m_rsp, tab_m_coeffs, \
    tab_m_elines, tab_m_PM, \
    tab_m_fe, hdu_m_sps, SPS = fit_individual_rss_spectrum_or_adopt_result(
        wl=wl__w,
        flux=int_flux_cw,
        eflux=int_eflux,
        lsf=lsf_mean,
        tab_PT=tab_PT,
        i_fib=i_fib,
        tab_el=tab_el,
        el_clusters=el_clusters,
        hdr_flux=hdr_flux,
        args_copy=args,
        z_guess=args.redshift[0],
        plot=args.plot,
        adopt_st_z = True,
        SN_cut_st_z = args.SN_CUT_INT, 
        SPS_master=None,
        verbose=args.verbose,
        fit_PM=True,
        fit_NP=True,
        bad_pixel_mask=(
            np.all(bad_pixels, axis=0)
            | ~np.isfinite(int_flux_cw)
            | ~np.isfinite(int_eflux)
            | (int_eflux <= 0)
        ),
        empirical_errors = empirical_errors,
        pm_strong_clusters=pm_strong_clusters,
    )
    print(f'### END RSP fitting the integrated spectrum ###')

    label_strongest_line = 'Halpha_6562.85_pm'

    f_strongest = []
    name_strongest = []
    for column in tab_m_PM.colnames:        
        if (column.startswith('flux_') and column.endswith('_pm')) and (column.find('[OI]') == -1):
            f_strongest.append(tab_m_PM[column][0])
            name_strongest.append(column)
    f_strongest = np.array(f_strongest)
    name_strongest = np.array(name_strongest)
    arg_strongest = np.nanargmax(f_strongest)
    label_strongest_line = name_strongest[arg_strongest]
    label_strongest_line = label_strongest_line.replace('flux_', '').replace('_pm', '')
    print(f'# Strongest PM line label: {label_strongest_line}')

    tab_el_strong_dynamic = select_tab_el_strong(
        tab_el, tab_m_PM,
        sn_min=float(getattr(args, "pm_strong_sn_min", 10.0)),
        max_lines=float(getattr(args, "pm_strong_max_lines", 10.0)),
        min_separation=float(getattr(args, "pm_strong_min_separation", 5.0)),
    )
    if len(tab_el_strong_dynamic):
        # This is the RSS FIT3D anchor selection.  The integrated fit above
        # deliberately used the original bootstrap clusters: tab_m_PM was not
        # available until that fit finished.  Rebuild for both RSS fitting paths.
        tab_el_strong = tab_el_strong_dynamic
        pm_strong_clusters = select_pm_strong_clusters(
            tab_el, tab_el_strong,
            max_cluster_span=float(getattr(args, "max_cluster_span", 300.0)),
            max_adjacent_distance=float(getattr(args, "max_adjacent_distance", 100.0)),
        )
        print(f'# RSS PM dynamically selected {len(tab_el_strong_dynamic)} / '
              f'{len(tab_el)} strong lines; FIT3D cluster members: '
              f'{sum(len(c) for c in pm_strong_clusters)}')
    else:
        # Preserve the existing predefined FIT3D strong-line clusters when
        # nothing meets the requested S/N and separation selection criteria.
        print(f'# RSS PM: no dynamic strong lines satisfy selection; using '
              f'{sum(len(c) for c in pm_strong_clusters)} predefined '
              f'FIT3D cluster members')
    # ------------------------------------------------------------
    # PEK fit of the integrated spectrum, using its own Halpha PM kinematics.
    # The independent RSS PEK algorithm and all fitted PM/NP parameters remain
    # unchanged; SPS is deliberately not modified (used later as SPS_master).
    # ------------------------------------------------------------
    tab_m_PEK = None
    m_pek_model = None
    if getattr(args, 'fit_PEK', True):
        ha_vel = float(tab_m_PM[f'vel_{label_strongest_line}_pm'][0])
        ha_disp = float(tab_m_PM[f'disp_{label_strongest_line}_pm'][0])
        if not (np.isfinite(ha_vel) and np.isfinite(ha_disp) and ha_disp > 0):
            print('# Integrated PEK: invalid Halpha velocity/dispersion; retaining PM model')
        else:
            bad_m_pek = (
                np.all(bad_pixels, axis=0)
                | ~np.isfinite(int_flux_cw)
                | ~np.isfinite(int_eflux)
                | (int_eflux <= 0)
            )
            tab_m_PEK, m_pek_model = fit_pek_rss(
                gas_spectrum=np.asarray(SPS.spectra['res_ssp'], dtype=float),
                error_spectrum=int_eflux,
                vel_interp=ha_vel,
                disp_interp=ha_disp,
                tab_el=tab_el,
                tab_PM=tab_m_PM,
                tab_fe=tab_m_fe,
                reference_line=label_strongest_line,
                wl=wl__w,
                lsf_spectrum=lsf_mean,  # same wavelength-dependent LSF as integrated PM
                bad_pixel_mask=bad_m_pek,
                velocity_tolerance=float(getattr(args, 'pek_velocity_tolerance', 5.0)),
                dispersion_tolerance=float(getattr(args, 'pek_dispersion_tolerance', 1.0)),
                fit_padding=float(getattr(args, 'pek_fit_padding', 20.0)),
                max_cluster_span=float(getattr(args, 'pek_max_cluster_span', 75.0)),
                max_adjacent_distance=float(getattr(args, 'pek_max_adjacent_distance', 25.0)),
                verbose=args.verbose,
            )
            add_emission_fit_quality(
                tab_m_PEK, wl__w, SPS.spectra['res_ssp'], m_pek_model,
                int_eflux, bad_pixel_mask=bad_m_pek, suffix='_pek')
            # With no successful cluster, the PEK routine returns a zero model.
            # Do not silently erase the original integrated emission model.
            valid_pek_flux = any(
                np.isfinite(tab_m_PEK[column][0])
                for column in tab_m_PEK.colnames
                if column.startswith('flux_') and column.endswith('_pek')
            )
            if not valid_pek_flux:
                print('# Integrated PEK: no fitted lines; retaining PM model')
                tab_m_PEK = None
                m_pek_model = None
            else:
                # Same convention as RSS: 7=PEK gas, 2=updated joint,
                # 4=updated residual, 8=explicit PEK joint spectrum.
                m_planes = np.asarray(hdu_m_sps[0].data, dtype=float).copy()
                if (m_planes.ndim != 2 or m_planes.shape[0] < 8 or
                        m_planes.shape[1] != len(wl__w)):
                    raise ValueError(f'Unexpected integrated MODEL shape: {m_planes.shape}')
                # SPS.output_fits_hdu currently has 8 planes (0..7).
                # Append plane 8 for the PEK joint spectrum; preserve 0..7.
                if m_planes.shape[0] == 8:
                    m_planes = np.concatenate((m_planes, m_planes[2:3, :].copy()), axis=0)
                old_m_gas = m_planes[7, :].copy()
                m_planes[7, :] = m_pek_model
                m_planes[2, :] = m_planes[2, :] - old_m_gas + m_pek_model
                m_planes[4, :] = m_planes[0, :] - m_planes[2, :]
                m_planes[8, :] = m_planes[2, :]
                for plane in (2, 4, 7, 8):
                    m_planes[plane, bad_m_pek] = np.nan
                hdu_m_sps[0].data = m_planes
                hdu_m_sps[0].header['NAME7'] = 'gas_model_PEK'
                hdu_m_sps[0].header['NAME8'] = 'mod_joint_spec_PEK'


    # Select a separate best line for each integrated-spectrum measurement,
    # using the original NP/PM/PEK per-line quality statistics.
    # The kinematics and individual method tables are left unchanged.
    add_gas_sigma_kms(tab_m_fe, wl__w, lsf_mean, suffix='')
    add_gas_sigma_kms(tab_m_PM, wl__w, lsf_mean, suffix='_pm')
    if tab_m_PEK is not None:
        add_gas_sigma_kms(tab_m_PEK, wl__w, lsf_mean, suffix='_pek')
    tab_m_best = select_best_emission_lines(tab_el, tab_m_fe, tab_m_PM, tab_m_PEK)
    m_best_gas = create_best_emission_model(wl__w, tab_m_best)
    bad_m_best = (np.all(bad_pixels, axis=0) |
                  ~np.isfinite(int_flux_cw) | ~np.isfinite(int_eflux) |
                  (int_eflux <= 0))
    # The integrated spectrum is 1D while the BEST model builder returns
    # (n_spectra, n_wave).  Select its only row for this single-spectrum QA;
    # the RSS path below correctly passes the full 2D BEST model.
    add_best_global_quality(
        tab_m_best, wl__w, np.asarray(hdu_m_sps[0].data[0]) -
        np.asarray(hdu_m_sps[0].data[1]), m_best_gas[0],
        int_eflux, bad_m_best)
    hdu_m_sps[0].data = adopt_best_model_planes(hdu_m_sps[0].data, m_best_gas)
    for plane in (2, 4, 5, 8):
        hdu_m_sps[0].data[plane, bad_m_best] = np.nan
    hdu_m_sps[0].header['NAME8'] = 'gas_model_best'


    #####################################################################
    # Test plot
    ##################################################################### 
    if (args.verbose == True):
        import matplotlib
        matplotlib.use('TkAgg')  # Must be set BEFORE importing pyplot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(wl__w, SPS.spectra['orig_flux_ratio'], color='black', lw=2.0, label='Integrated Spectrum')
        #ax.plot(wl__w, out_decompose_flux_rss['st_flux_cw'], color='yellow', lw=1.5,  alpha = 0.9, label='Decomposed Stellar')
        ax.plot(wl__w, hdu_m_sps[0].data[2], color='tab:blue', lw=1.5,  alpha = 0.9, label='RSP + PEK Model' if tab_m_PEK is not None else 'RSP + Emission Model')
        ax.plot(wl__w, SPS.spectra['model_ssp_min'], color='tab:red', lw=1.5,  alpha = 0.9, label='RSP Model (Smoothed)')
        ax.plot(wl__w, hdu_m_sps[0].data[4], color='tab:orange', lw=1.5,  alpha = 0.9, label='Residual Spectrum')
        ax.plot(wl__w, int_eflux, color='grey', lw=3,  alpha = 0.75, label='Error Spectrum')
        ax.set_xlabel('Wavelength (Angstrom)')
        ax.set_ylabel('Flux')
        ax.set_title('Integrated Spectrum Decomposition')
        ax.legend()
        plt.tight_layout()
        fig.savefig(f'{args.output_path}/output_m.{args.label}.decompose.{out_plot_format}', dpi=300)
        plt.show(block=True)  # Wait until the window is manually closed
        plt.close(fig)

#    exit(0)
    ####################################################################
    # Preliminar version of the output plot
    if (args.do_plots==1):
        y_off = -1.09
        y_off2 = -2.18
        plot_spec_art_hdu(dir='',hdu=hdu_m_sps,\
                      name=args.label,text=args.label,output=f'{args.output_path}/output_m.{args.label}.{out_plot_format}',\
                      c_map='CMRmap',\
                      insets=((0.00, y_off, 0.09, 0.9,3715,3739,-1,35,'[OII]'),\
                              (0.107, y_off, 0.09, 0.9,4088,4118,-0.5,10,r'H$\delta$'),\
                              (0.213, y_off, 0.09, 0.9,4324,4357,-0.5,18,r'H$\gamma$'),\
                              (0.320, y_off, 0.09, 0.9,4851,4872,-1.0,38,r'H$\beta$'),\
                              (0.425, y_off, 0.09, 0.9,4998,5015,-1.0,50,'[OIII]'),\
                              (0.530, y_off, 0.155, 0.9,6543,6590,-2,220,r'H$\alpha$+[NII]'),\
                              (0.700, y_off, 0.09, 0.9,6709,6737,-0.5,20,'[SII]'),\
                              (0.807, y_off, 0.09, 0.9,9062,9076,-1.0,28,'[SIII]'),\
                              (0.910, y_off, 0.09, 0.9,9526,9537,-1.5,82,'[SIII]'),\
                              (0.00, y_off2, 0.22, 0.9,3801,4099,-0.5,3.2,'3800-4100'),\
                              (0.23, y_off2, 0.22, 0.9,4801,5099,-0.2,2.8,'4800-5100'),\
                              (0.46, y_off2, 0.10, 0.9,4363-8.1,4363+7.9,-0.25,0.8,'[OIII]'),\
                              (0.57, y_off2, 0.10, 0.9,4472-8.1,4472+7.9,-0.25,0.8,'HeI'),\
                              (0.68, y_off2, 0.10, 0.9,7320-8,7320+8,-0.2,1.7,'[OII]'),\
                              (0.79, y_off2, 0.10, 0.9,5755-6,5755+6,-0.25,2.5,'[NII]'),\
                              (0.90, y_off2, 0.10, 0.9,6312-6,6312+6,-0.25,2.5,'[SIII]')
                              ),
                      y_min=-0.5,y_max=30,y0_d=0.3,y1_d=2.9,\
                      x_min=3600,x_max=9600,plot_el=True, tab_el=tab_el,plot_res=True,show_scale=False,n_ord=2,gamma=0.5)#,x_min=3600,x_max=9600)



    tab_m_fe_B, tab_m_fe_R, tab_m_fe_I = split_fe_table(tab_m_fe, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    tab_m_PM_B, tab_m_PM_R, tab_m_PM_I = split_fe_table(tab_m_PM, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    hdr_0['SPLITBR'] = (FIT_QUALITY_W_BR, 'B/R table wavelength boundary [A]')
    hdr_0['SPLITRI'] = (FIT_QUALITY_W_RI, 'R/I table wavelength boundary [A]')
    hdr_0['dap_ver']=dap_ver
    hdr_0['dap_date']=dap_date   
    hdu_m_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    hdu_m_FE_B = fits.BinTableHDU(tab_m_fe_B,name='NP_ELINES_B')
    hdu_m_FE_R = fits.BinTableHDU(tab_m_fe_R,name='NP_ELINES_R')
    hdu_m_FE_I = fits.BinTableHDU(tab_m_fe_I,name='NP_ELINES_I')
    hdu_m_PM_B = fits.BinTableHDU(tab_m_PM_B,name='PM_ELINES_B')
    hdu_m_PM_R = fits.BinTableHDU(tab_m_PM_R,name='PM_ELINES_R')
    hdu_m_PM_I = fits.BinTableHDU(tab_m_PM_I,name='PM_ELINES_I')
    tab_m_best_B, tab_m_best_R, tab_m_best_I = split_fe_table(
        tab_m_best, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    hdu_m_RSP = fits.BinTableHDU(tab_m_rsp,name='RSP')
    hdu_m_COEFFS = fits.BinTableHDU(tab_m_coeffs,name='COEFFS')
    hdu_m_dap =fits.HDUList([hdu_m_hdr_0,hdu_m_RSP,hdu_m_COEFFS,\
                           hdu_m_FE_B,hdu_m_FE_R,hdu_m_FE_I,hdu_m_PM_B,hdu_m_PM_R,hdu_m_PM_I,\
                            hdu_info])
    for band, table in (('B', tab_m_best_B), ('R', tab_m_best_R),
                        ('I', tab_m_best_I)):
        hdu_m_dap.append(fits.BinTableHDU(table, name=f'BEST_ELINES_{band}'))
    if tab_m_PEK is not None:
        tab_m_PEK_B, tab_m_PEK_R, tab_m_PEK_I = split_fe_table(
            tab_m_PEK, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
        for band, table in (('B', tab_m_PEK_B), ('R', tab_m_PEK_R), ('I', tab_m_PEK_I)):
            hdu_m_dap.append(fits.BinTableHDU(table, name=f'PEK_ELINES_{band}'))
    out_file_m_dap = os.path.join(args.output_path, f"m_{args.label}.dap.fits.gz")
    hdu_m_dap.writeto(out_file_m_dap,overwrite=True)

    if args.dump_model == True:
        out_file_m_model = os.path.join(args.output_path, f"m_{args.label}.model.fits.gz")
        fits.PrimaryHDU(
            data=hdu_m_sps[0].data,
            header=hdu_m_sps[0].header,
        ).writeto(out_file_m_model, overwrite=True)

    if (only_integrated == True):
      #
      # ***** NEED TO ADD WRITTING RESULTS FOR INTEGRATED SPECTRUM!!!!!
      #
      print("# Only mean spectrum analyzed: END ALL")
      quit()
    

    print("##############################################")
    print("# End fitting the integrated spectrum ########")
    print("##############################################")
    print("\n")


    print(f'### START analyzing the RSS for the strong stellar component ###')

    #
    #  We find the spectra with strong stellar component to be fitted with RSPs
    #     
    w1 = args.w_range_nl[0]*(1+args.redshift[0])
    w2 = args.w_range_nl[1]*(1+args.redshift[0])
    i1 = np.argmin(np.abs(wl__w-w1))
    i2 = np.argmin(np.abs(wl__w-w2))
    # Ignore repaired pixels when selecting reliable stellar fits.
    flux_c = np.nanmean(np.where(~bad_pixels[:, i1:i2],
                                 rss_flux[:, i1:i2], np.nan), axis=-1)
    flux_e = np.nanmean(np.where(~bad_pixels[:, i1:i2],
                                 rss_eflux[:, i1:i2], np.nan), axis=-1)
    mask_st = (flux_c>args.SN_CUT*flux_e) & (flux_c>0)
    len_mask_st = np.sum(mask_st)
    print(f'# Number of spectra with strong stellar component: {len_mask_st} / {len(rss_flux)}')
    print(f'### END analyzing the RSS for the strong stellar component ###')

    if (len_mask_st==0):
        print("# No spectra with strong stellar component: END ALL")
        tab_rsp_interp = []
        tab_coeffs_interp = []
        for (i,id_fib) in enumerate(tab_PT['id']):
            tab_m_rsp_now = tab_m_rsp.copy()
            tab_m_coeffs_now = tab_m_coeffs.copy()
            tab_m_rsp_now['id'] = np.repeat(id_fib, len(tab_m_rsp))
            tab_m_coeffs_now['id'] = np.repeat(id_fib, len(tab_m_coeffs))
            tab_rsp_interp.append(tab_m_rsp_now)
            tab_coeffs_interp.append(tab_m_coeffs_now)

        tab_rsp_interp = tab_vstack(tab_rsp_interp)
        tab_coeffs_interp = tab_vstack(tab_coeffs_interp)
        ############################################################
        # We will assume that all spaxels have as similar stellar component as the integrated spectrum
        ############################################################
    else:
        print("##########################################################################")
        print("# Start fitting the RSS spectra of bright stellar components with RSPs ##")
        print("##########################################################################")
        rss_flux_st = rss_flux[mask_st]
        rss_eflux_st = rss_eflux[mask_st]
        lsf_rss_st = LSF_rss[mask_st]
        tab_PT_st = tab_PT[mask_st]
        z_org = args.redshift[0]
        z_min = args.redshift[1]
        z_max = args.redshift[2]
        tab_rsp_st = []
        tab_coeffs_st = []
        tab_elines_st = []
        tab_PM_st = []
        tab_fe_st = [] 
        nspec = rss_flux_st.shape[0]
        for i, (f__w, ef__w) in enumerate(zip(rss_flux_st, rss_eflux_st)):
            if (args.verbose==True):
                print(f"\n### Fitting spaxel {i+1}/{nspec} (ID: {tab_PT_st['id'][i]})")
            else:
                print_progress_bar(i, nspec, label="St. Fit. progress", bar_width=40)
            tab_i_rsp, tab_i_coeffs, \
                tab_i_elines, tab_i_PM, \
                tab_i_fe, hdu_i_sps, SPS = fit_individual_rss_spectrum_or_adopt_result(
                wl=wl__w,
                flux=f__w,
                eflux=ef__w,
                # DRP per-fiber LSF where available; fall back to mean LSF
                # for generic RSS/simulator inputs without an LSF extension.
                lsf=np.where(np.isfinite(lsf_rss_st[i]) & (lsf_rss_st[i] > 0),
                             lsf_rss_st[i], lsf_mean),
                tab_PT=tab_PT_st,
                i_fib=i,
                tab_el=tab_el,
                el_clusters=el_clusters,
                hdr_flux=hdr_flux,
                args_copy=args,
                z_guess=args.redshift[0],
                plot=args.plot,
                adopt_st_z = True,
                SN_cut_st_z = args.SN_CUT, 
                SPS_master=SPS,
                verbose=args.verbose,
                fit_NP=False,
                fit_PM=True,
                bad_pixel_mask=bad_pixels[mask_st][i],
                empirical_errors = empirical_errors,
                pm_strong_clusters=pm_strong_clusters,
            )
            tab_rsp_st.append(tab_i_rsp)
            tab_coeffs_st.append(tab_i_coeffs)
            tab_elines_st.append(tab_i_elines)
            tab_PM_st.append(tab_i_PM)
            tab_fe_st.append(tab_i_fe)

        tab_rsp_st = tab_vstack(tab_rsp_st)
        tab_coeffs_st = tab_vstack(tab_coeffs_st)
        tab_elines_st = tab_vstack(tab_elines_st)
        tab_coeffs_st = tab_vstack(tab_coeffs_st)
        tab_fe_st = tab_vstack(tab_fe_st)
        tab_PM_st = tab_vstack(tab_PM_st)
        ####################################################################################
        # Now we propagate the velocity map for the remaining spaxels with low stellar S/N
        ####################################################################################
        
        #############################################################################
        # We create the interpolated RSP and coefficients tables for the remaining spaxels
        #############################################################################        
        tab_rsp_interp, tab_coeffs_interp = interpolate_rsp_coeffs(
            tab_PT_st=tab_PT_st,
            tab_rsp_st=tab_rsp_st,
            tab_coeffs_st=tab_coeffs_st,
            tab_PT=tab_PT,
            method="spline",
            smoothing=0,
            decimals=8,
             min_spline_points=10,
            k=4,
            power=2.0,
            verbose=args.verbose
        )


        tab_rsp_interp, tab_coeffs_interp = preserve_rsp_coeffs_samples(
            tab_PT_st,
            tab_rsp_st,
            tab_coeffs_st,
            tab_PT,
            tab_rsp_interp,
            tab_coeffs_interp
        )

        #####################################################
        # Comparison plots if required
        #####################################################

        if (args.do_plots==1):
            vmin = args.redshift[1]*__c__
            vmax = args.redshift[2]*__c__
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(12.5, 6))
            ax[0].scatter(tab_PT_st['ra'], tab_PT_st['dec'], c = tab_rsp_st['z']*__c__, \
                s= 150, cmap='Spectral', alpha = 0.5 , edgecolor='none')
            ax[1].scatter(tab_PT['ra'], tab_PT['dec'], c = tab_rsp_interp['z']*__c__, \
                s= 150, cmap='Spectral', alpha = 0.5 , edgecolor='none')
            ax[0].scatter(tab_PT['ra'], tab_PT['dec'], color='None', alpha = 0.5 , s = 150, edgecolor='Black')
            ax[1].scatter(tab_PT['ra'], tab_PT['dec'], color='None', alpha = 0.5 , s = 150,edgecolor='Black')
            fig.savefig(f'vel_map_interp_{args.label}.png', dpi=300)
        #plt.show()

        print("##########################################################################")
        print("# End fitting the RSS spectra with bright stellar components with RSPs ##")
        print("##########################################################################")


    if (args.verbose==True):
        print("##############################################")
        print("# Interpolated RSP and coefficients tables ###")
        print("##############################################")
        print(tab_rsp_interp.colnames)
        print(tab_coeffs_interp.colnames)
        print(tab_rsp_interp[0:5])
        print(tab_coeffs_interp[0:108])


    ##############################################################################
    # START: Fitting the full RSS spectra using the interpolated RSPs and coefficients
    ##############################################################################

    print("##############################################")
    print("# Start fitting full RSS spectra with interpolated RSPs and coefficients ##")
    print("##############################################")

    tab_rsp = []
    tab_coeffs = []
    tab_elines = []
    tab_PM = []
    tab_fe = [] 
    output_model = []
    gas_spectra_for_pek = []
    nspec = len(rss_flux)
    for i, (f__w, ef__w) in enumerate(zip(rss_flux, rss_eflux)):
        if (args.verbose==True):
            print(f"\n### EL Fitting spaxel {i+1}/{nspec} (ID: {tab_PT['id'][i]})")
        else:
            print_progress_bar(i, nspec, label="EL fit progress", bar_width=40)

        tab_i_rsp, tab_i_coeffs, \
        tab_i_elines, tab_i_PM, \
        tab_i_fe, hdu_i_sps, SPS = fit_individual_rss_spectrum_or_adopt_result(
            wl=wl__w,
            flux=f__w,
            eflux=ef__w,
            lsf=np.where(np.isfinite(LSF_rss[i]) & (LSF_rss[i] > 0),
                         LSF_rss[i], lsf_mean),
            tab_PT=tab_PT,
            i_fib=i,
            tab_el=tab_el,
            el_clusters=el_clusters,
            hdr_flux=hdr_flux,
            args_copy=args,
            z_guess=args.redshift[0],
            plot=args.plot,
            adopt_st_z = True,
            SN_cut_st_z = args.SN_CUT, 
            SPS_master=SPS,
            verbose=args.verbose,
            fit_NP=True,
            fit_PM=True,
            tab_rsp_adopt=tab_rsp_interp,
            tab_coeffs_adopt=tab_coeffs_interp,
            bad_pixel_mask=bad_pixels[i],
            empirical_errors = empirical_errors,
            pm_strong_clusters=pm_strong_clusters,
        )
        tab_rsp.append(tab_i_rsp)
        tab_coeffs.append(tab_i_coeffs)
        tab_elines.append(tab_i_elines)
        tab_PM.append(tab_i_PM)
        tab_fe.append(tab_i_fe)
        # Keep FIT3D/RSP/PEK calculations in their original precision;
        # store each completed MODEL fiber as float32 to halve cube RAM.
        output_model.append(np.asarray(hdu_i_sps[0].data, dtype=np.float32))
        if getattr(args, 'fit_PEK', True):
            # Fit PEK to the observed flux minus the FINAL stellar model.
            # The stored 'res_ssp' is stale after continuum correction.
            gas_spectra_for_pek.append(
                (np.asarray(f__w, float) -
                 np.asarray(SPS.spectra['model_ssp_min'], float)).copy()
            )


    tab_rsp = tab_vstack(tab_rsp)
    tab_coeffs = tab_vstack(tab_coeffs)
    tab_elines = tab_vstack(tab_elines)
    tab_coeffs = tab_vstack(tab_coeffs)
    tab_fe = tab_vstack(tab_fe)
    tab_PM = tab_vstack(tab_PM)

    ########################################################################
    # START: Interpolating the velocity map of the emission lines for the spaxels with low S/N
    ########################################################################





    vel_Ha = tab_PM[f'vel_{label_strongest_line}_pm']
    disp_Ha = tab_PM[f'disp_{label_strongest_line}_pm']
    SN_Ha = tab_PM[f'flux_{label_strongest_line}_pm']/tab_PM[f'e_flux_{label_strongest_line}_pm']
    vel_mask_Ha = (np.asarray(SN_Ha > args.SN_CUT, bool)
                   & np.isfinite(vel_Ha) & np.isfinite(disp_Ha)
                   & (np.asarray(disp_Ha) > 0)
                   & np.isfinite(tab_PT['ra']) & np.isfinite(tab_PT['dec']))
    if np.sum(vel_mask_Ha) >= 4:
        # Preserve the spline map introduced in dap(8).py.
        try:
            vel_map_Ha_interp = interp2d_scattered(
                tab_PT['ra'][vel_mask_Ha], tab_PT['dec'][vel_mask_Ha],
                vel_Ha[vel_mask_Ha], tab_PT['ra'], tab_PT['dec'],
                method='spline', smoothing=0)
            disp_map_Ha_interp = interp2d_scattered(
                tab_PT['ra'][vel_mask_Ha], tab_PT['dec'][vel_mask_Ha],
                disp_Ha[vel_mask_Ha], tab_PT['ra'], tab_PT['dec'],
                method='spline', smoothing=0)
        except (ValueError, RuntimeError):
            vel_map_Ha_interp, disp_map_Ha_interp = interpolate_ha_kinematics(
                tab_PT, tab_PM, sn_cut=float(args.SN_CUT),
                line_key=label_strongest_line)
    elif np.any(vel_mask_Ha):
        # The spline requires sufficient anchors; use linear/nearest interpolation.
        vel_map_Ha_interp, disp_map_Ha_interp = interpolate_ha_kinematics(
            tab_PT, tab_PM, sn_cut=float(args.SN_CUT),
                line_key=label_strongest_line)
    else:
        vel_map_Ha_interp = vel_Ha.copy()
        disp_map_Ha_interp = disp_Ha.copy()
    # A spline may extrapolate invalid or nonphysical values: replace only those.
    vel_map_Ha_interp = np.asarray(vel_map_Ha_interp, dtype=float)
    disp_map_Ha_interp = np.asarray(disp_map_Ha_interp, dtype=float)
    if (np.any(~np.isfinite(vel_map_Ha_interp)) or
            np.any(~np.isfinite(disp_map_Ha_interp) | (disp_map_Ha_interp <= 0))):
        v_fallback, s_fallback = interpolate_ha_kinematics(
            tab_PT, tab_PM, sn_cut=float(args.SN_CUT),
                line_key=label_strongest_line)
        bad_v = ~np.isfinite(vel_map_Ha_interp)
        bad_s = ~np.isfinite(disp_map_Ha_interp) | (disp_map_Ha_interp <= 0)
        vel_map_Ha_interp[bad_v] = v_fallback[bad_v]
        disp_map_Ha_interp[bad_s] = s_fallback[bad_s]

    




    output_model = np.asarray(output_model)
    output_model = np.swapaxes(output_model, 1, 0)  # (n_plane, n_fiber, n_wave)
    # Apply the ORIGINAL mask to every plane only AFTER all computations.
    if output_model.ndim != 3 or output_model.shape[1:] != bad_pixels.shape:
        raise ValueError(f'MODEL shape {output_model.shape} is inconsistent '
                         f'with MASK shape {bad_pixels.shape}')
    output_model[:, bad_pixels] = np.nan

#    print(tab_rsp_interp.colnames)
#    print(tab_coeffs_interp.colnames)
#    print(tab_rsp_interp[0:5])
#    print(tab_coeffs_interp[0:108])
#    print(tab_elines.colnames)
#    print(tab_elines[0:20])
#    print(tab_fe.colnames)
#    print(tab_fe[0:20])
#    print(tab_PM.colnames)
#    print(tab_PM[0:20])



    # PEK: constrained Gaussian + constant-background fits, independent of pyFIT3D.
    # Reuse the Halpha velocity/dispersion maps computed immediately above.
    tab_PEK = None
    pek_model = None
    if getattr(args, 'fit_PEK', True):
        tab_PEK, pek_model = fit_pek_rss(
            gas_spectrum=np.asarray(gas_spectra_for_pek),
            error_spectrum=rss_eflux,
            vel_interp=vel_map_Ha_interp,
            disp_interp=disp_map_Ha_interp,
            tab_el=tab_el,
            tab_PM=tab_PM,
            tab_fe=tab_fe,
            reference_line=label_strongest_line,
            wl=wl__w,
            lsf_spectrum=LSF_rss,       # per-fiber Gaussian sigma [Angstrom]
            bad_pixel_mask=bad_pixels,
            velocity_tolerance=float(getattr(args, 'pek_velocity_tolerance', 5.0)),
            dispersion_tolerance=float(getattr(args, 'pek_dispersion_tolerance', 1.0)),
            fit_padding=float(getattr(args, 'pek_fit_padding', 20.0)),
            max_cluster_span=float(getattr(args, 'pek_max_cluster_span', 75.0)),
            max_adjacent_distance=float(getattr(args, 'pek_max_adjacent_distance', 25.0)),
            verbose=args.verbose,
        )
        add_emission_fit_quality(
            tab_PEK, wl__w, np.asarray(gas_spectra_for_pek), pek_model,
            rss_eflux, bad_pixel_mask=bad_pixels, suffix='_pek')
        add_gas_sigma_kms(tab_PEK, wl__w, LSF_rss, suffix='_pek')
        tab_PEK_B, tab_PEK_R, tab_PEK_I = split_fe_table(
            tab_PEK, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)

    #################################################################################
    # Writting DAP file
    #################################################################################
    if (args.verbose==True):
        print("##############################################")
        print("# Writing output DAP file           ##########")
        print("##############################################")


    # Restored from the pre-2.0 legacy outputs: LSF-deconvolved velocity
    # dispersions, stored per line in the existing wide NP/PM FITS tables.
    # Unlike the old block below exit(0), this uses each fiber's actual LSF
    # at the observed wavelength and does not fabricate resolved dispersions.
    add_gas_sigma_kms(tab_fe, wl__w, LSF_rss, suffix='')
    add_gas_sigma_kms(tab_PM, wl__w, LSF_rss, suffix='_pm')

    # Each line/fiber adopts the lowest valid local reduced chi-square from
    # NP, PM and (if fitted) PEK, copying all quantities and formal errors
    # from that method. Rebuild the Gaussian composite from selected results.
    tab_best = select_best_emission_lines(tab_el, tab_fe, tab_PM, tab_PEK)
    gas_model_best = create_best_emission_model(wl__w, tab_best)
    add_best_global_quality(
        tab_best, wl__w, np.asarray(rss_flux, float) - output_model[1],
        gas_model_best, rss_eflux, bad_pixels)
    tab_best_B, tab_best_R, tab_best_I = split_fe_table(
        tab_best, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    print(f'# BEST selected lines per method: ' + ', '.join(
        f'{method}={sum(np.count_nonzero(np.asarray(tab_best[c]) == method) for c in tab_best.colnames if c.startswith("best_method_"))}'
        for method in ('NP', 'PM', 'PEK')))

    tab_fe_B, tab_fe_R, tab_fe_I = split_fe_table(tab_fe, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    tab_PM_B, tab_PM_R, tab_PM_I = split_fe_table(tab_PM, w_br=FIT_QUALITY_W_BR, w_ri=FIT_QUALITY_W_RI)
    hdr_0['SPLITBR'] = (FIT_QUALITY_W_BR, 'B/R table wavelength boundary [A]')
    hdr_0['SPLITRI'] = (FIT_QUALITY_W_RI, 'R/I table wavelength boundary [A]')
    hdr_0['dap_ver']=dap_ver
    hdr_0['dap_date']=dap_date   
    hdu_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    hdu_PT = fits.BinTableHDU(tab_PT,name='PT')
    hdu_FE_B = fits.BinTableHDU(tab_fe_B,name='NP_ELINES_B')
    hdu_FE_R = fits.BinTableHDU(tab_fe_R,name='NP_ELINES_R')
    hdu_FE_I = fits.BinTableHDU(tab_fe_I,name='NP_ELINES_I')
    hdu_PM_B = fits.BinTableHDU(tab_PM_B,name='PM_ELINES_B')
    hdu_PM_R = fits.BinTableHDU(tab_PM_R,name='PM_ELINES_R')
    hdu_PM_I = fits.BinTableHDU(tab_PM_I,name='PM_ELINES_I')
    hdu_RSP = fits.BinTableHDU(tab_rsp,name='RSP')
    hdu_COEFFS = fits.BinTableHDU(tab_coeffs,name='COEFFS')
    # Averages of all scalar measurements with an associated error column.
    # COEFFS is split by template (rsp); source tables are not modified.
    tab_avg_param_stats = build_average_measurement_statistics_table(
        tab_PT,
        {'PT': tab_PT, 'RSP': tab_rsp, 'COEFFS': tab_coeffs,
         'NP': tab_fe, 'PM': tab_PM, 'PEK': tab_PEK, 'BEST': tab_best},
        get_input_expnum(hdr_0, hdr_flux_org, tab_PT),
    )
    # Fit-quality columns have no paired e_* uncertainties, but should also
    # appear in both average-parameter products as fiber-averaged statistics.
    add_average_xsq_statistics(
        tab_avg_param_stats, {'NP': tab_fe, 'PM': tab_PM, 'PEK': tab_PEK, 'BEST': tab_best})
    hdu_avg_param_stats = fits.BinTableHDU(
        tab_avg_param_stats, name='AVG_PARAM_STATS')
    hdu_avg_param_stats.header['COMMENT'] = (
        'Fiber mean, sample std and combined mean error per measured parameter')
    hdu_mask = fits.ImageHDU(data=input_mask, name='MASK')
    hdu_mask.header['COMMENT'] = 'Original DRP bitmask; nonzero marks a flagged pixel'
    hdu_badpix = fits.ImageHDU(data=bad_pixels.astype(np.uint8), name='BADPIX')
    hdu_badpix.header['COMMENT'] = 'Effective bad-pixel mask; 1=excluded/filled'
    # Compare the masked mean RSS gas spectrum to NP, PM and (when
    # available) PEK gas models. This QA is stored in every RSS DAP file,
    # even when plots and separate model output are disabled.
    tab_avg_el_fit_qa = build_average_emission_quality_table(
        wl__w, np.asarray(rss_flux, float) - output_model[1],
        rss_eflux, bad_pixels,
        {'np': output_model[6], 'pm': output_model[7],
         **({'pek': pek_model} if pek_model is not None else {}),
         'best': gas_model_best},
        tab_el, tab_m_PM)
    hdu_avg_el_fit_qa = fits.BinTableHDU(
        tab_avg_el_fit_qa, name='AVG_EL_FIT_QA')
    hdu_avg_el_fit_qa.header['COMMENT'] = (
        'Mean RSS gas-spectrum quality by line; NP, PM, PEK compared')
    hdu_dap =fits.HDUList([hdu_hdr_0,hdu_PT,hdu_RSP,hdu_COEFFS,\
                           hdu_FE_B,hdu_FE_R,hdu_FE_I,hdu_PM_B,hdu_PM_R,hdu_PM_I])
    out_file_dap = os.path.join(args.output_path, f"{args.label}.dap.fits.gz")
    if tab_PEK is not None:
        for band, table in (('B', tab_PEK_B), ('R', tab_PEK_R), ('I', tab_PEK_I)):
            hdu_dap.append(fits.BinTableHDU(table, name=f'PEK_ELINES_{band}'))
    for band, table in (('B', tab_best_B), ('R', tab_best_R),
                        ('I', tab_best_I)):
        hdu_dap.append(fits.BinTableHDU(table, name=f'BEST_ELINES_{band}'))
    hdu_dap.append(hdu_avg_el_fit_qa)
    hdu_dap.append(hdu_avg_param_stats)
    hdu_dap.append(hdu_mask)
    hdu_dap.append(hdu_badpix)
    hdu_dap.append(hdu_info)
    hdu_dap.writeto(out_file_dap,overwrite=True)

    # A standalone, compact summary with the same primary header as the
    # full DAP file.  Both the per-fiber X_sq statistics (AVG_PARAM_STATS)
    # and the fit quality of the averaged gas spectrum (AVG_EL_FIT_QA)
    # are retained, including all global NP/PM/PEK X_sq measurements.
    out_file_avg_dap = os.path.join(
        args.output_path, f"avg_{args.label}.dap.fits.gz")
    fits.HDUList([
        fits.PrimaryHDU(header=hdu_hdr_0.header.copy()),
        fits.BinTableHDU(tab_avg_param_stats, name='AVG_PARAM_STATS'),
        fits.BinTableHDU(tab_avg_el_fit_qa, name='AVG_EL_FIT_QA'),
        hdu_info.copy(),
    ]).writeto(out_file_avg_dap, overwrite=True)

    # Restored final spatial maps. Keep independent of --dump_model and avoid
    # executing the legacy RSS analysis that remains below exit(0).
    if args.do_plots == 1:
        try:
            tab_DAP = read_DAP_file(out_file_dap, verbose=False)
        except Exception as exc:
            print(f'# Unable to read DAP output for final maps: {exc}')
        else:
            for field in ('flux_Halpha_6562.85', 'med_flux_st'):
                if field not in tab_DAP.colnames:
                    print(f'# Final map skipped: missing {field}')
                    continue
                try:
                    map_plot_DAP(
                        tab_DAP, line=field, vmin=0, vmax=0, title=None,
                        filename=f'{args.label}_{field}', cmap='Spectral',
                        fsize=8, figs_dir=args.output_path,
                        fig_type=out_plot_format, gamma=0.3)
                except Exception as exc:
                    print(f'# Unable to plot {field}: {exc}')

            rgb_keys = (
                'flux_[NII]_6583.45_pek',
                'flux_Halpha_6562.85_pek',
                'flux_[OIII]_5006.84_pek',
            )
            if tab_PEK is None:
                print('# RGB [NII]/Halpha/[OIII] map skipped: PEK not enabled')
            elif not all(key in tab_DAP.colnames for key in rgb_keys):
                missing = [key for key in rgb_keys if key not in tab_DAP.colnames]
                print(f'# RGB PEK map skipped: missing {missing}')
            else:
                try:
                    map_plot_DAP_rgb(
                        tab_DAP,
                        rgb_key=rgb_keys,
                        rgb_scale=(1, 1, 6),
                        vmin=0, vmax=0.0,
                        titles=('[NII]', r'H$\alpha$', '[OIII]'),
                        filename=args.label,
                        fsize=6, figs_dir=args.output_path,
                        gamma=0.35, fig_type='png', fs=15)
                except Exception as exc:
                    print(f'# Unable to plot PEK RGB map: {exc}')

    # Optional first/last RSS diagnostic spectra.  The average-RSS quality
    # comparison above is always stored in the AVG_EL_FIT_QA FITS extension.
    if args.do_plots == 1:
        try:
            files = plot_first_last_rss_diagnostics(
                wl__w, rss_flux, rss_eflux, bad_pixels, output_model,
                pek_model, args.output_path, args.label,
                fig_type=out_plot_format, best_model=gas_model_best)
            print(f'# Saved {len(files)} first/last RSS diagnostic figures')
        except Exception as exc:
            print(f'# Unable to write first/last RSS diagnostic figures: {exc}')

    #######################################################
    # Finalize the existing MODEL planes, and save the averaged spectrum.
    # Only the full RSS MODEL file remains controlled by --dump_model.
    #######################################################
    # Preserve channel 6=NP. Channel 7 holds PEK if available, otherwise PM.
    # The previous ninth channel redundantly duplicated the PEK joint model;
    # it now contains the independent, per-line selected BEST gas model.
    if pek_model is not None:
        output_model[7, :, :] = pek_model
        hdu_i_sps[0].header['NAME7'] = 'gas_model_PEK'
    output_model = adopt_best_model_planes(output_model, gas_model_best)
    hdu_i_sps[0].header['NAME8'] = 'gas_model_best'
    # Original DRP masks must apply to every final MODEL channel, including
    # the PEK-replaced and newly constructed BEST channels.
    output_model[:, bad_pixels] = np.nan

    # Mean each channel over valid fibers, without a full-sized temporary cube.
    # The original bad-pixel mask has already been applied to output_model.
    avg_output_model = average_model_channels(output_model)
    if avg_output_model.shape != (output_model.shape[0], len(wl__w)):
        raise ValueError('Average MODEL has an unexpected spectral shape')
    avg_model_header = hdu_i_sps[0].header.copy()
    avg_model_header['NFIBERS'] = (len(tab_PT), 'Number of averaged RSS fibers')

    # The integrated plot uses the pre-flux-scaled model, so draw the
    # matching average-spectrum figure before converting physical units.
    if args.do_plots == 1:
        y_off = -1.09
        y_off2 = -2.18
        avg_plot_hdu = fits.HDUList([
            fits.PrimaryHDU(data=avg_output_model, header=avg_model_header.copy())])
        try:
            plot_spec_art_hdu(
                dir='', hdu=avg_plot_hdu,
                name=args.label, text=args.label,
                output=f'{args.output_path}/output_avg.{args.label}.{out_plot_format}',
                c_map='CMRmap',
                insets=((0.00, y_off, 0.09, 0.9,3715,3739,-1,35,'[OII]'),
                        (0.107, y_off, 0.09, 0.9,4088,4118,-0.5,10,r'H$\delta$'),
                        (0.213, y_off, 0.09, 0.9,4324,4357,-0.5,18,r'H$\gamma$'),
                        (0.320, y_off, 0.09, 0.9,4851,4872,-1.0,38,r'H$\beta$'),
                        (0.425, y_off, 0.09, 0.9,4998,5015,-1.0,50,'[OIII]'),
                        (0.530, y_off, 0.155, 0.9,6543,6590,-2,220,r'H$\alpha$+[NII]'),
                        (0.700, y_off, 0.09, 0.9,6709,6737,-0.5,20,'[SII]'),
                        (0.807, y_off, 0.09, 0.9,9062,9076,-1.0,28,'[SIII]'),
                        (0.910, y_off, 0.09, 0.9,9526,9537,-1.5,82,'[SIII]'),
                        (0.00, y_off2, 0.22, 0.9,3801,4099,-0.5,3.2,'3800-4100'),
                        (0.23, y_off2, 0.22, 0.9,4801,5099,-0.2,2.8,'4800-5100'),
                        (0.46, y_off2, 0.10, 0.9,4363-8.1,4363+7.9,-0.25,0.8,'[OIII]'),
                        (0.57, y_off2, 0.10, 0.9,4472-8.1,4472+7.9,-0.25,0.8,'HeI'),
                        (0.68, y_off2, 0.10, 0.9,7320-8,7320+8,-0.2,1.7,'[OII]'),
                        (0.79, y_off2, 0.10, 0.9,5755-6,5755+6,-0.25,2.5,'[NII]'),
                        (0.90, y_off2, 0.10, 0.9,6312-6,6312+6,-0.25,2.5,'[SIII]')),
                y_min=-0.5, y_max=30, y0_d=0.3, y1_d=2.9,
                x_min=3600, x_max=9600, plot_el=True, tab_el=tab_el,
                plot_res=True, show_scale=False, n_ord=2, gamma=0.5)
        except Exception as exc:
            print(f'# Unable to plot averaged RSS spectrum: {exc}')

    # Scale both products in place.  Neither operation creates a float64
    # copy of the full MODEL cube.
    output_model /= flux_scaling
    avg_output_model /= flux_scaling
    out_file_avg_model = os.path.join(
        args.output_path, f"avg_{args.label}.model.fits.gz")
    fits.PrimaryHDU(
        header=avg_model_header, data=avg_output_model).writeto(
            out_file_avg_model, overwrite=True)

    if args.dump_model == True:
        if args.verbose == True:
            print("# Writing output model to FITS file")
        out_file_model = os.path.join(args.output_path, f"{args.label}.model.fits.gz")
        hdu_model = fits.PrimaryHDU(
            header=hdu_i_sps[0].header, data=output_model)
        hdu_model.writeto(out_file_model, overwrite=True)

    print("#******   ALL DONE ******#")


    
