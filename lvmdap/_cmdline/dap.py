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
from lvmdap.dap_tools import find_redshift_spec, replace_nan_inf_with_adjacent_avg
from lvmdap.dap_tools import find_continuum
from lvmdap.dap_tools import fit_legendre_polynomial
from lvmdap.dap_tools import read_RSS_PT
from lvmdap.dap_tools import sort_table_by_id
from lvmdap.dap_tools import find_closest_indices_different 
from lvmdap.dap_tools import get_DAP_tab
from lvmdap.flux_elines_tools import flux_elines_RSS_EW_cl
from lvmdap.dap_tools import read_DAP_file,map_plot_DAP,nanaverage,find_closest_indices
from lvmdap.pyFIT3D.common.io import array_to_fits, trim_waves, write_img_header, array_to_hdu
from lvmdap.dap_tools import eline, get_X_sq_out_mod
from lvmdap.modelling.auto_rsp_tools import auto_rsp_elines_rnd
from lvmdap.dap_tools import decompose_spectrum
from lvmdap.dap_tools import find_redshift_elines,print_solution
from lvmdap.dap_tools import find_redshift_weights,running_std_spec
from lvmdap.dap_tools import median_fit_spectrum,fit_emission_line_system
from lvmdap.pyFIT3D.common.io import print_verbose
from lvmdap.dap_tools import process_elines_spec, cluster_emission_lines
from lvmdap.dap_tools import interp2d_scattered
from lvmdap.pyFIT3D.common.stats import shift_convolve
from lvmdap.pyFIT3D.modelling.dust import spec_apply_dust
from astropy.table import vstack as tab_vstack
from scipy.spatial import cKDTree, Delaunay

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
def _no_traceback(type, value, traceback):
  print(value)



##############################################################################
# END: LVM-DAP main functions
##############################################################################


def load_input_data(args, ny_range=None, nx_range=None,
                    sky_hack=False, mask_to_val=True):
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
    """

    # ---------------------------------------------------------
    # 1. Load spectra according to the input format
    # ---------------------------------------------------------

    if not args.lvmsim:

        if not args.in_rss:

            print('# Reading data in the LVMCFrame format...')

            (wl__w, rss_flux_org, rss_eflux_org,
             hdr_flux_org, hdr_0, LSF_rss_org) = load_LVM_rss(
                args.lvm_file,
                ny_range=ny_range,
                nx_range=nx_range,
                sky_hack=sky_hack,
                m2a=1,
                get_lsf=True
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
                nx_range=nx_range
            )

            LSF_rss_org = np.zeros_like(rss_flux_org)

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
            nx_range=nx_range
        )

        LSF_rss_org = np.zeros_like(rss_flux_org)

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
    # 3. Replace non-finite values when requested
    # ---------------------------------------------------------

    if mask_to_val:

        print('# Modifying masked regions with dummy values')

        rss_flux_org = replace_nan_inf_with_adjacent_avg(
            rss_flux_org
        )

        rss_eflux_org = replace_nan_inf_with_adjacent_avg(
            rss_eflux_org
        )

        rss_eflux_org[rss_eflux_org == 0] = np.nanmedian(
            rss_eflux_org
        )

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
        LSF_mean
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
        'out_plot_format': 'pdf',
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
    cw_range=(6400, 6500)
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

    # ---------------------------------------------------------
    # 1. Integrated spectrum selected by emission-line S/N
    # ---------------------------------------------------------

    w0 = 6530 * (1 + auto_z_min)
    w1 = 6650 * (1 + auto_z_max)

    mask_line = (wl > w0) & (wl < w1)

    SN_map = np.array([
        np.nanmax(f[mask_line] / e[mask_line])
        for f, e in zip(rss_flux, rss_eflux)
    ])

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
    else:
        selected_flux = rss_flux[high_SN_mask]
        selected_eflux = rss_eflux[high_SN_mask]



    # ---------------------------------------------------------
    # Inverse-variance weighted mean and its propagated error
    # ---------------------------------------------------------

    valid = (
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

    int_flux = nanaverage(
        selected_flux,
        1 / selected_eflux**2,
        axis=0
    )


    # ---------------------------------------------------------
    # 2. Emission-line-weighted integrated spectrum
    # ---------------------------------------------------------

    we_min, we_max = we_range
    mask_we = (wl > we_min) & (wl < we_max)

    we = np.nanmean(
        rss_flux[:, mask_we],
        axis=1
    )

    we_spec = (
        we[:, np.newaxis]
        * np.ones_like(rss_flux)
        / we.sum()
    )

    int_flux_we = nanaverage(
        rss_flux,
        we_spec,
        axis=0
    )

    # ---------------------------------------------------------
    # 3. Continuum-weighted integrated spectrum
    # ---------------------------------------------------------

    wc_min, wc_max = cw_range
    mask_cw = (wl > wc_min) & (wl < wc_max)

    cw = np.nanmean(
        rss_flux[:, mask_cw],
        axis=1
    )

    cw_spec = (
        cw[:, np.newaxis]
        * np.ones_like(rss_flux)
        / cw.sum()
    )

    int_flux_cw = nanaverage(
        rss_flux,
        cw_spec,
        axis=0
    )

    # ---------------------------------------------------------
    # 4. Continuum-weighted instrumental LSF
    # ---------------------------------------------------------

    int_lsf = nanaverage(
        LSF_rss,
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
        verbose=0
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
            line_name = f"{line['id']}_{np.around(float(line['wl']), 2)}"

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
    SPS.spectra['gas_model_NP'] = create_elines_model_pm(
        wl, tab_fe, dap_fwhm=dap_fwhm, verbose=verbose
    )[0]
    hdu_sps = SPS.output_fits_hdu()


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
    verbose=False
):
    """
    Estimate redshift directly from a gas-only spectrum.

    Uses emission-line peak detection and weighted matching.
    No spectral decomposition or stellar-continuum fitting.

    Returns
    -------
    float
        Best redshift, or input z if no solution is found.
    """

    wl = np.asarray(wl, dtype=float)
    flux = np.asarray(flux, dtype=float)
    eflux = np.asarray(eflux, dtype=float)

    if wl.ndim != 1 or flux.shape != wl.shape or eflux.shape != wl.shape:
        raise ValueError("wl, flux and eflux must be 1D arrays of equal length")

    if z_min > z_max:
        raise ValueError("z_min must be <= z_max")

    # --------------------------------------------------
    # 1. Detect emission-line peaks
    # --------------------------------------------------

    _, _, w_peaks, f_peaks = peak_finder(
        wl, flux, 2,
        peak_threshold=peak_threshold,
        dmin=0.3,
        plot=int(bool(plot)),
        verbose=verbose
    )

    w_peaks = np.asarray(w_peaks, dtype=float)
    f_peaks = np.asarray(f_peaks, dtype=float)

    valid = np.isfinite(w_peaks) & np.isfinite(f_peaks)
    w_peaks = w_peaks[valid]
    f_peaks = f_peaks[valid]

    if len(w_peaks) < min_matches:
        return float(z)

    # --------------------------------------------------
    # 2. Select significant peaks
    # --------------------------------------------------

    peak_indices = np.array([
        np.argmin(np.abs(wl - w))
        for w in w_peaks
    ])

    good = (
        (f_peaks > 0) &
        np.isfinite(eflux[peak_indices]) &
        (eflux[peak_indices] > 0) &
        (f_peaks > n_cut_peaks * eflux[peak_indices])
    )

    obs_waves = w_peaks[good]

    if verbose:
        print(f"# Detected peaks: {len(w_peaks)}")
        print(f"# Significant peaks: {len(obs_waves)}")

    if len(obs_waves) < min_matches:
        return float(z)

    # --------------------------------------------------
    # 3. Match peaks with reference emission lines
    # --------------------------------------------------

    solutions = find_redshift_weights(
        rest_waves=np.array(
            list(REFERENCE_LINES.values()), dtype=float
        ),
        obs_waves=obs_waves,
        rest_names=list(REFERENCE_LINES.keys()),
        line_weights=LINE_WEIGHTS,
        line_groups=LINE_GROUPS,
        zmin=z_min,
        zmax=z_max,
        velocity_tolerance=velocity_tolerance,
        min_matches=min_matches
    )

    # --------------------------------------------------
    # 4. Return best redshift
    # --------------------------------------------------

    if len(solutions) == 0:
        if verbose:
            print(f"# No solution found. Using z={z}")
        return float(z)

    z_best = float(solutions[0].z)

    if verbose:
        print(f"# Best redshift: {z_best:.8f}")
        print(f"# Velocity: {solutions[0].velocity_kms:.2f} km/s")
        print(f"# Matched lines: {solutions[0].n_matches if hasattr(solutions[0], 'n_matches') else 'N/A'}")

    return z_best


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
    if (tab_rsp_adopt is not None) and (tab_coeffs_adopt is not None):
        eline_spec = np.zeros(wl.shape, dtype=float)
        spec_stellar = flux.copy()
        if verbose:
            print(f'# We do not create an eline_spec for {i_fib}')
    else:
        if verbose:
            print(f'# Decomposing the spectrum for {i_fib}')
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
    # wave, spec, dw, method="legendre", degree=11, spline_k=3, spline_s=0.0, min_points_bin=1, return_nodes=False
    res_smooth = median_fit_spectrum(
        wl,
        spec_stellar - model_uncorr,
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
    if (fit_PM) & (len(el_clusters)>0):
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
            verbose=args.verbose
        )

        z = z_gas


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

            for i_line, line in enumerate(cluster):
                pars = np.asarray(EL.final_fit_params_mean[i_line], dtype=float)
                errs = np.asarray(EL.final_fit_params_std[i_line], dtype=float)
                line_name = f"{line['id']}_{np.around(float(line['wl']), 2)}"

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

    #s['orig_flux_ratio'],
    #        s['model_ssp_min'],
    #        s['model_joint'],
    #        s['res_ssp_no_corr'],
    #        s['orig_flux_ratio'] - s['model_joint'],
    #        s['orig_flux_ratio'] - (s['model_joint'] - s['model_ssp_min']),


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
    if (tab_rsp_adopt is not None) and (tab_coeffs_adopt is not None):
        tab_rsp = tab_rsp_adopt[i_fib:i_fib+1]
        tab_coeffs = tab_coeffs_adopt[i_fib:i_fib+1]
    else:
        tab_rsp = create_rsp_table(i_fib, SPS)
        tab_coeffs = create_coeffs_table(i_fib, SPS)
    tab_elines = create_elines_table(i_fib, SPS)
    hdu_sps = SPS.output_fits_hdu()

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
    if (fit_NP) and (len(tab_el)>0):
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
    else:
        tab_fe = Table()
        tab_fe.add_column(np.array([i_fib], dtype=np.int64), name='id_fib', index=0)

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
        ax.plot(wl, spec_stellar, color='cyan', lw=4, label='Decomposed st. spec.')
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


def split_fe_table_org(tab_fe, w_br=5755.0, w_ri=7545.0):
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



import re

def split_fe_table(tab_fe, w_br=5755.0, w_ri=7545.0):
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
        if c in ('id', 'id_fib', 'fiberid', 'ID')
    ]

    colnames_B = []
    colnames_R = []
    colnames_I = []
    unknown_columns = []

    for colname in tab_fe.colnames:

        if colname in id_columns:
            continue

        # Extract wavelength before optional '_pm' suffix
        match = re.search(
            r'_(\d+(?:\.\d+)?)(?:_pm)?$',
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

            values = interpolate(
                tab_coeffs_st[col][ordered_indices]
            )

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



def print_progress_bar(i, total, label="Fit. progress", bar_width=40):
    """
    Display a progress bar updated on the same terminal line.

    Parameters
    ----------
    i : int
        Current iteration (zero-based).
    total : int
        Total number of iterations.
    label : str
        Progress description.
    bar_width : int
        Width of the progress bar.
    """
    if total <= 0:
        return

    fraction = min(1.0, (i + 1) / total)
    percent = 100 * fraction
    filled = int(bar_width * fraction)

    bar = '█' * filled + '░' * (bar_width - filled)

    print(
        f"\r### {label}: |{bar}| "
        f"{percent:5.1f}% ({i+1}/{total})",
        end='\n' if i + 1 >= total else '',
        flush=True
    )


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
     tab_PT_org, LSF_mean) = load_input_data(
         args,
         ny_range=ny_range,
         nx_range=nx_range,
         sky_hack=sky_hack,
         mask_to_val=mask_to_val
     )
    tab_el_strong = read_tab_EL(args.emission_lines_file)
    ######################################################################
    # We mask the non-science spectra and broken fibers ones
    ######################################################################
    rss_flux = rss_flux_org[tab_PT_org['mask']]
    rss_eflux = rss_eflux_org[tab_PT_org['mask']]
    tab_PT = tab_PT_org[tab_PT_org['mask']]
    LSF_rss = LSF_rss_org[tab_PT_org['mask']]
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
        cw_range=(6400, 6500)
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
    )
    print(f'### END RSP fitting the integrated spectrum ###')





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
        ax.plot(wl__w, SPS.spectra['model_joint'], color='tab:blue', lw=1.5,  alpha = 0.9, label='RSP + Emission Model')
        ax.plot(wl__w, SPS.spectra['model_ssp_min'], color='tab:red', lw=1.5,  alpha = 0.9, label='RSP Model (Smoothed)')
        ax.plot(wl__w, SPS.spectra['res_joint'], color='tab:orange', lw=1.5,  alpha = 0.9, label='Residual Spectrum')
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



    tab_m_fe_B, tab_m_fe_R, tab_m_fe_I = split_fe_table(tab_m_fe, w_br=5755.0, w_ri=7545.0)
    tab_m_PM_B, tab_m_PM_R, tab_m_PM_I = split_fe_table(tab_m_PM, w_br=5755.0, w_ri=7545.0)
    hdr_0['dap_ver']=dap_ver
    hdr_0['dap_date']=dap_date   
    hdu_m_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    hdu_m_ELINES = fits.BinTableHDU(tab_m_elines,name='PM_ELINES')
    hdu_m_FE_B = fits.BinTableHDU(tab_m_fe_B,name='NP_ELINES_B')
    hdu_m_FE_R = fits.BinTableHDU(tab_m_fe_R,name='NP_ELINES_R')
    hdu_m_FE_I = fits.BinTableHDU(tab_m_fe_I,name='NP_ELINES_I')
    hdu_m_PM_B = fits.BinTableHDU(tab_m_PM_B,name='PM_ELINES_B')
    hdu_m_PM_R = fits.BinTableHDU(tab_m_PM_R,name='PM_ELINES_R')
    hdu_m_PM_I = fits.BinTableHDU(tab_m_PM_I,name='PM_ELINES_I')
    hdu_m_RSP = fits.BinTableHDU(tab_m_rsp,name='RSP')
    hdu_m_COEFFS = fits.BinTableHDU(tab_m_coeffs,name='COEFFS')
    hdu_m_dap =fits.HDUList([hdu_m_hdr_0,hdu_m_RSP,hdu_m_COEFFS,hdu_m_ELINES,\
                           hdu_m_FE_B,hdu_m_FE_R,hdu_m_FE_I,hdu_m_PM_B,hdu_m_PM_R,hdu_m_PM_I,\
                            hdu_info])
    out_file_m_dap = os.path.join(args.output_path, f"m_{args.label}.dap.fits.gz")
    hdu_m_dap.writeto(out_file_m_dap,overwrite=True)
    
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
    flux_c = np.nanmean(rss_flux[:,i1:i2],axis=-1)
    flux_e = np.nanmean(rss_eflux[:,i1:i2],axis=-1)
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
                lsf=lsf_mean,
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
            fig.savefig(f'vel_map_interp_{args.label}.pdf')
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
    nspec = len(rss_flux)
    for i, (f__w, ef__w) in enumerate(zip(rss_flux, rss_eflux)):
        if (args.verbose==True):
            print(f"\n### EL Fitting spaxel {i+1}/{nspec} (ID: {tab_PT_st['id'][i]})")
        else:
            print_progress_bar(i, nspec, label="EL fit progress", bar_width=40)

        tab_i_rsp, tab_i_coeffs, \
        tab_i_elines, tab_i_PM, \
        tab_i_fe, hdu_i_sps, SPS = fit_individual_rss_spectrum_or_adopt_result(
            wl=wl__w,
            flux=f__w,
            eflux=ef__w,
            lsf=lsf_mean,
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
            tab_coeffs_adopt=tab_coeffs_interp
        )
        tab_rsp.append(tab_i_rsp)
        tab_coeffs.append(tab_i_coeffs)
        tab_elines.append(tab_i_elines)
        tab_PM.append(tab_i_PM)
        tab_fe.append(tab_i_fe)
        output_model.append(hdu_i_sps[0].data)


    tab_rsp = tab_vstack(tab_rsp)
    tab_coeffs = tab_vstack(tab_coeffs)
    tab_elines = tab_vstack(tab_elines)
    tab_coeffs = tab_vstack(tab_coeffs)
    tab_fe = tab_vstack(tab_fe)
    tab_PM = tab_vstack(tab_PM)
    output_model = np.swapaxes(np.array(output_model), 2, 3)


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

    #################################################################################
    # Writting DAP file
    #################################################################################
    if (args.verbose==True):
        print("##############################################")
        print("# Writing output DAP file           ##########")
        print("##############################################")


    tab_fe_B, tab_fe_R, tab_fe_I = split_fe_table(tab_fe, w_br=5755.0, w_ri=7545.0)
    tab_PM_B, tab_PM_R, tab_PM_I = split_fe_table(tab_PM, w_br=5755.0, w_ri=7545.0)
    hdr_0['dap_ver']=dap_ver
    hdr_0['dap_date']=dap_date   
    hdu_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    hdu_PT = fits.BinTableHDU(tab_PT,name='PT')
    hdu_ELINES = fits.BinTableHDU(tab_elines,name='PM_ELINES')
    hdu_FE_B = fits.BinTableHDU(tab_fe_B,name='NP_ELINES_B')
    hdu_FE_R = fits.BinTableHDU(tab_fe_R,name='NP_ELINES_R')
    hdu_FE_I = fits.BinTableHDU(tab_fe_I,name='NP_ELINES_I')
    hdu_PM_B = fits.BinTableHDU(tab_PM_B,name='PM_ELINES_B')
    hdu_PM_R = fits.BinTableHDU(tab_PM_R,name='PM_ELINES_R')
    hdu_PM_I = fits.BinTableHDU(tab_PM_I,name='PM_ELINES_I')
    hdu_RSP = fits.BinTableHDU(tab_rsp,name='RSP')
    hdu_COEFFS = fits.BinTableHDU(tab_coeffs,name='COEFFS')
    hdu_dap =fits.HDUList([hdu_hdr_0,hdu_PT,hdu_RSP,hdu_COEFFS,hdu_ELINES,\
                           hdu_FE_B,hdu_FE_R,hdu_FE_I,hdu_PM_B,hdu_PM_R,hdu_PM_I,\
                            hdu_info])
    out_file_dap = os.path.join(args.output_path, f"{args.label}.dap.fits.gz")
    hdu_dap.writeto(out_file_dap,overwrite=True)
    #######################################################
    # Write output model
    ########################################################
    print(f'SHAPE models: {output_model.shape}')
    print(f'NAXIS1 {hdu_i_sps[0].header["NAXIS1"]}')
    print(f'NAXIS2 {hdu_i_sps[0].header["NAXIS2"]}')
    if (args.dump_model==True):
        if (args.verbose==True):
            print("##############################################")
            print("# Writing output model to fits file ##########")
            print("##############################################")
        out_file_model = os.path.join(args.output_path, f"{args.label}.model.fits.gz") 
        hdi_i_sps[0].header['NAXIS'] = 3              
        hdu_i_sps[0].header['NAXIS1'] = output_model.shape[2]
        hdu_i_sps[0].header['NAXIS2'] = output_model.shape[1]
        hdu_i_sps[0].header['NAXIS3'] = output_model.shape[0]
        hdu_model = fits.PrimaryHDU(header=hdu_i_sps[0].header,data=output_model)
        hdu_model.writeto(out_file_model, overwrite=True)



    exit(0)


    ############################################################################
    # START: Fitting the full RSS spectra with RSPs
    ############################################################################
    print("##############################################")
    print("# Start fitting full RSS spectra  with RSPs ##")
    print("##############################################")
    out_file_fe = os.path.join(args.output_path, f"{args.label}.fe.txt")
    out_file_elines = os.path.join(args.output_path, f"{args.label}.elines.txt")
    out_file_kel = os.path.join(args.output_path, f"{args.label}.kel.txt")
    out_file_single = os.path.join(args.output_path, f"{args.label}.single.txt")
    out_file_coeffs = os.path.join(args.output_path, f"{args.label}.coeffs.txt")
    out_file_fit = os.path.join(args.output_path, f"{args.label}.output.fits.gz")
    out_file_ps = os.path.join(args.output_path, f"{args.label}.rsp.txt")
    out_file_dap = os.path.join(args.output_path, f"{args.label}.dap.fits.gz")
    if args.clear_outputs:
        clean_preview_results_files(out_file_ps, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
        clean_preview_results_files(out_file_fe, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
        clean_preview_results_files(out_file_dap, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
        clean_preview_results_files(out_file_kel, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
    # ----------------------------------------------------------------------------------------------
    # OUTPUT NAMES ---------------------------------------------------------------------------------
    is_guided_sigma = False
    guided_nl = False
    guided_errors = None
    sigma_seq = []
    input_delta_sigma = args.sigma[1]
    input_min_sigma = args.sigma[2]
    input_max_sigma = args.sigma[3]
    model_spectra = []
    tab_coeffs = []
    tab_rsp = []
    tab_elines = []

    y_ratio = None
    ns = rss_flux.shape[0]
    for i, (f__w, ef__w) in enumerate(zip(rss_flux, rss_eflux)):
        if i % max(1, ns // 100) == 0:
           print('.', end='', flush=True)
        print_verbose(f"\n# ID {i}/{ns - 1} ===============================================\n", verbose=args.verbose)
        if i > 0 and is_guided_sigma:
            if SPS.best_sigma > 0:
                sigma_seq.append(SPS.best_sigma)
            guided_sigma = SPS.best_sigma
            k_seq = len(sigma_seq)
            n_seq_last = int(0.2*i)
            if n_seq_last < 10:
                n_seq_last = 10
            if k_seq > n_seq_last:
                guided_sigma = np.median(np.asarray(sigma_seq)[-n_seq_last:])
            input_sigma = guided_sigma
            min_sigma = guided_sigma - input_delta_sigma
            max_sigma = guided_sigma + input_delta_sigma
            delta_sigma = 0.25*input_delta_sigma
            if min_sigma < input_min_sigma:
                min_sigma = input_min_sigma
            if max_sigma > input_max_sigma:
                max_sigma = input_max_sigma
        #
        # We fit all in a single fit 
        #

        #
        # We fix the input parameters
        #
        args.redshift[0] = SPS.best_redshift
        args.sigma[0] = SPS.best_sigma
        args.AV[0] = SPS.best_AV
        cf = SPS.config
        cf.redshift = SPS.best_redshift
        cf.sigma = SPS.best_sigma
        cf.AV = SPS.best_AV
        SPS.full_verbose = args.full_verbose
        _, SPS = auto_rsp_elines_rnd(
            wl__w, f__w, ef__w, ssp_file=args.rsp_file, ssp_nl_fit_file=args.rsp_nl_file,
            config_file=args.config_file,
            w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0],
            nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
            min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
            fit_gas=not args.ignore_gas, refine_gas=not args.single_gas_fit, sigma_gas=args.sigma_gas,
            input_redshift=args.redshift[0], delta_redshift=args.redshift[1], min_redshift=args.redshift[2], max_redshift=args.redshift[3],
            input_sigma=args.sigma[0], delta_sigma=args.sigma[1], min_sigma=args.sigma[2], max_sigma=args.sigma[3],
            input_AV=args.AV[0], delta_AV=args.AV[1], min_AV=args.AV[2], max_AV=args.AV[3], y_ratio=y_ratio,
            sigma_inst=args.sigma_inst, spaxel_id=f"{args.label}_{i}", out_path=args.output_path, plot=args.plot,
            SPS_master=SPS,SN_CUT=SN_CUT,verbose=args.verbose, full_verbose=args.full_verbose
        )
#        y_ratio = SPS.ratio_master

        tab_i_coeffs=create_coeffs_table(i,SPS)
        tab_i_elines=create_elines_table(i,SPS)
        tab_i_rsp=create_rsp_table(i,SPS)

            
        model_spectra.append(SPS.output_spectra_list) 
        tab_coeffs.append(tab_i_coeffs)
        tab_rsp.append(tab_i_rsp)   
        tab_elines.append(tab_i_elines)

    print("\n")
    tab_elines = tab_vstack(tab_elines)
    tab_rsp = tab_vstack(tab_rsp)
    tab_coeffs = tab_vstack(tab_coeffs)


    #
    # This needs to be reevaluatdel']=='eline')
    mask_elines = (tab_elines['model']=='eline')
    tab_elines = tab_elines[mask_elines]
    
    a_wl = np.unique(tab_elines['wl'])
  
    #print(f'a_wl: {a_wl}')
    #print(f'#a_wl: {a_wl.shape}ed, but for now we just save the results
    # 
    model_spectra = np.array(model_spectra).transpose(1, 0, 2)

    #
    # We re-arange the results
    #
    gas_spectra = model_spectra[0,:,:] - model_spectra[1,:,:]
    l_smooth_spectra = gas_spectra*0.0
    try:
      for idx,gas in enumerate(gas_spectra): 
        smooth = median_filter(gas, size=smooth_size, mode='reflect')
        l_smooth = fit_legendre_polynomial(wl__w, smooth,n_leg)
        l_smooth_spectra[idx,:] = l_smooth
    except:
      print("# Warning: something went wrong with the smoothing of the gas spectra, no smooth applied")
      l_smooth_spectra = gas_spectra*0.0
    model_spectra[2,:,:] = model_spectra[2,:,:] + l_smooth_spectra
    model_spectra[3,:,:] = model_spectra[0,:,:] - (model_spectra[1,:,:] + l_smooth_spectra)
    model_spectra[4,:,:] = model_spectra[0,:,:] - model_spectra[2,:,:]


#    tab_rsp=read_rsp(file_ssp=out_file_ps)
#    tab_coeffs=read_coeffs_RSP(coeffs_file=out_file_coeffs)
#    tab_elines=read_elines_RSP(elines_file=out_file_elines)

    id_elines=[]    
    for id_fib in tab_elines['id_fib']:
        id_elines.append(tab_PT['id'].value[id_fib])
    id_elines=np.array(id_elines)
    tab_elines.add_column(id_elines,name='id',index=0)
    
    print("##############################################")
    print("# End fitting full RSS spectra with RSPs #####")
    print("##############################################")
    print("\n")
    print("#####################################################")
    print("# START: Flux_elines analysis on full RSS spectra ###")
    print("#####################################################")

    ############################################################################################
    # END: FIT all RSS spectra
    ############################################################################################


    ############################################################################
    # Run flux_elines full SSP spectrum spectrum
    #
    mask_elines = (tab_elines['model']=='eline')
    tab_elines = tab_elines[mask_elines]
    
    a_wl = np.unique(tab_elines['wl'])
  
    #print(f'a_wl: {a_wl}')
    #print(f'#a_wl: {a_wl.shape}')
    print("#####################")
    print(f'# START: Ord. ELINES table ###')
    print("#####################")
#    print(tab_elines)
#    print(a_wl)
#    exit(0)


    I=0
    tab_PE_ord=Table()
    tab_PE_ord['id']=tab_PT['id']
    for wl_now in a_wl:
      if (wl_now>0.0):
        tab_PE_now=tab_elines[tab_elines['wl']==wl_now]
        tab_PE_tmp=tab_PE_now['id','flux','e_flux','disp','e_disp','vel','e_vel']
        for cols in tab_PE_tmp.colnames:
          if (cols != 'id'):
            tab_PE_tmp.rename_column(cols,f'{cols}_{wl_now}')
        tab_PE_ord=tab_join(tab_PE_ord,tab_PE_tmp,keys=['id'],join_type='left')
        I=I+1
    tab_PE_ord = sort_table_by_id(tab_PE_ord,tab_PT)


        

    print("####################")
    print(f'# END:  Ord. ELINES table ###')
    print("####################")
    w_Ha= a_wl.flat[np.abs(a_wl - 6562.68).argmin()]
    #
    # Patch 25.12.02
    #
    if (np.abs(w_Ha-6562.85)>3):
      w_Ha = 6562.85
    print(f'# WAVELENGTH Ha : {w_Ha}')
    

    vel_mean=np.nanmean(tab_PE_ord[f'vel_{w_Ha}'])
    disp_mean=np.nanmean(tab_PE_ord[f'disp_{w_Ha}'])
    vel__yx=np.zeros(hdr_flux['NAXIS2'])+vel_mean
    sigma__yx=disp_mean
    print('# ELINES Ha kinematics parameters: #')
    print(f'# vel_mean: {vel_mean} #')
    print(f'# disp_mean: {disp_mean} #')


    for I,val in enumerate(tab_PE_ord[f'vel_{w_Ha}'].value):
      if (np.isfinite(val)==True):
        try:
            vel__yx[I]=val
        except:
            vel__yx[I]=vel_mean
      else:
        vel__yx[I] = vel_mean

    #sigma_we = np.nanmedian(sigma__yx)
    sigma_we = 1.0
    fe_data, fe_hdr =flux_elines_RSS_EW_cl(model_spectra[0,:,:]-model_spectra[1,:,:]-l_smooth_spectra, hdr_flux, 5,\
                                           tab_el, vel__yx,\
                                              sigma__yx,eflux__wyx=rss_eflux,\
                                              flux_ssp__wyx=model_spectra[1,:,:],w_range=w_range_FE,\
                                                sigma_we = sigma_we, verbose=args.verbose)

    colnames=[]
    for i in range(fe_data.shape[0]):   
      colname=str(fe_hdr[f'NAME{i}'])+'_'+str(fe_hdr[f'WAVE{i}'])
      colname = colname.replace(' ','_')
      colnames.append(colname)
    colnames=np.array(colnames)
    tab_fe=Table(np.transpose(fe_data),names=colnames)
    tab_fe_B, tab_fe_R, tab_fe_I = split_fe_table(tab_fe, w_br=5755.0, w_ri=7545.0)
    
    print("##################################################")
    print("# END Flux_elines analysis on full RSS spectra ###")
    print("##################################################")
 

    #
    # 27.11.2024
    #
    print("#####################################################")
    print("# START: KIN_ELINES analysis on full RSS spectra ###")
    print("#####################################################")
 
    cf = SPS.config

    # We use as guide the non-parametric derivation
    # vel_map = np.array(tab_PE_ord[f'vel_{w_Ha}'])
    vel_map = np.array(tab_fe[f'vel_Halpha_{w_Ha}'])
    vel_hr = 5*np.nanstd(vel_map)
    vel_min = vel_mean - vel_hr 
    vel_max = vel_mean + vel_hr

    vel_mask_map = None
    # We use as guide the non-parametric derivation
    #    sigma_map = np.array(tab_PE_ord[f'disp_{w_Ha}'])
    #    disp_sigma=np.nanstd(tab_PE_ord[f'disp_{w_Ha}'])
    sigma_map = np.array(tab_fe[f'disp_Halpha_{w_Ha}'])
    disp_sigma=np.nanstd(tab_fe[f'disp_Halpha_{w_Ha}'])
    sigma_fixed = None
    vel_fixed = None


    print(f'# ELINES Ha kinematics parameters: #')
    print(f'# vel_mean: {vel_mean} #')
    print(f'# disp_mean: {disp_mean} #')
    print(f'# Number of systems: {cf.n_systems} #')
    print(f'# vel_min: {vel_min} #')
    print(f'# vel_max: {vel_max} #')
#    plot=1
    tab_kel=[]
    for i_s in range(cf.n_systems):
      syst = cf.systems[i_s]
      z_fact = 1 + vel_mean/__c__
      w_min = syst['start_w']
      w_max = syst['end_w']
      w_min_corr = int(w_min*z_fact)
      w_max_corr = int(w_max*z_fact)
      elcf = cf.systems_config[i_s]
      # v0
      elcf.guess[0][_MODELS_ELINE_PAR['v0']] = vel_mean
      elcf.to_fit[0][_MODELS_ELINE_PAR['v0']] = 1
      elcf.pars_0[0][_MODELS_ELINE_PAR['v0']] = vel_min
      elcf.pars_1[0][_MODELS_ELINE_PAR['v0']] = vel_max
      elcf.links[0][_MODELS_ELINE_PAR['v0']] = -1
      # sigma
      elcf.guess[0][_MODELS_ELINE_PAR['sigma']] = disp_mean
      elcf.to_fit[0][_MODELS_ELINE_PAR['sigma']] = 1
      elcf.pars_0[0][_MODELS_ELINE_PAR['sigma']] = disp_mean-0.5*disp_sigma
      elcf.pars_1[0][_MODELS_ELINE_PAR['sigma']] = disp_mean+2.5*disp_sigma
      elcf.links[0][_MODELS_ELINE_PAR['sigma']] = -1
      sel_wl_range = trim_waves(wl__w, [w_min_corr, w_max_corr])
      #out_file_kel = os.path.join(args.output_path, f"{args.label}.kel_{i_s}")

      #
      # This has to be re-evaluated, but for now we just save the results
      #
      gas_spectra = (model_spectra[0,:,:]-model_spectra[1,:,:]-l_smooth_spectra)
      sec_flux = gas_spectra.T[sel_wl_range]
      sec_eflux = rss_eflux.T[sel_wl_range]
      sec_flux = sec_flux.T
      sec_eflux = sec_eflux.T
      w_minmax_prefix = f'{w_min}_{w_max}'
      _m_str = 'model'
      if elcf.n_models > 1:
        _m_str += 's'
      print(f'# analyzing {elcf.n_models} {_m_str} in {w_minmax_prefix.replace("_", "-")} wavelength range')
      tab_kel_now=kin_rss_elines_main_return(wl__w[sel_wl_range], \
                          sec_flux,\
                          elcf, \
                          rss_eflux=sec_eflux, run_mode='BOTH', guided=False, memo=False, \
                          vel_map=vel_map, vel_mask_map=None, vel_fixed=kin_fixed,\
                          sigma_map=sigma_map, sigma_fixed=kin_fixed, \
                          n_MC=N_MC, n_loops=n_loops, scale_ini=0.15, \
                          fine_search=None, redefine_max=False, max_factor=2, \
                          redefine_min=False, min_factor=0.012, plot=args.plot)
      
      tab_kel.append(tab_kel_now)
    tab_kel=tab_vstack(tab_kel)
#    tab_kel=read_elines_RSP(elines_file=out_file_kel)
    id_kel=[]    
    for id_fib in tab_kel['id_fib']:
        id_kel.append(tab_PT['id'].value[id_fib])
    id_kel=np.array(id_kel)
    tab_kel.add_column(id_kel,name='id',index=0)
      
    print("#####################################################")
    print("# END: KIN_ELINES analysis on full RSS spectra ###")
    print("#####################################################")


    

    print("##################################################")
    print("# START: Storing the results in a single file  ###")
    print("##################################################")


    #print('tab_PT: ',tab_PT)
 #   print('tab_PT.row: ',len(tab_PT))
  #  print('tab_PT.colnames: ', len(tab_PT.colnames))
    #print('tab_PT: ',tab_PT)
 #   print('tab_fe.row: ',len(tab_fe))
 #   print('tab_fe.colnames: ', len(tab_fe.colnames))     
 #   print('tab_fe_B.row: ',len(tab_fe_B))
 #   print('tab_fe_R.row: ',len(tab_fe_R))
 #   print('tab_fe_I.row: ',len(tab_fe_I))
#    print('tab_fe_B.colnames: ', len(tab_fe_B.colnames))     
    
    tab_rsp.add_column(tab_PT['id'].value,name='id',index=0)
    tab_fe.add_column(tab_PT['id'].value,name='id',index=0)
    tab_fe_B.add_column(tab_PT['id'].value,name='id',index=0)
    tab_fe_R.add_column(tab_PT['id'].value,name='id',index=0)
    tab_fe_I.add_column(tab_PT['id'].value,name='id',index=0)
#    print('FE_tab: ',tab_fe)
#    print('FE_tab_rows: ',len(tab_fe))
#    print('FE_tab.colnames: ', len(tab_fe.colnames))     

      

    
    id_coeffs=[]    
    for id_fib in tab_coeffs['id_fib']:
        id_coeffs.append(tab_PT['id'].value[id_fib])
    id_coeffs=np.array(id_coeffs)
    tab_coeffs.add_column(id_coeffs,name='id',index=0)

    #########################################################################
    # We determine and correct for the instrumental resolution
    #########################################################################

    tab_DAP = get_DAP_tab(tab_PT.copy(),tab_rsp.copy(),tab_coeffs.copy(),\
                      tab_elines.copy(),tab_fe_B.copy(),tab_fe_R.copy(),\
                        tab_fe_I.copy(),tab_kel.copy(), kel_ext = 1, verbose=args.verbose)
    

    tab_el_chi = Table()
    el_names = []
    el_wave = []
    el_inst_disp = []
    a_sig_name = []
    for cols in tab_DAP.colnames:
      if (cols.find('cor_') != -1):
          tab_DAP.remove_column(cols)

    a_sig_name.append('id')
    for cols in tab_DAP.colnames:
      if ((cols.find('disp_') != -1) & ~(cols.find('_st') !=-1) & ~(cols.find('e_disp') !=-1)):
          name = cols.replace('disp_','')
          a_wave = cols.split('_')
          wave = float(a_wave[-1])
          el_names.append(name)
          el_wave.append(wave)
          i_wl_el = find_closest_indices(wl__w, wave)
          lsf_now = LSF_rss[:,i_wl_el[0]]
#          lsf_now = lsf_now[pt_table['mask']]
          el_inst_disp.append(lsf_now)
          tab_DAP[f'lsf_{name}'] = lsf_now
          a_sig_name.append(f'sigma_kms_{name}')
          f_fwhm = 1
          sigma_obs = np.abs((tab_DAP[cols]/f_fwhm))
          min_sigma_obs = np.nanmin(sigma_obs)
          min_lsf = np.nanmin(lsf_now)
          if (min_sigma_obs<min_lsf):
              lsf_now = (min_sigma_obs/min_lsf)*lsf_now
          tab_DAP[f'sigma_kms_{name}'] = np.float32(np.sqrt(np.abs((tab_DAP[cols]/f_fwhm)**2 - lsf_now**2))*__c__ /(wave))

    tab_el_chi['name'] = np.array(el_names)
    tab_el_chi['wave'] = np.array(el_wave)
    tab_el_chi['inst_disp'] = np.array(el_inst_disp)
    tab_sig_kms = tab_DAP[a_sig_name]


    ########################################################
    # Dump the final model
    ########################################################
    model_spectra_rsp=np.array(model_spectra[1])

    crpix1 = 1
    crval1 = wl__w[0]
    cdelt1 = wl__w[1]-wl__w[0]
    (ny,nx) = rss_flux.shape
    hdr_out = fits.Header()
    dap_fwhm = 1.0
    f_scale = 1e16
    hdr_out['CRPIX1'] = crpix1
    hdr_out['CRVAL1'] = crval1
    hdr_out['CDELT1'] = cdelt1
    hdr_out['NAME0'] = 'org_spec'
    hdr_out['NAME1'] = 'model_spec'
    hdr_out['NAME2'] = 'mod_joint_spec'
    hdr_out['NAME3'] = 'gas_spec'
    hdr_out['NAME4'] = 'res_joint_spec'
    hdr_out['NAME5'] = 'no_gas_spec'
    hdr_out['NAME6'] = 'np_mod_spec'
    hdr_out['NAME7'] = 'pe_mod_spec'
    hdr_out['NAME8'] = 'pk_mod_spec'

    model_spectra = np.zeros((9,ny,nx), dtype=np.float32)
    model_spectra[0,:,:]=np.copy(rss_flux)
    #
    # 18.12.2025
    # Model equal to gen-mod
    #

    pe_e=[]
    w_e=[]
    for td_col in tab_DAP.columns:
      if ((td_col.find("_pe_")==-1) & (td_col.find("_pek_")==-1) &  (td_col.find("e_flux_")>-1)):
        l_name = td_col.replace('e_flux_','')
        w_name = float(l_name.split("_")[-1])
        pe_e.append(l_name)
        w_e.append(w_name)
    print(f'# Number of emission lines in the model: {len(pe_e)}')
    #
    # Create emission lines!
    #
    print("##############################################")
    print('# Creating NP the emission line models\n')
    spec2D_elines=np.copy(0.0*rss_flux)
    for i,spec2D_now in enumerate(spec2D_elines):
      for j,(pe_now,w_now) in enumerate(zip(pe_e,w_e)):        
        F=tab_DAP[f'flux_{pe_now}'][i]
        D=tab_DAP[f'disp_{pe_now}'][i]
        V=tab_DAP[f'vel_{pe_now}'][i]
        spec_eline = eline(wl__w,w_now,F,D,V,dap_fwhm=dap_fwhm)
        spec2D_now += spec_eline
      
    print("# Done...")
    print("##############################################")
    model_spectra[6,:,:]=np.copy(spec2D_elines)


    pe_e=[]
    w_e=[]
    for td_col in tab_DAP.columns:
      if ((td_col.find("_pe_")>-1) & (td_col.find("_pek_")==-1) &  (td_col.find("e_flux_")>-1)):
        l_name = td_col.replace('e_flux_','')
        w_name = float(l_name.split("_")[-1])
        pe_e.append(l_name)
        w_e.append(w_name)
    print(f'# Number of emission lines in the model: {len(pe_e)}')
    #
    # Create emission lines!
    #
    print("##############################################")
    print('# Creating the emission line models\n')
    spec2D_elines=np.copy(0.0*rss_flux)
    for i,spec2D_now in enumerate(spec2D_elines):
      for j,(pe_now,w_now) in enumerate(zip(pe_e,w_e)):        
        F=tab_DAP[f'flux_{pe_now}'][i]
        D=tab_DAP[f'disp_{pe_now}'][i]
        V=tab_DAP[f'vel_{pe_now}'][i]
        spec_eline = eline(wl__w,w_now,F,D,V,dap_fwhm=dap_fwhm)
        spec2D_now += spec_eline      
    print("# Done...")
    print("##############################################")
    model_spectra[7,:,:]=np.copy(spec2D_elines)
    model_spectra[1]=model_spectra_rsp
    res_spectra = model_spectra[0,:,:] - model_spectra[1,:,:] - model_spectra[6,:,:]
    l_smooth_spectra = res_spectra*0.0
    print(f'# smooth_size, n_legrendre : {smooth_size},{n_leg}')
    for idx,res in enumerate(res_spectra): 
      try:
        smooth = median_filter(res, size=smooth_size)
        l_smooth = fit_legendre_polynomial(wl__w, smooth, n_leg)
        l_smooth_spectra[idx] = l_smooth
      except:
        print("# Warning: something went wrong with the smoothing of the residual spectra, no smooth applied")
    print("# Done ...")
    print("##############################################")
    model_spectra[2,:,:] = model_spectra[1,:,:] + model_spectra[6,:,:] + l_smooth_spectra
    model_spectra[3,:,:] = model_spectra[0,:,:] - (model_spectra[1,:,:] + l_smooth_spectra)
    model_spectra[4,:,:] = model_spectra[0,:,:] - model_spectra[2,:,:]
    model_spectra[5,:,:] = model_spectra[0,:,:] - model_spectra[6,:,:]
    model_spectra[8,:,:] = model_spectra[1,:,:] + model_spectra[7,:,:] + l_smooth_spectra
    gas_spectra = model_spectra[0,:,:] - model_spectra[1,:,:] - l_smooth_spectra
    fits_name = out_file_fit if out_file_fit.endswith(".gz") else out_file_fit+".gz"
    h = {}
    h['CRPIX1'] = 1
    h['CRVAL1'] = wl__w[0]
    h['CDELT1'] = wl__w[1] - wl__w[0]
    h['NAME0'] = 'org_spec'
    h['NAME1'] = 'model_spec'
    h['NAME2'] = 'mod_joint_spec'
    h['NAME3'] = 'gas_spec'
    h['NAME4'] = 'res_joint_spec'
    h['NAME5'] = 'no_gas_spec'
    h['NAME6'] = 'gas_model_NP'
    h['NAME7'] = 'gas_model_PEK'
    h['NAME8'] = 'mod_joint_spec_PEK'
    h["FILENAME"] = fits_name
    #model_spectra_out = model_spectra.astype(np.float32)
    

    ##########################################################
    # Estimating the X_sq of the emission line model
    # and stellar population + emission line model
    #
    ##########################################################

    print("####################################################")
    print("# Estimating the X_sq of the emission line models ###")
    print("####################################################")
    n_rsp = len(np.unique(tab_coeffs['rsp']))
    n_free_st = nx-n_rsp-3-1
    n_free_el = nx-3-1
    error_map = np.abs(rss_eflux.copy())+np.nanmedian(np.abs(rss_eflux))#+np.sqrt(np.abs(model_spectra[1,:,:]+model_spectra[6,:,:]))*np.sqrt(20)
    #error_map = error_map_all.copy()
    #error_map = np.abs(rss_eflux.copy())+np.sqrt(np.abs(model_spectra[1,:,:]+model_spectra[6,:,:]))*np.sqrt(20)
    #error_map = np.abs(rss_eflux.copy())+np.abs(np.nanmedian(model_spectra[1,:,:]))+\
    #  np.sqrt(np.abs(model_spectra[1,:,:]))    
#    error_map = np.sqrt(((np.abs(rss_eflux.copy()))**2 + (np.abs(model_spectra[0,:,:])))/2)
#    error_map = np.sqrt(((np.abs(rss_eflux.copy()))**2 + (np.abs(model_spectra[0,:,:])))/2)
#    error_map = np.abs(rss_eflux.copy())+np.abs(np.nanmedian(model_spectra[4,:,:]))+\
#      np.sqrt(np.abs(model_spectra[0,:,:]))

    X_sq_st = (1/n_free_st) * np.nansum(((model_spectra[0,:,:] - model_spectra[1,:,:])**2/((error_map**2))),axis=1)
    X_sq_st_np = (1/n_free_el) * np.nansum(((model_spectra[4,:,:])**2/((error_map**2))),axis=1)
    X_sq_st_pek = (1/n_free_el) * np.nansum(((model_spectra[8,:,:])**2/((error_map**2))),axis=1)

    tab_sig_kms['X_sq_st'] = X_sq_st
    tab_sig_kms['X_sq_st_np'] = X_sq_st_np
    tab_sig_kms['X_sq_st_pek'] = X_sq_st_pek

    

    print(f"mean_X_sq_st = {np.nanmean(X_sq_st):.2f} (n_free_st = {n_free_st})")
    print(f"mean_X_sq_st_np = {np.nanmean(X_sq_st_np):.2f} (n_free_el = {n_free_el})")
    print(f"mean_X_sq_st_pek = {np.nanmean(X_sq_st_pek):.2f} (n_free_el = {n_free_el})")

    w_range_fe=10*int(np.nanmedian(tab_el_chi['inst_disp']))
    #0.75*w_range_FE
    if (w_range_fe<6):
      w_range_fe=6.0
    a_X_sq = []
    a_fr = []
    zz = args.redshift[0]
    for name,wave_now,inst_disp in tab_el_chi['name','wave', 'inst_disp']:
      mask_wl = (wl__w > (wave_now* (1+zz) - w_range_fe)) & (wl__w < (wave_now* (1+zz) + w_range_fe)) 
      mask_2d = np.tile(mask_wl, (ny, 1))
      weight = np.ones_like(model_spectra[0,:,:])
      weight[~mask_2d] = 0.0
      fa_now = np.nansum(np.abs(gas_spectra)*weight,axis=1)
      #error_map = error_map_all.copy()+np.sqrt(np.nanmax((model_spectra[1,:,:]+model_spectra[6,:,:])*weight))
      if (name.find('pe') != -1):
        if (name.find('pek') != -1):
            rat_set = ((gas_spectra-model_spectra[7,:,:])**2/((error_map**2)))*weight
            fr_set = ((np.abs(gas_spectra-model_spectra[7,:,:]))*weight)/fa_now[:, np.newaxis]
        else:
            rat_set = ((model_spectra[4,:,:])**2/((error_map**2)))*weight
            fr_set = ((np.abs(model_spectra[4,:,:]))*weight)/fa_now[:, np.newaxis]
      else:
        rat_set = ((gas_spectra-model_spectra[6,:,:])**2/((error_map**2)))*weight
        fr_set = ((np.abs(gas_spectra-model_spectra[6,:,:]))*weight)/fa_now[:, np.newaxis]
      n_free = len(wl__w[mask_wl])-3-1
      X_sq_now = (1/n_free) * np.nansum(rat_set*weight,axis=1)
      tab_sig_kms[f'X_sq_{name}'] = X_sq_now
      fr_now = np.nansum(fr_set,axis=1)
      tab_sig_kms[f'fr_{name}'] = fr_now
      a_X_sq.append(np.nanmedian(X_sq_now))
      a_fr.append(np.nanmedian(fr_now))
    tab_el_chi['X_sq'] = np.array(a_X_sq)
    tab_el_chi['fr'] = np.array(a_fr)

    ###############################################
    # Additional X-sq analysis
    avg_model_spectra = np.nanmean(model_spectra,axis=1)
    a_X_sq_org = []
    a_X_sq_pek = []
    a_X_sq_np = []
    for name,wave_now in tab_el_chi['name','wave']:
      wmin_X = (wave_now* (1+zz) - w_range_fe)
      wmax_X = (wave_now* (1+zz) + w_range_fe)
      print(f'# Analyzing line: {name} at {wave_now} A (wmin_X, wmax_X)=({wmin_X},{wmax_X})')
      a_X_now = get_X_sq_out_mod(wl__w,avg_model_spectra,wmin_X,wmax_X)
      print(f'# X_sq_org: {a_X_now[0]:.2f}, X_sq_pek: {a_X_now[1]:.2f}, X_sq_np: {a_X_now[2]:.2f}')
      a_X_sq_org.append(a_X_now[0])
      a_X_sq_pek.append(a_X_now[1])
      a_X_sq_np.append(a_X_now[2])
    tab_el_chi['X_sq_avg_org'] = np.array(a_X_sq_org)
    tab_el_chi['X_sq_avg_pek'] = np.array(a_X_sq_pek)
    tab_el_chi['X_sq_avg_np'] = np.array(a_X_sq_np)


    #####################################################
    # Storing the results
    #####################################################


    #
    # We add new entries to the header
    #
    #print('##############################')
    #print('# Updating the header ###############')
    #hdr_0['dap_ver']=1.240208
    #hdr_0['dap_ver']=1.240730
    #hdr_0['dap_ver']='1.1.0.241106'
    #hdr_0['dap_ver']='1.1.0.241127'
    #hdr_0['dap_ver']='1.1.0.250322'
    #hdr_0['dap_ver']='1.1.0.250507'
    #hdr_0['dap_ver']='1.2.0.251218'
    #hdr_0['dap_ver']='1.2.0.260405'
    #hdr_0['dap_ver']='1.2.0.260421'
    #hdr_0['dap_ver']='1.2.1'


    hdr_0['dap_ver']=dap_ver
    hdr_0['dap_date']=dap_date
    #for key in dict_param.keys():
    #  val=dict_param[key]
    #  hdr_0[key]=val
    #   # print(dict_param)
    # quit()
    #print('# Header updated    #################')
    #print('##############################')


    
    hdu_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    hdu_PT = fits.BinTableHDU(tab_PT,name='PT')
    hdu_ELINES = fits.BinTableHDU(tab_elines,name='PM_ELINES')
    #hdu_FE = fits.BinTableHDU(tab_fe,name='FE')
    hdu_FE_B = fits.BinTableHDU(tab_fe_B,name='NP_ELINES_B')
    hdu_FE_R = fits.BinTableHDU(tab_fe_R,name='NP_ELINES_R')
    hdu_FE_I = fits.BinTableHDU(tab_fe_I,name='NP_ELINES_I')
    hdu_RSP = fits.BinTableHDU(tab_rsp,name='RSP')
    hdu_COEFFS = fits.BinTableHDU(tab_coeffs,name='COEFFS')
    hdu_KEL = fits.BinTableHDU(tab_kel,name='PM_KEL')
    hdu_SIGMA = fits.BinTableHDU(tab_sig_kms, name="ELINES_SIGMA_CHI")
    hdu_el_chi = fits.BinTableHDU(tab_el_chi, name="ELINES_CHI2_AVG")

    hdu_dap =fits.HDUList([hdu_hdr_0,hdu_PT,hdu_RSP,hdu_COEFFS,hdu_ELINES,\
                           hdu_FE_B,hdu_FE_R,hdu_FE_I,hdu_KEL,hdu_SIGMA,\
                            hdu_el_chi, hdu_info])
    hdu_dap.writeto(out_file_dap,overwrite=True)
    print(f'# dap_file: {out_file_dap} written')
    print("##################################################")
    print("# END: Storing the results in a single file    ###")
    print("##################################################")

    if (args.do_plots==1):
      print("##################################################")
      print("# START: Plotting Ha and continous flux maps                                                   ###")
      print("##################################################")
      tab_DAP=read_DAP_file(out_file_dap,verbose=args.verbose)
      param='flux_Halpha_6562.85'
      try:
        map_plot_DAP(tab_DAP,line=param, \
                     vmin=0, vmax=0, title=None, filename=f'{args.label}_{param}',\
                     cmap='Spectral', fsize=8, figs_dir=args.output_path,fig_type=out_plot_format,\
                     gamma=0.3)    
      except Exception as error:
        print(f'{param} does not exits?',error)
        
      param='med_flux_st'
      try:
        map_plot_DAP(tab_DAP,line=param, \
                     vmin=0, vmax=0, title=None, filename=f'{args.label}_{param}',\
                     cmap='Spectral', fsize=8, figs_dir=args.output_path,fig_type=out_plot_format,\
                     gamma=0.3)
      except Exception as error:  
        print(f'{param} does not exits',error)

    #args.output_path, f"elines_{args.label}")     
      
      print("##################################################")
      print("# End:  Plotting Ha and continous flux maps                                                    ###")
      print("##################################################")

    if (dump_model==True):
      print("###################################################");
      print(f"# Start: Dumping the final model: {out_file_fit}                                                      #");
      #
      # Patch model
      #
      res_spec = model_spectra[0,:,:]-(model_spectra[8,:,:])
      s_res_spec = res_spec.copy()
      smooth_spec = res_spec.copy()
      smooth_size = 21
      for idx,res in enumerate(res_spec): 
        smooth = median_filter(res, size=smooth_size)
        smooth_spec[idx] = smooth
        s_res_spec[idx] = res-smooth
      mask_NP_neg = model_spectra[6,:,:]<0
      NP_neg = model_spectra[6,:,:].copy()
      NP_neg[~mask_NP_neg] = 0  
      NP_pos = model_spectra[6,:,:].copy()
      NP_pos[mask_NP_neg] = 0
      model_spectra[1,:,:] = model_spectra[8,:,:]-model_spectra[7,:,:]+smooth_spec+NP_neg
      model_spectra[2,:,:] = model_spectra[8,:,:]+smooth_spec+NP_neg
      model_spectra[3,:,:] = model_spectra[0,:,:]-model_spectra[1,:,:]
      model_spectra[6,:,:] = NP_pos



      array_to_fits(out_file_fit, model_spectra/f_scale, overwrite=True)
      write_img_header(out_file_fit, list(h.keys()), list(h.values()))
      #dump_rss_output(out_file_fit=out_file_fit, wavelength=wl__w, model_spectra=model_spectra)
      print("# End: Dumping the final model                                                                               #");
      if (args.do_plots==1):
        plot_spectra(dir='',n_sp=0, file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_first.{args.label}.{out_plot_format}',\
                     insets=((0.25, 0.5, 0.22, 0.47,4840,5020,-0.5,16,''),\
                             (0.01, 0.5, 0.22, 0.47,3851,3999,-0.5,3,''),\
                             (0.52, 0.5, 0.22, 0.47,6303,6322,0.5,2,'[SIII]6312'),\
                             (0.76, 0.5, 0.22, 0.47,9055,9083,-0.5,10,'[SIII]9069')),\
                     y_min=-3,y_max=66,y0_d=0.3,y1_d=2.9,\
                     x_min=3600,x_max=9600,plot_el=True, tab_el=tab_el)
        plot_spectra(dir='',n_sp=0,file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     x_min=6500,x_max=6600,y_min=-0.2,y_max=15.5,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_first_6500.{args.label}.{out_plot_format}')
        plot_spectra(dir='',n_sp=0,file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     x_min=6700,x_max=6750,y_min=-0.2,y_max=15.5,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_first_6700.{args.label}.{out_plot_format}')
        plot_spectra(dir='',n_sp=0,file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     x_min=4800,x_max=5030,y_min=-0.2,y_max=15.5,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_first_5000.{args.label}.{out_plot_format}')        
        
        
        plot_spectra(dir='',n_sp=hdr_flux['NAXIS2']-1, file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_last.{args.label}.{out_plot_format}',\
                     insets=((0.25, 0.5, 0.22, 0.47,4840,5020,-0.5,16,''),\
                             (0.01, 0.5, 0.22, 0.47,3851,3999,-0.5,3,''),\
                             (0.52, 0.5, 0.22, 0.47,6303,6322,0.5,2,'[SIII]6312'),\
                             (0.76, 0.5, 0.22, 0.47,9055,9083,-0.5,10,'[SIII]9069')),\
                     y_min=-3,y_max=66,y0_d=0.3,y1_d=2.9,\
                     x_min=3600,x_max=9600,plot_el=True, tab_el=tab_el)
        plot_spectra(dir='',n_sp=hdr_flux['NAXIS2']-1,file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     x_min=6500,x_max=6600,y_min=-0.2,y_max=15.5,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_last_6500.{args.label}.{out_plot_format}')
        plot_spectra(dir='',n_sp=hdr_flux['NAXIS2']-1,file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     x_min=6700,x_max=6750,y_min=-0.2,y_max=15.5,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_last_6700.{args.label}.{out_plot_format}')
        plot_spectra(dir='',n_sp=hdr_flux['NAXIS2']-1,file=out_file_fit,\
                     file_ssp = out_file_ps,\
                     x_min=4800,x_max=5030,y_min=-0.2,y_max=15.5,\
                     name=args.label,text=args.label,output=f'{args.output_path}/output_last_5000.{args.label}.{out_plot_format}')        
        
        



    
    print("#******   ALL DONE ******#")


    
