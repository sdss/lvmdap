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
# START: Import required packages                                            #
##############################################################################
'''

#################################################################
# Defining the number of cores to to use in paralelization      #
#################################################################

import sys, os
import wave
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
from lvmdap.pyFIT3D.common.gas_tools import detect_create_ConfigEmissionModel
from lvmdap.pyFIT3D.common.io import create_ConfigAutoSSP_from_lists
from lvmdap.pyFIT3D.common.io import create_emission_lines_file_from_list
from lvmdap.pyFIT3D.common.io import create_emission_lines_mask_file_from_list
from lvmdap.pyFIT3D.common.gas_tools import kin_rss_elines_main
from lvmdap.pyFIT3D.common.constants import __c__, __Ha_central_wl__, _MODELS_ELINE_PAR
from lvmdap.modelling.synthesis import StellarSynthesis
from lvmdap.dap_tools import load_LVM_rss, read_PT, plot_spec, read_rsp
from lvmdap.dap_tools import plot_spec_art, Table_mean_rows
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
from lvmdap.pyFIT3D.common.io import array_to_fits, trim_waves, write_img_header
from lvmdap.dap_tools import eline, get_X_sq_out_mod
from lvmdap.modelling.auto_rsp_tools import auto_rsp_elines_rnd




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
        type=int,
        default=0,
        help="Plot mode: 0=no plot, 1=interactive, 2=save to file"
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

    int_flux = nanaverage(
        selected_flux,
        1 / selected_eflux**2,
        axis=0
    )

    # Preserve the notebook error estimator
    int_eflux = np.sqrt(
        np.nanmedian(rss_eflux**2, axis=0)
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


#----------------------------------------------------------------



####################################################
# MAIN script. Uses the entries from commands lines: Depcreated since version
####################################################################
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

    # Select the integrated spectrum for RSP fitting  
    m_flux = integrated['int_flux_cw'].copy()
    e_flux = integrated['int_eflux'].copy()

    # Prepare the integrated spectrum
    m_flux, e_flux, s_flux, m_flux_SN, m_flux_rss = \
      prepare_integrated_spectrum(m_flux, e_flux, smooth_box=51)

    # Initialize kinematics
    vel__yx = np.zeros(1)
    sigma__yx = 0.8
    m_flux_rss = m_flux[np.newaxis, :].copy()
    
    if ((args.flux_scale[0]==-1) and (args.flux_scale[1]==1)):
      args.flux_scale[0]=-0.1*np.abs(np.median(m_flux_rss))
      args.flux_scale[1]=3*np.abs(np.median(m_flux_rss))+10*np.std(m_flux_rss)

    #####################################################################
    # END: Generate integrated spectra
    #####################################################################

    #
    # Redefine the redshift
    #

    #
    # This is hard coded, what is not a good idea!
    #
    if (auto_redshift == True):
      #,w_min=6500,w_max=6800,\
      #                 w_min_ne=6350,w_max_ne=6500,\
      #                 w_ref=(6548.05,6562.85,6583.45,6678.15,6716.44,6730.82),do_plot=0,\

      #
      # We remove the continuum to the m_flux 
      #
      m_flux_bgs = m_flux.copy()
      # print(f'auto_z_bg: {args.auto_z_bg}')
      if (auto_z_bg == True):
        m_flux_bg = find_continuum(m_flux,niter=15,thresh=0.8,median_box_max=75,median_box_min=5)
        m_flux_bgs = m_flux-m_flux_bg
        print('# Back ground removed from auto-z spec')


      if (auto_z_sc == True):
        tab_el_sky=read_tab_EL(args.emission_lines_file_sky)
        specSky_e_mod = np.zeros(m_flux_bgs.shape)
        for w in tab_el_sky['wl']:
          delta_w = np.abs(wl__w-w)
          i_w = delta_w.argmin()
          sigma_inst=0.05+0.75*np.sqrt(((w-3000)/10000))          
          if (np.isfinite(m_flux_bgs[i_w]) == True):
            G = m_flux_bgs[i_w]*np.exp(-0.5*(delta_w/sigma_inst)**2)
            specSky_e_mod = specSky_e_mod+G
#        plt.plot(wl__w,m_flux)
#        plt.plot(wl__w,m_flux_bg)
#        if (args.plot==1):
#          plt.plot(wl__w,m_flux_bgs)
#        if           plt.plot(wl__w,m_flux_bg)    
#        m_flux_bgs = m_flux_bgs-specSky_e_mod


        m_flux_bgs_max = np.nanmax(m_flux_bgs[int(0.45*m_flux.shape[0]):int(0.55*m_flux.shape[0])])
        m_flux_bgs_median = np.nanmedian(m_flux_bgs[int(0.45*m_flux.shape[0]):int(0.55*m_flux.shape[0])])

        m_flux_bgs_std = np.nanstd(m_flux_bgs[int(0.45*m_flux.shape[0]):int(0.55*m_flux.shape[0])])
        m_flux_bgs_SN = m_flux_bgs_max / m_flux_bgs_std
        print(f'# m_flux_bgs: {m_flux_bgs_median} +- {m_flux_bgs_std}');
        print(f'# m_flux_bgs_max: {m_flux_bgs_max},  S/N: {m_flux_bgs_SN}');
        WA0 = 6500.0
        WA1 = 6650.0

        WAb0 = WA0-60.0
#        WAb1 = WA0-30.0
        mask_WA = (wl__w > WA0) & (wl__w < WA1)
#        mask_WAb = (wl__w > WAb0) & (wl__w < WAb1)
        m_flux_bgs_max_WA = np.nanmax(m_flux_bgs[mask_WA])
        m_flux_bgs_median_WA = np.nanmedian(m_flux_bgs[mask_WA])
        m_flux_bgs_std_WA = np.nanmedian(e_flux[mask_WA])
        m_flux_bgs_SN_WA = m_flux_bgs_max_WA / m_flux_bgs_std_WA
        print(f'# m_flux_bgs auto_z     window: {m_flux_bgs_median_WA} +- {m_flux_bgs_std_WA}');
        print(f'# m_flux_bgs_max auto_z window: {m_flux_bgs_max_WA},  S/N: {m_flux_bgs_SN_WA}');
        
#        if (args.plot==1):
#          plt.plot(wl__w,m_flux_bgs)    
#          plt.plot(wl__w,specSky_e_mod)
#          plt.plot(wl__w,m_flux,linewidth=2,color='black')    
#          plt.plot(wl__w,m_flux_bg,alpha=0.7,color='cyan')
#          plt.plot(wl__w,m_flux_bgs,alpha=0.7,color='red')

#          plt.show()
        #
        # We remove the residuals of the emission lines
        # from the integrated spectrum

        if (m_flux_SN>0):
          m_flux = m_flux-specSky_e_mod
        else:
          m_flux = m_flux-m_flux_bg


        
        print('# Night sky emission lines removed from auto-z spec')
      auto_z=find_redshift_spec(wl__w,m_flux_bgs,z_min=auto_z_min,z_max=auto_z_max,d_z=auto_z_del,\
                                w_min=6500,w_max=6650,w_ref=(6548.05,6562.85,6583.45),do_plot=args.do_plots)
      if (auto_z != auto_z_min):
        if (m_flux_bgs_SN_WA>10):
          args.redshift[0]=auto_z
          range_z = args.redshift[3]-args.redshift[2]
          args.redshift[2]=auto_z-range_z
          args.redshift[3]=auto_z+range_z
          args.w_range_nl[0]=args.w_range_nl[0]*(1+auto_z)
          args.w_range_nl[1]=args.w_range_nl[1]*(1+auto_z)     
          print(f'# Auto_z derivation ({auto_z_min},{auto_z_max},{auto_z_del}) :{auto_z}')
        else:
          print(f'# No auto_z derived, S/N is too low')
      else:
        if (m_flux_SN>20):
          args.redshift[0]=0.0
          args.redshift[2]=auto_z_min
          args.redshift[3]=auto_z_max
          args.w_range_nl[0]=args.w_range_nl[0]*(1+auto_z_min)
          args.w_range_nl[1]=args.w_range_nl[1]*(1+auto_z_max)
          print(f'# No auto_z peaks found, SN={m_flux_SN} use auto_z parameters')
        else:
          print(f'# No auto_z peaks found, SN={m_flux_SN} use configuration file parameters')



    
      
    #
    #
      
    ############################################################################
    # Run flux_elines on the mean spectrum
    #
#    fe_m_data, fe_m_hdr =flux_elines_RSS_EW(m_flux_rss, hdr_flux_org, 5, args.emission_lines_file, vel__yx,\
#                                              sigma__yx,eflux__wyx=m_e_flux_rss,\
#                                              flux_ssp__wyx=m_s_flux_rss,w_range=15)
#    colnames=[]
#    for i in range(fe_m_data.shape[0]):
#      colname=str(fe_m_hdr[f'NAME{i}'])+'_'+str(fe_m_hdr[f'WAVE{i}'])
#      colname=colname.replace(' ','_')
#      colnames.append(colname)
#    colnames=np.array(colnames)
#    tab_fe_m=Table(np.transpose(fe_m_data),names=colnames)  
#    print(tab_fe_m)
    
    #########################################################
    # We fit the mean spectrum with RSPs 
    #
    # OUTPUT NAMES ---------------------------------------------------------------------------------
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



    # remove previous outputs with the same label
    if args.clear_outputs:
        clean_preview_results_files(out_file_ps, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
        clean_preview_results_files(out_file_fe, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
    # ----------------------------------------------------------------------------------------------

    seed = print_time(print_seed=False, get_time_only=True)
    # initial time used as the seed of the random number generator.
    np.random.seed(seed)

    #
    # We start the fitting
    #
#    print(f'ignore_gas = {args.ignore_gas}')
 #   print(f'single_gas_fit = {args.single_gas_fit}')
    print(f'### START RSP fitting the integrated spectrum...')

       
    #
    # Reading the emission line file
    #
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





    #    print('EL file:',args.emission_lines_file_long)
    
    #print('tab_EL:',tab_el['id'])
    #quit()


    #
    # We fit everything in a single run
    #

    #
    # We create the SPS:
    #
    #print('Test loading the SPS')
    #print(f'{args.config_file}')
    #cf_master = ConfigAutoSSP(args.config_file, redshift_set=args.redshift, sigma_set=args.sigma, AV_set=args.AV) 
    #SPS_master = StellarSynthesis(config=cf_master,
    #                              wavelength=wl__w, flux=m_flux, eflux=e_flux,
    #                             mask_list=args.mask_file, elines_mask_file=args.emission_lines_file,
    #                              sigma_inst=args.sigma_inst, ssp_file=args.rsp_file,
    #                              ssp_nl_fit_file=args.rsp_nl_file, out_file=None,
    #                              w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0], nl_w_max=args.w_range_nl[1],                          
    #                              R_V=None, extlaw=None, spec_id=None, min=args.flux_scale[0], max=args.flux_scale[1],
    #                             guided_errors=None, ratio_master=None,
    #                              fit_gas=True, plot=None, verbose=False)
    #print('SPS_master loaded')
    
#
# (1) Lets fit just the velocity and the dispersion!
#
#    print('##############################');
#    print('# *** Start SPS_kin fitting...');
#    mask_w_nl = (wl__w>args.w_range_nl[0]*(1+args.redshift[0])) & (wl__w<args.w_range_nl[1]*(1+args.redshift[0]))
#    kin_pars, SPS_kin = auto_rsp_elines_rnd(
#      wl__w[mask_w_nl], m_flux[mask_w_nl], e_flux[mask_w_nl],\
#      ssp_file=args.rsp_nl_file, ssp_nl_fit_file=args.rsp_nl_file,
#      config_file=args.config_file,
#      w_min=args.w_range_nl[0], w_max=args.w_range_nl[1], nl_w_min=args.w_range_nl[0],
#      nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
#      min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
#      fit_gas=False, refine_gas=False, sigma_gas=args.sigma_gas,
#      input_redshift=args.redshift[0], delta_redshift=args.redshift[1], min_redshift=args.redshift[2], max_redshift=args.redshift[3],
#            input_sigma=args.sigma[0], delta_sigma=args.sigma[1], min_sigma=args.sigma[2], max_sigma=args.sigma[3],
#            input_AV=0, delta_AV=0, min_AV=args.AV[2], max_AV=args.AV[3],
#      sigma_inst=args.sigma_inst, spaxel_id=args.label, out_path=args.output_path, plot=args.plot) #, SPS_master=SPS_master)
#    print(f'SPS_kin: {SPS_kin.best_redshift} , {SPS_kin.best_sigma}')
#    print('##############################')


#
# (2) Now we fit the rest
#
#    print('# **** Start SPS_full fitting...');
#    _, SPS = auto_rsp_elines_rnd(
#      wl__w, m_flux, e_flux, ssp_file=args.rsp_file, ssp_nl_fit_file=args.rsp_nl_file,
#      config_file=args.config_file,
#      w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0],
#      nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
#      min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
#      fit_gas=not args.ignore_gas, refine_gas=not args.single_gas_fit, sigma_gas=args.sigma_gas,
#      input_redshift=SPS_kin.best_redshift, delta_redshift=0.0, min_redshift=args.redshift[2], max_redshift=args.redshift[3],
#            input_sigma=SPS_kin.best_sigma, delta_sigma=0.0, min_sigma=args.sigma[2], max_sigma=args.sigma[3],
#            input_AV=args.AV[0], delta_AV=args.AV[1], min_AV=args.AV[2], max_AV=args.AV[3],
#      sigma_inst=args.sigma_inst, spaxel_id=args.label, out_path=args.output_path, plot=args.plot, SPS_master=SPS_kin)

    _, SPS = auto_rsp_elines_rnd(
      wl__w, m_flux, e_flux, ssp_file=args.rsp_file, ssp_nl_fit_file=args.rsp_nl_file,
      config_file=args.config_file,
      w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0],
      nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
      min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
      fit_gas=not args.ignore_gas, refine_gas=not args.single_gas_fit,sigma_gas=args.sigma_gas,
      input_redshift=args.redshift[0], delta_redshift=args.redshift[1], min_redshift=args.redshift[2], max_redshift=args.redshift[3],
            input_sigma=args.sigma[0], delta_sigma=args.sigma[1], min_sigma=args.sigma[2], max_sigma=args.sigma[3],
            input_AV=args.AV[0], delta_AV=args.AV[1], min_AV=args.AV[2], max_AV=args.AV[3],
      sigma_inst=args.sigma_inst, spaxel_id=args.label, out_path=args.output_path, plot=args.plot,SN_CUT=SN_CUT_INT
    )

    #print(f'SPS nl ssp models: {SPS.ssp_nl_fit.flux_models.shape}')
    #print(f'SPS ssp models: {SPS.ssp.flux_models.shape}')

    y_ratio = None #SPS.ratio_master
    print(f'#### END RSP fitting the integrated spectrum...')
    #quit()
    
    # WRITE OUTPUTS --------------------------------------------------------------------------------
    SPS.output_gas_emission(filename=out_file_elines)
    if args.single_rsp:
      SPS.output_single_ssp(filename=out_file_single)
    else:
      SPS.output_fits(filename=out_file_fit)
      SPS.output_coeffs_MC(filename=out_file_coeffs)
      try:
        SPS.output(filename=out_file_ps)
      except:
        SPS.mass_to_light = np.nan
        SPS.teff_min = np.nan
        SPS.logg_min = np.nan
        SPS.meta_min = np.nan
        SPS.alph_min = np.nan
        SPS.AV_min = np.nan
        SPS.mass_to_light = np.nan
        SPS.teff_min_mass = np.nan
        SPS.logg_min_mass = np.nan
        SPS.meta_min_mass = np.nan
        SPS.alph_min_mass = np.nan
        SPS.AV_min_mass = np.nan
        SPS.e_teff_min = np.nan
        SPS.e_logg_min = np.nan
        SPS.e_meta_min = np.nan
        SPS.e_alph_min = np.nan
        SPS.e_AV_min = np.nan
        SPS.e_teff_min_mass = np.nan
        SPS.e_logg_min_mass = np.nan
        SPS.e_meta_min_mass = np.nan
        SPS.e_alph_min_mass = np.nan
        SPS.e_AV_min_mass = np.nan
        SPS.output(filename=out_file_ps)#, write_header=i==0, block_plot=False)
      

    tab_m_coeffs=read_coeffs_RSP(coeffs_file=out_file_coeffs)
    tab_m_elines=read_elines_RSP(elines_file=out_file_elines)
    tab_m_rsp=read_rsp(file_ssp=out_file_ps)

    ####################################################################
    # Preliminar version of the output plot
    if (args.do_plots==1):
        y_off = -1.09
        y_off2 = -2.18
        plot_spec_art(dir='',file=out_file_fit,\
                      file_ssp = out_file_ps,\
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

      
        plot_spec(dir='',file=out_file_fit,\
            file_ssp = out_file_ps,\
                  name=args.label,text=args.label,output=f'{args.output_path}/output_m_simple.{args.label}.{out_plot_format}',\
                  insets=((0.25, 0.5, 0.22, 0.47,4840,5020,-0.5,16,''),\
                      (0.01, 0.5, 0.22, 0.47,3851,3999,-0.5,3,''),\
                      (0.52, 0.5, 0.22, 0.47,6303,6322,0.5,2,'[SIII]6312'),\
                      (0.76, 0.5, 0.22, 0.47,9055,9083,-0.5,10,'[SIII]9069')),\
              y_min=-3,y_max=66,y0_d=0.3,y1_d=2.9,\
              x_min=3600,x_max=9600,plot_el=True, tab_el=tab_el)
        plot_spec(dir='',file=out_file_fit,\
            file_ssp = out_file_ps,\
            x_min=6500,x_max=6600,y_min=-0.2,y_max=15.5,\
            name=args.label,text=args.label,output=f'{args.output_path}/output_m_6500.{args.label}.{out_plot_format}')
        plot_spec(dir='',file=out_file_fit,\
            file_ssp = out_file_ps,\
            x_min=6700,x_max=6750,y_min=-0.2,y_max=15.5,\
            name=args.label,text=args.label,output=f'{args.output_path}/output_m_6700.{args.label}.{out_plot_format}')
        plot_spec(dir='',file=out_file_fit,\
            file_ssp = out_file_ps,\
            x_min=4800,x_max=5030,y_min=-0.2,y_max=15.5,\
            name=args.label,text=args.label,output=f'{args.output_path}/output_m_5000.{args.label}.{out_plot_format}')

    ############################################################################
    # Run flux_elines on the mean spectrum once subtracted the RSP model
    #
    out_model = np.array(SPS.output_spectra_list)
    #
    # We re-arange the results
    #
    gas = out_model[0,:] - out_model[1,:]
    smooth = median_filter(gas, size=smooth_size, mode='reflect')
    l_smooth = fit_legendre_polynomial(wl__w, smooth,n_leg)
    out_model[2,:] = out_model[2,:]+l_smooth
    out_model[3,:] = out_model[0,:]-(out_model[1,:]+l_smooth)
    out_model[4,:] = out_model[0,:]-out_model[2,:]
    
    m_flux_rss = np.zeros((1,m_flux.shape[0]))
    m_flux_rss[0,:]=out_model[0,:]-out_model[1,:]+l_smooth
    m_e_flux_rss = np.zeros((1,m_flux.shape[0]))
    m_e_flux_rss[0,:]=e_flux
    m_s_flux_rss = np.zeros((1,s_flux.shape[0]))
    m_s_flux_rss[0,:]=out_model[1,:]+l_smooth




    ############################################################################
    # Run flux_elines on the mean spectrum
    #
    #vel__yx=np.zeros(1)+SPS.best_redshift*__c__
    vel__yx=np.zeros(1)+args.redshift[0]*__c__
    sigma__yx=0.8
    sigma_we = 1.0
    print(f'# Guess Kin param. for NP elines analysis: {vel__yx} {sigma__yx}')
    
    fe_m_data, fe_m_hdr =flux_elines_RSS_EW_cl(m_flux_rss, hdr_flux_org, 5, tab_el, vel__yx,\
                                              sigma__yx,eflux__wyx=m_e_flux_rss,\
                                              flux_ssp__wyx=m_s_flux_rss,w_range=w_range_FE,\
                                            plot=args.do_plots, sigma_we=sigma_we)
#m_flux_rss[0,:]=out_model[0,:]-out_model[1,:]
    

    
    colnames=[]
    for i in range(fe_m_data.shape[0]):
      colname=str(fe_m_hdr[f'NAME{i}'])+'_'+str(fe_m_hdr[f'WAVE{i}'])
      colname=colname.replace(' ','_')
      colnames.append(colname)
    colnames=np.array(colnames)
    tab_m_fe=Table(np.transpose(fe_m_data),names=colnames)  
    tab_m_fe.write(out_file_fe, overwrite=True, delimiter=',')  

    m_tab_PT = Table_mean_rows(tab_PT)
    m_tab_PT['fiberid']=[args.label] 
    m_tab_PT.write(out_file_PT, overwrite=True, delimiter=',')
    
    if (only_integrated == True):
      print("# Only mean spectrum analyzed: END ALL")
      quit()
    

    print("##############################################")
    print("# End fitting the integrated spectrum ########")
    print("##############################################")
    print("\n")
    print("##############################################")
    print("# Start fitting full RSS spectra  with RSPs ##")
    print("##############################################")

    ############################################################################
    # FIT all the RSS
    ############################################################################
    out_file_fe = os.path.join(args.output_path, f"{args.label}.fe.txt")
    out_file_elines = os.path.join(args.output_path, f"{args.label}.elines.txt")
    out_file_kel = os.path.join(args.output_path, f"{args.label}.kel.txt")
    out_file_single = os.path.join(args.output_path, f"{args.label}.single.txt")
    out_file_coeffs = os.path.join(args.output_path, f"{args.label}.coeffs.txt")
    out_file_fit = os.path.join(args.output_path, f"{args.label}.output.fits.gz")
    out_file_ps = os.path.join(args.output_path, f"{args.label}.rsp.txt")
    out_file_dap = os.path.join(args.output_path, f"{args.label}.dap.fits.gz")

    # remove previous outputs with the same label
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
    y_ratio = None
    ns = rss_flux.shape[0]
    for i, (f__w, ef__w) in enumerate(zip(rss_flux, rss_eflux)):
        print(f"\n# ID {i}/{ns - 1} ===============================================\n")
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
            SPS_master=SPS,SN_CUT=SN_CUT
        )
#        y_ratio = SPS.ratio_master
        SPS.output_gas_emission(filename=out_file_elines, spec_id=i)
        SPS.output_coeffs_MC(filename=out_file_coeffs, write_header=i==0)
#        print(f'Teff test = {SPS.teff_min}')
        try:
            SPS.output(filename=out_file_ps, write_header=i==0, block_plot=False)
        except:
            SPS.mass_to_light = np.nan
            SPS.teff_min = np.nan
            SPS.logg_min = np.nan
            SPS.meta_min = np.nan
            SPS.alph_min = np.nan
            SPS.AV_min = np.nan
            SPS.mass_to_light = np.nan
            SPS.teff_min_mass = np.nan
            SPS.logg_min_mass = np.nan
            SPS.meta_min_mass = np.nan
            SPS.alph_min_mass = np.nan
            SPS.AV_min_mass = np.nan
            SPS.e_teff_min = np.nan
            SPS.e_logg_min = np.nan
            SPS.e_meta_min = np.nan
            SPS.e_alph_min = np.nan
            SPS.e_AV_min = np.nan
            SPS.e_teff_min_mass = np.nan
            SPS.e_logg_min_mass = np.nan
            SPS.e_meta_min_mass = np.nan
            SPS.e_alph_min_mass = np.nan
            SPS.e_AV_min_mass = np.nan
            SPS.output(filename=out_file_ps, write_header=i==0, block_plot=False)
        model_spectra.append(SPS.output_spectra_list) 



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


    tab_rsp=read_rsp(file_ssp=out_file_ps)
    tab_coeffs=read_coeffs_RSP(coeffs_file=out_file_coeffs)
    tab_elines=read_elines_RSP(elines_file=out_file_elines)

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
                                                sigma_we = sigma_we, verbose=False)

    
    colnames=[]
    colnames_B=[]
    colnames_R=[]
    colnames_I=[]
    wr_fe=np.array((3500,5755,7545,1000))
    for i in range(fe_data.shape[0]):
      colname=str(fe_hdr[f'NAME{i}'])+'_'+str(fe_hdr[f'WAVE{i}'])
      colname=colname.replace(' ','_')
      colnames.append(colname)
      wave_now=float(fe_hdr[f'WAVE{i}'])
      if (wave_now<5755):
        colnames_B.append(colname)
      if ((wave_now>=5755) & (wave_now<7545)):
        colnames_R.append(colname)
      if (wave_now>=7545):
        colnames_I.append(colname)  
    colnames=np.array(colnames)
    #colnames_B=np.array(colnames_B)
    #colnames_R=np.array(colnames_R)
    #colnames_I=np.array(colnames_I)
    tab_fe=Table(np.transpose(fe_data),names=colnames)
    tab_fe_B=tab_fe[colnames_B]
    tab_fe_R=tab_fe[colnames_R]
    tab_fe_I=tab_fe[colnames_I]
    
    print("##################################################")
    print("# END Flux_elines analysis on full RSS spectra ###")
    print("##################################################")
 

    #
    # 27.11.2024
    #
    print("#####################################################")
    print("# START: KIN_ELINES analysis on full RSS spectra ###")
    print("#####################################################")
    # out_file_kel = os.path.join(args.output_path, f"{args.label}.kel.txt")
    #    fe_data, fe_hdr =flux_elines_RSS_EW_cl(model_spectra[0,:,:]-model_spectra[1,:,:], hdr_flux, 5,\
    #    tab_el, vel__yx,\
    #                                          sigma__yx,eflux__wyx=rss_eflux,\
    #                                          flux_ssp__wyx=model_spectra[1,:,:],w_range=15,verbose=False)
    #     vel_mean=np.nanmean(tab_PE_ord[f'vel_{w_Ha}'])
    # disp_mean=np.nanmean(tab_PE_ord[f'disp_{w_Ha}'])
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


    
#    plot=1
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
      kin_rss_elines_main(wl__w[sel_wl_range], \
                          sec_flux,\
                          elcf, out_file=out_file_kel, \
                          rss_eflux=sec_eflux, run_mode='BOTH', guided=False, memo=False, \
                          vel_map=vel_map, vel_mask_map=None, vel_fixed=kin_fixed,\
                          sigma_map=sigma_map, sigma_fixed=kin_fixed, \
                          n_MC=N_MC, n_loops=n_loops, scale_ini=0.15, \
                          fine_search=None, redefine_max=False, max_factor=2, \
                          redefine_min=False, min_factor=0.012, plot=args.plot)



    tab_kel=read_elines_RSP(elines_file=out_file_kel)
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
                        tab_fe_I.copy(),tab_kel.copy(), kel_ext = 1, verbose=False)
    

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
      tab_DAP=read_DAP_file(out_file_dap,verbose=True)
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


    
