#!/usr/bin/env python3

import sys, os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1



import time
from astropy.io.fits.column import _parse_tdim
import numpy as np
import argparse
from copy import deepcopy as copy
from pprint import pprint

# pyFIT3D dependencies
from lvmdap.pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra

# 18.11.2023
# So far we were ysing the auto_ssp_tools from lvmdap.pyFIT3D
# We will attempt to modify them
from lvmdap.pyFIT3D.common.auto_ssp_tools import auto_ssp_elines_single_main
from lvmdap.pyFIT3D.common.auto_ssp_tools import load_rss, dump_rss_output
from lvmdap.pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra

from lvmdap.pyFIT3D.common.gas_tools import detect_create_ConfigEmissionModel
from lvmdap.pyFIT3D.common.io import create_ConfigAutoSSP_from_lists
from lvmdap.pyFIT3D.common.io import create_emission_lines_file_from_list
from lvmdap.pyFIT3D.common.io import create_emission_lines_mask_file_from_list
#from lvmdap.pyFIT3D.common.tools import read_coeffs_CS

from lvmdap.modelling.synthesis import StellarSynthesis
from lvmdap.dap_tools import load_LVM_rss, read_PT, rsp_print_header, plot_spec, read_rsp
from lvmdap.dap_tools import plot_spectra, read_coeffs_RSP, read_elines_RSP, read_tab_EL
from lvmdap.flux_elines_tools import flux_elines_RSS_EW

from scipy.ndimage import gaussian_filter1d,median_filter

from astropy.table import Table
from astropy.table import vstack as vstack_table
from astropy.io import fits, ascii

import yaml
import re
from collections import Counter

CWD = os.path.abspath(".")
EXT_CHOICES = ["CCM", "CAL"]
EXT_CURVE = EXT_CHOICES[0]
EXT_RV = 3.1
N_MC = 20

def _no_traceback(type, value, traceback):
  print(value)


#######################################################
# RSP version of the auto_ssp_elines_rnd from lvmdap.pyFIT3D
#######################################################
def auto_rsp_elines_rnd(
    wl__w, f__w, ef__w, ssp_file, spaxel_id, config_file=None, plot=None,
    ssp_nl_fit_file=None, sigma_inst=None, mask_list=None,
    min=None, max=None, w_min=None, w_max=None,
    nl_w_min=None, nl_w_max=None, elines_mask_file=None, fit_gas=True, refine_gas=True,
    input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None, sigma_gas=None,
    input_AV=None, delta_AV=None, min_AV=None, max_AV=None, ratio=True, y_ratio=None,
    fit_sigma_rnd=True, out_path=None):

  #
  # If there is no RSP for the Non-Linear (nl) fitting, they it is 
  # used the one for the Linear Fitting (that it is slower)
  #
    ssp_nl_fit_file = ssp_file if ssp_nl_fit_file is None else ssp_nl_fit_file
    if delta_redshift == 0:
      cc_redshift_boundaries = None
    else:
      cc_redshift_boundaries = [min_redshift, max_redshift]

  #
  # If the emission lines are fitted, but there is no config file, then
  # the program creates a set of configuraton files for the detected emission lines
  # NOTE: I think this is overdoing having the flux_elines script
  # But needs to be explored!
  #
    if fit_gas and config_file is None:
        print("##############################");
        print("# START: Autodectecting emission lines...");
        if sigma_gas is None: sigma_gas = 3.0
        if out_path is None: out_path = "."
        wl_mask = (w_min<=wl__w)&(wl__w<=w_max)
        config_filenames, wl_chunks, _, wave_peaks_tot_rf = detect_create_ConfigEmissionModel(
            wl__w[wl_mask], f__w[wl_mask],
            redshift=input_redshift,
            sigma_guess=sigma_gas,
            chunks=4,
            polynomial_order=1,
            polynomial_coeff_guess=[0.000, 0.001],
            polynomial_coeff_boundaries=[[-1e13, 1e13], [-1e13, 1e13]],
            flux_boundaries_fact=[0.001, 1000],
            sigma_boundaries_fact=[0.1, 1.5],
            v0_boundaries_add=[-1000, 1000],
            peak_find_nsearch=1,
            peak_find_threshold=0.2,
            peak_find_dmin=1,
            crossmatch_list_filename=elines_mask_file,
            crossmatch_absdmax_AA=5,
            crossmatch_redshift_search_boundaries=cc_redshift_boundaries,
            sort_by_flux=True,
            output_path=out_path,
            label=spaxel_id,
            verbose=0,
            plot=0,
        )

        create_emission_lines_mask_file_from_list(wave_peaks_tot_rf, eline_half_range=3*sigma_gas, output_path=out_path, label=spaxel_id)
        create_emission_lines_file_from_list(wave_peaks_tot_rf, output_path=out_path, label=spaxel_id)
        create_ConfigAutoSSP_from_lists(wl_chunks, config_filenames, output_path=out_path, label=spaxel_id)

        config_file = os.path.join(out_path, f"{spaxel_id}.autodetect.auto_ssp_several.config")
        if not refine_gas: elines_mask_file = os.path.join(out_path, f"{spaxel_id}.autodetect.emission_lines.txt")
        print("# END: Autodectecting emission lines...");     
    else:
      print("# Using predefined configuration file for the emission lines");
    #
    # The spectrum is fitted for the 1st time in here
    #
    print("##############################");
    print(f"# START: fitting the continuum+emission lines, fit_gas:{fit_gas} ...");
    cf, SPS = auto_ssp_elines_single_main(
        wl__w, f__w, ef__w, ssp_file,
        config_file=config_file,
        ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file="NOT_USED",
        mask_list=mask_list, elines_mask_file=elines_mask_file, fit_gas=fit_gas,
        min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
        input_redshift=input_redshift, delta_redshift=delta_redshift,
        min_redshift=min_redshift, max_redshift=max_redshift,
        input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
        input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
        plot=plot, single_ssp=False, ratio=ratio, y_ratio=y_ratio, fit_sigma_rnd=fit_sigma_rnd,
        sps_class=StellarSynthesis
    )
    print(f"# END: fitting the continuum+emission lines, fit_gas:{fit_gas} ...");
    print("##############################");
    #
    # There is refinement in the fitting
    #
    print(f"# refine_gas: {refine_gas}");
    if refine_gas:
        print(f"# START: refining gas fitting, refine_gas:{refine_gas} ...");
        if sigma_gas is None: sigma_gas = 3.0
        if out_path is None: out_path = "."
        wl_mask = (w_min<=wl__w)&(wl__w<=w_max)
        gas_wl, gas_fl = SPS.spectra["orig_wave"][wl_mask], (SPS.output_spectra_list[0] - SPS.output_spectra_list[1])[wl_mask]
        config_filenames, wl_chunks, _, wave_peaks_tot_rf = detect_create_ConfigEmissionModel(
            gas_wl, gas_fl,
            redshift=input_redshift,
            sigma_guess=sigma_gas,
            chunks=4,
            polynomial_order=1,
            polynomial_coeff_guess=[0.000, 0.001],
            polynomial_coeff_boundaries=[[-1e13, 1e13], [-1e13, 1e13]],
            flux_boundaries_fact=[0.001, 1000],
            sigma_boundaries_fact=[0.1, 1.5],
            v0_boundaries_add=[-1000, 1000],
            peak_find_nsearch=1,
            peak_find_threshold=0.2,
            peak_find_dmin=1,
            crossmatch_list_filename=elines_mask_file,
            crossmatch_absdmax_AA=5,
            crossmatch_redshift_search_boundaries=cc_redshift_boundaries,
            sort_by_flux=True,
            output_path=out_path,
            label=spaxel_id,
            verbose=0,
            plot=0,
        )

        create_emission_lines_mask_file_from_list(wave_peaks_tot_rf, eline_half_range=3*sigma_gas, output_path=out_path, label=spaxel_id)
        create_emission_lines_file_from_list(wave_peaks_tot_rf, output_path=out_path, label=spaxel_id)
        create_ConfigAutoSSP_from_lists(wl_chunks, config_filenames, output_path=out_path, label=spaxel_id)

        config_file = os.path.join(out_path, f"{spaxel_id}.autodetect.auto_ssp_several.config")
        elines_mask_file = os.path.join(out_path, f"{spaxel_id}.autodetect.emission_lines.txt")

        cf, SPS = auto_ssp_elines_single_main(
            wl__w, f__w, ef__w, ssp_file,
            config_file=config_file,
            ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file="NOT_USED",
            mask_list=mask_list, elines_mask_file=elines_mask_file, fit_gas=fit_gas,
            min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
            input_redshift=input_redshift, delta_redshift=delta_redshift,
            min_redshift=min_redshift, max_redshift=max_redshift,
            input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
            input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
            plot=plot, single_ssp=False, ratio=ratio, y_ratio=y_ratio, fit_sigma_rnd=fit_sigma_rnd,
            sps_class=StellarSynthesis
        )
        print(f"# END: refining gas fitting, refine_gas:{refine_gas} ...");
        print("########################################");
    print("# END RSP fitting...");
    print("########################################");
    return cf, SPS

####################################################
# MAIN script. Uses the entries from commands lines
####################################################
def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run the spectral fitting procedure for the LVM"
    )
    parser.add_argument(
        "spec_file", metavar="spectrum-file",
        help="input spectrum to fit"
    )
    parser.add_argument(
        "rsp_file", metavar="rsp-file",
        help="the resolved stellar population basis"
    )
    parser.add_argument(
        "sigma_inst", metavar="sigma-inst", type=np.float,
        help="the standard deviation in wavelength of the Gaussian kernel to downgrade the resolution of the models to match the observed spectrum. This is: sigma_inst^2 = sigma_obs^2 - sigma_mod^2"
    )
    parser.add_argument(
        "label",
        help="string to label the current run"
    )
    parser.add_argument(
        "--input-fmt",
        help="the format of the input file. It can be either 'single' or 'rss'. Defaults to 'single'",
        default="single"
    )
    parser.add_argument(
	"--error-file",
        help="the error file"
    )
    parser.add_argument(
        "--config-file",
        help="the configuration file used to set the parameters for the emission line fitting"
    )
    parser.add_argument(
        "--emission-lines-file",
        help="file containing emission lines list"
    )
    parser.add_argument(
        "--mask-file",
        help="the file listing the wavelength ranges to exclude during the fitting"
    )
    parser.add_argument(
        "--sigma-gas", type=np.float,
        help="the guess velocity dispersion of the gas"
    )
    parser.add_argument(
        "--single-gas-fit",
        help="whether to run a single fit of the gas or refine fitting. Defaults to False",
        action="store_true"
    )
    parser.add_argument(
        "--ignore-gas",
        help="whether to ignore gas during the fitting or not. Defaults to False",
        action="store_true"
    )
    parser.add_argument(
        "--rsp-nl-file",
        help="the resolved stellar population *reduced* basis, for non-linear fitting"
    )
    parser.add_argument(
        "--plot", type=np.int,
        help="whether to plot (1) or not (0, default) the fitting procedure. If 2, a plot of the result is store in a file without display on screen",
        default=0
    )
    parser.add_argument(
        "--flux-scale", metavar=("min","max"), type=np.float, nargs=2,
        help="scale of the flux in the input spectrum",
        default=[-1,1]
    )
    parser.add_argument(
        "--w-range", metavar=("wmin","wmax"), type=np.float, nargs=2,
        help="the wavelength range for the fitting procedure",
        default=[-np.inf, np.inf]
    )
    parser.add_argument(
        "--w-range-nl", metavar=("wmin2","wmax2"), type=np.float, nargs=2,
        help="the wavelength range for the *non-linear* fitting procedure"
    )

    parser.add_argument(
        "--redshift", metavar=("input_redshift","delta_redshift","min_redshift","max_redshift"), type=np.float, nargs=4,
        help="the guess, step, minimum and maximum value for the redshift during the fitting",
        default=(0.00, 0.01, 0.00, 0.30)
    )
    parser.add_argument(
        "--sigma", metavar=("input_sigma","delta_sigma","min_sigma","max_sigma"), type=np.float, nargs=4,
        help="same as the redshift, but for the line-of-sight velocity dispersion",
        default=(0, 10, 0, 450)
    )
    parser.add_argument(
        "--AV", metavar=("input_AV","delta_AV","min_AV","max_AV"), type=np.float, nargs=4,
        help="same as the redshift, but for the dust extinction in the V-band",
        default=(0.0, 0.1, 0.0, 3.0)
    )
    parser.add_argument(
        "--ext-curve",
        help=f"the extinction model to choose for the dust effects modelling. Choices are: {EXT_CHOICES}",
        choices=EXT_CHOICES, default=EXT_CURVE
    )
    parser.add_argument(
        "--RV", type=np.float,
        help=f"total to selective extinction defined as: A_V / E(B-V). Default to {EXT_RV}",
        default=EXT_RV
    )
    parser.add_argument(
        "--single-rsp",
        help="whether to fit a single stellar template to the target spectrum or not. Default to False",
        action="store_true"
    )
    parser.add_argument(
        "--n-mc", type=np.int,
        help="number of MC realisations for the spectral fitting",
        default=N_MC
    )
    parser.add_argument(
        "-o", "--output-path", metavar="path",
        help=f"path to the outputs. Defaults to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "-c", "--clear-outputs",
        help="whether to remove or not a previous run with the same label (if present). Defaults to false",
        action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="if given, shows information about the progress of the script. Defaults to false.",
        action="store_true"
    )
    parser.add_argument(
        "-d", "--debug",
        help="debugging mode. Defaults to false.",
        action="store_true"
    )

#    print(cmd_args)

    args = parser.parse_args(cmd_args)
    if not args.debug:
        sys.excepthook = _no_traceback
    else:
        pprint("COMMAND LINE ARGUMENTS")
        pprint(f"{args}\n")
    if args.rsp_nl_file is None:
        args.rsp_nl_file = args.rsp_file
    if args.w_range_nl is None:
        args.w_range_nl = copy(args.w_range)

    # OUTPUT NAMES ---------------------------------------------------------------------------------
    out_file_elines = os.path.join(args.output_path, f"elines_{args.label}")
    out_file_single = os.path.join(args.output_path, f"single_{args.label}")
    out_file_coeffs = os.path.join(args.output_path, f"coeffs_{args.label}")
    out_file_fit = os.path.join(args.output_path, f"output.{args.label}.fits.gz")
    out_file_ps = os.path.join(args.output_path, args.label)
    # remove previous outputs with the same label
    if args.clear_outputs:
        clean_preview_results_files(out_file_ps, out_file_elines, out_file_single, out_file_coeffs, out_file_fit)
    # ----------------------------------------------------------------------------------------------

    seed = print_time(print_seed=False, get_time_only=True)
    # initial time used as the seed of the random number generator.
    np.random.seed(seed)

    # FITTING --------------------------------------------------------------------------------------
    if args.input_fmt == "single":
        wl__w, f__w, ef__w = read_spectra(args.spec_file, f_error=lambda x: 0.1*np.sqrt(np.abs(x)))

        _, SPS = auto_rsp_elines_rnd(
            wl__w, f__w, ef__w, ssp_file=args.rsp_file, ssp_nl_fit_file=args.rsp_nl_file,
            config_file=args.config_file,
            w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0],
            nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
            min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
            fit_gas=not args.ignore_gas, refine_gas=not args.single_gas_fit, sigma_gas=args.sigma_gas,
            input_redshift=args.redshift[0], delta_redshift=args.redshift[1], min_redshift=args.redshift[2], max_redshift=args.redshift[3],
            input_sigma=args.sigma[0], delta_sigma=args.sigma[1], min_sigma=args.sigma[2], max_sigma=args.sigma[3],
            input_AV=args.AV[0], delta_AV=args.AV[1], min_AV=args.AV[2], max_AV=args.AV[3],
            sigma_inst=args.sigma_inst, spaxel_id=args.label, out_path=args.output_path, plot=args.plot
        )
        # WRITE OUTPUTS --------------------------------------------------------------------------------
        SPS.output_gas_emission(filename=out_file_elines)
        if args.single_rsp:
            SPS.output_single_ssp(filename=out_file_single)
        else:
            SPS.output_fits(filename=out_file_fit)
            SPS.output_coeffs_MC(filename=out_file_coeffs)
            SPS.output(filename=out_file_ps)
    elif args.input_fmt == "rss":
        wl__w, rss_flux, rss_eflux = load_rss(spec_file=args.spec_file, error_file=args.error_file)

        is_guided_sigma = False
        guided_nl = False
        guided_errors = None
        # if input_guided is not None:
        #     guided_nl = True
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
                sigma_inst=args.sigma_inst, spaxel_id=f"{args.label}_{i}", out_path=args.output_path, plot=args.plot
            )
            y_ratio = SPS.ratio_master
            SPS.output_gas_emission(filename=out_file_elines, spec_id=i)
            SPS.output_coeffs_MC(filename=out_file_coeffs, write_header=i==0)
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
        dump_rss_output(out_file_fit=out_file_fit, wavelength=wl__w, model_spectra=model_spectra)
    else:
        raise(NotImplementedError(f"--input-fmt='{args.input_fmt}'"))





####################################################
# MAIN script. Uses the entries from commands lines
####################################################
def _dap_yaml(cmd_args=sys.argv[1:]):
    PLATESCALE = 112.36748321030637

#    print(f'n_MC = {__n_Monte_Carlo__}')
#    quit()

    parser = argparse.ArgumentParser(
        description="lvm-dap-yaml LVM_FILE OUTPUT_LABEL CONFIG.YAML"
    )
    parser.add_argument(
        "lvm_file", metavar="lvm_file",
        help="input LVM spectrum to fit"
    )

    parser.add_argument(
        "label",
        help="string to label the current run"
    )

    parser.add_argument(
        "config_yaml",
        help="config_yaml with the fitting parameters"
    )

    parser.add_argument(
        "-d", "--debug",
        help="debugging mode. Defaults to false.",
        action="store_true"
    )


    
    args = parser.parse_args(cmd_args)
    print(cmd_args)
    print(args)
    if not args.debug:
        sys.excepthook = _no_traceback
    else:
        pprint("COMMAND LINE ARGUMENTS")
        pprint(f"{args}\n")


    #
    # Read the YAML file
    #
    print(args.config_yaml)
    with open(args.config_yaml, 'r') as yaml_file:
      dap_config_args = yaml.safe_load(yaml_file)
    #
    # We add the full list of arguments
    #
    for k, v in dap_config_args.items():
      if(isinstance(v, str)):
        v=v.replace("..",dap_config_args['lvmdap_dir'])
      parser.add_argument(
        '--' + k, default=v
      )
    #
    # We transform it to a set of arguments
    #

    parser.add_argument(
        "--flux-scale", metavar=("min","max"), type=np.float, nargs=2,
        help="scale of the flux in the input spectrum",
        default=[-1, +1]
    )

    parser.add_argument(
      "--plot", type=np.int,
      help="whether to plot (1) or not (0, default) the fitting procedure. If 2, a plot of the result is store in a file without display on screen",
      default=0
    )
    
    args = parser.parse_args(cmd_args)

    try:
        ny_range=args.ny_range
    except:
        ny_range=None

    try:
        out_plot_format=args.out_plot_format
    except:
        out_plot_format="pdf"        

    try:
        only_integrated=args.only_integrated
    except:
        only_integrated=False







#    if args.rsp_nl_file is None:
#        args.rsp_nl_file = args.rsp_file
#    if args.w_range_nl is None:
#        args.w_range_nl = copy(args.w_range)

    wl__w, rss_flux_org, rss_eflux_org, hdr_flux_org, hdr_0 = load_LVM_rss(args.lvm_file,ny_range=ny_range)
#   just a check
#    print(rss_flux.shape)

    # Include here the AG-cam meadiand file!
    #
    tab_PT_org = read_PT(args.lvm_file,'none',ny_range=ny_range)


    
#    print(args.rsp_file)


    if ((args.flux_scale[0]==-1) and (args.flux_scale[1]==1)):
      args.flux_scale[0]=args.flux_scale_org[0]
      args.flux_scale[1]=args.flux_scale_org[1]
      
    #
    # We mask the bad spectra
    #
    rss_flux = rss_flux_org[tab_PT_org['mask']]
    rss_eflux = rss_eflux_org[tab_PT_org['mask']]
    tab_PT = tab_PT_org[tab_PT_org['mask']]
    hdr_flux=copy(hdr_flux_org)
    hdr_flux['NAXIS2']=len(tab_PT)

    print(f'# Number of spectra to analyze : {len(tab_PT)}')
    #
    # First we create a mean spectrum
    #
    #m_flux = rss_flux.mean(axis=0)
    #e_flux = rss_eflux.mean(axis=0)/np.sqrt(rss_flux.shape[0])
    m_flux = np.median(rss_flux,axis=0)
    e_flux = rss_eflux.mean(axis=0)/np.sqrt(rss_flux.shape[0])
    s_flux = median_filter(m_flux,51)

    vel__yx=np.zeros(1)
    sigma__yx=1.5
    print(f'# Number of spectral pixels : {m_flux.shape[0]}')
    m_flux_rss = np.zeros((1,m_flux.shape[0]))
    m_flux_rss[0,:]=m_flux
#    m_e_flux_rss = np.zeros((1,m_flux.shape[0]))
#    m_e_flux_rss[0,:]=e_flux
#    m_s_flux_rss = np.zeros((1,s_flux.shape[0]))
#    m_s_flux_rss[0,:]=s_flux

    if ((args.flux_scale[0]==-1) and (args.flux_scale[1]==1)):
      args.flux_scale[0]=-0.1*np.abs(np.median(m_flux_rss))
      args.flux_scale[1]=3*np.abs(np.median(m_flux_rss))+10*np.std(m_flux_rss)


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
    out_file_fe = os.path.join(args.output_path, f"m_{args.label}.fe.txt")
    out_file_elines = os.path.join(args.output_path, f"m_{args.label}.elines.txt")
    out_file_single = os.path.join(args.output_path, f"m_{args.label}.single.txt")
    out_file_coeffs = os.path.join(args.output_path, f"m_{args.label}.coeffs.txt")
    out_file_fit = os.path.join(args.output_path, f"m_{args.label}.output.fits.gz")
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
#    print('EL file:',args.emission_lines_file_long)
#    print('tab_EL:',tab_el)
#    quit()
    
    _, SPS = auto_rsp_elines_rnd(
      wl__w, m_flux, e_flux, ssp_file=args.rsp_file, ssp_nl_fit_file=args.rsp_nl_file,
      config_file=args.config_file,
      w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0],
      nl_w_max=args.w_range_nl[1], mask_list=args.mask_file,
      min=args.flux_scale[0], max=args.flux_scale[1], elines_mask_file=args.emission_lines_file,
      fit_gas=not args.ignore_gas, refine_gas=not args.single_gas_fit, sigma_gas=args.sigma_gas,
      input_redshift=args.redshift[0], delta_redshift=args.redshift[1], min_redshift=args.redshift[2], max_redshift=args.redshift[3],
            input_sigma=args.sigma[0], delta_sigma=args.sigma[1], min_sigma=args.sigma[2], max_sigma=args.sigma[3],
            input_AV=args.AV[0], delta_AV=args.AV[1], min_AV=args.AV[2], max_AV=args.AV[3],
      sigma_inst=args.sigma_inst, spaxel_id=args.label, out_path=args.output_path, plot=args.plot
    )
    print(f'#### END RSP fitting the integrated spectrum...')
    
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
        plot_spec(dir='',file=out_file_fit,\
            file_ssp = out_file_ps,\
                  name=args.label,text=args.label,output=f'{args.output_path}/output_m.{args.label}.{out_plot_format}',\
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
    m_flux_rss = np.zeros((1,m_flux.shape[0]))
    m_flux_rss[0,:]=out_model[0,:]-out_model[1,:]
    m_e_flux_rss = np.zeros((1,m_flux.shape[0]))
    m_e_flux_rss[0,:]=e_flux
    m_s_flux_rss = np.zeros((1,s_flux.shape[0]))
    m_s_flux_rss[0,:]=out_model[1,:]




    ############################################################################
    # Run flux_elines on the mean spectrum
    #
    fe_m_data, fe_m_hdr =flux_elines_RSS_EW(m_flux_rss, hdr_flux_org, 5, args.emission_lines_file, vel__yx,\
                                              sigma__yx,eflux__wyx=m_e_flux_rss,\
                                              flux_ssp__wyx=m_s_flux_rss,w_range=15)
    colnames=[]
    for i in range(fe_m_data.shape[0]):
      colname=str(fe_m_hdr[f'NAME{i}'])+'_'+str(fe_m_hdr[f'WAVE{i}'])
      colname=colname.replace(' ','_')
      colnames.append(colname)
    colnames=np.array(colnames)
    tab_m_fe=Table(np.transpose(fe_m_data),names=colnames)  

    if (args.only_integrated == True):
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
            sigma_inst=args.sigma_inst, spaxel_id=f"{args.label}_{i}", out_path=args.output_path, plot=args.plot
        )
        y_ratio = SPS.ratio_master
        SPS.output_gas_emission(filename=out_file_elines, spec_id=i)
        SPS.output_coeffs_MC(filename=out_file_coeffs, write_header=i==0)
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
    dump_rss_output(out_file_fit=out_file_fit, wavelength=wl__w, model_spectra=model_spectra)

    tab_rsp=read_rsp(file_ssp=out_file_ps)
    tab_coeffs=read_coeffs_RSP(coeffs_file=out_file_coeffs)
    tab_elines=read_elines_RSP(elines_file=out_file_elines)
    #
    # Do a plot of the 1st and the last spectra
    #
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
    print(f'model_spectra_shape {model_spectra.shape}')
    vel__yx=np.zeros(hdr_flux['NAXIS2'])
    sigma__yx=1.5


    fe_data, fe_hdr =flux_elines_RSS_EW(model_spectra[0,:,:]-model_spectra[1,:,:], hdr_flux, 5, args.emission_lines_file_long, vel__yx,\
                                              sigma__yx,eflux__wyx=rss_eflux,\
                                              flux_ssp__wyx=model_spectra[1,:,:],w_range=15)
    colnames=[]
    for i in range(fe_data.shape[0]):
      colname=str(fe_hdr[f'NAME{i}'])+'_'+str(fe_hdr[f'WAVE{i}'])
      colname=colname.replace(' ','_')
      colnames.append(colname)
    colnames=np.array(colnames)
    tab_fe=Table(np.transpose(fe_data),names=colnames)  
    print("##################################################")
    print("# END Flux_elines analysis on full RSS spectra ###")
    print("##################################################")
 


    print("##################################################")
    print("# START: Storing the results in a single file  ###")
    print("##################################################")

    tab_rsp.add_column(tab_PT['id'].value,name='id',index=0)
    tab_fe.add_column(tab_PT['id'].value,name='id',index=0)

    id_elines=[]    
    for id_fib in tab_elines['id_fib']:
        id_elines.append(tab_PT['id'].value[id_fib])
    id_elines=np.array(id_elines)
    tab_elines.add_column(id_elines,name='id',index=0)
    
    id_coeffs=[]    
    for id_fib in tab_coeffs['id_fib']:
        id_coeffs.append(tab_PT['id'].value[id_fib])
    id_coeffs=np.array(id_coeffs)
    tab_coeffs.add_column(id_coeffs,name='id',index=0)
    

    hdu_hdr_0 = fits.PrimaryHDU(header=hdr_0) 
    hdu_PT = fits.BinTableHDU(tab_PT,name='PT')
    hdu_ELINES = fits.BinTableHDU(tab_elines,name='ELINES')
    hdu_FE = fits.BinTableHDU(tab_fe,name='FE')
    hdu_RSP = fits.BinTableHDU(tab_rsp,name='RSP')
    hdu_COEFFS = fits.BinTableHDU(tab_coeffs,name='COEFFS')

    hdu_dap =fits.HDUList([hdu_hdr_0,hdu_PT,hdu_RSP,hdu_COEFFS,hdu_ELINES,hdu_FE])
    hdu_dap.writeto(out_file_dap,overwrite=True)
    print(f'# dap_file: {out_file_dap} written')
    print("##################################################")
    print("# END: Storing the results in a single file    ###")
    print("##################################################")


    print("#******   ALL DONE ******#")
