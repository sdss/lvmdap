#!/usr/bin/env python3

import sys, os
import time
from astropy.io.fits.column import _parse_tdim
import numpy as np
import argparse
from copy import deepcopy as copy
from pprint import pprint

# pyFIT3D dependencies
from pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra
from pyFIT3D.common.auto_ssp_tools import auto_ssp_elines_single_main, dump_rss_output
from pyFIT3D.common.auto_ssp_tools import load_rss
from pyFIT3D.common.io import clean_preview_results_files, print_time, read_spectra

from pyFIT3D.common.gas_tools import detect_create_ConfigEmissionModel
from pyFIT3D.common.io import create_ConfigAutoSSP_from_lists
from pyFIT3D.common.io import create_emission_lines_file_from_list
from pyFIT3D.common.io import create_emission_lines_mask_file_from_list

from lvmdap.modelling.synthesis import StellarSynthesis


CWD = os.path.abspath(".")
EXT_CHOICES = ["CCM", "CAL"]
EXT_CURVE = EXT_CHOICES[0]
EXT_RV = 3.1
N_MC = 20

def _no_traceback(type, value, traceback):
  print(value)

def auto_rsp_elines_rnd(
    wl__w, f__w, ef__w, ssp_file, spaxel_id, config_file=None, plot=None,
    ssp_nl_fit_file=None, sigma_inst=None, mask_list=None,
    min=None, max=None, w_min=None, w_max=None,
    nl_w_min=None, nl_w_max=None, elines_mask_file=None, fit_gas=True, refine_gas=True,
    input_redshift=None, delta_redshift=None, min_redshift=None, max_redshift=None,
    input_sigma=None, delta_sigma=None, min_sigma=None, max_sigma=None, sigma_gas=None,
    input_AV=None, delta_AV=None, min_AV=None, max_AV=None, ratio=True, y_ratio=None,
    fit_sigma_rnd=True, out_path=None):

    ssp_nl_fit_file = ssp_file if ssp_nl_fit_file is None else ssp_nl_fit_file

    if fit_gas and config_file is None:
        if delta_redshift == 0:
            cc_redshift_boundaries = None
        else:
            cc_redshift_boundaries = [min_redshift, max_redshift]
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

    if refine_gas:
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

    return cf, SPS

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
        default=[-np.inf, +np.inf]
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
        # results = []
        # results_coeffs = []
        y_ratio = None
        ns = rss_flux.shape[0]
        # output_el_models = {}
        # if not guided_nl:
        #     _tmpcf = ConfigAutoSSP(config_file)
        #     for i_s in range(_tmpcf.n_systems):
        #         system = _tmpcf.systems[i_s]
        #         elcf = _tmpcf.systems_config[i_s]
        #         k = f'{system["start_w"]}_{system["end_w"]}'
        #         output_el_models[k] = create_emission_lines_parameters(elcf, ns)
        #     del _tmpcf
        for i, (f__w, ef__w) in enumerate(zip(rss_flux, rss_eflux)):
            print(f"\n# ID {i}/{ns - 1} ===============================================\n")
            # if guided_nl:
            #     input_redshift = input_guided[0][i]
            #     delta_redshift = 0
            #     input_sigma = input_guided[1][i]
            #     delta_sigma = 0
            #     input_AV = input_guided[2][i]
            #     delta_AV = 0
            #     # (e_AV, e_sigma, e_redshift)
            #     guided_errors = (input_guided_errors[0][i], input_guided_errors[1][i], input_guided_errors[2][i])
            #     print(f'-> Forcing non-linear fit parameters (input guided):')
            #     print(f'-> input_guided_redshift:{input_redshift} e:{guided_errors[0]}')
            #     print(f'-> input_guided_sigma:{input_sigma} e:{guided_errors[1]}')
            #     print(f'-> input_guided_AV:{input_AV} e:{guided_errors[2]}')
            # if i > 0 and (not guided_nl and is_guided_sigma):
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
            # cf, SPS = auto_ssp_elines_single_main(
            #     wavelength=wavelength, flux=f__w, eflux=ef__w, ssp_file=ssp_file, config_file=config_file,
            #     ssp_nl_fit_file=ssp_nl_fit_file, sigma_inst=sigma_inst, out_file=out_file,
            #     mask_list=mask_list, elines_mask_file=elines_mask_file,
            #     min=min, max=max, w_min=w_min, w_max=w_max, nl_w_min=nl_w_min, nl_w_max=nl_w_max,
            #     input_redshift=input_redshift, delta_redshift=delta_redshift,
            #     min_redshift=min_redshift, max_redshift=max_redshift,
            #     input_sigma=input_sigma, delta_sigma=delta_sigma, min_sigma=min_sigma, max_sigma=max_sigma,
            #     input_AV=input_AV, delta_AV=delta_AV, min_AV=min_AV, max_AV=max_AV,
            #     plot=plot, single_ssp=False,
            #     is_guided_sigma=is_guided_sigma,
            #     # guided_sigma=guided_sigma,
            #     spec_id=spec_id,
            #     guided_errors=guided_errors, ratio=ratio, y_ratio=y_ratio,
            #     fit_sigma_rnd=fit_sigma_rnd, sps_class=sps_class
            # )
            print('PASO POR AQUI 1')
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
            # if not guided_nl:
            #     SPS.output_gas_emission(filename=out_file_elines, spec_id=i)
            print('PASO POR AQUI 2')
            SPS.output_gas_emission(filename=out_file_elines, spec_id=i)
            print('PASO POR AQUI 3')
            SPS.output_coeffs_MC(filename=out_file_coeffs, write_header=i==0)
            print('PASO POR AQUI 4')
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
            # if not guided_nl:
            #     for system in SPS.config.systems:
            #         if system['EL'] is not None:
            #             k = f'{system["start_w"]}_{system["end_w"]}'
            #             append_emission_lines_parameters(system['EL'], output_el_models[k], i)
            print('PASO POR AQUI 5')
            model_spectra.append(SPS.output_spectra_list)
            # results.append(SPS.output_results)
            # results_coeffs.append(SPS.output_coeffs)

        # output model_spectra has dimensions (n_output_spectra_list, n_s, n_wavelength)
        model_spectra = np.array(model_spectra).transpose(1, 0, 2)
        # output results has dimensions (n_results, n_s)
        # results = np.array(results).T
        # output results_coeffs has dimensions (n_results_coeffs, n_models, n_s)
        # results_coeffs = np.array(results_coeffs).transpose(1, 2, 0)
        print('PASO POR AQUI 6')
        dump_rss_output(out_file_fit=out_file_fit, wavelength=wl__w, model_spectra=model_spectra)
        print('PASO POR AQUI 7')
    else:
        raise(NotImplementedError(f"--input-fmt='{args.input_fmt}'"))
