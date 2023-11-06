import sys
import os
import numpy as np

import argparse
from pprint import pprint

from astropy.io import fits

from lvmdap.config import ConfigRSP
from lvmdap.analysis.stats import downgrade_resolution

from lvmdap.modelling.synthesis import StellarSynthesis
from .dap import _no_traceback


SPECTRA_TYPES = ["FLUX", "ERROR", "MASK", "SIGINST"]
WAVELENGTH_NORM = 5490, 5510
EXT_CHOICES = ["CCM", "CAL"]
EXT_CURVE = EXT_CHOICES[0]
EXT_RV = 3.1
N_MC = 20
CWD = os.path.abspath(".")

def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="analysis of MWM stellar spectrum using the LVM-DAP")
    parser.add_argument(
        "input_path", metavar="input-path",
        help="path to FITS file containing MWM spectrum"
    )
    parser.add_argument(
        "rsp_file", metavar="rsp-file",
        help="the resolved stellar population basis"
    )
    parser.add_argument(
        "--rsp-nl-file",
        help="the resolved stellar population *reduced* basis, for non-linear fitting"
    )
    parser.add_argument(
        "--sigma-inst",
        help=f"instrumental resolution in AA. Defaults to '0.0'",
        default=0.0
    )
    parser.add_argument(
        "--wavelength-norm", metavar=("wl_norm_ini", "wl_norm_fin"),
        help=f"the normalization windows in wavelength. The processed fluxes (and errors) will be normalized using the median within this range. Defaults to {WAVELENGTH_NORM}",
        nargs=2,
        default=WAVELENGTH_NORM,
        type=np.float
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
        default=(0.00, 0.00005, -35.14/3e5, 42.54/3e5)
    )
    parser.add_argument(
        "--sigma", metavar=("input_sigma","delta_sigma","min_sigma","max_sigma"), type=np.float, nargs=4,
        help="same as the redshift, but for the line-of-sight velocity dispersion",
        default=(0, 0, 0, 450)
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
        "--force-analyse",
        help="whether to force analysis or not. Defaults to False",
        action="store_true"
    )
    parser.add_argument(
        "-o", "--output-path", metavar="path",
        help=f"path to the outputs. Defaults to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "-d", "--debug",
        help="run in debugging mode (meant to test changes in the code)",
        action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="print useful information on screen",
        action="store_true"
    )

    args = parser.parse_args(cmd_args)
    if not args.debug:
        sys.excepthook = _no_traceback
    else:
        pprint("COMMAND LINE ARGUMENTS")
        pprint(f"{args}\n")

    # check if preprocessed file exists and pre-process only if does not or forced
    sed_name = os.path.basename(args.input_path).replace(".fits", "").replace(".gz", "")
    out_path = os.path.join(args.output_path, f"{sed_name}.lvmdap.fits.gz")
    if args.force_analyse or not os.path.isfile(out_path):
        sed_fits = fits.open(args.input_path)
        wl__w = 10**sed_fits[1].data["LOGLAM"]
        if sed_fits[0].header["VACUUM"]:
            sigma_2 = (1e4/wl__w)**2
            f = 1.0 + 0.05792105 / (238.0185 - sigma_2) + 0.00167917 / (57.362 - sigma_2)
            wl__w = wl__w / f

        f__w = sed_fits[1].data["FLUX"] * 1e-17
        ef__w = sed_fits[1].data["IVAR"]
        ef__w = np.sqrt(np.divide(1, ef__w, where=ef__w>0.0, out=np.zeros_like(ef__w))) * 1e-17
        sg__w = sed_fits[1].data["WRESL"] / 2.355

        # remove invalid flux values
        mask = (f__w > 0)&(ef__w > 0)&(sg__w > 0)
        f__w = np.interp(wl__w, wl__w[mask], f__w[mask], left=f__w[mask][0], right=f__w[mask][-1])
        ef__w = np.interp(wl__w, wl__w[mask], ef__w[mask], left=ef__w[mask][0], right=ef__w[mask][-1])
        sg__w = np.interp(wl__w, wl__w[mask], sg__w[mask], left=sg__w[mask][0], right=sg__w[mask][-1])

        diff_sg = np.sqrt(args.sigma_inst**2 - sg__w**2, where=args.sigma_inst>=sg__w, out=np.zeros_like(sg__w))
        f__w = downgrade_resolution(wl__w, f__w, sigma=diff_sg, verbose=args.verbose)
        ef__w = downgrade_resolution(wl__w, ef__w, sigma=diff_sg, verbose=args.verbose)

        # normalize spectrum
        mask_norm = (wl__w>=args.wavelength_norm[0])&(wl__w<=args.wavelength_norm[1])
        fl_norm = np.nanmedian(f__w[mask_norm])

        f__w /= fl_norm
        ef__w /= fl_norm

        cf = ConfigRSP(
            config_file=None,
            redshift_set=args.redshift,
            sigma_set=args.sigma,
            AV_set=args.AV,
            gas_fit=False
        )
        SPS = StellarSynthesis(config=cf,
            wavelength=wl__w, flux=f__w, eflux=ef__w,
            mask_list=None, elines_mask_file=None,
            sigma_inst=0.001, ssp_file=args.rsp_file,
            ssp_nl_fit_file=args.rsp_nl_file, out_file=None,
            w_min=args.w_range[0], w_max=args.w_range[1], nl_w_min=args.w_range_nl[0], nl_w_max=args.w_range_nl[1],
            R_V=args.RV, extlaw=args.ext_curve, spec_id=None, min=None, max=None,
            guided_errors=None, ratio_master=None,
            fit_gas=False, plot=0, verbose=0
        )
        msg_cut = f' - cut value: {cf.CUT_MEDIAN_FLUX:6.4f}'
        if cf.CUT_MEDIAN_FLUX == 0:
            msg_cut = ' - Warning: no cut (CUT_MEDIAN_FLUX = 0)'
        print(f'-> median raw flux = {SPS.median_flux:6.4f}{msg_cut}')
        SPS.cut = False
        if (SPS.valid_flux > 0) and (SPS.median_flux > cf.CUT_MEDIAN_FLUX):  # and (median_flux > cf.ABS_MIN):
            # redshift, sigma and AV fit
            SPS.non_linear_fit(False, fit_sigma_rnd=True, sigma_rnd_medres_merit=False)
            # SPS.non_linear_fit(guided_sigma)
            SPS.calc_SN_norm_window()
            min_chi_sq = 1e12
            n_iter = 0
            while ((min_chi_sq > cf.MIN_DELTA_CHI_SQ) & (n_iter < cf.MAX_N_ITER)):
                print(f'Deriving SFH... attempt {n_iter + 1} of {cf.MAX_N_ITER}')
                # Emission lines fit
                SPS.gas_fit(ratio=False)
                # stellar population synthesis based solely on chi_sq determination
                if args.single_rsp:
                    min_chi_sq_now = SPS.ssp_single_fit()
                else:
                    min_chi_sq_now = SPS.ssp_fit(n_MC=args.n_mc)
                SPS.resume_results()
                print(f'Deriving SFH... attempt {n_iter + 1} DONE!')
                if not args.single_rsp:
                    coeffs_table = SPS.output_coeffs_MC_to_screen()
                    SPS.output_to_screen(block_plot=False)
                if min_chi_sq_now < min_chi_sq:
                    min_chi_sq = min_chi_sq_now
                n_iter += 1
        else:
            SPS.cut = True
            print('-> median flux below cut: unable to perform analysis.')

        rsp_model = SPS.output_spectra_list[1]
        teff, logg, feh, afe, teff_m, logg_m, feh_m, afe_m = SPS.models.moments_from_coeffs(SPS.coeffs_norm)
        teff = 10**teff
        teff_m = 10**teff_m

        header_new = sed_fits[0].header.copy()
        del header_new["*COMMENT*"]
        header_new["FNORM"] = (fl_norm, "[erg/cm^2/s/AA] normalization flux")
        header_new.add_comment("", before="FNORM")
        header_new.add_comment("*** LVM-DAP ***", before="FNORM")
        header_new.add_comment("", before="FNORM")
        header_new["TEFF"] = (teff, "[K] effective temperature from LVM-DAP")
        header_new["LOGG"] = (logg, "[log/cm/s^2] surface gravity from LVM-DAP")
        header_new["FEH"] = (feh, "[Fe/H] metallicity from LVM-DAP")
        header_new["ALPHAM"] = (afe, "[alpha/Fe] alpha-to-iron abundance from LVM-DAP")
        header_new["MWTEFF"] = (teff_m, "[K] MW effective temperature from LVM-DAP")
        header_new["MWLOGG"] = (logg_m, "[log/cm/s^2] MW surface gravity from LVM-DAP")
        header_new["MWFEH"] = (feh_m, "[Fe/H] MW metallicity from LVM-DAP")
        header_new["MWALPHAM"] = (afe_m, "[alpha/Fe] MW alpha-to-iron abundance from LVM-DAP")
        header_new["SIGMA"] = (SPS.best_sigma, "[km/s] LOSVD from LVM-DAP")
        header_new["REDSHIFT"] = (SPS.best_redshift, "redshift from LVM-DAP")
        header_new["AVDUST"] = (SPS.best_AV, "[mag] V dust extinction from LVM-DAP")
        hdu_0 = fits.PrimaryHDU(header=header_new)

        columns = [
            fits.Column(name="WAVELENGTH", unit="AA", format="E", array=wl__w),
            fits.Column(name=SPECTRA_TYPES[0], unit="FNORM", format="E", array=f__w),
            fits.Column(name=SPECTRA_TYPES[1], unit="FNORM", format="E", array=ef__w),
            fits.Column(name=SPECTRA_TYPES[3], unit="AA", format="E", array=np.sqrt(sg__w**2+diff_sg**2)),
            fits.Column(name="MODEL", unit="FNORM", format="E", array=rsp_model)
        ]
        hdu_1 = fits.BinTableHDU.from_columns(fits.ColDefs(columns), name="SPECTRA")
        hdu_2 = fits.BinTableHDU(coeffs_table, name="COEFFS")
        
        hdulist = fits.HDUList([hdu_0, hdu_1, hdu_2])
        hdulist.writeto(out_path, overwrite=True, output_verify="silentfix")