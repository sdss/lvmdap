
from pickle import NONE
import sys
import os
import itertools as it
import numpy as np
import argparse

from pprint import pprint
from requests import head
from tqdm import tqdm
from specutils import Spectrum1D
from specutils.manipulation import gaussian_smooth
from astropy.io import fits
import astropy.units as u

from pyFIT3D.common.tools import get_wave_from_header


CWD = os.getcwd()
FWHM_MUSE = 2.2
FWHM_NEW = 6.0
MUSE_SCALE = 0.2
MUSE_PSF = 1.0

def _no_traceback(type, value, traceback):
  print(value)

def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run RSS preprocessing of MUSE data"
    )
    parser.add_argument(
        "--sigma-inst", type=float,
        help=f"the target instrumental dispersion to downgrade the input spectra to. Defaults to {FWHM_NEW}AA",
        default=FWHM_NEW
    )
    parser.add_argument(
        "-i", "--input-path", metavar="path",
        help=f"path to the inputs. Defaults to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "-o", "--output-path", metavar="path",
        help=f"path to the outputs. Defaults to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "-t", "--input-type", metavar="data format",
        help=f"type of input file to preprocess, it can be either 'cube' or 'CS' (default)",
        choices=("cube", "CS"), default="CS"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="if given, shows information about the progress of the script. Defaults to false",
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
    
    res_correction = np.sqrt(args.sigma_inst**2 - FWHM_MUSE**2)/2.355

    if args.input_type == "CS":
        sed_rss = sorted([os.path.join(root,file) for root, _, files in os.walk(args.input_path) for file in files if file.startswith("CS.LMC") and file.endswith(".RSS.fits.gz")])
        err_rss = [os.path.join(os.path.dirname(file), f"e_{os.path.basename(file)}") for file in sed_rss]

        no_error_file = False
        for ipoint in tqdm(range(len(sed_rss)), desc="writing RSS", ascii=True, unit="pointing"):
            f = fits.open(sed_rss[ipoint], memmap=False)
            if not os.path.isfile(err_rss[ipoint]):
                no_error_file = True
            else:
                e = fits.open(err_rss[ipoint], memmap=False)

            wl = get_wave_from_header(f[0].header)
            dummy_err = np.abs(1e-7*np.ones_like(wl))
            
            sed_rss_cor, err_rss_cor = [], []
            for ised in range(f[0].data.shape[0]):
                sed_rss_cor.append(gaussian_smooth(Spectrum1D(
                        spectral_axis=wl * u.AA,
                        flux=f[0].data[ised] * u.erg/u.s/u.AA
                    ), stddev=res_correction/np.diff(wl)[0]).flux.value
                )
                err_rss_cor.append(dummy_err if no_error_file else np.where(e[0].data[ised]>0, e[0].data[ised], dummy_err))

            no_error_file = False
        
            f[0].data = np.asarray(sed_rss_cor)
            e[0].data = np.asarray(err_rss_cor)
            f.writeto(os.path.join(args.output_path,os.path.basename(sed_rss[ipoint])), overwrite=True)
            e.writeto(os.path.join(args.output_path,os.path.basename(err_rss[ipoint])), overwrite=True)
    elif args.input_type == "cube":
        sed_cube = sorted([os.path.join(root,file) for root, _, files in os.walk(args.input_path) for file in files if file.startswith("LMC_") and file.endswith(".cube.fits.gz")])

        for ipoint in tqdm(range(len(sed_cube)), desc="writing RSS", ascii=True, unit="pointing"):
            cube_path = sed_cube[ipoint]
            label = os.path.basename(cube_path).replace(".cube.fits.gz", "")
            f = fits.open(cube_path, memmap=False)
            
            wl = get_wave_from_header(f[0].header, wave_axis=3)
            dummy_err = np.abs(np.ones_like(wl))

            sed_rss_cor, err_rss_cor = [], []
            pt = ["C 1 1 0"]
            for ised, (ix, iy) in enumerate(it.product(range(f[0].data.shape[2]), range(f[0].data.shape[1]))):
                pt.append(f"{ised} {ix} {iy} 1")
                sed = f[0].data[:, iy, ix]
                err = f[1].data[:, iy, ix]
                if np.isnan(sed).all() or np.isnan(sed).sum() / sed.size > 0.1: continue

                sed_rss_cor.append(gaussian_smooth(Spectrum1D(
                        spectral_axis=wl*u.AA,
                        flux=sed*u.erg/u.s/u.AA
                    ), stddev=res_correction/np.diff(wl)[0]).flux.value
                )

                if np.isnan(err).all() or np.isnan(err).sum() / err.size > 0.1: err = dummy_err
                err_rss_cor.append(gaussian_smooth(Spectrum1D(
                        spectral_axis=wl*u.AA,
                        flux=err*u.erg/u.s/u.AA
                    ), stddev=res_correction/np.diff(wl)[0]).flux.value
                )

            header = fits.Header()
            header["CDELT1"] = f[0].header["CDELT3"]
            header["CRPIX1"] = f[0].header["CRPIX3"]
            header["CRVAL1"] = f[0].header["CRVAL3"]
            f_rss = fits.PrimaryHDU(data=np.asarray(sed_rss_cor), header=header)
            e_rss = fits.PrimaryHDU(data=np.asarray(err_rss_cor), header=header)
            f_rss.writeto(os.path.join(args.output_path,f"CS.{label}.RSS.fits.gz"), overwrite=True)
            e_rss.writeto(os.path.join(args.output_path,f"e_CS.{label}.RSS.fits.gz"), overwrite=True)
            with open(os.path.join(args.output_path,f"CS.{label}.RSS.pt.txt"), "w") as pt_file:
                for record in pt:
                    pt_file.write(f"{record}\n")
            