import os
import sys
import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS


CWD = os.path.abspath(".")
WL_RANGE = (5450, 5550)


def _no_traceback(type, value, traceback):
  print(value)


def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Convert a MUSE cube into a 2D map within a given spectral range"
    )
    parser.add_argument(
        "cube_path", metavar="cube-path",
        help="path to input cube"
    )
    parser.add_argument(
        "-w", "--wl-range", metavar=("wli","wlf"), type=float, nargs=2,
        help=f"wavelength range within which the resulting map should be computed. Defaults to {WL_RANGE}",
        default=WL_RANGE
    )
    parser.add_argument(
        "-o", "--output-path", metavar="path",
        help=f"path to the outputs. Defaults to '{CWD}'",
        default=CWD
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

    raw = fits.open(args.cube_path)
    wavelength = raw[1].header["CRVAL3"] + raw[1].header["CD3_3"] * \
        (np.arange(raw[1].header["NAXIS3"]) + 1 - raw[1].header["CRPIX3"])

    wli, wlf = args.wl_range
    mask = (wli <= wavelength) & (wavelength <= wlf)

    flux_cube = raw[1].data
    error_cube = raw[2].data

    flux_map = np.nanmean(flux_cube[mask], axis=0)
    error_map = np.sqrt(np.nansum(error_cube[mask]**2, axis=0)/mask.sum())
    snr_map = np.divide(flux_map, error_map, where=error_map>0,
                        out=np.zeros_like(error_map))

    wcs = WCS(header=raw[1].header, naxis=2)
    new_header = wcs.to_header()
    # copy over important metadata from original header to new header
    cards = ["OBJECT"]
    for card in cards:
        new_header[card] = raw[0].header[card]
    new_header["CUBEPATH"] = args.cube_path
    new_header["WLINI"] = (wli, "[AA] initial wavelength in window")
    new_header["WLFIN"] = (wlf, "[AA] final wavelength in window")

    # write output maps
    flux_hdu = fits.PrimaryHDU(data=flux_map, header=new_header)
    error_hdu = fits.ImageHDU(data=error_map, name="ERROR_MAP")
    snr_hdu = fits.ImageHDU(data=snr_map, name="SNR_MAP")
    hdu_list = fits.HDUList([flux_hdu, error_hdu, snr_hdu])

    label = os.path.basename(args.cube_path).replace(".fits", "")
    output_path = os.path.join(
        args.output_path, f"{label}_{int(wli)}-{int(wlf)}.map.fits.gz")
    hdu_list.writeto(output_path, overwrite=True)
    return None
