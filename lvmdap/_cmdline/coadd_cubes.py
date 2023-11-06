# 3. coadd cubes of the same pointing
#   - inputs:
#       * list of maps
#   - outputs:
#       * coadded cubes
#   - steps:
#       * read maps
#       * store primary headers
#       * calculate coadded mosaic
#       * display coadded mosaic
#       * identify cubes to coadd
#       * device strategy for coadding cubes (e.g., slice-by-slice)
#       * coadd cubes and propagate errors
#       * save coadded cubes
import os
import sys
import argparse
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS

from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking import reproject_and_coadd


CWD = os.path.abspath(".")
WL_RANGE = (5450, 5550)


def _no_traceback(type, value, traceback):
  print(value)


def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Convert a cube FITS into a 2D map within a given spectral range"
    )
    parser.add_argument(
        "maps-path", metavar="maps_path",
        help="path to input maps"
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

    maps_paths = sorted([os.path.join(root, file) for root, _, files in os.walk(args.maps_path) for file in files if file.endswith(".map.fits.gz")])
    flux_hdus, cube_paths = [], []
    for map_path in os.path.listdir(maps_paths):
        image = fits.open(map_path)
        flux_hdus.append(image[0])
        cube_paths.append(image[0].header["CUBEPATH"])

    wcs_out, shape_out = find_optimal_celestial_wcs(
        flux_hdus, auto_rotate=True)


    array, footprint = reproject_and_coadd(flux_hdus,
                                        wcs_out, shape_out=shape_out,
                                        reproject_function=reproject_interp)
    
    plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(1, 2, 1)
    im1 = ax1.imshow(array, origin='lower')
    ax1.set_title('Mosaic')
    ax2 = plt.subplot(1, 2, 2)
    im2 = ax2.imshow(footprint, origin='lower')
    ax2.set_title('Footprint')

        
    return None
