
import sys
import os
import numpy as np
import argparse

from copy import deepcopy as copy
from pprint import pprint
from tqdm import tqdm
from astropy.io import fits

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import patches
from scipy.ndimage import gaussian_filter

from pyFIT3D.common.tools import get_wave_from_header


CWD = os.getcwd()
VOXEL_RADIUS = 2.5
ALPHA = 1.06

MANGA_SCALE = 0.5 # arcsec/pix
# MANGA_PSF = 1.44 / np.sqrt(3)
MANGA_PSF = 2.4 / 2.355

CUBE_PATTERN = ".cube.fits.gz"
SPECTRA_RSS_NAME = "CS.{}.RSS.fits.gz"
ERRORS_RSS_NAME = "e_CS.{}.RSS.fits.gz"
CS_NAME = "cont_seg.{}.fits.gz"

def _no_traceback(type, value, traceback):
  print(value)

def circle_packing(center, radius, points, centers_precision=2):
    hexagon = patches.RegularPolygon(center, 6, 2*radius, np.pi/2, lw=1, fc="none", ec="0.5")
    points_in_hexagons = hexagon.contains_points(points)

    hexagons, centers = [hexagon], [center]
    while not np.all(points_in_hexagons):
        new_hexagons = []
        for hexagon in hexagons:
            new_centers = hexagon.get_verts()
            for new_center in new_centers:
                new_hexagon = patches.RegularPolygon(new_center, 6, 2*radius, np.pi/2, lw=1, fc="none", ec="0.5")
                points_in_hexagons = np.logical_or(points_in_hexagons, new_hexagon.contains_points(points))
                new_hexagons.append(new_hexagon)
            centers.extend(new_centers)
        hexagons = new_hexagons

    centers = np.unique(np.round(centers, centers_precision), axis=0)
    return centers

def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run RSS preprocessing of MUSE data"
    )
    parser.add_argument(
        "--pointing",
        help=f"optional pointing for which gas cube will be extracted. If not given, run analysis on all cubes found"
    )
    parser.add_argument(
        "-r", "--voxel-radius", type=float,
        help=f"the circular voxel radius. Defaults to {VOXEL_RADIUS} arcsec",
        default=VOXEL_RADIUS
    )
    parser.add_argument(
        "--alpha", type=float,
        help=f"the slope in the covariance correction in noise propagation for each voxel. Defaults to {ALPHA}",
        default=ALPHA
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
        "--overwrite",
        help="whether to overwrite output files or not (default)",
        action="store_true"
    )
    parser.add_argument(
        "-l", "--cube-list-file", metavar="file",
        help="file listing the cube names to preprocess"
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

    radius = args.voxel_radius / MANGA_SCALE
    
    # list original cubes
    if args.cube_list_file is None:
        if args.pointing is not None:
            org_cubes_path = sorted([os.path.join(root, file) for root, _, files in os.walk(args.input_path) for file in files if file.startswith(args.pointing) and file.endswith(CUBE_PATTERN)])
        else:
            org_cubes_path = sorted([os.path.join(root, file) for root, _, files in os.walk(args.input_path) for file in files if file.endswith(CUBE_PATTERN)])
    else:
        org_cubes_path = []
        with open(args.cube_list_file) as cube_list:
            for cube_name in cube_list.readlines():
                cube_file = os.path.join(args.input_path, cube_name[:-1])
                if not os.path.exists(cube_file): continue
                org_cubes_path.append(cube_file)

    for cube_file in tqdm(org_cubes_path, desc="preprocessing MaNGA cubes", unit="cube", ascii=True):
        
        label = os.path.basename(cube_file).replace(CUBE_PATTERN, "")

        rss_path = os.path.join(args.output_path,SPECTRA_RSS_NAME.format(label))
        err_path = os.path.join(args.output_path,ERRORS_RSS_NAME.format(label))
        cs_path = os.path.join(args.output_path,CS_NAME.format(label))
        existing_outputs = list(filter(os.path.isfile, [rss_path,err_path,cs_path]))
        if len(existing_outputs) == 3 and not args.overwrite: continue

        # read cube
        obs_cube = fits.getdata(cube_file)
        err_cube = fits.getdata(cube_file, ext=1)
        err_cube = gaussian_filter(err_cube, sigma=(
            0, MANGA_PSF/MANGA_SCALE, MANGA_PSF/MANGA_SCALE))
        good_mask = (np.nan_to_num(obs_cube) != 0).any(axis=0)
        _, ny, nx = obs_cube.shape

        X, Y = np.meshgrid(np.arange(nx)+0.5, np.arange(ny)+0.5)
        spaxels = np.column_stack((X.ravel(), Y.ravel()))

        center = np.mean(spaxels, axis=0)
        centers = circle_packing(center, radius, np.column_stack(np.where(good_mask.T))+0.5)

        nv = len(centers)
        voxels, voxel_masks = [], []
        spectra_rss, errors_rss = [], []
        remaining_mask = copy(good_mask)
        spaxel_mask = np.zeros((ny,nx), dtype=bool)
        for i, center in enumerate(centers):
            # compute bins
            voxel = patches.Circle(xy=center, radius=radius, fc="none", ec="magenta", lw=1)
            voxels.append(voxel)

            # update remaining spaxels mask to remove last voxel
            remaining_mask = remaining_mask & (~spaxel_mask)
            # calculate current voxel mask
            spaxel_mask = voxel.contains_points(spaxels).reshape(ny,nx)
            # intersect current voxel mask & remaining spaxels mask to remove redundant spaxels
            voxel_mask = spaxel_mask & remaining_mask
            # plotting for debugging
            # _, ax = plt.subplots(figsize=(10,10))
            # ax.scatter(*spaxels[voxel_mask.ravel()].T, lw=0, s=10, c="k")
            # ax.add_patch(copy(voxel))
            # ax.set_xlim(0, nx)
            # ax.set_ylim(0, ny)
            # ax.set_aspect("equal")
            # plt.show()
            voxel_masks.append(voxel_mask)

            # sum spectra within each bin
            spectra_rss.append(np.mean(obs_cube, where=voxel_mask[None], axis=(1,2)))
            # propagate the errors
            N = voxel_mask.sum()
            covar = 1 + args.alpha * np.log10(N)
            errors_rss.append(np.sqrt(np.sum(err_cube**2, where=voxel_mask[None], axis=(1,2))/N)*covar)
        
        # build CS
        cs_map = np.nansum([voxel_masks[ivox].astype(int)*(ivox+1) for ivox in range(nv)], axis=0)

        # store resulting spectra in RSS
        cube_header = fits.getheader(cube_file)
        hdr = fits.Header({k.replace("3","1"):cube_header[k] for k in ["CDELT3", "CRVAL3", "CRPIX3"]})

        fits.PrimaryHDU(spectra_rss, header=hdr).writeto(rss_path, overwrite=True)
        fits.PrimaryHDU(errors_rss, header=hdr).writeto(err_path, overwrite=True)
        fits.PrimaryHDU(cs_map).writeto(cs_path, overwrite=True)
        
        # plot bins over V-band image =============================================================
        wl = get_wave_from_header(cube_header, wave_axis=3)
        wl_mask = (5450<=wl)&(wl<=5550)
        V_image = np.nanmedian(obs_cube[wl_mask], axis=0)

        _, ax = plt.subplots(figsize=(10,10))
        ax.imshow(V_image, origin="lower")
        [ax.add_patch(copy(voxel)) for voxel in voxels]
        plt.savefig(os.path.join(args.output_path,f"{label}-v-map_voxels.png"), bbox_inches="tight")
        # plot voxel mask =========================================================================
        _, ax = plt.subplots(figsize=(10,10))
        plt.imshow(np.logical_or.reduce(voxel_masks), origin="lower", cmap="binary")
        [ax.add_patch(copy(voxel)) for voxel in voxels]
        plt.savefig(os.path.join(args.output_path,f"{label}-cs-mask.png"), bbox_inches="tight")

        plt.close("all")

