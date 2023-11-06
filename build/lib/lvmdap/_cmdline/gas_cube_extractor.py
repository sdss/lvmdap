import sys
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

from pprint import pprint
from scipy.interpolate import griddata
from scipy.ndimage import median_filter, gaussian_filter
from pyFIT3D.common.constants import __sigma_to_FWHM__
from pyFIT3D.common.tools import rss_seg2cube, get_slice, smooth_spec_clip_cube, spec_extract_cube_mean
from pyFIT3D.common.io import array_to_fits
from pyFIT3D.common.io import get_wave_from_header
from pyFIT3D.common.tools import flux_elines_cube_EW

from copy import deepcopy as copy
from astropy.io import fits

from astropy.visualization import make_lupton_rgb
from tqdm import tqdm

from lvmdap._cmdline.dap import N_MC
from lvmdap._cmdline.preprocess_manga import MANGA_SCALE, MANGA_PSF
from lvmdap._cmdline.preprocess_muse import MUSE_SCALE, MUSE_PSF


KIND_CHOICES = ["manga", "muse"]
KIND_DEFAULT = KIND_CHOICES[0]

CWD = os.getcwd()
SLICE_CONFIG_PATH = "../../_fitting-data/_configs/slice_V.conf"
EMISSION_LINES_LIST = "../../_fitting-data/_configs/MaNGA/emission_lines_momana.txt"
RGB_ELINES = "[NII] Halpha [OIII]".split()
HALPHA_WL = 6562.85
CUBE_PATTERN = ".cube.fits.gz"

def _no_traceback(type, value, traceback):
  print(value)

def get_cs_map(filename, good_pix_mask):
    cs = fits.open(filename)

    yo, xo = np.where(cs[0].data>0)
    values = cs[0].data[yo,xo]

    yi, xi = np.where(cs[0].data<9999)

    seg_map__yx = griddata(np.column_stack((xo,yo)), values, np.column_stack((xi,yi)), method="nearest").reshape(cs[0].data.shape)*good_pix_mask.astype(int)

    return seg_map__yx

def read_fit_elines_rss(filename):
    iseg = 0
    wave__s, flux__s, eflux__s, vel__s, evel__s, sig__s, esig__s = [], [], [], [], [], [], []
    mtypes, wlc, flx, e_flx, sig, e_sig, vel, e_vel = [], [], {}, {}, {}, {}, {}, {}
    with open(filename) as f:
        for line in f:
            if line.startswith("#"):
                last_seg = iseg
                iseg = int(line.split()[-1])
                if iseg != last_seg:
                    mtypes, wlc, flx, e_flx, sig, e_sig, vel, e_vel = [], [], {}, {}, {}, {}, {}, {}
            nmod, _ = tuple(eval(v) for v in f.readline()[:-1].split())
            for _ in range(nmod):
                line_model = f.readline()[:-1].split()
                mtype = line_model[0]
                if mtype == "eline":
                    mtypes.append(mtype)
                    wl = eval(line_model[1])
                    wlc.append(wl)
                    
                    flx[wl] = eval(line_model[3])
                    e_flx[wl] = eval(line_model[4])
                    sig[wl] = eval(line_model[5])
                    e_sig[wl] = eval(line_model[6])
                    vel[wl] = eval(line_model[7])
                    e_vel[wl] = eval(line_model[8])
            wave__s.append(wlc)
            flux__s.append(flx)
            eflux__s.append(e_flx)
            vel__s.append(vel)
            evel__s.append(e_vel)
            sig__s.append(sig)
            esig__s.append(e_sig)

    ns = last_seg + 1
    wave__m = np.unique(np.concatenate(wave__s))
    flux__ms = np.zeros((wave__m.size, ns))
    eflux__ms = np.zeros((wave__m.size, ns))
    sig__ms = np.zeros((wave__m.size, ns))
    esig__ms = np.zeros((wave__m.size, ns))
    vel__ms = np.zeros((wave__m.size, ns))
    evel__ms = np.zeros((wave__m.size, ns))

    for iwl in range(wave__m.size):
        for iseg in range(ns):
            flux__ms[iwl, iseg] = flux__s[iseg].get(wave__m[iwl], 0.0)
            eflux__ms[iwl, iseg] = eflux__s[iseg].get(wave__m[iwl], 0.0)
            sig__ms[iwl, iseg] = sig__s[iseg].get(wave__m[iwl], 0.0)
            esig__ms[iwl, iseg] = esig__s[iseg].get(wave__m[iwl], 0.0)
            vel__ms[iwl, iseg] = vel__s[iseg].get(wave__m[iwl], 0.0)
            evel__ms[iwl, iseg] = evel__s[iseg].get(wave__m[iwl], 0.0)
    
    return wave__m, flux__ms, eflux__ms, vel__ms, evel__ms, sig__ms, esig__ms

def get_gas_cube(org_cube__wyx, err_cube__wyx, org_wave__w, out_rss__tsw, wave_rss__w, seg_map__yx, label, slice_conf, spatial_psf):
    slice_prefix = f'img_{label}'
    slices = get_slice(copy(org_cube__wyx), org_wave__w, slice_prefix, slice_conf)
    V__yx = list(slices.values())[0]
    # TODO: mask all emission lines in the original cube,
    #       this map in V-band assumes that all emission is stellar emission
    V__yx[np.isnan(V__yx)] = -1
    mV__yx = median_filter(V__yx, size=(2, 2), mode='reflect')

    org_rss__sw, _, _, _, _ = spec_extract_cube_mean(copy(org_cube__wyx), seg_map__yx)
    cube__wyx = rss_seg2cube(org_rss__sw, seg_map__yx)

    slice_prefix = f'SEG_img_{label}'
    slices = get_slice(copy(cube__wyx), wave_rss__w, slice_prefix, slice_conf)
    V_slice__yx = list(slices.values())[0]
    V_slice__yx[np.isnan(V_slice__yx)] = -1
    scale_seg__yx = np.divide(mV__yx, V_slice__yx, where=V_slice__yx!=0, out=np.zeros_like(V_slice__yx))

    rsp_mod_tmp__wyx = rss_seg2cube(copy(out_rss__tsw[1]), seg_map__yx)
    rsp_mod_cube__wyx = rsp_mod_tmp__wyx*scale_seg__yx

    # BUG: this is a patch for a scale problem in the RSP cube
    #      not matching the scale of the original cube
    slice_prefix = f'img_{label}'
    slices = get_slice(copy(rsp_mod_cube__wyx), org_wave__w,
                       slice_prefix, slice_conf)
    rV__yx = list(slices.values())[0]
    rV__yx[np.isnan(rV__yx)] = -1
    # rsp_mod_cube__wyx = rsp_mod_cube__wyx / rV__yx * mV__yx
    rsp_mod_cube__wyx = np.divide(
        rsp_mod_cube__wyx * mV__yx, rV__yx, where=rV__yx != 0.0, out=rsp_mod_cube__wyx)
    # ----------------------------------------------------------

    # clean rsp cube
    # TODO: remove whole rsp spectrum if is within the level of noise. This is essentially no underlying SP. Run this in the V-band
    # TODO: perform penalization using the SNR instead of a sigma clipping
    
    # sigma clip criterium -------------------------------------------------------------------------
    # mask_negative = rsp_mod_cube__wyx<0
    # negative_fluxes = np.zeros(2*mask_negative.sum())
    # negative_fluxes[:negative_fluxes.size//2] = rsp_mod_cube__wyx[mask_negative].ravel()
    # negative_fluxes[negative_fluxes.size//2:] = -1*rsp_mod_cube__wyx[mask_negative].ravel()
    # sigma_clip = 1*negative_fluxes.std()
    # mask_clean = (rsp_mod_cube__wyx>sigma_clip).all(axis=0)
    # SNR criterium --------------------------------------------------------------------------------
    snr_cube__wyx = org_cube__wyx/err_cube__wyx
    slices = get_slice(copy(snr_cube__wyx), org_wave__w,
                       slice_prefix, slice_conf)
    snrV__yx = list(slices.values())[0]
    # snrV__yx[np.isnan(snrV__yx)] = -1
    mask_clean = snrV__yx>40
    
    rsp_mod_cube__wyx = gaussian_filter(rsp_mod_cube__wyx, sigma=(0, 2*spatial_psf, 2*spatial_psf)) * mask_clean[None]

    # plt.imshow(rsp_mod_cube__wyx.mean(axis=0), origin="lower")
    # plt.colorbar()
    # plt.show()

    # plt.imshow(org_cube__wyx.mean(axis=0), origin="lower")
    # plt.colorbar()
    # plt.show()

    # plt.imshow((org_cube__wyx-rsp_mod_cube__wyx).mean(axis=0), origin="lower")
    # plt.colorbar()
    # plt.show()
    # exit()

    tmp_cube__wyx = org_cube__wyx - rsp_mod_cube__wyx

    # smooth
    smooth_cube__wyx = smooth_spec_clip_cube(copy(tmp_cube__wyx), wavebox_width=75, sigma=1.5, wavepix_min=10, wavepix_max=1860)

    # generate GAS cube
    gas_cube__wyx = tmp_cube__wyx - smooth_cube__wyx
    # gas_cube__wyx[gas_cube__wyx<0] = 0
    # gas_cube__wyx += np.nanmin(gas_cube__wyx)
    gas_cube__wyx[:, np.isnan(org_cube__wyx).all(axis=0)] = np.nan
    rsp_mod_cube__wyx[:, np.isnan(org_cube__wyx).all(axis=0)] = np.nan
    scale_seg__yx[np.isnan(org_cube__wyx).all(axis=0)] = np.nan
    
    # plt.imshow(tmp_cube__wyx.mean(axis=0), origin="lower")
    # plt.colorbar()
    # plt.show()

    # plt.imshow(gas_cube__wyx.mean(axis=0), origin="lower")
    # plt.colorbar()
    # plt.show()
    # exit()

    return gas_cube__wyx, rsp_mod_cube__wyx, scale_seg__yx

def _main(cmd_args=sys.argv[1:]):
    parser = argparse.ArgumentParser(
        description="Run gas cube extraction from RSS analysis"
    )
    parser.add_argument(
        "--pointing",
        help=f"optional pointing for which gas cube will be extracted. If not given, run analysis on all cubes found"
    )
    parser.add_argument(
        "-i", "--input-path", metavar="path",
        help=f"path to the inputs. Defaults to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "-p", "--dataproducts-path", metavar="path",
        help=f"path where to find Pipe3D maps. Defatuls to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "-o", "--output-path", metavar="path",
        help=f"path to the analyzed RSS spectra and where to save outputs of this script. Defaults to '{CWD}'",
        default=CWD
    )
    parser.add_argument(
        "--kind", metavar="survey",
        help=f"survey name. This will define the path structure for Pipe3D dataproducts. Choices are: {KIND_CHOICES}, defaults to '{KIND_DEFAULT}'",
        choices=KIND_CHOICES, default=KIND_DEFAULT
    )
    parser.add_argument(
        "--slice-config-file", metavar="filename",
        help=f"filename of the slice configuration file. Defaults to '{SLICE_CONFIG_PATH}'",
        default=SLICE_CONFIG_PATH
    )
    parser.add_argument(
        "--elines-list-file", metavar="filename",
        help=f"filename of the emission lines list to use in moment analysis. Defaults to '{EMISSION_LINES_LIST}'",
        default=EMISSION_LINES_LIST
    )
    parser.add_argument(
        "-n", "--n-mc", type=int,
        help=f"number of MC realisations for the moment analysis. Defaults to {N_MC}",
        default=N_MC
    )
    parser.add_argument(
        "--rgb-elines", nargs=3,
        help=f"name of the emission lines to use for RGB composed image of gas cube. Defaults to {', '.join(RGB_ELINES)}",
        default=RGB_ELINES
    )
    parser.add_argument(
        "--overwrite",
        help="whether to overwrite output files or not (default)",
        action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose",
        help="if given, shows information about the progress of the script",
        action="store_true"
    )
    parser.add_argument(
        "-d", "--debug",
        help="run in debugging mode",
        action="store_true"
    )
    args = parser.parse_args(cmd_args)
    
    if not args.debug:
        sys.excepthook = _no_traceback
    else:
        pprint("COMMAND LINE ARGUMENTS")
        pprint(f"{args}\n")

    # read moment analysis lines list
    with open(args.elines_list_file, "r") as elines_file:
        records = elines_file.readlines()
        mom_elines = []
        for rec in records:
            if rec.startswith("#"): continue
            split = rec.split()
            mom_elines.append([float(split[0]), " ".join(split[1:]) if len(split[1:])>1 else split[1]])
    mom_elines = dict(mom_elines)

    if args.kind == "manga":
        spatial_psf = MANGA_PSF / MANGA_SCALE
    elif args.kind == "muse":
        spatial_psf = MUSE_PSF / MUSE_SCALE

    if args.pointing is not None:
        org_cubes_path = sorted([os.path.join(root,file) for root, _, files in os.walk(args.input_path) for file in files if file.startswith(args.pointing) and file.endswith(CUBE_PATTERN)])
    else:
        org_cubes_path = sorted([os.path.join(root, file) for root, _, files in os.walk(args.input_path) for file in files if file.endswith(CUBE_PATTERN)])

    for i, cube_path in enumerate(org_cubes_path):
        label = os.path.basename(cube_path).replace(CUBE_PATTERN, "") if args.pointing is None else args.pointing
        cs_paths = [os.path.join(args.dataproducts_path, label, f"cont_seg.{label}.fits.gz"), os.path.join(args.dataproducts_path, f"cont_seg.{label}.fits.gz")]
        out_rss_path = os.path.join(args.output_path, f"output.{label}.fits.gz")
        elines_path = os.path.join(args.output_path, f"elines_{label}")

        cs_paths_exists = list(filter(lambda path: os.path.isfile(path), cs_paths))
        print(out_rss_path)
        if len(cs_paths_exists) == 0:
            print(f"CS map file for cube {label} is missing")
            continue
        else:
            cs_path = cs_paths_exists[0]
        if not os.path.isfile(out_rss_path):
            print(f"output RSS for cube {label} is missing")
            continue
        if not os.path.isfile(elines_path):
            print(f"output elines for cube {label} is missing")
            continue

        gas_cube_path = os.path.join(args.output_path, f"{label}-gas.cube.fits.gz")
        rsp_cube_path = os.path.join(args.output_path, f"{label}-rsp.cube.fits.gz")
        # TODO: add original header to moments cube
        mom_cube_path = os.path.join(args.output_path, f"{label}-moments.cube.fits.gz")
        dez_map_path = os.path.join(args.output_path, f"{label}-dezonification.map.fits.gz")
        existing_outputs = list(filter(os.path.isfile, [gas_cube_path,rsp_cube_path,mom_cube_path,dez_map_path]))
        if len(existing_outputs) == 4 and not args.overwrite: continue

        # read original cube
        cube = fits.open(cube_path, memmap=False)
        org_cube__wyx = np.nan_to_num(cube[0].data)
        err_cube__wyx = gaussian_filter(np.nan_to_num(cube[1].data), sigma=(0, spatial_psf, spatial_psf))
        org_wave__w = get_wave_from_header(cube[0].header, wave_axis=3)
        # BUG: since I'm converting nans to zeros above, at this point
        #      I need to check if it is necessary to treat nans and zero
        #      flux in different ways
        mask = (org_cube__wyx!=0).any(axis=0)

        # read RSS fitting output
        out_rss = fits.open(out_rss_path, memmap=False)
        wave_rss__w = get_wave_from_header(out_rss[0].header)
        out_rss__tsw = out_rss[0].data

        # read segmentation map
        seg_map__yx = get_cs_map(cs_path, good_pix_mask=mask)

        # compute gas cube
        gas_cube__wyx, rsp_mod_cube__wyx, dez_map__yx = get_gas_cube(org_cube__wyx, err_cube__wyx, org_wave__w, out_rss__tsw, wave_rss__w, seg_map__yx, label, slice_conf=args.slice_config_file, spatial_psf=spatial_psf)
        
        # run moment analysis

        # plt.imshow(org_cube__wyx.mean(axis=0), origin="lower")
        # plt.colorbar()
        # plt.show()

        # plt.imshow((gas_cube__wyx / err_cube__wyx).mean(axis=0), origin="lower")
        # plt.colorbar()
        # plt.show()
        # exit()

        # generate RSS of emission lines fitting
        wave__m, _, _, vel__ms, _, sig__ms, _ = read_fit_elines_rss(elines_path)

        vel_map__yx = gaussian_filter(vel__ms[np.where(wave__m == HALPHA_WL)].ravel()[seg_map__yx.astype(int)-1], sigma=2*spatial_psf)
        vel_map__yx[seg_map__yx == 0] = np.nan
        sig_map__yx = gaussian_filter(sig__ms[np.where(wave__m == HALPHA_WL)].ravel()[seg_map__yx.astype(int)-1], sigma=2*spatial_psf)
        sig_map__yx[seg_map__yx == 0] = np.nan

        # plt.imshow(vel_map__yx, origin="lower")
        # plt.colorbar()
        # plt.show()

        # plt.imshow(sig_map__yx, origin="lower")
        # plt.colorbar()
        # plt.show()
        # exit()

        mom_cube__wyx, mom_header = flux_elines_cube_EW(
            flux__wyx=gas_cube__wyx,
            input_header=cube[0].header,
            n_MC=args.n_mc,
            elines_list=args.elines_list_file,
            vel__yx=vel_map__yx, sigma__yx=sig_map__yx,
            flux_ssp__wyx=rsp_mod_cube__wyx, eflux__wyx=err_cube__wyx
        )
        
        # make RGB gas image
        mom_elines_names = list(mom_elines.values())
        if np.all(np.isin(args.rgb_elines,mom_elines_names)):
            ir = mom_elines_names.index(args.rgb_elines[0])
            ig = mom_elines_names.index(args.rgb_elines[1])
            ib = mom_elines_names.index(args.rgb_elines[2])

            nlines = mom_cube__wyx.shape[0]//8
            fluxes = np.nan_to_num(mom_cube__wyx[:nlines], nan=0)
            rgb_image = make_lupton_rgb(fluxes[ir], fluxes[ig], fluxes[ib], stretch=1, Q=0)

            # plt.imshow(mom_cube__wyx[nlines:2*nlines][ig], origin="lower")
            # plt.colorbar()
            # plt.show()
            # exit()

            plt.figure(figsize=(10,10))
            plt.imshow(rgb_image, origin="lower")
            plt.savefig(os.path.join(args.output_path, f"{label}-RGB.jpeg"), bbox_inches="tight")
            plt.close(fig="all")

        # write cubes
        array_to_fits(gas_cube_path, gas_cube__wyx, header=cube[0].header, overwrite=True)
        array_to_fits(rsp_cube_path, rsp_mod_cube__wyx, header=cube[0].header, overwrite=True)
        array_to_fits(mom_cube_path, mom_cube__wyx, header=mom_header, overwrite=True)
        array_to_fits(dez_map_path, dez_map__yx, overwrite=True)

        tqdm.write(f"**** done cube {label} ({i+1}/{len(org_cubes_path)}) ****")
