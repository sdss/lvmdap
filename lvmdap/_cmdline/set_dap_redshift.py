#!/usr/bin/env python3

import sys, os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=1

import argparse
import numpy as np
import yaml
from pprint import pprint

from astropy.io import fits

# lvmdap dependencies
from lvmdap.dap_tools import read_PT

try:
  from lvmdap.gaia_velocity_lookup import load_velocity_map
except ImportError:
  from gaia_velocity_lookup import load_velocity_map

try:
  from lvmdap.yaml_edit import edit_yaml
except ImportError:
  from yaml_edit import edit_yaml


CWD = os.path.abspath(".")

# Speed of light in km/s, matching the value used elsewhere in the DAP.
__c__ = 299792.00

# Header card holding the heliocentric velocity correction of the science
# telescope, written by the LVM DRP.
HVCOR_KEY = "HIERARCH WAVE HELIORV_SCI"


def _no_traceback(type, value, traceback):
  print(value)


####################################################
# MAIN script. Set the redshift prior of a DAP
# config YAML from the Gaia all-sky velocity map
####################################################

def _main(cmd_args=sys.argv[1:]):

  parser = argparse.ArgumentParser(
    description="lvm-dap-set-z LVM_FILE GAIA_RV_MAP.fits CONFIG_IN.YAML CONFIG_OUT.YAML"
  )

  parser.add_argument(
    "lvm_file", metavar="lvm_file",
    help="input LVM spectrum, used for the pointing and the heliocentric correction"
  )

  parser.add_argument(
    "gaia_rv_file",
    help="Gaia velocity map (FITS or ECSV) built by gaia_velocity_map.py"
  )

  parser.add_argument(
    "config_yaml",
    help="config_yaml with the fitting parameters (input)"
  )

  parser.add_argument(
    "output_yaml",
    help="config_yaml with the updated redshift prior (output)"
  )

  parser.add_argument(
    "-d", "--debug",
    help="debugging mode. Defaults to false.",
    action="store_true"
  )

  parser.add_argument(
    "--ra", type=np.float64, default=None,
    help="RA in degrees, overriding the mean of the good fibres"
  )
  parser.add_argument(
    "--dec", type=np.float64, default=None,
    help="DEC in degrees, overriding the mean of the good fibres"
  )
  parser.add_argument(
    "--hvcor", type=np.float64, default=None,
    help=f"heliocentric velocity correction in km/s, overriding "
         f"'{HVCOR_KEY}'. Pass 0 to disable the correction."
  )
  parser.add_argument(
    "--hvcor-sign", type=np.float64, default=+1.0, choices=[+1.0, -1.0],
    help="sign with which HVCOR is added to the Gaia velocity. Use -1 if "
         "the spectra being fitted are already in the heliocentric frame. "
         "Defaults to +1."
  )
  parser.add_argument(
    "--ny-range", type=np.int64, nargs=2, default=None,
    help="fibre range passed to read_PT, as in gen_output_model"
  )

  parser.add_argument(
    "--nsigma", type=np.float64, default=3.0,
    help="half-width of the search range, in units of the velocity "
         "dispersion of the cell. Defaults to 3."
  )
  parser.add_argument(
    "--sigma-floor", type=np.float64, default=20.0,
    help="minimum velocity dispersion in km/s, applied when the cell value "
         "is NaN or unusably small. Defaults to 20."
  )
  parser.add_argument(
    "--sigma-max", type=np.float64, default=None,
    help="cap on the velocity dispersion in km/s, to stop contaminated "
         "cells opening an enormous search range. Defaults to no cap."
  )
  parser.add_argument(
    "--z-del", type=np.float64, default=None,
    help="step of the redshift grid. Defaults to the range divided by "
         "--n-steps."
  )
  parser.add_argument(
    "--auto-z-del", type=np.float64, default=None,
    help="step of the auto_z grid. Defaults to --z-del."
  )
  parser.add_argument(
    "--n-steps", type=np.int64, default=40,
    help="number of steps across the full range, used when --z-del is not "
         "given. Defaults to 40."
  )
  parser.add_argument(
    "--weighted", action="store_true",
    help="use avg_vel_w / std_vel_intr (inverse-variance weighted mean and "
         "error-corrected dispersion) instead of avg_vel / std_vel"
  )
  parser.add_argument(
    "--min-stars", type=np.int64, default=1,
    help="cells with fewer stars are treated as empty. Defaults to 1."
  )
  parser.add_argument(
    "--max-sep", type=np.float64, default=None,
    help="fail if the nearest populated cell is farther than this, in "
         "degrees. Defaults to no limit."
  )
  parser.add_argument(
    "--create-missing", action="store_true",
    help="create the YAML keys if they are absent, instead of failing"
  )
  parser.add_argument(
    "--dry-run", action="store_true",
    help="report the values but do not write the output YAML"
  )

  args = parser.parse_args(cmd_args)
  print(cmd_args)
  print(args)
  if not args.debug:
    sys.excepthook = _no_traceback
  else:
    pprint("COMMAND LINE ARGUMENTS")
    pprint(f"{args}\n")

  lvm_file = args.lvm_file

  print("##############################################")
  print("# Reading input files...")

  ##################################################################
  # Heliocentric velocity correction
  ##################################################################
  if args.hvcor is not None:
    HVCOR = float(args.hvcor)
    print(f"# Heliocentric velocity correction (command line): {HVCOR:.2f} km/s")
  else:
    hdr_0 = fits.open(lvm_file)[0].header
    if HVCOR_KEY not in hdr_0:
      raise SystemExit(
        f"'{HVCOR_KEY}' not found in the primary header of '{lvm_file}'.\n"
        f"  Pass --hvcor VALUE, or --hvcor 0 to skip the correction."
      )
    HVCOR = float(hdr_0[HVCOR_KEY])
    print(f"Heliocentric velocity correction: {HVCOR:.2f} km/s")

  ##################################################################
  # Pointing: mean of the good fibres in the PT table
  ##################################################################
  if (args.ra is not None) and (args.dec is not None):
    ra_sci, dec_sci = float(args.ra), float(args.dec)
    print("# RA/DEC given on the command line")
  else:
    ny_range = tuple(args.ny_range) if args.ny_range is not None else None
    pt_table = read_PT(lvm_file, 'none', ny_range=ny_range)
    mask_good = pt_table['mask'] == True
    n_good = int(np.sum(mask_good))
    print(len(pt_table), "spectra in the PT table")
    print(n_good, "good spectra in the PT table")
    if n_good == 0:
      raise SystemExit(
        f"No good fibres in the PT table of '{lvm_file}'. "
        f"Pass --ra/--dec explicitly."
      )
    ra_sci = np.mean(pt_table['ra'][mask_good])
    dec_sci = np.mean(pt_table['dec'][mask_good])

  print(f"Mean RA: {ra_sci:.6f}, Mean DEC: {dec_sci:.6f}")
  print("# Done...")
  print("##############################################")

  ##################################################################
  # Look up the Gaia velocity map
  ##################################################################
  value_col = "avg_vel_w" if args.weighted else "avg_vel"
  sigma_col = "std_vel_intr" if args.weighted else "std_vel"
  vel = load_velocity_map(args.gaia_rv_file, min_stars=int(args.min_stars),
                          value_col=value_col, sigma_col=sigma_col)
  print(f"# Map: {args.gaia_rv_file}  nside={vel.nside}  "
        f"{vel.n_valid}/{vel.npix} valid cells")

  cell = vel(ra_sci, dec_sci, max_sep_deg=args.max_sep, full=True)
  a_vel_pos = (float(cell[value_col]), float(cell[sigma_col]))
  n_stars = int(cell["n_stars"])
  sep = float(cell["sep_deg"])
  print(f"Mean Gaia RV at position (RA: {ra_sci:.6f}, DEC: {dec_sci:.6f}): "
        f"{a_vel_pos} km/s")

  if not np.isfinite(a_vel_pos[0]):
    raise SystemExit(
      f"No velocity available at RA={ra_sci:.6f}, DEC={dec_sci:+.6f} "
      f"(nearest valid cell is {sep:.3f} deg away, beyond --max-sep). "
      f"Relax --max-sep or lower --min-stars."
    )

  # The dispersion is the spread of stellar velocities inside the cell, not
  # the error on the mean: it is the spread that says how far the true
  # line-of-sight velocity may sit from the average.
  v_std = a_vel_pos[1]
  note = ""
  if not np.isfinite(v_std):
    v_std = float(args.sigma_floor)
    note = f" (was NaN, n_stars={n_stars}; floored)"
  elif v_std < args.sigma_floor:
    v_std = float(args.sigma_floor)
    note = " (raised to --sigma-floor)"
  if (args.sigma_max is not None) and (v_std > args.sigma_max):
    v_std = float(args.sigma_max)
    note = " (capped at --sigma-max)"
  if note:
    print(f"# Dispersion {a_vel_pos[1]:.3f} -> using {v_std:.3f} km/s{note}")

  print(f"# Cell hpx={cell['hpx']}  n_stars={n_stars}  sep={sep:.4f} deg")
  print("##############################################")


  ##################################################################
  # Velocity -> redshift
  ##################################################################
  v_tot = a_vel_pos[0] + args.hvcor_sign * HVCOR
  z_guess = v_tot / __c__
  d_z_guess = v_std / __c__
  print(f"Redshift guess: {z_guess:.6f}, Redshift uncertainty: {d_z_guess:.6f}")

  dz = args.nsigma * d_z_guess
  z_min = z_guess - dz
  z_max = z_guess + dz

  z_del = float(args.z_del) if args.z_del is not None else (2 * dz) / args.n_steps
  auto_z_del = float(args.auto_z_del) if args.auto_z_del is not None else z_del

  sgn = "+" if args.hvcor_sign > 0 else "-"
  print("##############################################")
  print("# Redshift prior")
  print(f"#   v_gaia {sgn} HVCOR = {a_vel_pos[0]:+9.3f} {sgn} "
        f"{abs(HVCOR):.3f} = {v_tot:+9.3f} km/s")
  print(f"#   z              = {z_guess:+.7f}")
  print(f"#   sigma          = {v_std:9.3f} km/s  ->  d_z = {d_z_guess:.7f}")
  print(f"#   {args.nsigma:g} sigma range  = [{z_min:+.7f}, {z_max:+.7f}]"
        f"  = [{v_tot - args.nsigma * v_std:+.1f}, "
        f"{v_tot + args.nsigma * v_std:+.1f}] km/s")
  print(f"#   step           = {z_del:.7f} ({z_del * __c__:.2f} km/s), "
        f"auto step = {auto_z_del:.7f}")
  print("##############################################")

  ##################################################################
  # Patch the YAML
  ##################################################################
  # redshift is the 4-element list [input, delta, min, max].
  overrides = {
    "redshift[0]": float(z_guess),
    "redshift[1]": float(z_del),
    "redshift[2]": float(z_min),
    "redshift[3]": float(z_max),
    "auto-redshift": True,
    "auto_z_min": float(z_min),
    "auto_z_max": float(z_max),
    "auto_z_del": float(auto_z_del),
  }

  with open(args.config_yaml, "r") as f:
    before = yaml.safe_load(f)

  print("##############################################")
  print("# YAML entries")
  for k, v_new in overrides.items():
    key = k.split("[")[0]
    old = before.get(key, "<absent>") if isinstance(before, dict) else "<absent>"
    if "[" in k and isinstance(old, list):
      idx = int(k.split("[")[1].rstrip("]"))
      old = old[idx] if idx < len(old) else "<absent>"
    print(f"#   {k:16s} {str(old):>14s}  ->  {v_new}")

  if args.dry_run:
    print("# --dry-run: nothing written")
    print("##############################################")
    return 0

  edit_yaml(args.config_yaml, args.output_yaml, overrides,
            create_missing=args.create_missing)
  print(f"# Written: {args.output_yaml}")
  print("##############################################")

  print("#############################")
  print('# ALL DONE !!! ')
  print("#############################")
  return 0


if __name__ == "__main__":
  sys.exit(_main())
