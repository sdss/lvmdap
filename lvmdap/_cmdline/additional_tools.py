#
#
#

#!/usr/bin/env python3
# path: yaml_edit.py
"""
Edit a YAML config by overriding existing keys from the command line.

Usage:
  yaml_edit.py input.yaml output.yaml --auto-redshift=False --redshift=(0.01,0.002,0.005,0.02)

Rules:
- Overrides are passed as: --key=value OR --key value
- Keys must already exist in the YAML (unless --create-missing is used).
- Values are parsed as YAML scalars/lists/dicts when possible.
  Additionally supports Python literals like tuples: (1,2,3) -> [1,2,3].

Nested keys:
- Dot paths for nested dicts: --section.subkey=123
- List indexing with brackets: --items[0].name="x"
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from typing import Any, List, Tuple

import yaml


def _yaml_edit(cmd_args: List[str] = sys.argv[1:]) -> int:
    _PATH_TOKEN_RE = re.compile(
        r"""
        (?P<name>[^.\[\]]+)     # dict key
        (?P<idx>(\[[0-9]+\])*)  # optional [0][1]...
        """,
        re.VERBOSE,
    )

    @dataclass(frozen=True)
    class _SetOptions:
        create_missing: bool = False

    def _parse_value(raw: str) -> Any:
        """Parse CLI value into Python types (YAML first, then Python literal for tuples)."""
        try:
            val = yaml.safe_load(raw)
        except Exception:
            val = raw

        if isinstance(val, str):
            s = val.strip()
            if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")) or (
                s.startswith("{") and s.endswith("}")
            ):
                try:
                    val2 = ast.literal_eval(s)
                    val = val2
                except Exception:
                    pass

        if isinstance(val, tuple):
            val = list(val)

        return val

    def _parse_override_tokens(argv: List[str]) -> List[Tuple[str, str]]:
        """Parse unknown argv tokens into (key, raw_value). Supports --k=v and --k v."""
        out: List[Tuple[str, str]] = []
        i = 0
        while i < len(argv):
            tok = argv[i]
            if not tok.startswith("--"):
                raise SystemExit(f"Unexpected token '{tok}'. Overrides must start with '--'.")
            tok = tok[2:]
            if "=" in tok:
                k, v = tok.split("=", 1)
                out.append((k, v))
                i += 1
                continue
            if i + 1 >= len(argv):
                raise SystemExit(f"Missing value for '--{tok}'. Use --{tok}=VALUE or --{tok} VALUE.")
            out.append((tok, argv[i + 1]))
            i += 2
        return out

    def _split_path(key_path: str) -> List[Any]:
        """Split dotted path with optional list indices into tokens: a.b[0].c -> ['a','b',0,'c']."""
        parts: List[Any] = []
        for seg in key_path.split("."):
            m = _PATH_TOKEN_RE.fullmatch(seg)
            if not m:
                raise ValueError(f"Invalid key path segment: '{seg}' in '{key_path}'")
            parts.append(m.group("name"))
            idx_part = m.group("idx") or ""
            if idx_part:
                for idx_str in re.findall(r"\[([0-9]+)\]", idx_part):
                    parts.append(int(idx_str))
        return parts

    def _set_by_path(obj: Any, key_path: str, value: Any, opts: _SetOptions) -> None:
        """Set obj[key_path] = value, where key_path may contain dots and [idx]."""
        tokens = _split_path(key_path)
        cur = obj
        for t in tokens[:-1]:
            if isinstance(t, int):
                if not isinstance(cur, list):
                    raise KeyError(f"Path '{key_path}': expected list, got {type(cur).__name__}")
                if t < 0 or t >= len(cur):
                    raise IndexError(f"Path '{key_path}': list index {t} out of range (len={len(cur)})")
                cur = cur[t]
            else:
                if not isinstance(cur, dict):
                    raise KeyError(f"Path '{key_path}': expected dict, got {type(cur).__name__}")
                if t not in cur:
                    if not opts.create_missing:
                        raise KeyError(f"Key '{t}' not found while setting '{key_path}'.")
                    cur[t] = {}
                cur = cur[t]

        last = tokens[-1]
        if isinstance(last, int):
            if not isinstance(cur, list):
                raise KeyError(f"Path '{key_path}': expected list at final parent, got {type(cur).__name__}")
            if last < 0 or last >= len(cur):
                raise IndexError(f"Path '{key_path}': list index {last} out of range (len={len(cur)})")
            cur[last] = value
            return

        if not isinstance(cur, dict):
            raise KeyError(f"Path '{key_path}': expected dict at final parent, got {type(cur).__name__}")
        if (last not in cur) and (not opts.create_missing):
            raise KeyError(f"Key '{last}' not found while setting '{key_path}'.")
        cur[last] = value

    def _edit_yaml_file(
        input_path: str,
        output_path: str,
        overrides: List[Tuple[str, str]],
        *,
        create_missing: bool = False,
    ) -> None:
        with open(input_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise SystemExit("Top-level YAML must be a mapping (dict).")

        opts = _SetOptions(create_missing=create_missing)
        for k, raw_v in overrides:
            _set_by_path(data, k, _parse_value(raw_v), opts)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                data,
                f,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=True,
            )

    parser = argparse.ArgumentParser(
        description="Edit YAML by overriding existing keys from the command line.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_yaml", help="Input YAML file")
    parser.add_argument("output_yaml", help="Output YAML file")
    parser.add_argument(
        "--create-missing",
        action="store_true",
        help="Allow creating missing keys along a path (default: error if key missing).",
    )
    args, unknown = parser.parse_known_args(cmd_args)

    overrides = _parse_override_tokens(unknown) if unknown else []
    _edit_yaml_file(
        args.input_yaml,
        args.output_yaml,
        overrides,
        create_missing=args.create_missing,
    )
    return 0


"""
gaia_find_stars_cli.py

Command-line version of the gaia_find_stars notebook:
- reads an LVM-DAP ".output.fits(.gz)" model cube (inferred from dap_file, unless present)
- reads the corresponding ".dap.fits(.gz)" table
- queries Gaia for stars in the IFU footprint (with optional caching)
- matches stars to fibers
- builds a background (ISM) model from low-S/N spectra and subtracts it
- for matched fibers, subtracts a median of nearest-neighbor spectra to remove stellar contamination
- writes a "clean" FITS file with:
    * PrimaryHDU: cleaned 2D spectra (ny, nx)
    * ERR      : running-std estimate (ny, nx)
    * WAVE     : wavelength vector (nx,)
    * PT       : fiber table including star matching columns

No plotting is performed.

Example
-------
python gaia_find_stars_cli.py \
  output_dap_v1.2.0/dap-rsp108-sn20-00012180.clean.fits.gz \
  output_dap_v1.2.0/dap-rsp108-sn20-00012180.dap.fits.gz \
  15.0 5000 --use-cache --DIR_DAP output_dap_v1.2.0
"""

#import sys


def _gaia_find_stars_cli(cmd_args=sys.argv[1:]) -> int:
    """
    gaia_find_stars_cli.py
    
    Command-line version of the gaia_find_stars notebook:
    - reads an LVM-DAP ".output.fits(.gz)" model cube (inferred from dap_file, unless present)
    - reads the corresponding ".dap.fits(.gz)" table
    - queries Gaia for stars in the IFU footprint (with optional caching)
    - matches stars to fibers
    - builds a background (ISM) model from low-S/N spectra and subtracts it
    - for matched fibers, subtracts a median of nearest-neighbor spectra to remove stellar contamination
    - writes a "clean" FITS file with:
        * PrimaryHDU: cleaned 2D spectra (ny, nx)
        * ERR      : running-std estimate (ny, nx)
        * WAVE     : wavelength vector (nx,)
        * PT       : fiber table including star matching columns
    
    No plotting is performed.
    
    Example
    -------
    python gaia_find_stars_cli.py \
      output_dap_v1.2.0/dap-rsp108-sn20-00012180.clean.fits.gz \
      output_dap_v1.2.0/dap-rsp108-sn20-00012180.dap.fits.gz \
      15.0 5000 --use-cache --DIR_DAP output_dap_v1.2.0
    """
    
    import os
    
    # Match the style of gen_output_model.py: force single-threaded BLAS/OpenMP to avoid oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    
    import argparse
    import pathlib
    import re
    from typing import Optional, Dict, Tuple
    
    import numpy as np
    from astropy.io import fits
    from astropy.table import Table, Row
    
    from astroquery.gaia import Gaia
    
    from scipy.ndimage import median_filter
    
    from lvmdap.dap_tools import read_DAP_file
    
    
    def _no_traceback(exc_type, value, traceback):
        print(value)
    
    
    def _require_astroquery() -> None:
        try:
            import astroquery  # noqa: F401
        except Exception as e:
            raise SystemExit(
                "Missing dependency: astroquery\n"
                "Install with: pip install astroquery\n"
                f"Original error: {e}"
            )
    
    
    def extract_expnum_and_prefix(dap_file: str) -> Tuple[str, str]:
        """
        From a dap_file like:
          dap-rsp108-sn20-00012180.dap.fits.gz
        return:
          expnum='00012180'
          prefix='dap-rsp108-sn20-'
        """
        base = os.path.basename(dap_file)
    
        # remove common suffixes
        base = re.sub(r"\.fits(\.gz)?$", "", base)
        base = re.sub(r"\.dap$", "", base)
    
        # find the trailing number (expnum)
        m = re.search(r"(\d+)$", base)
        if m:
            expnum = m.group(1)
            prefix = base[: -len(expnum)]
        else:
            expnum = base
            prefix = base + "-"
        return expnum, prefix
    
    
    def infer_outmod_from_dap(dap_file: str) -> str:
        """
        Infer the DAP out-model cube from dap_file by swapping the suffix:
          *.dap.fits(.gz) -> *.output.fits(.gz)
        """
        if dap_file.endswith(".dap.fits.gz"):
            return dap_file.replace(".dap.fits.gz", ".output.fits.gz")
        if dap_file.endswith(".dap.fits"):
            return dap_file.replace(".dap.fits", ".output.fits")
        # fallback: best-effort
        return re.sub(r"\.dap(\.fits(\.gz)?)?$", ".output.fits.gz", dap_file)
    
    
    def retrieve_gaia_star_positions_in_field(
        expnum: str | int,
        ra_tile: float,
        dec_tile: float,
        *,
        lim_mag: float = 14.0,
        n_spec: int = 200,
        GAIA_CACHE_DIR: str = "./gaia_cache",
        require_xp_continuous: bool = False,
        use_cache: bool = True,
    ) -> Table:
        """
        Retrieve Gaia stars (positions + magnitudes) in the LVM IFU field.
    
        Caches the Gaia query result into:
          <GAIA_CACHE_DIR>/<expnum>_ids.ecsv
        """
        pathlib.Path(GAIA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        cache_path = os.path.join(GAIA_CACHE_DIR, f"{expnum}_ids.ecsv")
    
        if use_cache and os.path.exists(cache_path):
            r = Table.read(cache_path)
        else:
            # Inner radius of hexagon in degrees for margin (same geometry as notebook)
            r_ifu = (3.0**0.5) / 2.0 * (30.2 / 2.0) / 60.0
    
            select_tile = f"DISTANCE({float(ra_tile)}, {float(dec_tile)}, ra, dec) < {r_ifu} "
            xp_clause = " AND has_xp_continuous = 'True' " if require_xp_continuous else ""
    
            adql = (
                f"SELECT TOP {int(n_spec)} "
                "source_id, ra, dec, phot_g_mean_mag"
                + (", has_xp_continuous" if require_xp_continuous else "")
                + " FROM gaiadr3.gaia_source_lite WHERE "
                + select_tile
                + f"AND phot_g_mean_mag < {float(lim_mag)}"
                + xp_clause
                + " ORDER BY phot_g_mean_mag ASC"
            )
    
            job = Gaia.launch_job(adql)
            r = job.get_results()
            r.write(cache_path, overwrite=True)
    
        # match notebook behavior: lowercase columns
        r.rename_columns(r.colnames, [c.lower() for c in r.colnames])
        return r
    
    
    def assign_stars_to_fibers(tab_PT: Table, stars: Table, r_max_arcsec: float = 20.0) -> Table:
        """
        For each fiber, assign the closest Gaia star within r_max_arcsec.
        Adds/overwrites columns:
          - star_id (object)
          - star_sep_arcsec (float)
        """
        if "star_id" not in tab_PT.colnames:
            tab_PT["star_id"] = np.full(len(tab_PT), 0, dtype=int)
        if "star_sep_arcsec" not in tab_PT.colnames:
            tab_PT["star_sep_arcsec"] = np.full(len(tab_PT), np.nan, dtype=float)
    
        ra_fib = np.asarray(tab_PT["ra"], dtype=float)
        dec_fib = np.asarray(tab_PT["dec"], dtype=float)
        ra_star = np.asarray(stars["ra"], dtype=float) if len(stars) else np.array([], dtype=float)
        dec_star = np.asarray(stars["dec"], dtype=float) if len(stars) else np.array([], dtype=float)
        sid = np.asarray(stars["source_id"], dtype=object) if len(stars) else np.array([], dtype=object)
    
        if len(stars) == 0:
            return tab_PT
    
        for i_fib in range(len(tab_PT)):
            dra = ra_star - ra_fib[i_fib]
            dra = (dra + 180.0) % 360.0 - 180.0  # handle RA wrap
            dx = dra * np.cos(np.deg2rad(dec_fib[i_fib]))
            dy = dec_star - dec_fib[i_fib]
            sep_arcsec = 3600.0 * np.hypot(dx, dy)
    
            j = int(np.nanargmin(sep_arcsec))
            if np.isfinite(sep_arcsec[j]) and sep_arcsec[j] <= float(r_max_arcsec):
                tab_PT["star_id"][i_fib] = sid[j]
                tab_PT["star_sep_arcsec"][i_fib] = float(sep_arcsec[j])
    
        return tab_PT
    
    
    def create_avg_spec(
        spec2D_orig: np.ndarray,
        tab_PT: Table,
        N_nearest: int,
        mask_st: np.ndarray,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        """
        For each star-matched fiber, find N_nearest nearest *non-star* fibers on-sky,
        and compute the median spectrum of those neighbors.
    
        Returns
        -------
        neighbor_masks : dict
            i_fib -> boolean mask over fibers used as neighbors
        median_specs : dict
            i_fib -> median spectrum (nx,)
        """
        ra = np.asarray(tab_PT["ra"], dtype=float)
        dec = np.asarray(tab_PT["dec"], dtype=float)
    
        neighbor_masks: Dict[int, np.ndarray] = {}
        median_specs: Dict[int, np.ndarray] = {}
    
        idx_star = np.where(mask_st)[0]
        idx_bg_pool = np.where(~mask_st)[0]
    
        if len(idx_star) == 0 or len(idx_bg_pool) == 0:
            return neighbor_masks, median_specs
    
        for i_fib in idx_star:
            dra = ra[idx_bg_pool] - ra[i_fib]
            dra = (dra + 180.0) % 360.0 - 180.0
            dx = dra * np.cos(np.deg2rad(dec[i_fib]))
            dy = dec[idx_bg_pool] - dec[i_fib]
            dist = np.hypot(dx, dy)
    
            order = np.argsort(dist)
            take = idx_bg_pool[order[: int(N_nearest)]]
            m = np.zeros(len(tab_PT), dtype=bool)
            m[take] = True
    
            neighbor_masks[int(i_fib)] = m
            median_specs[int(i_fib)] = np.nanmedian(spec2D_orig[take, :], axis=0)
    
        return neighbor_masks, median_specs
    
    
    def running_std_x(spec2d: np.ndarray, window: int = 31) -> np.ndarray:
        """
        Running standard deviation along spectral axis (x-axis), NaN-safe.
        """
        w = int(window)
        if w < 1:
            raise ValueError("window must be >= 1")
        if w % 2 == 0:
            w += 1
    
        arr = np.asarray(spec2d, dtype=np.float32)
    
        out = np.empty_like(arr)
        pad = w // 2
        for i in range(arr.shape[0]):
            row = np.pad(arr[i], (pad, pad), mode="edge")
            sw = np.lib.stride_tricks.sliding_window_view(row, window_shape=w)
            out[i] = np.nanstd(sw, axis=-1)
        return out
    
    
    def build_gas_background_interp(outmod_cube: np.ndarray, mask_bg: np.ndarray) -> np.ndarray:
        """
        Build an interpolated gas/background model across fibers by taking
        max(plane6, plane7) and interpolating along the fiber index for each wavelength.
        """
        # Element-wise max between the two gas model components
        spec2D_gas = np.maximum(outmod_cube[6, :, :], outmod_cube[7, :, :])
    
        spec2D_gas_interp = spec2D_gas.copy()
        x = np.arange(spec2D_gas.shape[0], dtype=float)
    
        for iw in range(spec2D_gas.shape[1]):
            y = spec2D_gas[:, iw]
            good = mask_bg & np.isfinite(y)
    
            if np.count_nonzero(good) >= 2:
                spec2D_gas_interp[~good, iw] = np.interp(x[~good], x[good], y[good])
            elif np.count_nonzero(good) == 1:
                spec2D_gas_interp[~good, iw] = y[good][0]
    
        return spec2D_gas_interp
    
    
    parser = argparse.ArgumentParser(
        description="Clean LVM-DAP spectra using Gaia star matching (no plots)."
    )
    
    parser.add_argument(
        "outputfile",
        help="Output clean FITS filename to write (e.g., *.clean.fits.gz).",
    )
    parser.add_argument(
        "dap_file",
        help="Input DAP table file (e.g., *.dap.fits.gz).",
    )
    parser.add_argument(
        "m_lim", type=float,
        help="Gaia G-band magnitude limit (phot_g_mean_mag < m_lim).",
    )
    parser.add_argument(
        "n_spec", type=int,
        help="Maximum number of Gaia stars to query (brightest first).",
    )
    
    parser.add_argument(
        "--use-cache",
        dest="use_cache",
        action="store_true",
        help="If set, reuse Gaia cached query (<gaia_cache>/<expnum>_ids.ecsv) when present.",
        default=False,
    )
    parser.add_argument(
        "--DIR_DAP",
        dest="DIR_DAP",
        default=None,
        help="Optional base directory. If provided, relative paths for outputfile/dap_file are resolved inside it.",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Debug mode (full tracebacks).",
        default=False,
    )
    
    args = parser.parse_args(cmd_args)
    if not args.debug:
        sys.excepthook = _no_traceback
    
    _require_astroquery()
    
    # Resolve paths with optional DIR_DAP
    def _resolve(p: str) -> str:
        if args.DIR_DAP and not os.path.isabs(p):
            return os.path.join(args.DIR_DAP, p)
        return p
    
    outputfile = _resolve(args.outputfile)
    dap_file = _resolve(args.dap_file)
    outmod_file = infer_outmod_from_dap(dap_file)
    
    expnum, dap_prefix = extract_expnum_and_prefix(dap_file)
    dap_dir = os.path.dirname(os.path.abspath(dap_file)) or os.path.abspath(".")
    gaia_cache_dir = os.path.join(args.DIR_DAP if args.DIR_DAP else dap_dir, "gaia_cache")
    
    if args.use_cache:
        pathlib.Path(gaia_cache_dir).mkdir(parents=True, exist_ok=True)
    
    print("##############################################")
    print("# Inputs")
    print(f"  dap_file     = {dap_file}")
    print(f"  outmod_file  = {outmod_file}")
    print(f"  outputfile   = {outputfile}")
    print(f"  dap_prefix   = {dap_prefix}")
    print(f"  expnum       = {expnum}")
    print(f"  gaia_cache   = {gaia_cache_dir} (use_cache={args.use_cache})")
    print(f"  m_lim        = {args.m_lim}")
    print(f"  n_spec       = {args.n_spec}")
    print("##############################################")
    
    print("Reading DAP out-model cube...")
    hdu_outmod = fits.open(outmod_file)
    cube = np.asarray(hdu_outmod[0].data)
    hdr = hdu_outmod[0].header
    
    print("Reading DAP table...")
    tab_DAP = read_DAP_file(dap_file)
    tab_DAP.sort("fiberid")
    tab_PT = tab_DAP["id", "ra", "dec", "mask", "fiberid", "exposure"]
    
    ra_c = float(np.nanmean(tab_PT["ra"]))
    dec_c = float(np.nanmean(tab_PT["dec"]))
    print(f"Field center (RA, Dec) = {ra_c:.6f}, {dec_c:.6f}")
    
    # Wavelength axis from header
    ny = cube.shape[1]
    nx = cube.shape[2]
    wave = hdr["CRVAL1"] + np.arange(nx) * hdr["CDELT1"]
    
    # Select background fibers (low S/N continuum + outer region)
    w1, w2 = 6300.0, 6800.0
    i1 = int(np.argmin(np.abs(wave - w1)))
    i2 = int(np.argmin(np.abs(wave - w2)))
    
    flux_c = np.nanmean(cube[1, :, i1:i2], axis=-1)
    noise = np.nanstd(np.abs(cube[4, :, i1:i2]), axis=-1)
    SN_c = flux_c / noise
    
    mask_SN_c = SN_c > 1.5
    
    map_dist = np.sqrt((np.asarray(tab_PT["ra"]) - ra_c) ** 2 + (np.asarray(tab_PT["dec"]) - dec_c) ** 2)
    mask_dist = map_dist > 0.2
    
    mask_bg = (~mask_SN_c) & mask_dist
    if np.count_nonzero(mask_bg) < 10:
        mask_bg = mask_dist
    
    # Build and subtract interpolated ISM background
    spec2D_gas_interp = build_gas_background_interp(cube, mask_bg=mask_bg)
    spec2D_bg_clean = cube[0, :, :] - spec2D_gas_interp
    
    # Query Gaia stars and match to fibers
    print("Querying Gaia...")
    stars = retrieve_gaia_star_positions_in_field(
        expnum=int(expnum) if str(expnum).isdigit() else str(expnum),
        ra_tile=ra_c,
        dec_tile=dec_c,
        lim_mag=args.m_lim,
        n_spec=args.n_spec,
        GAIA_CACHE_DIR=gaia_cache_dir,
        require_xp_continuous=False,
        use_cache=args.use_cache,
    )
    print(f"  Gaia stars returned: {len(stars)}")
    
    tab_PT = assign_stars_to_fibers(tab_PT, stars, r_max_arcsec=20.0)
    mask_st = np.isfinite(tab_PT["star_sep_arcsec"])
    
    # Build median neighbor spectra for star fibers (using model plane 1, as in notebook)
    print("Computing neighbor medians for star-matched fibers...")
    print(f'  Star-matched fibers: {np.count_nonzero(mask_st)}')
    neighbor_masks, median_specs = create_avg_spec(cube[1, :, :], tab_PT, N_nearest=10, mask_st=mask_st)
    
    print("Subtracting background and neighbor medians...")
    # Subtract neighbor median from background-clean spectra for star fibers
    spec2D_full_clean = spec2D_bg_clean.copy()
    for i_fib in np.where(mask_st)[0]:
        spec2D_full_clean[i_fib, :] = spec2D_bg_clean[i_fib, :] - median_specs.get(int(i_fib), np.full_like(wave, np.nan))
    
    print("Writing output FITS...")
    # Error estimate (running std) from abs(residual plane 4), matching notebook
    spec2D_runstd = running_std_x(np.abs(cube[4, :, :]), window=51)
    print(f"  Running std shape: {spec2D_runstd.shape}")
    # Prepare PT columns for writing (replace None with 0 for star_id)
    if tab_PT["star_id"].dtype.kind == "O":
        tab_PT["star_id"][tab_PT["star_id"] == None] = 0  # noqa: E711
    #print(f"  PT table columns: {tab_PT.colnames}")
    # Write output
    pathlib.Path(os.path.dirname(os.path.abspath(outputfile)) or ".").mkdir(parents=True, exist_ok=True)


    # Make sure PT is a Table (not a Row / scalar record)
    #if tab_PT is None:
    #    tab_PT = Table()
    #elif isinstance(tab_PT, Row):
    #    tab_PT = Table(tab_PT)  # wrap single row into a 1-row table
    #elif not isinstance(tab_PT, Table):
    #    tab_PT = Table(tab_PT)

    # Convert to FITS-friendly structured array
    #pt_data = tab_PT.as_array()


    tab_PT['star_id'][tab_PT['star_id'] == None] = 0

    hdu_sel = fits.PrimaryHDU(data=spec2D_full_clean.astype(np.float32), header=hdr)
    hdu_list_sel = []
    hdu_list_sel.append(hdu_sel)
    hdu_sel = fits.ImageHDU(data=spec2D_runstd.astype(np.float32), header=hdr, name='ERR')
    hdu_list_sel.append(hdu_sel)
    hdu_sel = fits.ImageHDU(data=wave.astype(np.float32), name='WAVE')
    hdu_list_sel.append(hdu_sel)
    hdu_sel = fits.BinTableHDU(data=tab_PT,name='PT')
    hdu_list_sel.append(hdu_sel)
    hdu_list = fits.HDUList(hdu_list_sel)
    hdu_list.writeto(outputfile, overwrite=True)
    
    print("#############################")
    print("# ALL DONE !!!")
    print("#############################")
    return 
    
    


