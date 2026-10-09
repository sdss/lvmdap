#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gaia_velocity_lookup.py
=======================

Self-contained version of GaiaVelocityMap: a single public function, no
classes, no scipy, no healpy and no astropy-healpix.  Only numpy and
astropy are required.

    from gaia_velocity_lookup import load_velocity_map

    vel = load_velocity_map("data/gaia_dr3_rv_map.fits")
    vel(266.4, -29.0)                  # -> (-12.727, 35.798)
    vel([80.9, 13.2], [-69.8, -72.8])  # -> (array([...]), array([...]))
    vel(80.9, -69.8, full=True)        # -> dict with every column

The NESTED HEALPix index is computed here (Gorski et al. 2005), so the
lookup is O(1) per position: no tree and no brute force, except for empty
pixels, which do not occur in the complete DR3 map.

Velocities are heliocentric/barycentric, as published in the Gaia archive.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table

__all__ = ["load_velocity_map"]


def load_velocity_map(path, min_stars=1, value_col="avg_vel",
                      sigma_col="std_vel"):
    """Load the map and return a lookup function ``f(ra, dec)``.

    Equivalent to ``GaiaVelocityMap.from_file(path)``, but it returns a
    closure instead of an instance.

    Parameters
    ----------
    path : str
        A .fits or .ecsv file written by gaia_velocity_map.py.  It must
        contain the ``hpx`` column and the velocity columns; ``ra`` and
        ``dec`` are used only by the empty-pixel fallback.
    min_stars : int
        Cells holding fewer stars than this are treated as empty.
    value_col, sigma_col : str
        Columns returned by the lookup function, in that order.  Switch to
        ``value_col="avg_vel_w"`` or ``sigma_col="std_vel_intr"`` for the
        inverse-variance weighted mean or the error-corrected dispersion.

    Returns
    -------
    callable
        ``f(ra, dec, max_sep_deg=None, full=False)``.  Returns
        ``(avg, std)`` as scalars for scalar input and as arrays for
        vector input.  With ``full=True`` it returns a dict holding every
        column of the map plus ``hpx`` and ``sep_deg``.
    """
    fmt = "ascii.ecsv" if str(path).lower().endswith((".ecsv", ".csv")) else "fits"
    tab = Table.read(path, format=fmt)
    tab.rename_columns(tab.colnames, [c.lower() for c in tab.colnames])

    hpx = np.asarray(tab["hpx"], dtype=np.int64)

    nside = int(tab.meta.get("NSIDE", 0)) or int(
        2 ** np.ceil(np.log2(np.sqrt((hpx.max() + 1) / 12.0)))
    )
    order = int(round(np.log2(nside)))
    if 2 ** order != nside:
        raise ValueError(f"NSIDE={nside} is not a power of two.")
    npix = 12 * nside * nside

    # --- scatter the columns into arrays indexed by pixel -----------------
    counts = np.zeros(npix, dtype=np.int64)
    if "n_stars" in tab.colnames:
        counts[hpx] = np.asarray(tab["n_stars"], dtype=np.int64)
    else:
        counts[hpx] = 1

    data = {}
    for name in tab.colnames:
        if name in ("hpx", "n_stars"):
            continue
        col = np.asarray(tab[name], dtype=float)
        arr = np.full(npix, np.nan, dtype=float)
        arr[hpx] = col
        data[name] = arr

    for need in (value_col, sigma_col):
        if need not in data:
            raise KeyError(f"The map has no column {need!r}; "
                           f"available: {sorted(data)}")

    valid = (counts >= int(min_stars)) & np.isfinite(data[value_col])
    good = np.flatnonzero(valid)
    if good.size == 0:
        raise ValueError("The map contains no valid cell.")

    # Centres of the valid cells, used by the nearest-neighbour fallback.
    if "ra" in data and "dec" in data:
        g_ra, g_dec = data["ra"][good], data["dec"][good]
    else:  # the map carried no ra/dec: rebuild them from the index itself
        g_ra, g_dec = _pix2radec_nest(nside, good)
    g_vec = _unit_vec(g_ra, g_dec)

    # ---------------------------------------------------------------------
    def velocity_at(ra, dec, max_sep_deg=None, full=False):
        ra_a = np.atleast_1d(np.asarray(ra, dtype=float))
        dec_a = np.atleast_1d(np.asarray(dec, dtype=float))
        if ra_a.shape != dec_a.shape:
            raise ValueError("ra and dec must have the same shape.")
        scalar = np.ndim(ra) == 0 and np.ndim(dec) == 0

        pix = _ang2pix_nest(nside, order, ra_a, dec_a)
        sep = np.zeros(ra_a.size, dtype=float)

        miss = ~valid[pix]
        if miss.any():
            # Blocked brute force over the valid cells.  This only runs if
            # the map has holes; the complete DR3 map has none.
            q = _unit_vec(ra_a[miss], dec_a[miss])
            best = np.empty(q.shape[0], dtype=np.int64)
            bsep = np.empty(q.shape[0], dtype=float)
            step = max(1, int(4e7 // max(g_vec.shape[0], 1)))
            for i in range(0, q.shape[0], step):
                dots = q[i:i + step] @ g_vec.T
                j = np.argmax(dots, axis=1)
                best[i:i + step] = good[j]
                bsep[i:i + step] = np.degrees(np.arccos(
                    np.clip(dots[np.arange(j.size), j], -1.0, 1.0)))
            pix[miss] = best
            sep[miss] = bsep

        out_avg = data[value_col][pix].copy()
        out_std = data[sigma_col][pix].copy()
        if max_sep_deg is not None:
            bad = sep > float(max_sep_deg)
            out_avg[bad] = np.nan
            out_std[bad] = np.nan

        if full:
            res = {"hpx": pix, "sep_deg": sep, "n_stars": counts[pix]}
            for name, arr in data.items():
                res[name] = arr[pix]
            if scalar:
                res = {k: (v[0].item() if hasattr(v[0], "item") else v[0])
                       for k, v in res.items()}
            return res

        if scalar:
            return float(out_avg[0]), float(out_std[0])
        return out_avg, out_std

    velocity_at.nside = nside
    velocity_at.npix = npix
    velocity_at.n_valid = int(good.size)
    velocity_at.columns = sorted(data)
    velocity_at.table = tab
    velocity_at.__doc__ = (
        f"Lookup for map {path!r}: nside={nside} "
        f"({np.degrees(np.sqrt(4 * np.pi / npix)):.4f} deg/pixel), "
        f"{good.size}/{npix} valid cells.\n"
        f"f(ra, dec, max_sep_deg=None, full=False) -> "
        f"({value_col}, {sigma_col})"
    )
    return velocity_at


# ---------------------------------------------------------------------------
# NESTED HEALPix in pure numpy (Gorski et al. 2005)
# ---------------------------------------------------------------------------


def _unit_vec(ra_deg, dec_deg):
    """Unit vectors (N, 3) from RA/DEC in degrees."""
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    cd = np.cos(dec)
    return np.column_stack((cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)))


def _spread_bits(v):
    """Interleave zeros between the bits (Morton encoding)."""
    v = v.astype(np.uint64)
    v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
    v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
    return v


def _compress_bits(v):
    """Inverse of _spread_bits: gather the even bits."""
    v = v.astype(np.uint64) & np.uint64(0x5555555555555555)
    v = (v | (v >> np.uint64(1))) & np.uint64(0x3333333333333333)
    v = (v | (v >> np.uint64(2))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    v = (v | (v >> np.uint64(4))) & np.uint64(0x00FF00FF00FF00FF)
    v = (v | (v >> np.uint64(8))) & np.uint64(0x0000FFFF0000FFFF)
    v = (v | (v >> np.uint64(16))) & np.uint64(0x00000000FFFFFFFF)
    return v


def _xyf2nest(order, ix, iy, face):
    """Combine face number and within-face (x, y) into a NESTED index."""
    return (
        (face.astype(np.int64) << np.int64(2 * order))
        + (_spread_bits(ix) | (_spread_bits(iy) << np.uint64(1))).astype(np.int64)
    )


def _ang2pix_nest(nside, order, ra_deg, dec_deg):
    """NESTED HEALPix pixel containing each (RA, DEC) given in degrees.

    Exact-tie warning: on a face corner or edge the containing pixel is
    formally undefined, and this routine may pick a different one from
    healpy or astropy-healpix.  The set of measure-zero positions where
    that happens is irrelevant for real data.
    """
    ra = np.asarray(ra_deg, dtype=float)
    dec = np.asarray(dec_deg, dtype=float)

    z = np.sin(np.radians(dec))
    za = np.abs(z)
    phi = np.mod(np.radians(ra), 2.0 * np.pi)
    tt = np.mod(phi * (2.0 / np.pi), 4.0)          # in [0, 4)

    ix = np.zeros(ra.shape, dtype=np.int64)
    iy = np.zeros(ra.shape, dtype=np.int64)
    face = np.zeros(ra.shape, dtype=np.int64)

    eq = za <= 2.0 / 3.0

    # --- equatorial belt --------------------------------------------------
    if eq.any():
        temp1 = nside * (0.5 + tt[eq])
        temp2 = nside * z[eq] * 0.75
        jp = np.floor(temp1 - temp2).astype(np.int64)   # ascending edge line
        jm = np.floor(temp1 + temp2).astype(np.int64)   # descending edge line
        ifp = jp >> order                               # face in {0, 4}
        ifm = jm >> order
        f = np.where(ifp == ifm, (ifp & 3) + 4,
                     np.where(ifp < ifm, ifp & 3, (ifm & 3) + 8))
        face[eq] = f
        ix[eq] = jm & (nside - 1)
        iy[eq] = nside - (jp & (nside - 1)) - 1

    # --- polar caps -------------------------------------------------------
    po = ~eq
    if po.any():
        ntt = np.minimum(3, tt[po].astype(np.int64))
        tp = tt[po] - ntt
        tmp = nside * np.sqrt(np.maximum(3.0 * (1.0 - za[po]), 0.0))
        jp = np.minimum((tp * tmp).astype(np.int64), nside - 1)
        jm = np.minimum(((1.0 - tp) * tmp).astype(np.int64), nside - 1)
        north = z[po] >= 0
        face[po] = np.where(north, ntt, ntt + 8)
        ix[po] = np.where(north, nside - jm - 1, jp)
        iy[po] = np.where(north, nside - jp - 1, jm)

    return _xyf2nest(order, ix, iy, face)


def _pix2radec_nest(nside, ipix):
    """Centre (RA, DEC) in degrees of NESTED HEALPix pixels."""
    order = int(round(np.log2(nside)))
    ipix = np.asarray(ipix, dtype=np.int64)
    npface = nside * nside
    face = ipix // npface
    p = (ipix % npface).astype(np.uint64)
    ix = _compress_bits(p).astype(np.int64)
    iy = _compress_bits(p >> np.uint64(1)).astype(np.int64)

    # Ring index counted from the north pole.
    jr = (np.array([2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4], dtype=np.int64)[face]
          * nside - ix - iy - 1)

    z = np.empty(ipix.shape, dtype=float)
    nr = np.empty(ipix.shape, dtype=np.int64)      # pixels per quadrant ring
    kshift = np.empty(ipix.shape, dtype=np.int64)  # ring offset, 0 or 1

    north = jr < nside
    south = jr > 3 * nside
    mid = ~(north | south)

    nr[north] = jr[north]
    z[north] = 1.0 - (nr[north] ** 2) / (3.0 * nside * nside)
    kshift[north] = 0

    nr[south] = 4 * nside - jr[south]
    z[south] = (nr[south] ** 2) / (3.0 * nside * nside) - 1.0
    kshift[south] = 0

    nr[mid] = nside
    z[mid] = (2 * nside - jr[mid]) * (2.0 / (3.0 * nside))
    kshift[mid] = (jr[mid] - nside) & 1

    jpll = np.array([1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int64)
    jp = (jpll[face] * nr + ix - iy + 1 + kshift) // 2
    jp = np.where(jp > 4 * nside, jp - 4 * nside, jp)
    jp = np.where(jp < 1, jp + 4 * nside, jp)

    phi = (jp - (kshift + 1) * 0.5) * (np.pi / 2.0 / nr)
    return np.degrees(np.mod(phi, 2 * np.pi)), np.degrees(np.arcsin(z))
