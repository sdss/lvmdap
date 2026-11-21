"""Opt-in peak-based spatial gas seeds for the per-fiber fitting pass."""

import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from lvmdap.dap_tools import find_redshift
from lvmdap.pyFIT3D.common.tools import peak_finder

C_KMS = 299792.458


def measure_peak_seed(wave, gas, error, good, rest, zmin, zmax, halpha, peak_threshold):
    """Return redshift and wavelength, flux density, and S/N of its strongest support."""
    absent = (np.nan,) * 4
    if np.count_nonzero(good) < 5:
        return absent
    filled = np.interp(wave, wave[good], gas[good])
    pixels, _, centers, _ = peak_finder(wave, filled, 2, peak_threshold=peak_threshold, dmin=2)
    pixels = np.asarray(pixels, dtype=int)
    centers = np.asarray(centers, dtype=float)
    if not pixels.size:
        return absent
    # Interpolation is for detection. Use the measured pixel wavelength if
    # the subpixel centroid's immediate neighborhood contains missing data.
    interior = (pixels > 0) & (pixels < len(wave) - 1)
    neighbors = np.zeros(len(pixels), dtype=bool)
    neighbors[interior] = good[pixels[interior]-1] & good[pixels[interior]+1]
    centers = np.where(neighbors, centers, wave[pixels])
    keep = good[pixels] & np.isfinite(centers) & (gas[pixels] > 0)
    keep &= np.any((centers[:, None] >= rest * (1 + zmin)) &
                   (centers[:, None] <= rest * (1 + zmax)), axis=1)
    pixels, centers = pixels[keep], centers[keep]
    if not pixels.size:
        return absent
    if len(pixels) == 1:
        if halpha is None or not zmin <= centers[0] / halpha - 1 <= zmax:
            return absent
        rest = np.array([halpha])
    z = find_redshift(w_peak=centers, f_peak=gas[pixels], w_ref=rest,
                      z_min=zmin, z_max=zmax, d_z=5 / C_KMS)
    matched = np.min(np.abs(centers[:, None] - rest * (1 + z)), axis=1) <= .5
    if not np.any(matched):
        return absent
    best = np.flatnonzero(matched)[np.argmax(gas[pixels[matched]])]
    p = pixels[best]
    return z, centers[best], gas[p], gas[p] / error[p]


def filter_spatial_seeds(raw, peak_sn, coords, *, sn_cut, difference_kms):
    """Replace low-S/N disagreements, using only high-S/N interpolation anchors."""
    valid = np.all(np.isfinite(coords), axis=1)
    measured = valid & np.isfinite(raw)
    anchors = measured & np.isfinite(peak_sn) & (peak_sn >= sn_cut)
    seed = np.where(measured, raw, np.nan)
    source = np.where(measured, 'measured', 'unsupported').astype('U12')
    rejected = np.zeros(len(raw), dtype=bool)
    supported = np.zeros(len(raw), dtype=bool)
    local = np.full(len(raw), np.nan)
    if np.count_nonzero(anchors) < 3:
        return seed, source, anchors, rejected, supported, local

    # Scale RA by cos(dec) on this small field, and wrap around the RA seam.
    x, y = coords[valid].T
    dx = (x - x[0] + 180) % 360 - 180
    xy = np.column_stack((dx * np.cos(np.deg2rad(np.median(y))), y))
    rows = np.flatnonzero(valid)
    subset = anchors[valid]
    anchor_xy, anchor_z = xy[subset], raw[anchors]
    tree = cKDTree(anchor_xy)
    dist, indices = tree.query(xy, k=min(6, len(anchor_xy)))
    radius = 2 * cKDTree(xy).query(xy, k=min(7, len(xy)))[0][:, -1]
    close = dist <= radius[:, None]
    supported[rows] = np.count_nonzero(close, axis=1) >= 3
    local[rows] = np.ma.median(np.ma.array(anchor_z[indices], mask=~close), axis=1).filled(np.nan)
    rejected = measured & ~anchors & supported & (np.abs(raw - local) * C_KMS > difference_kms)
    fill = rows[(~measured | rejected)[rows] & supported[rows]]
    if fill.size:
        target = xy[np.searchsorted(rows, fill)]
        values = anchor_z[tree.query(target)[1]]
        if np.linalg.matrix_rank(anchor_xy - anchor_xy.mean(axis=0)) == 2:
            from scipy.spatial import QhullError
            try:
                linear = griddata(anchor_xy, anchor_z, target, method='linear')
                values = np.where(np.isfinite(linear), linear, values)
            except QhullError:
                pass  # Nearest anchors remain available for degenerate geometry.
        seed[fill], source[fill] = values, 'interpolated'
    return seed, source, anchors, rejected, supported, local


def filter_joint_spatial_seeds(raw, peak_flux, peak_sn, coords, *, sn_cut, neighbors, n_sigma):
    """Reject joint velocity/peak-flux outliers and interpolate all retained seeds."""
    valid = np.all(np.isfinite(coords), axis=1)
    measured = valid & np.isfinite(raw) & np.isfinite(peak_flux)
    reference = measured & np.isfinite(peak_sn) & (peak_sn >= sn_cut)
    seed = np.where(measured, raw, np.nan)
    source = np.where(measured, 'measured', 'unsupported').astype('U12')
    rejected = np.zeros(len(raw), bool)
    supported = np.zeros(len(raw), bool)
    local_z, local_flux, std_z, std_flux = np.full((4, len(raw)), np.nan)
    rows = np.flatnonzero(valid)
    ref = np.flatnonzero(reference)
    if ref.size >= neighbors:
        from astropy.coordinates import SkyCoord
        sky = SkyCoord(coords[rows, 0], coords[rows, 1], unit='deg')
        xyz = sky.cartesian.xyz.value.T
        take = min(neighbors + 1, len(ref))
        indices = ref[cKDTree(xyz[reference[rows]]).query(xyz, k=take)[1]]
        for i, candidates in zip(rows, indices):
            peers = candidates[candidates != i][:neighbors]
            if len(peers) < neighbors:
                continue
            local_z[i], local_flux[i] = np.median(raw[peers]), np.median(peak_flux[peers])
            std_z[i], std_flux[i] = np.std(raw[peers], ddof=1), np.std(peak_flux[peers], ddof=1)
            supported[i] = True
        rejected = (measured & supported &
                    (np.abs(raw - local_z) > n_sigma * std_z) &
                    (np.abs(peak_flux - local_flux) > n_sigma * std_flux))
    anchors = measured & ~rejected
    missing = (~measured | rejected) & valid
    seed[rejected] = np.nan
    source[rejected] = 'unsupported'
    if np.any(anchors) and np.any(missing):
        x, y = coords[rows].T
        dx = (x-x[0]+180) % 360 - 180
        xy = np.c_[dx*np.cos(np.deg2rad(np.median(y))), y]
        anchor_xy, target = xy[anchors[rows]], xy[missing[rows]]
        values = raw[anchors][cKDTree(anchor_xy).query(target)[1]]
        if np.linalg.matrix_rank(anchor_xy-anchor_xy.mean(axis=0)) == 2:
            from scipy.spatial import QhullError
            try:
                linear = griddata(anchor_xy, raw[anchors], target, method='linear')
                values = np.where(np.isfinite(linear), linear, values)
            except QhullError:
                pass  # Use nearest for degenerate geometry.
        seed[missing], source[missing] = values, 'interpolated'
    return dict(z_gas_interpolated=seed, source=source, spatial_anchor=anchors,
                spatial_rejected=rejected, spatial_supported=supported,
                neighbor_z=local_z, neighbor_peak_flux=local_flux,
                neighbor_velocity_std=std_z*C_KMS, neighbor_peak_flux_std=std_flux)
