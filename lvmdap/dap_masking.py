"""Conservative, optional LSF-aware supplemental masking for extracted LVM RSS.

This is NOT the CCD-level PyCosmic algorithm.  It detects only isolated
1--2-pixel excursions that are significantly narrower than the measured
instrumental LSF.  In particular it does not mask an unusual line because
of its wavelength, flux sign, line list membership, or spatial isolation.

All decisions use the *original* observed flux, errors and DRP mask.  The
input arrays are not changed; returned True means an additional bad pixel.
Broad row/column artifacts cannot safely be distinguished from genuine
astrophysical structure from a single resampled RSS and are not masked.
"""

import numpy as np
from scipy.ndimage import median_filter


def detect_lvm_artifacts(flux, error, wavelength, lsf, drp_mask=None, *,
                         spike_sigma=8.0, shape_sigma=3.0,
                         support_fraction=0.35, min_lsf_pixels=1.1,
                         median_window=11):
    """Return a boolean extra mask for strictly sub-LSF 1--2-pixel defects.

    Parameters
    ----------
    flux, error : ndarray (n_fiber, n_wave)
        Original extracted flux density and its positive standard error.
    wavelength : ndarray (n_wave,)
        Increasing wavelength in Angstroms.
    lsf : ndarray (n_fiber, n_wave) or (n_wave,)
        Instrumental *Gaussian sigma* in Angstroms, not FWHM.  Pixels with
        missing or under-sampled LSF are intentionally left unclassified.
    drp_mask : ndarray, optional
        Existing DRP bit mask; never altered. Nonzero pixels are excluded
        from detection, including the two-pixel neighborhood.
    spike_sigma : float
        Minimum absolute excursion relative to original error.
    shape_sigma : float
        Minimum significance of the missing LSF-expected wing support.
    support_fraction : float
        Conservative fraction of the minimum LSF-model wing strength that
        must be absent before the pixel(s) can be flagged.
    min_lsf_pixels : float
        Minimum instrumental sigma / wavelength-pixel spacing required
        for shape discrimination (avoid misclassifying unresolved lines).
    median_window : odd int
        Width of per-fiber local baseline estimate, in pixels.

    Notes
    -----
    For a genuine unresolved Gaussian centered on one sample, its immediate
    wing sum is at least 2*A*exp(-1/(2*sigma_pix**2)), where A is its excess
    at the central pixel.  A two-pixel peak centered between samples has
    outside wings >= A*exp(-1/sigma_pix**2) on each side.  The detector asks
    for a large *deficit* of these expected wings, well above the original
    errors. No negative-flux restriction is imposed: negative single-pixel
    detector defects can also be flagged; LSF-shaped absorption is protected.
    """
    f = np.asarray(flux, dtype=float)
    e = np.asarray(error, dtype=float)
    w = np.asarray(wavelength, dtype=float)
    if f.ndim != 2 or f.shape != e.shape or w.shape != (f.shape[1],):
        raise ValueError('flux/error must share RSS shape and wavelength match columns')
    if w.size < 7 or not np.all(np.isfinite(w)) or np.any(np.diff(w) <= 0):
        raise ValueError('wavelength must be strictly increasing and contain >=7 samples')
    if median_window < 7 or median_window % 2 != 1:
        raise ValueError('median_window must be odd and >=7')
    if not all(np.isfinite(x) and x > 0 for x in
               (spike_sigma, shape_sigma, support_fraction, min_lsf_pixels)):
        raise ValueError('detection parameters must be finite and positive')
    if support_fraction >= 1:
        raise ValueError('support_fraction must be < 1')
    if drp_mask is None:
        bad = np.zeros(f.shape, dtype=bool)
    else:
        bad = np.asarray(drp_mask) != 0
        if bad.shape != f.shape:
            raise ValueError('DRP mask must match RSS shape')
    ls = np.asarray(lsf, dtype=float)
    if ls.shape == w.shape:
        ls = np.broadcast_to(ls, f.shape)
    if ls.shape != f.shape:
        raise ValueError('LSF must be (n_wave,) or (n_fiber, n_wave)')

    dw = np.gradient(w)
    extra = np.zeros(f.shape, dtype=bool)
    n_fib, n_wave = f.shape
    # Per-fiber operation: avoids confusing detector-row order with sky position
    # and does not form several temporary full-RSS images in memory.
    for i in range(n_fib):
        valid = (~bad[i] & np.isfinite(f[i]) & np.isfinite(e[i]) &
                 (e[i] > 0))
        if valid.sum() < median_window:
            continue
        # Fill masked samples ONLY in the temporary baseline-estimation array.
        # Their observed values remain untouched, and they are never detected.
        xp = np.flatnonzero(valid)
        baseline_data = np.interp(np.arange(n_wave), xp, f[i, xp])
        baseline = median_filter(baseline_data, size=median_window, mode='nearest')
        residual = f[i] - baseline
        sigma_pix = ls[i] / dw
        lsf_good = np.isfinite(sigma_pix) & (sigma_pix >= min_lsf_pixels)

        # 1 pixel: both nearest neighbors must be valid and consistent with
        # too little LSF-shaped support; a real emission or absorption line
        # with expected LSF wings is not selected.
        centers = np.arange(2, n_wave - 2)
        if len(centers):
            r = residual[centers]
            absolute = np.abs(r)
            sign = np.sign(r)
            neighbor = sign * (residual[centers-1] + residual[centers+1])
            # Minimum wing sum expected if the peak were a sampled LSF line.
            sig = sigma_pix[centers]
            expected = (2.0 * np.exp(-0.5 / np.maximum(sig, 1e-12)**2) *
                        absolute * support_fraction)
            missing = expected - neighbor
            uncertainty = np.hypot(e[i, centers-1], e[i, centers+1])
            ok = (valid[centers-1] & valid[centers] & valid[centers+1] &
                  lsf_good[centers-1] & lsf_good[centers] & lsf_good[centers+1] &
                  (absolute >= spike_sigma * e[i, centers]) &
                  (missing >= shape_sigma * uncertainty) &
                  (missing > 0))
            for c in centers[ok]:
                c = int(c)
                radius = int(np.ceil(3.0 * sigma_pix[c])) + 2
                lo, hi = max(0, c-radius), min(n_wave, c+radius+1)
                near = np.abs(residual[lo:hi]).copy()
                near[c-lo] = 0.0
                if np.any(valid[lo:hi] & (near >= abs(residual[c]))):
                    continue
                extra[i, c] = True

        # 2 adjacent pixels: a cosmic may be strong in both, so neither
        # isolated-pixel test detects it. Compare the *outer* wing pixels
        # with the minimum two-sample LSF profile, requiring large deficits
        # on BOTH sides (to preserve real lines near another feature).
        a = np.arange(2, n_wave - 3)
        if len(a):
            b = a + 1
            r1, r2 = residual[a], residual[b]
            same_sign = (r1 * r2 > 0)
            sign = np.sign(r1)
            peak = np.minimum(np.abs(r1), np.abs(r2))
            sig = np.minimum(sigma_pix[a], sigma_pix[b])
            wing_min = peak * support_fraction * np.exp(-1.0 / np.maximum(sig, 1e-12)**2)
            left = wing_min - sign * residual[a-1]
            right = wing_min - sign * residual[b+1]
            ok = (same_sign & valid[a-1] & valid[a] & valid[b] & valid[b+1] &
                  lsf_good[a-1] & lsf_good[a] & lsf_good[b] & lsf_good[b+1] &
                  (np.abs(r1) >= spike_sigma * e[i, a]) &
                  (np.abs(r2) >= spike_sigma * e[i, b]) &
                  (left >= shape_sigma * e[i, a-1]) &
                  (right >= shape_sigma * e[i, b+1]))
            # Refuse to mask the shoulder of a genuine nearby strong line.
            # If a stronger peak exists within three LSF sigmas of the pair,
            # the pair cannot safely be distinguished from that line.
            for idx in np.flatnonzero(ok):
                a0, b0 = int(a[idx]), int(b[idx])
                radius = int(np.ceil(3.0 * sig[idx])) + 2
                lo, hi = max(0, a0-radius), min(n_wave, b0+radius+1)
                near = np.abs(residual[lo:hi]).copy()
                near[a0-lo:b0-lo+1] = 0.0
                near_valid = valid[lo:hi]
                if np.any(near_valid & (near >= peak[idx])):
                    continue
                extra[i, a0:b0+1] = True

    # Never flag pixels already masked by the DRP and never alter input arrays.
    return extra & ~bad
