"""3DCubeGen cube input and spatially registered LVM-DAP parameter products.

This module handles only input/output representation. It does not change any
LVM-DAP fitting, quality calculation, masking thresholds or fitted precision.
"""
from __future__ import annotations

import gzip
import os
import shutil
import tempfile

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u


def _cube_flux_factor(bunit, user_factor, native_flux_scale):
    """Multiplier taking a 3DCubeGen flux value into DAP's scaled fit units.

    An explicit `cube_input_scale` always wins.  The existing DAP subsequently
    divides its MODEL products by `flux_scaling`, irrespective of cube mode.
    """
    if user_factor is not None:
        factor = float(user_factor)
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError('cube_input_scale must be a finite positive multiplier')
        return factor
    unit = str(bunit).replace(' ', '').lower()
    if any(label in unit for label in ('10^-16', '10**-16', '1e-16', '10-16', '10^(-16)')):
        return float(native_flux_scale) * 1e-16
    if ('erg' in unit and ('cm' in unit or 'centimeter' in unit)):
        return float(native_flux_scale)
    raise ValueError(
        f'Unrecognized cube BUNIT={bunit!r}. Specify cube_input_scale '
        '(1.0 if values are already in 10^-16 flux units, or '
        'flux_scaling if values are in physical erg units).'
    )


def _spatial_range(value, length, label):
    if value is None:
        return 0, length
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f'{label} must be [first, stop], zero-based and stop-exclusive')
    a, b = value
    if not isinstance(a, (int, np.integer)) or not isinstance(b, (int, np.integer)):
        raise ValueError(f'{label} entries must be integer pixel coordinates')
    if not 0 <= a < b <= length:
        raise ValueError(f'{label}={value!r} outside valid [0,{length}) interval')
    return a, b



def _spatial_cube_invalid_mask(flux, error, x0, x1, y0, y1, *,
                               radius=1, wavelength_block=64):
    """Dilate invalid samples in x/y independently at each wavelength.

    The returned boolean mask has shape (n_wave, y1-y0, x1-x0). A square
    (2*radius+1)^2 footprint masks the immediately adjacent spatial pixels.
    A halo from the original full cube is included around the ROI so that
    spatially tiled runs agree with an uncropped cube. The footprint's
    spectral dimension is exactly one: other wavelengths are never masked.
    Small wavelength blocks bound temporary allocations for large cubes.
    """
    from scipy.ndimage import binary_dilation

    if flux.shape != error.shape or flux.ndim != 3:
        raise ValueError('flux and error cubes must have identical 3D shape')
    nw, ny, nx = flux.shape
    if not (0 <= x0 < x1 <= nx and 0 <= y0 < y1 <= ny):
        raise ValueError('cube mask ROI exceeds the input cube dimensions')
    if not isinstance(radius, int) or radius < 0:
        raise ValueError('spatial mask radius must be a nonnegative integer')
    if wavelength_block < 1:
        raise ValueError('wavelength_block must be positive')

    halo_x0, halo_x1 = max(0, x0-radius), min(nx, x1+radius)
    halo_y0, halo_y1 = max(0, y0-radius), min(ny, y1+radius)
    iy0, iy1 = y0-halo_y0, y1-halo_y0
    ix0, ix1 = x0-halo_x0, x1-halo_x0
    footprint = np.ones((1, 2*radius+1, 2*radius+1), dtype=bool)
    mask = np.empty((nw, y1-y0, x1-x0), dtype=bool)
    initially_bad = 0
    for w0 in range(0, nw, wavelength_block):
        w1 = min(nw, w0+wavelength_block)
        f = flux[w0:w1, halo_y0:halo_y1, halo_x0:halo_x1]
        e = error[w0:w1, halo_y0:halo_y1, halo_x0:halo_x1]
        invalid = (~np.isfinite(f) | (f == 0) |
                   ~np.isfinite(e) | (e <= 0))
        initially_bad += np.count_nonzero(invalid[:, iy0:iy1, ix0:ix1])
        expanded = binary_dilation(invalid, structure=footprint)
        mask[w0:w1] = expanded[:, iy0:iy1, ix0:ix1]
    return mask, initially_bad


def load_3dcubegen_cube(path, *, flux_scaling=1e16,
                         cube_input_scale=None, cube_lsf_sigma=None,
                         cube_x_range=None, cube_y_range=None,
                         create_cube_mask=True):
    """Return (wave, RSS flux, RSS error, header, primary, LSF, PT, MASK).

    3DCubeGen stores (wavelength, y, x) in Python array order, with flux in
    extension 0 and 1-sigma *errors*, NOT inverse variance, in extension 1.
    An RSS row is formed for every spaxel with at least one valid sample.
    Invalid/zero-valued pixels are masked, not silently dropped from the line.
    If create_cube_mask is True (default), invalid flux/error samples also
    mask the immediately adjacent spaxels (3x3 square) in the SAME
    wavelength plane. False retains the previous pointwise invalid mask.
    Spatial ROI uses full-cube zero-based, end-exclusive pixel coordinates.
    """
    with fits.open(path, memmap=True, do_not_scale_image_data=False) as hdu:
        flux_cube = hdu[0].data
        error_cube = hdu[1].data if len(hdu) > 1 else None
        if flux_cube is None or flux_cube.ndim != 3:
            raise ValueError('Input cube HDU 0 must contain (n_wave, n_y, n_x) FLUX')
        if error_cube is None or error_cube.shape != flux_cube.shape:
            raise ValueError('Input cube HDU 1 must contain matching 1-sigma ERROR')
        header = hdu[0].header.copy()
        nw, ny, nx = flux_cube.shape
        x0, x1 = _spatial_range(cube_x_range, nx, 'cube_x_range')
        y0, y1 = _spatial_range(cube_y_range, ny, 'cube_y_range')

        wcs = WCS(header)
        # The FITS cube spectral coordinate is axis 3 (NumPy axis 0).
        # 3DCubeGen cubes can omit CTYPE3, and astropy.wcs.spectral may
        # consequently fail to recognize the wavelength axis. Read the
        # explicitly provided FITS axis-3 linear solution, prioritizing
        # CDELT3 as stored in these cubes; never infer it from spatial axes.
        if 'CRVAL3' not in header or 'CRPIX3' not in header:
            raise ValueError('Cube wavelength needs CRVAL3 and CRPIX3')
        if 'CDELT3' in header:
            spectral_step = float(header['CDELT3'])
        elif 'CD3_3' in header:
            spectral_step = float(header['CD3_3'])
        else:
            raise ValueError('Cube wavelength needs CDELT3 or CD3_3')
        spect_unit = u.Unit(header.get('CUNIT3', 'Angstrom') or 'Angstrom')
        if not spect_unit.is_equivalent(u.AA):
            raise ValueError('Cube axis 3 must be a wavelength axis')
        wave = ((float(header['CRVAL3']) +
                 (np.arange(nw, dtype=float) + 1.0 - float(header['CRPIX3']))
                 * spectral_step) * spect_unit).to_value(u.AA)
        print(f'# Cube wavelength from FITS axis 3: '
              f'CRVAL3={header["CRVAL3"]}, CRPIX3={header["CRPIX3"]}, '
              f'CDELT3/CD3_3 used={spectral_step:g} {spect_unit}; '
              f'first={wave[0]:.3f} A, last={wave[-1]:.3f} A, '
              f'delta={wave[1]-wave[0]:.6f} A/pixel')
        if not np.all(np.isfinite(wave)) or np.any(np.diff(wave) <= 0):
            raise ValueError('Cube wavelength must be finite and increasing')
        # DAP uses a linear FITS wavelength header for some subroutines.
        if not np.allclose(np.diff(wave), wave[1] - wave[0], rtol=1e-8, atol=1e-8):
            raise ValueError('Cube has nonlinear spectral WCS, unsupported by DAP RSS routines')
        if not wcs.has_celestial:
            raise ValueError('Cube must have celestial WCS for the spaxel PT')

        factor = _cube_flux_factor(header.get('BUNIT', ''), cube_input_scale,
                                   flux_scaling)
        flux_window = flux_cube[:, y0:y1, x0:x1]
        err_window = error_cube[:, y0:y1, x0:x1]
        # A spaxel is absent only if NO sampled wavelength is valid.  Negative
        # fluxes are retained; no flux detection threshold is introduced.
        if create_cube_mask:
            # A 3x3 spatial neighborhood is applied INDEPENDENTLY to each
            # wavelength plane. This includes pixels just outside a cropped
            # tile, to reproduce the same mask as a full-cube analysis.
            cube_bad, n_original_bad = _spatial_cube_invalid_mask(
                flux_cube, error_cube, x0, x1, y0, y1,
                radius=1,
            )
            print(f'# Cube spatial mask: 1-pixel radius (3x3), '
                  f'{n_original_bad} originally invalid voxels, '
                  f'{int(np.count_nonzero(cube_bad))} masked after dilation '
                  f'in ROI; no wavelength-axis dilation')
            valid = ~cube_bad
        else:
            # Preserve the pre-patch cube mask exactly when switched off.
            cube_bad = None
            valid = (np.isfinite(flux_window) & np.isfinite(err_window) &
                     (flux_window != 0) & (err_window > 0))
        in_spatial = np.any(valid, axis=0)
        yy, xx = np.nonzero(in_spatial)
        if not len(xx):
            raise ValueError('No valid spaxels in the requested cube footprint')
        xx = xx.astype(np.int32) + x0
        yy = yy.astype(np.int32) + y0
        # Gather rows in y-major, x-minor order. We copy only selected spaxels.
        rss_flux = np.asarray(flux_cube[:, yy, xx].T, dtype=np.float64)
        rss_error = np.asarray(error_cube[:, yy, xx].T, dtype=np.float64)
        if factor != 1.0:
            rss_flux *= factor
            rss_error *= factor
        # Original zero/NaN/inf/error<=0 samples are flagged, and will be
        # handled by the existing `repair_masked_rss_spectra` machinery.
        if create_cube_mask:
            # Position indices yy/xx are absolute coordinates; cube_bad is
            # indexed within the ROI. Propagate the expanded mask unchanged
            # to the same RSS rows consumed by load_input_data().
            input_mask = cube_bad[:, yy-y0, xx-x0].T.astype(np.uint8)
            del cube_bad
        else:
            bad = (~np.isfinite(rss_flux) | (rss_flux == 0) |
                   ~np.isfinite(rss_error) | (rss_error <= 0))
            input_mask = bad.astype(np.uint8)
        ra, dec = wcs.celestial.all_pix2world(xx.astype(float), yy.astype(float), 0)
        pt = Table()
        pt['id'] = (yy.astype(np.int64) * nx + xx).astype(np.int64)
        pt['ID'] = pt['id'].copy()
        pt['fiberid'] = pt['id'].copy()
        pt['xpix'] = xx
        pt['ypix'] = yy
        pt['ra'] = np.asarray(ra, float)
        pt['dec'] = np.asarray(dec, float)
        pt['mask'] = np.ones(len(xx), dtype=bool)
        # No exposure number is fabricated for a dither-combined cube.

        if cube_lsf_sigma is None:
            # Same placeholder semantics as the existing generic RSS reader:
            # zero means unavailable, not a measured sigma=0 instrument.
            lsf = np.zeros(rss_flux.shape, dtype=np.float32)
            print('# WARNING: reconstructed cube has no LSF. Per-pixel LSF is '
                  'unavailable. Set cube_lsf_sigma to an explicitly justified '
                  'instrumental Gaussian sigma in Angstrom before scientific use.')
        else:
            sigma = float(cube_lsf_sigma)
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('cube_lsf_sigma must be finite and >0 Angstrom')
            lsf = np.full(rss_flux.shape, sigma, dtype=np.float32)
            print(f'# Cube LSF explicitly supplied: Gaussian sigma={sigma} Angstrom')

    # FIT3D/RSP reads wavelength via CRVAL1/CDELT1/CRPIX1 on a 2D RSS.
    rss_header = fits.Header()
    rss_header['NAXIS'] = 2
    rss_header['NAXIS1'] = int(nw)
    rss_header['NAXIS2'] = int(len(xx))
    rss_header['CRPIX1'] = 1.0
    rss_header['CRVAL1'] = float(wave[0])
    rss_header['CDELT1'] = float(wave[1] - wave[0])
    print(f'# Cube->RSS wavelength: CRVAL1={rss_header["CRVAL1"]:.3f} A, '
          f'CRPIX1={rss_header["CRPIX1"]:.0f}, '
          f'CDELT1={rss_header["CDELT1"]:.6f} A/pixel')
    rss_header['CTYPE1'] = 'WAVE'
    rss_header['CUNIT1'] = 'Angstrom'
    rss_header['BUNIT'] = 'scaled DAP fit flux'
    rss_header['CUBEMASK'] = (bool(create_cube_mask), 'Spatial dilation of cube invalid samples')
    if create_cube_mask:
        rss_header['CMBRAD'] = (1, 'Spatial dilation radius, pixels (x/y only)')
    rss_header['CUBENX'] = nx
    rss_header['CUBENY'] = ny
    rss_header['CUBEX0'] = x0
    rss_header['CUBEX1'] = x1
    rss_header['CUBEY0'] = y0
    rss_header['CUBEY1'] = y1
    primary = header.copy()
    primary['CUBEMODE'] = (True, 'Input is a 3DCubeGen spatial cube')
    primary['CUBEMASK'] = (bool(create_cube_mask), 'Spatial dilation of cube invalid samples')
    if create_cube_mask:
        primary['CMBRAD'] = (1, 'Spatial dilation radius, pixels (x/y only)')
    primary['CUBENX'] = nx
    primary['CUBENY'] = ny
    primary['CUBEX0'] = x0
    primary['CUBEX1'] = x1
    primary['CUBEY0'] = y0
    primary['CUBEY1'] = y1
    primary['POSCIRA'] = float(header.get('CRVAL1', np.median(ra)))
    primary['POSCIDE'] = float(header.get('CRVAL2', np.median(dec)))
    print(f'# Cube (wave,y,x)={nw,ny,nx}; ROI x=[{x0},{x1}), '
          f'y=[{y0},{y1}); selected {len(xx)} nonempty spaxels; '
          f'fit-flux input factor={factor:g}')
    return wave, rss_flux, rss_error, rss_header, primary, lsf, pt, input_mask


def _numeric_and_code(values):
    """Return a masked numeric vector and optional category names."""
    arr = np.ma.asarray(values)
    if np.issubdtype(arr.dtype, np.number) or arr.dtype.kind == 'b':
        out = np.ma.asarray(arr, dtype=np.float64).filled(np.nan)
        return np.asarray(out, dtype=np.float64), None
    strings = np.ma.asarray(arr).astype(str)
    mask = np.ma.getmaskarray(arr)
    names = sorted(set(np.asarray(strings, dtype=str)[~mask].tolist()))
    codes = np.full(len(arr), np.nan, dtype=np.float64)
    for i, name in enumerate(names, 1):
        codes[(np.asarray(strings, dtype=str) == name) & (~mask)] = float(i)
    return codes, names


def _make_parameter_hdu(name, table, xpix, ypix, full_nx, full_ny, celestial,
                        *, coeffs=False):
    """Map all numeric/categorical fields into a 3D [plane,y,x] FITS image."""
    if table is None or len(table) == 0:
        return None, [], []
    columns = list(table.colnames)
    if coeffs:
        for key in ('id_fib', 'rsp'):
            if key not in columns:
                raise KeyError(f'COEFFS requires {key} to map template rows')
        template_ids = sorted(np.unique(np.asarray(table['rsp'], dtype=int)))
        fields = [(c, t) for t in template_ids for c in columns
                  if c not in ('id_fib', 'rsp', 'id', 'ID', 'fiberid')]
    else:
        if len(table) != len(xpix):
            raise ValueError(f'{name}: {len(table)} rows, expected {len(xpix)}')
        fields = [(c, None) for c in columns]
    if not fields:
        return None, [], []
    result = np.full((len(fields), full_ny, full_nx), np.nan, dtype=np.float64)
    catalogue = []
    categories = []
    if coeffs:
        fiber_indices = np.asarray(table['id_fib'], dtype=np.int64)
        if np.any(fiber_indices < 0) or np.any(fiber_indices >= len(xpix)):
            raise ValueError('COEFFS id_fib contains out-of-range RSS row index')
        rss_template = np.asarray(table['rsp'], dtype=int)
    for k, (column, template_id) in enumerate(fields):
        if template_id is None:
            values, codes = _numeric_and_code(table[column])
            result[k, ypix, xpix] = values
        else:
            indices = np.flatnonzero(rss_template == template_id)
            fi = fiber_indices[indices]
            if len(set(fi.tolist())) != len(fi):
                raise ValueError(f'Duplicate COEFFS fiber/template pair: {template_id}')
            values, codes = _numeric_and_code(table[column][indices])
            result[k, ypix[fi], xpix[fi]] = values
        unit = getattr(table[column], 'unit', None)
        label = str(column) if template_id is None else f'{column}@rsp={template_id}'
        catalogue.append((name, k, label, '' if unit is None else str(unit),
                          'CATEGORY' if codes is not None else 'NUMERIC'))
        if codes is not None:
            categories.extend((name, k, j, category)
                              for j, category in enumerate(codes, 1))
    out_header = celestial.copy()
    out_header['WCSAXES'] = 3
    out_header['CTYPE3'] = 'PARAM'
    out_header['CRVAL3'] = 0.0
    out_header['CRPIX3'] = 1.0
    out_header['CDELT3'] = 1.0
    out_header['NPLANE'] = len(fields)
    # Pipe3D-style, directly discoverable labels for every parameter plane.
    # NAME_<plane> is EXACTLY the original DAP Astropy-table column name.
    # COEFFS repeats column names for each SSP template; RSP_<plane> gives
    # its template ID without changing the original name. PLANE_INDEX keeps
    # the longer unique "column@rsp=<ID>" description as before.
    for k, (column, template_id) in enumerate(fields):
        out_header[f'NAME_{k}'] = (str(column), 'Original DAP table column')
        if template_id is not None:
            out_header[f'RSP_{k}'] = (int(template_id), 'RSP template index')
    out_header['COMMENT'] = 'Axis 3 is parameter index; see PLANE_INDEX extension'
    return fits.ImageHDU(result, header=out_header, name=name), catalogue, categories


def write_dap_parameter_cubes(path, source_cube, pt, tables, *,
                              primary_header=None, info=None,
                              avg_param_stats=None, avg_el_fit_qa=None,
                              bad_pixel_mask=None):
    """Write spatial parameter cubes and an explicit catalogue of every plane.

    `tables` maps FITS extension names (RSP, COEFFS, NP_ELINES_B, ...)
    to DAP Tables. Spatial pixel addresses come exclusively from the PT.
    The source 3D cube WCS is copied as 2D celestial WCS per parameter HDU.
    """
    original = fits.getheader(source_cube, 0)
    source_wcs = WCS(original)
    celestial = source_wcs.celestial.to_header(relax=True)
    ny = int(original['NAXIS2'])
    nx = int(original['NAXIS1'])
    xpix = np.asarray(pt['xpix'], dtype=np.int64)
    ypix = np.asarray(pt['ypix'], dtype=np.int64)
    if np.any(xpix < 0) or np.any(xpix >= nx) or np.any(ypix < 0) or np.any(ypix >= ny):
        raise ValueError('PT spaxel coordinates outside original cube')
    if len(np.unique(ypix * nx + xpix)) != len(pt):
        raise ValueError('PT contains duplicate cube spaxel coordinates')
    hdr = fits.Header() if primary_header is None else primary_header.copy()
    hdr['CUBEMODE'] = True
    hdr['CUBENX'] = nx
    hdr['CUBENY'] = ny
    hdr['COMMENT'] = 'Spatial parameter maps: see PLANE_INDEX for each plane'
    # Streaming to a temporary *uncompressed* FITS file avoids retaining all
    # potentially hundreds of MB parameter images in an in-memory HDUList.
    # gzip the completed file only once; appending directly to .fits.gz would
    # repeatedly re-compress it. The temp file is on the output filesystem.
    destination = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(destination))
    os.makedirs(directory, exist_ok=True)
    fd, tempfile_path = tempfile.mkstemp(prefix='.lvmdap_cube_',
                                          suffix='.fits', dir=directory)
    os.close(fd)
    index = []
    cats = []
    try:
        fits.PrimaryHDU(header=hdr).writeto(tempfile_path, overwrite=True)
        for name, tab in tables.items():
            hdu, fields, encodings = _make_parameter_hdu(
                name, tab, xpix, ypix, nx, ny, celestial,
                coeffs=(name == 'COEFFS'))
            if hdu is not None:
                fits.append(tempfile_path, hdu.data, header=hdu.header)
                index.extend(fields)
                cats.extend(encodings)
                del hdu

        def append_table(tab, name):
            hdu = fits.BinTableHDU(tab, name=name)
            fits.append(tempfile_path, hdu.data, header=hdu.header)

        append_table(Table(rows=index, names=(
            'HDU', 'PLANE', 'PARAMETER', 'UNIT', 'ENCODING')),
            'PLANE_INDEX')
        if cats:
            append_table(Table(rows=cats, names=(
                'HDU', 'PLANE', 'CODE', 'VALUE')), 'CATEGORIES')
        append_table(pt[['id', 'xpix', 'ypix', 'ra', 'dec']], 'SPAXEL_INDEX')
        if bad_pixel_mask is not None:
            bad = np.asarray(bad_pixel_mask, bool)
            if bad.ndim != 2 or bad.shape[0] != len(pt):
                raise ValueError('bad_pixel_mask must have shape (n_spaxel,n_wave)')
            goodmap = np.full((ny, nx), np.nan, dtype=np.float32)
            goodmap[ypix, xpix] = np.count_nonzero(~bad, axis=1)
            hdu = fits.ImageHDU(goodmap, header=celestial.copy(), name='NGOODPIX')
            fits.append(tempfile_path, hdu.data, header=hdu.header)
        if avg_param_stats is not None:
            append_table(avg_param_stats, 'AVG_PARAM_STATS')
        if avg_el_fit_qa is not None:
            append_table(avg_el_fit_qa, 'AVG_EL_FIT_QA')
        if info is not None:
            fits.append(tempfile_path, info.data, header=info.header)
        if destination.endswith('.gz'):
            # An atomic rename avoids an incomplete final .gz on failure.
            temp_gz = tempfile_path + '.gz'
            try:
                with open(tempfile_path, 'rb') as src, gzip.open(temp_gz, 'wb', compresslevel=4) as out:
                    shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
                os.replace(temp_gz, destination)
            finally:
                if os.path.exists(temp_gz):
                    os.unlink(temp_gz)
        else:
            os.replace(tempfile_path, destination)
        return path
    finally:
        if os.path.exists(tempfile_path):
            os.unlink(tempfile_path)
