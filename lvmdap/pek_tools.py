"""LSF-aware constrained Gaussian-cluster fits for SDSS-V LVM-DAP.

Independent of pyFIT3D. Requires numpy, scipy and astropy.table.

`fit_pek_rss` returns a wide Astropy table whose columns end in `_pek`
and the corresponding gas-emission-only RSS model (not the fitted constant
baselines). Each cluster shares a velocity and an intrinsic variance; the
observed Gaussian sigma is adjusted at every line for its actual LSF.
"""

import re
import warnings

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import griddata

_C = 299792.458
_ROOT2PI = np.sqrt(2 * np.pi)
_FIELDS = ('flux', 'e_flux', 'vel', 'e_vel', 'disp', 'e_disp', 'EW', 'e_EW')


def print_progress_bar(i, total, label="Fit. progress", bar_width=40):
    """
    Display a progress bar updated on the same terminal line.

    Parameters
    ----------
    i : int
        Current iteration (zero-based).
    total : int
        Total number of iterations.
    label : str
        Progress description.
    bar_width : int
        Width of the progress bar.
    """
    if total <= 0:
        return

    fraction = min(1.0, (i + 1) / total)
    percent = 100 * fraction
    filled = int(bar_width * fraction)

    bar = '█' * filled + '░' * (bar_width - filled)

    print(
        f"\r### {label}: |{bar}| "
        f"{percent:5.1f}% ({i+1}/{total})",
        end='\n' if i + 1 >= total else '',
        flush=True
    )







def _line_key(line):
    name = line['id']
    if isinstance(name, bytes):
        name = name.decode('utf-8')
    return f'{str(name).strip()}_{float(line["wl"]):.2f}'


def _table_value(tab, i, field, linekey, suffix='', default=np.nan):
    """Read PM/NP values, accepting old filenames missing a trailing zero."""
    if tab is None:
        return default
    names = tab.colnames if hasattr(tab, 'colnames') else tab.keys()
    key = f'{field}_{linekey}{suffix}'
    if key not in names:
        prefix, w = linekey.rsplit('_', 1)
        pattern = re.compile(rf'^{re.escape(field)}_{re.escape(prefix)}_(\d+(?:\.\d+)?){re.escape(suffix)}$')
        candidates = [n for n in names if (m := pattern.match(n)) is not None
                      and abs(float(m.group(1)) - float(w)) < 0.004]
        if len(candidates) != 1:
            return default
        key = candidates[0]
    try:
        v = float(tab[key][i])
    except (ValueError, TypeError, IndexError):
        return default
    return v if np.isfinite(v) else default


def _clusters(tab_el, max_span, max_gap):
    order = np.argsort(np.asarray(tab_el['wl'], dtype=float))
    groups = []
    group = []
    waves = np.asarray(tab_el['wl'], dtype=float)
    for k in order:
        k = int(k)
        if group and (waves[k] - waves[group[0]] > max_span or
                      waves[k] - waves[group[-1]] > max_gap):
            groups.append(group)
            group = []
        group.append(k)
    if group:
        groups.append(group)
    return groups


def interpolate_ha_kinematics(tab_pt, tab_pm, sn_cut=3.0, line_key="Halpha_6562.85"):
    """Interpolate observed velocities (km/s) and sigmas (Angstrom) for a PM line.

    ID/row order must agree between tab_pt and tab_pm. Low-S/N or nonfinite
    points are excluded as interpolation anchors. Use linear interpolation
    where possible, with nearest-neighbor extrapolation (spline oscillations
    and negative dispersion are avoided).
    """
    x = np.asarray(tab_pt['ra'], dtype=float)
    y = np.asarray(tab_pt['dec'], dtype=float)
    vel = np.array([_table_value(tab_pm, i, 'vel', line_key, '_pm')
                    for i in range(len(tab_pm))], float)
    sig = np.array([_table_value(tab_pm, i, 'disp', line_key, '_pm')
                    for i in range(len(tab_pm))], float)
    flux = np.array([_table_value(tab_pm, i, 'flux', line_key, '_pm')
                     for i in range(len(tab_pm))], float)
    eflux = np.array([_table_value(tab_pm, i, 'e_flux', line_key, '_pm')
                      for i in range(len(tab_pm))], float)
    if not (len(x) == len(vel) == len(y)):
        raise ValueError('tab_pt and tab_pm must have one row per same fiber in same order')
    with np.errstate(divide='ignore', invalid='ignore'):
        sn = flux / eflux
    source = (np.isfinite(x) & np.isfinite(y) & np.isfinite(vel) &
              np.isfinite(sig) & (sig > 0) & np.isfinite(sn) & (sn > sn_cut))
    if not source.any():
        warnings.warn(f'No usable {line_key} anchors; returning original kinematics')
        return vel.copy(), sig.copy()
    coords = np.column_stack((x, y))
    src = coords[source]

    def interpolate(values):
        nearest = griddata(src, values[source], coords, method='nearest')
        # A linear triangulation requires at least three noncollinear points.
        if source.sum() >= 3 and np.linalg.matrix_rank(src - src.mean(axis=0)) >= 2:
            try:
                linear = griddata(src, values[source], coords, method='linear')
                nearest[np.isfinite(linear)] = linear[np.isfinite(linear)]
            except (ValueError, RuntimeError):
                pass
        nearest[source] = values[source]
        return nearest

    return interpolate(vel), interpolate(sig)


def _pek_weighted_linear_fit(x, y, ey, rest, inst, velocity,
                             intrinsic_var_ref, reference_wavelength):
    """Solve signed Gaussian fluxes and constant directly at fixed kinematics.

    ``intrinsic_var_ref`` is the intrinsic Gaussian variance in Angstrom^2 at
    the selected reference line. Every component uses the corresponding
    wavelength-scaled intrinsic width, equivalent to a single intrinsic
    velocity dispersion for all lines in the cluster. Numerical optimization
    remains in reference-Angstrom^2 for conditioning.

    Returns (coefficients, rank, design), where design has the original
    flux-density units (not whitened). A rank-deficient solution is never
    accepted as an independent measurement of all the line fluxes.
    """
    width_ratio_sq = (rest / reference_wavelength)**2
    sig = np.sqrt(intrinsic_var_ref * width_ratio_sq + inst**2)
    centers = rest * (1.0 + velocity / _C)
    dx = (x[:, None] - centers[None, :]) / sig[None, :]
    profiles = np.exp(-0.5 * dx**2) / (_ROOT2PI * sig[None, :])
    design = np.column_stack((profiles, np.ones(len(x))))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design / ey[:, None], y / ey, rcond=None
    )
    return coefficients, rank, design


def fit_pek_rss(gas_spectrum, error_spectrum, vel_interp, disp_interp,
                tab_el, tab_PM, tab_fe, *, wl, lsf_spectrum=None,
                bad_pixel_mask=None, velocity_tolerance=5.0,
                dispersion_tolerance=1.0, fit_padding=20.0,
                max_cluster_span=75.0, max_adjacent_distance=25.0,
                min_valid_pixels=8, max_nfev=120, verbose=False,
                reference_line='Halpha_6562.85'):
    """Fit Gaussian emission lines plus a constant per cluster.

    Inputs
    ------
    gas_spectrum, error_spectrum: arrays (n_fiber,n_wave) or (n_wave,).
    vel_interp: interpolated PM reference-line LOS velocity, km/s (one per fiber).
    disp_interp: interpolated PM reference-line observed Gaussian sigma, Angstrom.
    reference_line: selected PM line key, e.g. '[NII]_6583.45'. If omitted,
        the historical Halpha_6562.85 reference is used for compatibility.
    tab_el: line catalog Table with columns 'id', 'wl', optionally 'lsf'.
    tab_PM, tab_fe: previous PM/NP wide tables. Their line fluxes are not used
        to initialize or constrain PEK amplitudes. NP reference-line velocity
        and dispersion, when available, only expand the kinematic bounds.
        Their equivalent widths are used for the original continuum estimate.
    wl: observed wavelength grid, Angstrom (required; not inferred from shape).
    lsf_spectrum: instrumental Gaussian sigma in Angstrom, a vector (n_wave,)
        or RSS (n_fiber,n_wave), strongly preferred. If absent, use
        tab_el['lsf'] interpolated at the selected reference wavelength.
    bad_pixel_mask: 0=valid, nonzero=invalid; same shape as gas spectrum.
    velocity_tolerance: km/s margin around both the interpolated PM and
        available NP reference-line velocity (per fiber).
    dispersion_tolerance: Angstrom margin around observed reference-line
        widths from PM and, if available, NP; intrinsic widths are deconvolved
        using the per-fiber LSF at each measurement's reference wavelength.
    fit_padding: +-Angstrom around the first/last line in a cluster.

    Returns
    -------
    tab_PEK: Astropy Table with id/id_fib and flux, e_flux, vel, e_vel,
        disp, e_disp, EW, e_EW columns ending in '_pek'. 'disp' is observed
        Gaussian sigma in Angstrom, not FWHM and not intrinsic sigma.
        EW is derived from an existing PM/NP continuum estimate; if no
        trustworthy continuum is available it is NaN. e_EW propagates only
        line-flux uncertainty because the original stellar spectrum is absent.
    model: fitted emission-only RSS with shape matching gas_spectrum.
        The fitted per-cluster constant background is not included, to avoid
        piecewise background offsets in the summed emission spectrum.

    Each trial velocity and intrinsic variance determines the Gaussian
    templates. Signed line fluxes and the constant background are solved by
    weighted linear least squares, with NO PM/NP flux initialization. The
    initial kinematics use interpolated PM at ``reference_line`` unless the
    NP reference-line local X_sq is lower (or the PM score is unavailable)
    and the NP velocity/observed dispersion are usable. The chosen source
    initializes all clusters of that fiber, without changing their bounds.
    Bounds also enclose valid NP reference-line measurements, when present,
    with the existing margins and no additional S/N cuts. Intrinsic velocity
    dispersion is shared within each cluster (not intrinsic Angstrom width).
    A component with no valid samples in its Gaussian half-maximum core is
    unmeasurable and remains NaN; it does not contaminate the other lines.
    Errors use the full weighted local covariance without chi-square scaling
    when nonsingular. If kinematics are degenerate, line-flux errors are
    conditional on the fitted kinematics; kinematic errors remain NaN.
    """
    from astropy.table import Table

    flux = np.asarray(gas_spectrum, float)
    err = np.asarray(error_spectrum, float)
    single = flux.ndim == 1
    if single:
        flux = flux[None, :]
        err = err[None, :]
    if flux.ndim != 2 or flux.shape != err.shape:
        raise ValueError('gas_spectrum and error_spectrum shapes must agree')
    nf, nw = flux.shape
    wave = np.asarray(wl, float)
    if wave.shape != (nw,) or not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        raise ValueError('wl must be a finite, strictly increasing 1D wavelength array')
    if len(tab_PM) != nf or len(tab_fe) != nf:
        raise ValueError('PM and NP tables must have exactly one row per RSS fiber in the same order')
    vs = np.broadcast_to(np.asarray(vel_interp, float), (nf,))
    ss = np.broadcast_to(np.asarray(disp_interp, float), (nf,))
    if velocity_tolerance <= 0 or dispersion_tolerance <= 0 or fit_padding <= 0:
        raise ValueError('Tolerances and fit_padding must be positive')
    if bad_pixel_mask is None:
        bad = np.zeros((nf,nw), dtype=bool)
    else:
        bad = np.asarray(bad_pixel_mask, bool)
        if single and bad.ndim == 1:
            bad = bad[None, :]
        if bad.shape != flux.shape:
            raise ValueError('bad_pixel_mask shape mismatch')
    if lsf_spectrum is None:
        if 'lsf' not in tab_el.colnames:
            raise ValueError('lsf_spectrum or tab_el["lsf"] is required for LSF corrections')
        cat_lsf = np.asarray(tab_el['lsf'], float)
        cat_w = np.asarray(tab_el['wl'], float)
        if np.any(~np.isfinite(cat_lsf)) or np.any(cat_lsf <= 0):
            raise ValueError('tab_el["lsf"] must contain positive finite Gaussian sigmas (Angstrom)')
        order = np.argsort(cat_w)
        cat_lsf = cat_lsf[order]
        cat_w = cat_w[order]
        lsf_mode = 'catalog'
    else:
        lsf = np.asarray(lsf_spectrum, float)
        if lsf.shape == (nw,):
            lsf = np.broadcast_to(lsf, (nf,nw))
        if lsf.shape != (nf,nw):
            raise ValueError('lsf_spectrum must be (n_wave,) or (n_fiber,n_wave) in Gaussian sigma Angstrom')
        lsf_mode = 'spectrum'

    # The rest wavelength of the SAME line used to derive vel_interp/disp_interp.
    # It must not silently revert to Halpha when [NII] or [OIII] was selected.
    try:
        reference_wavelength = float(str(reference_line).rsplit('_', 1)[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError(f'Invalid PEK reference_line: {reference_line!r}') from exc
    if not np.isfinite(reference_wavelength) or reference_wavelength <= 0:
        raise ValueError('PEK reference_line must have a positive finite wavelength')

    line_names = [_line_key(line) for line in tab_el]
    if len(set(line_names)) != len(line_names):
        raise ValueError('Two catalog lines have the same two-decimal identifier')
    groups = _clusters(tab_el, max_cluster_span, max_adjacent_distance)
    result = Table()
    for name in ('id_fib', 'id'):
        if name in tab_PM.colnames:
            result[name] = tab_PM[name].copy()
    if 'id_fib' not in result.colnames:
        result['id_fib'] = np.arange(nf, dtype=int)
    for line_key in line_names:
        for field in _FIELDS:
            result[f'{field}_{line_key}_pek'] = np.full(nf, np.nan)
    model = np.zeros((nf,nw), float)
    w_lines = np.asarray(tab_el['wl'], float)

    for i in range(nf):
        if (verbose==True):
            print(f"\n### Fitting PEK lines {i+1}/{nf}")
        else:
            print_progress_bar(i, nf, label="Fitting PEK lines", bar_width=40)
        if not (np.isfinite(vs[i]) and np.isfinite(ss[i]) and ss[i] > 0):
            continue
        # The original PM interpolation still defines the baseline kinematic
        # bounds. Select a better *initial guess* from the same reference line
        # when its measured NP local reduced chi-square beats PM. This does
        # not change the admissible bounds, line amplitudes, or their signs.
        np_velocity = _table_value(tab_fe, i, 'vel', reference_line)
        np_dispersion = _table_value(tab_fe, i, 'disp', reference_line)
        np_chi = _table_value(tab_fe, i, 'X_sq', reference_line)
        pm_chi = _table_value(tab_PM, i, 'X_sq', reference_line, '_pm')
        start_from_np = (
            np.isfinite(np_chi)
            and (not np.isfinite(pm_chi) or np_chi < pm_chi)
            and np.isfinite(np_velocity)
            and np.isfinite(np_dispersion) and np_dispersion > 0
        )
        ref_center_pm = reference_wavelength * (1.0 + vs[i] / _C)
        if lsf_mode == 'spectrum':
            lsf_ref = float(np.interp(ref_center_pm, wave, lsf[i],
                                      left=np.nan, right=np.nan))
        else:
            # Match the previous catalog-LSF endpoint behavior: the selected
            # reference line itself may lie at the catalog's extreme wavelength.
            lsf_ref = float(np.interp(ref_center_pm, cat_w, cat_lsf))
        if not (np.isfinite(lsf_ref) and lsf_ref > 0):
            continue

        velocity_lo = vs[i] - velocity_tolerance
        velocity_hi = vs[i] + velocity_tolerance
        if np.isfinite(np_velocity):
            velocity_lo = min(vs[i], np_velocity) - velocity_tolerance
            velocity_hi = max(vs[i], np_velocity) + velocity_tolerance

        # Work in intrinsic Angstrom^2 at the reference wavelength. For any
        # other line the intrinsic variance is multiplied by (lambda/ref)^2,
        # which is equivalent to a common physical dispersion in km/s.
        pm_sigma_lo = max(0.0, ss[i] - dispersion_tolerance)
        pm_sigma_hi = ss[i] + dispersion_tolerance
        var_lo = max(0.0, pm_sigma_lo**2 - lsf_ref**2)
        var_hi = max(0.0, pm_sigma_hi**2 - lsf_ref**2)
        var0 = max(0.0, ss[i]**2 - lsf_ref**2)
        start_velocity = vs[i]
        used_np_start = False

        if np.isfinite(np_dispersion) and np_dispersion > 0:
            # NP and PM can have slightly different reference velocities;
            # deconvolve with the instrumental LSF at each observed center.
            ref_center_np = (reference_wavelength *
                             (1.0 + (np_velocity if np.isfinite(np_velocity)
                                      else vs[i]) / _C))
            if lsf_mode == 'spectrum':
                lsf_np = float(np.interp(ref_center_np, wave, lsf[i],
                                         left=np.nan, right=np.nan))
            else:
                lsf_np = float(np.interp(ref_center_np, cat_w, cat_lsf))
            if np.isfinite(lsf_np) and lsf_np > 0:
                # Convert the NP intrinsic variance to the PM reference center
                # so both limits describe the same velocity dispersion.
                ratio_sq = (ref_center_pm / ref_center_np)**2
                np_sigma_lo = max(0.0, np_dispersion - dispersion_tolerance)
                np_sigma_hi = np_dispersion + dispersion_tolerance
                np_var_lo = max(0.0, np_sigma_lo**2 - lsf_np**2) * ratio_sq
                np_var_hi = max(0.0, np_sigma_hi**2 - lsf_np**2) * ratio_sq
                var_lo = min(var_lo, np_var_lo)
                var_hi = max(var_hi, np_var_hi)
                if start_from_np:
                    # Express the NP intrinsic variance at the same reference
                    # wavelength as PM, using measured LSF at the NP centroid.
                    var0 = max(0.0, np_dispersion**2 - lsf_np**2) * ratio_sq
                    start_velocity = np_velocity
                    used_np_start = True

        if var_hi <= var_lo:
            # There is no allowed intrinsic dispersion; do not fabricate one.
            if verbose:
                print(f'PEK fiber {i}: no physical dispersion interval')
            continue
        var0 = np.clip(var0, var_lo, var_hi)
        if verbose:
            choice = 'NP' if used_np_start else 'PM'
            print(f'PEK fiber {i}: initial {choice} kinematics '
                  f'(X_sq_NP={np_chi:.4g}, X_sq_PM={pm_chi:.4g})')

        for cluster in groups:
            rest = w_lines[cluster]
            names = [line_names[k] for k in cluster]
            n = len(cluster)
            if lsf_mode == 'spectrum':
                inst = np.interp(rest*(1+vs[i]/_C), wave, lsf[i], left=np.nan, right=np.nan)
            else:
                inst = np.interp(rest, cat_w, cat_lsf)
            if np.any(~np.isfinite(inst)) or np.any(inst <= 0):
                continue
            low = rest.min()*(1+velocity_lo/_C) - fit_padding
            high = rest.max()*(1+velocity_hi/_C) + fit_padding
            a,b = np.searchsorted(wave,[low,high])
            if b-a < min_valid_pixels:
                continue
            fit_good = (~bad[i,a:b] & np.isfinite(flux[i,a:b]) &
                        np.isfinite(err[i,a:b]) & (err[i,a:b] > 0))
            if fit_good.sum() < min_valid_pixels:
                continue
            x,y,ey = wave[a:b][fit_good],flux[i,a:b][fit_good],err[i,a:b][fit_good]

            # A masked line core cannot constrain its integrated amplitude.
            # This is a geometric pixel-coverage condition (half-maximum of
            # the predicted Gaussian), NOT a flux/SNR detection threshold.
            # Fit the remaining measurable lines in this same cluster.
            predicted = rest * (1.0 + vs[i] / _C)
            width_ratio_sq = (rest / reference_wavelength)**2
            sig0 = np.sqrt(var0 * width_ratio_sq + inst**2)
            core_halfwidth = np.sqrt(2.0 * np.log(2.0)) * sig0
            has_core_pixels = np.any(
                np.abs(x[:, None] - predicted[None, :]) <= core_halfwidth[None, :],
                axis=0
            )
            if np.isfinite(np_velocity):
                # If a PM core is masked but the NP center is not, do not
                # reject that line solely on the old PM-centered test.
                predicted_np = rest * (1.0 + np_velocity / _C)
                has_core_pixels |= np.any(
                    np.abs(x[:, None] - predicted_np[None, :])
                    <= core_halfwidth[None, :], axis=0)
            if not np.any(has_core_pixels):
                continue
            rest_fit = rest[has_core_pixels]
            inst_fit = inst[has_core_pixels]
            ratio_fit_sq = (rest_fit / reference_wavelength)**2
            names_fit = [name for name, good in zip(names, has_core_pixels) if good]
            n_fit = len(rest_fit)
            if len(y) < max(min_valid_pixels, n_fit + 4):
                continue

            # Variable projection: the optimizer sees only the two nonlinear
            # kinematic parameters. Fluxes and constant background are solved
            # exactly, with their signs free, at every (velocity, variance).
            def residual(kinematics):
                velocity, intrinsic_var = kinematics
                coefficients, _, design = _pek_weighted_linear_fit(
                    x, y, ey, rest_fit, inst_fit, velocity, intrinsic_var,
                    reference_wavelength
                )
                return (design @ coefficients - y) / ey

            try:
                fit = least_squares(
                    residual, np.array([start_velocity, var0]),
                    bounds=(np.array([velocity_lo, var_lo]),
                            np.array([velocity_hi, var_hi])),
                    x_scale='jac', max_nfev=max_nfev, method='trf'
                )
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if not fit.success or not np.all(np.isfinite(fit.x)):
                continue
            velocity, intrinsic_var = fit.x
            coefficients, rank, design = _pek_weighted_linear_fit(
                x, y, ey, rest_fit, inst_fit, velocity, intrinsic_var,
                reference_wavelength
            )
            # Never put a mathematically unidentifiable amplitude in the model.
            if rank != n_fit + 1 or not np.all(np.isfinite(coefficients)):
                if verbose:
                    print(f'PEK fiber {i}, cluster {names_fit[0]}: '
                          'rank-deficient linear templates; no fit retained')
                continue
            # Retain the numerical rank tolerance already used by the old
            # PEK covariance; a barely distinguishable blend must not create
            # enormous compensating positive and negative amplitudes.
            try:
                singular_linear = np.linalg.svd(
                    design / ey[:, None], compute_uv=False
                )
            except np.linalg.LinAlgError:
                continue
            if (not len(singular_linear) or
                    singular_linear[-1] <= singular_linear[0] * 1.e-10):
                if verbose:
                    print(f'PEK fiber {i}, cluster {names_fit[0]}: '
                          'ill-conditioned line templates; no fit retained')
                continue
            fitted_fluxes = coefficients[:n_fit]
            sig_out = np.sqrt(intrinsic_var * ratio_fit_sq + inst_fit**2)

            # Marginalize over velocity, intrinsic variance and the constant
            # background in the full weighted Jacobian (as in the old PEK).
            profiles = design[:, :n_fit]
            centers = rest_fit * (1.0 + velocity/_C)
            delta = (x[:, None] - centers[None, :]) / sig_out[None, :]
            d_velocity = np.sum(
                fitted_fluxes[None, :] * profiles *
                (delta / sig_out[None, :]) * (rest_fit[None, :]/_C), axis=1
            )
            d_var = np.sum(
                fitted_fluxes[None, :] * profiles *
                (delta**2 - 1.0) * ratio_fit_sq[None, :] /
                (2.0 * sig_out[None, :]**2), axis=1
            )
            full_jac = np.column_stack((profiles, d_velocity, d_var,
                                        np.ones(len(x)))) / ey[:, None]
            uncertainty = np.full(n_fit + 3, np.nan)
            try:
                _, singular, vt = np.linalg.svd(full_jac, full_matrices=False)
                if (len(singular) != n_fit+3 or singular[-1] <= singular[0]*1.e-10):
                    raise np.linalg.LinAlgError('Rank-deficient full covariance')
                cov = (vt.T * (singular**-2)) @ vt
                uncertainty = np.sqrt(np.maximum(np.diag(cov), 0.0))
            except np.linalg.LinAlgError:
                # Flux is still measured at reference-constrained kinematics even
                # if velocity/dispersion errors cannot be estimated. Return its
                # fixed-kinematics uncertainty, not a spurious NaN flux error.
                try:
                    whitened = design / ey[:, None]
                    _, singular, vt = np.linalg.svd(whitened, full_matrices=False)
                    if singular[-1] > singular[0] * 1.e-10:
                        cov_linear = (vt.T * (singular**-2)) @ vt
                        uncertainty[:n_fit] = np.sqrt(np.maximum(
                            np.diag(cov_linear)[:n_fit], 0.0))
                except np.linalg.LinAlgError:
                    pass

            for j, key in enumerate(names_fit):
                prefix = lambda field: f'{field}_{key}_pek'
                result[prefix('flux')][i] = fitted_fluxes[j]
                result[prefix('e_flux')][i] = uncertainty[j]
                result[prefix('vel')][i] = velocity
                result[prefix('e_vel')][i] = uncertainty[n_fit]
                result[prefix('disp')][i] = sig_out[j]
                result[prefix('e_disp')][i] = (
                    uncertainty[n_fit+1]*ratio_fit_sq[j]/(2*sig_out[j])
                    if sig_out[j] > 0 else np.nan
                )
                # Preserve previous EW definition and PM/NP-continuum lookup.
                continua = []
                for table,suff in ((tab_PM,'_pm'),(tab_fe,'')):
                    original_f = _table_value(table,i,'flux',key,suff)
                    original_ew = _table_value(table,i,'EW',key,suff)
                    if (np.isfinite(original_f) and np.isfinite(original_ew)
                            and abs(original_ew)>1e-20):
                        continua.append(-original_f/original_ew)
                if continua:
                    continuum = float(np.median(continua))
                    if abs(continuum)>1e-20:
                        result[prefix('EW')][i] = -fitted_fluxes[j]/abs(continuum)
                        result[prefix('e_EW')][i] = uncertainty[j]/abs(continuum)
                # Only emission lines: do not add fitted cluster background.
                xx = (wave[a:b] - rest_fit[j]*(1+velocity/_C))/sig_out[j]
                model[i,a:b] += (fitted_fluxes[j]/(_ROOT2PI*sig_out[j]) *
                                  np.exp(-0.5*xx**2))
            if verbose:
                reduced = np.sum(fit.fun**2)/max(len(y)-(n_fit+3),1)
                print(f'PEK fiber {i} cluster {names_fit[0]}..{names_fit[-1]}: '
                      f'chi2_red={reduced:.3f}, v={velocity:.2f} km/s')

    model[bad | ~np.isfinite(flux) | ~np.isfinite(err) | (err<=0)] = np.nan
    return result, model[0] if single else model
