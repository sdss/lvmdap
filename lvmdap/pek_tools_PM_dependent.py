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


def interpolate_ha_kinematics(tab_pt, tab_pm, sn_cut=3.0):
    """Interpolate observed Halpha velocities (km/s) and sigmas (Angstrom).

    ID/row order must agree between tab_pt and tab_pm. Low-S/N or nonfinite
    points are excluded as interpolation anchors. Use linear interpolation
    where possible, with nearest-neighbor extrapolation (spline oscillations
    and negative dispersion are avoided).
    """
    x = np.asarray(tab_pt['ra'], dtype=float)
    y = np.asarray(tab_pt['dec'], dtype=float)
    vel = np.array([_table_value(tab_pm, i, 'vel', 'Halpha_6562.85', '_pm')
                    for i in range(len(tab_pm))], float)
    sig = np.array([_table_value(tab_pm, i, 'disp', 'Halpha_6562.85', '_pm')
                    for i in range(len(tab_pm))], float)
    flux = np.array([_table_value(tab_pm, i, 'flux', 'Halpha_6562.85', '_pm')
                     for i in range(len(tab_pm))], float)
    eflux = np.array([_table_value(tab_pm, i, 'e_flux', 'Halpha_6562.85', '_pm')
                      for i in range(len(tab_pm))], float)
    if not (len(x) == len(vel) == len(y)):
        raise ValueError('tab_pt and tab_pm must have one row per same fiber in same order')
    with np.errstate(divide='ignore', invalid='ignore'):
        sn = flux / eflux
    source = (np.isfinite(x) & np.isfinite(y) & np.isfinite(vel) &
              np.isfinite(sig) & (sig > 0) & np.isfinite(sn) & (sn > sn_cut))
    if not source.any():
        warnings.warn('No usable Halpha anchors; returning original kinematics')
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


def fit_pek_rss(gas_spectrum, error_spectrum, vel_interp, disp_interp,
                tab_el, tab_PM, tab_fe, *, wl, lsf_spectrum=None,
                bad_pixel_mask=None, velocity_tolerance=5.0,
                dispersion_tolerance=1.0, fit_padding=20.0,
                max_cluster_span=75.0, max_adjacent_distance=25.0,
                min_valid_pixels=8, max_nfev=120, verbose=False):
    """Fit Gaussian emission lines plus a constant per cluster.

    Inputs
    ------
    gas_spectrum, error_spectrum: arrays (n_fiber,n_wave) or (n_wave,).
    vel_interp: interpolated Halpha LOS velocity, km/s (one per fiber).
    disp_interp: interpolated Halpha observed Gaussian sigma, Angstrom.
    tab_el: line catalog Table with columns 'id', 'wl', optionally 'lsf'.
    tab_PM, tab_fe: wide per-fiber tables from current DAP PM/NP processing;
        starting integrated flux = mean of finite PM and NP fluxes (signed).
    wl: observed wavelength grid, Angstrom (required; not inferred from shape).
    lsf_spectrum: instrumental Gaussian sigma in Angstrom, a vector (n_wave,)
        or RSS (n_fiber,n_wave), strongly preferred. If absent, use
        tab_el['lsf'] at each reference line, with an interpolation for Halpha.
    bad_pixel_mask: 0=valid, nonzero=invalid; same shape as gas spectrum.
    velocity_tolerance: +-km/s around each interpolated Halpha velocity.
    dispersion_tolerance: +-Angstrom around interpolated *observed* Halpha sigma.
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

    Errors are from the weighted fit's local (J^T W J)^-1 covariance, without
    rescaling by chi-squared. Bound-active/degenerated parameters can have
    unreliable symmetric errors; those are NaN when SVD is singular.
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
        if lsf_mode == 'spectrum':
            lsf_ha = float(np.interp(6562.85*(1+vs[i]/_C), wave, lsf[i], left=np.nan, right=np.nan))
        else:
            lsf_ha = float(np.interp(6562.85, cat_w, cat_lsf))
        if not (np.isfinite(lsf_ha) and lsf_ha > 0):
            continue
        # Keep the intrinsic variance physical, while respecting observed Halpha sigma bounds.
        ha_min = max(0.0, ss[i] - dispersion_tolerance)
        ha_max = ss[i] + dispersion_tolerance
        var_lo = max(0.0, ha_min**2 - lsf_ha**2)
        var_hi = max(0.0, ha_max**2 - lsf_ha**2)
        if var_hi <= var_lo:
            # No physical intrinsic variance satisfies the requested bounds.
            # Rather than fabricate a result, skip the fiber.
            if verbose:
                print(f'PEK fiber {i}: no physical dispersion interval')
            continue
        var0 = np.clip(max(0.0, ss[i]**2-lsf_ha**2),var_lo,var_hi)

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
            low = rest.min()*(1+vs[i]/_C) - fit_padding
            high = rest.max()*(1+vs[i]/_C) + fit_padding
            a,b = np.searchsorted(wave,[low,high])
            if b-a < min_valid_pixels:
                continue
            fit_good = (~bad[i,a:b] & np.isfinite(flux[i,a:b]) &
                        np.isfinite(err[i,a:b]) & (err[i,a:b] > 0))
            if fit_good.sum() < max(min_valid_pixels,n+4):
                continue
            x,y,ey = wave[a:b][fit_good],flux[i,a:b][fit_good],err[i,a:b][fit_good]
            # Robust initial constant background, from outside predicted cores.
            predicted = rest*(1+vs[i]/_C)
            core = np.any(np.abs(x[:,None]-predicted[None,:]) < 3*np.sqrt(var0+inst[None,:]**2),axis=1)
            background0 = np.median(y[~core]) if (~core).sum() >= 3 else np.median(y)
            starting_flux = []
            for key in names:
                candidates = [_table_value(tab_PM,i,'flux',key,'_pm'),
                              _table_value(tab_fe,i,'flux',key)]
                valid = [v for v in candidates if np.isfinite(v)]
                starting_flux.append(float(np.mean(valid)) if valid else 0.0)
            starting_flux = np.asarray(starting_flux)

            # Shared velocity and intrinsic variance, signed integrated fluxes,
            # and one unconstrained constant background.
            def gaussian_matrix(v, v_intr):
                sig = np.sqrt(v_intr + inst**2)
                centers = rest*(1+v/_C)
                dx = (x[:,None]-centers[None,:])/sig[None,:]
                return np.exp(-0.5*dx**2)/(_ROOT2PI*sig[None,:])

            def residual(p):
                return (gaussian_matrix(p[n],p[n+1]) @ p[:n] + p[n+2] - y)/ey

            p0 = np.r_[starting_flux,vs[i],var0,background0]
            lo_bounds = np.r_[np.full(n,-np.inf),vs[i]-velocity_tolerance,var_lo,-np.inf]
            hi_bounds = np.r_[np.full(n,np.inf),vs[i]+velocity_tolerance,var_hi,np.inf]
            try:
                fit = least_squares(residual, p0, bounds=(lo_bounds,hi_bounds),
                                    x_scale='jac',max_nfev=max_nfev, method='trf')
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if not fit.success or not np.all(np.isfinite(fit.x)):
                continue
            p = fit.x
            sig_out = np.sqrt(p[n+1]+inst**2)
            # Heteroscedastic Fisher covariance; fitted constant is marginalized.
            try:
                _, singular, vt = np.linalg.svd(fit.jac, full_matrices=False)
                if len(singular) != n+3 or singular[-1] <= singular[0]*1.e-10:
                    raise np.linalg.LinAlgError('Rank-deficient Gaussian fit')
                cov = (vt.T * (singular**-2)) @ vt
                uncertainty = np.sqrt(np.maximum(np.diag(cov),0))
            except np.linalg.LinAlgError:
                uncertainty = np.full(n+3,np.nan)
            for j,key in enumerate(names):
                prefix = lambda field: f'{field}_{key}_pek'
                result[prefix('flux')][i] = p[j]
                result[prefix('e_flux')][i] = uncertainty[j]
                result[prefix('vel')][i] = p[n]
                result[prefix('e_vel')][i] = uncertainty[n]
                result[prefix('disp')][i] = sig_out[j]
                result[prefix('e_disp')][i] = (uncertainty[n+1]/(2*sig_out[j])
                                              if sig_out[j] > 0 else np.nan)
                # EW from PM/NP continuum, not inferred from gas-only spectrum.
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
                        result[prefix('EW')][i] = -p[j]/abs(continuum)
                        result[prefix('e_EW')][i] = uncertainty[j]/abs(continuum)
                # Model is emission only; never sum cluster constant offsets.
                xx = (wave[a:b]-rest[j]*(1+p[n]/_C))/sig_out[j]
                model[i,a:b] += p[j]/(_ROOT2PI*sig_out[j])*np.exp(-0.5*xx**2)
            if verbose:
                reduced = np.sum(fit.fun**2)/max(len(y)-len(p),1)
                print(f'PEK fiber {i} cluster {names[0]}..{names[-1]}: '
                      f'chi2_red={reduced:.3f}, v={p[n]:.2f} km/s')

    model[bad | ~np.isfinite(flux) | ~np.isfinite(err) | (err<=0)] = np.nan
    return result, model[0] if single else model
