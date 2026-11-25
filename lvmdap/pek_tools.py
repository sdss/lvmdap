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
from scipy.interpolate import griddata
from lvmdap.linked_line_fitter import fit_linked_lines

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


def _measurement_column(tab, field, linekey, suffix=''):
    """Read PM/NP values, accepting old filenames missing a trailing zero."""
    if tab is None:
        return None
    names = tab.colnames if hasattr(tab, 'colnames') else tab.keys()
    key = f'{field}_{linekey}{suffix}'
    if key not in names:
        prefix, w = linekey.rsplit('_', 1)
        pattern = re.compile(rf'^{re.escape(field)}_{re.escape(prefix)}_(\d+(?:\.\d+)?){re.escape(suffix)}$')
        candidates = [n for n in names if (m := pattern.match(n)) is not None
                      and abs(float(m.group(1)) - float(w)) < 0.004]
        if len(candidates) != 1:
            return None
        key = candidates[0]
    return tab[key]


def _table_value(tab, i, field, linekey, suffix='', default=np.nan):
    column = _measurement_column(tab, field, linekey, suffix)
    if column is None:
        return default
    try:
        v = float(column[i])
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


def weighted_np_velocity(tab_np, names):
    """Return the cluster NP velocity and propagated velocity uncertainty."""
    def measurements(field):
        columns = [_measurement_column(tab_np, field, name) for name in names]
        return np.column_stack([
            np.ma.asarray(col, dtype=float).filled(np.nan) if col is not None
            else np.full(len(tab_np), np.nan) for col in columns
        ])

    flux = measurements('flux')
    e_flux = measurements('e_flux')
    velocity = measurements('vel')
    e_velocity = measurements('e_vel')
    eligible = (np.isfinite(flux) & np.isfinite(e_flux) & (e_flux > 0)
                & (flux >= 3 * e_flux) & np.isfinite(velocity))
    weights = np.where(eligible, flux, 0.0)
    total = weights.sum(axis=1)
    use = total > 0
    mean = np.full(len(tab_np), np.nan)
    error = mean.copy()
    mean[use] = (weights * np.where(eligible, velocity, 0.0)).sum(axis=1)[use] / total[use]
    valid_error = np.isfinite(e_velocity) & (e_velocity >= 0)
    known = use & np.all(~eligible | valid_error, axis=1)
    weighted_error = weights * np.where(valid_error, e_velocity, 0.0)
    error[known] = np.linalg.norm(weighted_error, axis=1)[known] / total[known]
    return mean, error


def add_weighted_np_velocities(tab_np, tab_el, max_cluster_span, max_adjacent_distance):
    """Append cluster diagnostics to the existing per-line NP table."""
    line_names = [_line_key(line) for line in tab_el]
    for group in _clusters(tab_el, max_cluster_span, max_adjacent_distance):
        names = [line_names[k] for k in group]
        velocity, error = weighted_np_velocity(tab_np, names)
        for name in names:
            tab_np[f'vel_w_{name}'] = velocity.copy()
            tab_np[f'e_vel_w_{name}'] = error.copy()


def _weighted_velocity_bounds(velocity, error, halfwidth, map_bounds):
    if np.isfinite(error):
        halfwidth = max(halfwidth, 3 * error)
    bounds = (velocity - halfwidth, velocity + halfwidth)
    if map_bounds is None:
        return bounds, False
    lo, hi = max(bounds[0], map_bounds[0]), min(bounds[1], map_bounds[1])
    # The optimizer requires a nonzero interval.
    return ((lo, hi), False) if lo < hi else (map_bounds, True)


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
    """Solve signed Gaussian fluxes directly at fixed kinematics.

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
    design = profiles
    coefficients, _, rank, _ = np.linalg.lstsq(
        design / ey[:, None], y / ey, rcond=None
    )
    return coefficients, rank, design


def fit_pek_rss(gas_spectrum, error_spectrum, vel_interp, disp_interp,
                tab_el, tab_PM, tab_fe, *, wl, lsf_spectrum=None,
                bad_pixel_mask=None, velocity_tolerance=None,
                dispersion_tolerance=None, fit_padding=20.0,
                max_cluster_span=75.0, max_adjacent_distance=25.0,
                min_valid_pixels=8, max_nfev=120, verbose=False,
                reference_line='Halpha_6562.85', fixed_velocity=None,
                map_velocity_bounds=None, restart_velocity=None,
                velocity_start='np_weighted', np_weighted_halfwidth=20.0):
    """Refine the PM Gaussian solution with the common SciPy linked-line fitter.

    The default start is the flux-weighted NP cluster velocity. Its bounds
    intersect the gas-map interval when available. The PM start and original
    error-defined bounds are used when no valid weighted seed is available
    or when velocity_start='pm'. Explicit fixed velocities take precedence.

    The optimized model is gas-only (no hidden background) and uses the same
    input spectrum, error vector and bad-pixel mask supplied by the DAP.
    After optimization, the SciPy candidate and the saved PM Gaussian model
    are compared on exactly the same usable pixels.  When PM is valid, PEK
    retains the SciPy result only if it has a strictly smaller chi-square;
    otherwise the PEK table/model are exact copies of PM for that cluster.
    Thus a valid PM cluster cannot be degraded by PEK.

    With a finite map bound and a supplied restart velocity, Halpha
    groups can retry from the map when their starts differ by >=10 km/s.
    The restart must improve the cluster and the local Halpha residual.

    Line fluxes remain signed and are solved linearly by ``fit_linked_lines``;
    the NP S/N cut applies only to the starting velocity.
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
    if (wave.shape != (nw,) or not np.all(np.isfinite(wave))
            or not np.all(np.diff(wave) > 0)):
        raise ValueError('wl must be a finite, strictly increasing 1D wavelength array')
    if len(tab_PM) != nf or len(tab_fe) != nf:
        raise ValueError('PM and NP tables must have exactly one row per RSS fiber in the same order')

    if velocity_tolerance is None or dispersion_tolerance is None:
        raise ValueError(
            'PEK requires YAML values for pek_velocity_tolerance and '
            'pek_dispersion_tolerance')
    velocity_tolerance = float(velocity_tolerance)
    dispersion_tolerance = float(dispersion_tolerance)
    fit_padding = float(fit_padding)
    if (not np.isfinite(velocity_tolerance) or velocity_tolerance <= 0 or
            not np.isfinite(dispersion_tolerance) or dispersion_tolerance <= 0 or
            not np.isfinite(fit_padding) or fit_padding <= 0):
        raise ValueError('PEK YAML tolerances and fit_padding must be positive finite values')

    if velocity_start not in ('pm', 'np_weighted'):
        raise ValueError("pek_velocity_start must be 'pm' or 'np_weighted'")
    np_weighted_halfwidth = float(np_weighted_halfwidth)
    if not np.isfinite(np_weighted_halfwidth) or np_weighted_halfwidth <= 0:
        raise ValueError('pek_np_weighted_halfwidth must be positive and finite')

    vs = np.broadcast_to(np.asarray(vel_interp, float), (nf,)).copy()
    ss = np.broadcast_to(np.asarray(disp_interp, float), (nf,))
    fixed_vel = (np.full(nf, np.nan) if fixed_velocity is None else
                 np.broadcast_to(np.asarray(fixed_velocity, float), (nf,)))
    fixed_now = np.isfinite(fixed_vel)
    if map_velocity_bounds is not None:
        map_velocity_bounds = np.broadcast_to(np.asarray(map_velocity_bounds, float), (nf, 2))
    restart = (None if restart_velocity is None else
               np.broadcast_to(np.asarray(restart_velocity, float), (nf,)))
    vs[fixed_now] = fixed_vel[fixed_now]

    if bad_pixel_mask is None:
        bad = np.zeros((nf, nw), dtype=bool)
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
            lsf = np.broadcast_to(lsf, (nf, nw))
        if lsf.shape != (nf, nw):
            raise ValueError(
                'lsf_spectrum must be (n_wave,) or (n_fiber,n_wave) '
                'in Gaussian sigma Angstrom')
        lsf_mode = 'spectrum'

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
    w_lines = np.asarray(tab_el['wl'], float)

    result = Table()
    for name in ('id_fib', 'id'):
        if name in tab_PM.colnames:
            result[name] = tab_PM[name].copy()
    if 'id_fib' not in result.colnames:
        result['id_fib'] = np.arange(nf, dtype=int)
    for line_key in line_names:
        for field in _FIELDS:
            result[f'{field}_{line_key}_pek'] = np.full(nf, np.nan)
    weighted = {}
    if velocity_start == 'np_weighted':
        for cluster in groups:
            names = [line_names[k] for k in cluster]
            weighted[cluster[0]] = weighted_np_velocity(tab_fe, names)
        for key in line_names:
            result[f'vel_w_conflict_{key}_pek'] = np.zeros(nf, dtype=bool)
    model = np.zeros((nf, nw), float)

    def lsf_row(i):
        if lsf_mode == 'spectrum':
            return np.asarray(lsf[i], float)
        return np.interp(wave, cat_w, cat_lsf)

    def measurement_margin(table, i, key, suffix, field, fallback):
        value = _table_value(table, i, field, key, suffix)
        error = _table_value(table, i, f'e_{field}', key, suffix)
        if not np.isfinite(value):
            return None
        margin = error if np.isfinite(error) and error > 0 else fallback
        return float(value), float(margin)

    def intrinsic_sigma_interval(obs_sigma, obs_margin, center, inst_sigma):
        if not (np.isfinite(obs_sigma) and obs_sigma > 0 and
                np.isfinite(obs_margin) and obs_margin > 0 and
                np.isfinite(center) and center > 0 and
                np.isfinite(inst_sigma) and inst_sigma > 0):
            return None
        lo_obs = max(0.0, obs_sigma - obs_margin)
        hi_obs = obs_sigma + obs_margin
        lo = _C * np.sqrt(max(0.0, lo_obs**2 - inst_sigma**2)) / center
        hi = _C * np.sqrt(max(0.0, hi_obs**2 - inst_sigma**2)) / center
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            return None
        return float(lo), float(hi)

    def pm_cluster_arrays(i, names, rest):
        f = np.array([_table_value(tab_PM, i, 'flux', key, '_pm') for key in names], float)
        ef = np.array([_table_value(tab_PM, i, 'e_flux', key, '_pm') for key in names], float)
        v = np.array([_table_value(tab_PM, i, 'vel', key, '_pm') for key in names], float)
        ev = np.array([_table_value(tab_PM, i, 'e_vel', key, '_pm') for key in names], float)
        d = np.array([_table_value(tab_PM, i, 'disp', key, '_pm') for key in names], float)
        ed = np.array([_table_value(tab_PM, i, 'e_disp', key, '_pm') for key in names], float)
        ew = np.array([_table_value(tab_PM, i, 'EW', key, '_pm') for key in names], float)
        eew = np.array([_table_value(tab_PM, i, 'e_EW', key, '_pm') for key in names], float)
        valid = (np.all(np.isfinite(f)) and np.all(np.isfinite(v)) and
                 np.all(np.isfinite(d)) and np.all(d > 0))
        return valid, f, ef, v, ev, d, ed, ew, eew

    def gaussian_model(grid, rest, fluxes, velocities, dispersions):
        out = np.zeros_like(grid, dtype=float)
        for rw, ff, vv, dd in zip(rest, fluxes, velocities, dispersions):
            if not (np.isfinite(ff) and np.isfinite(vv) and
                    np.isfinite(dd) and dd > 0):
                return None
            center = rw * (1.0 + vv / _C)
            dx = (grid - center) / dd
            out += ff / (_ROOT2PI * dd) * np.exp(-0.5 * dx * dx)
        return out

    def store_pm(i, names, rest, pm_values, a, b):
        valid, f, ef, v, ev, d, ed, ew, eew = pm_values
        if not valid:
            return False
        for j, key in enumerate(names):
            pref = lambda field: f'{field}_{key}_pek'
            result[pref('flux')][i] = f[j]
            result[pref('e_flux')][i] = ef[j]
            result[pref('vel')][i] = v[j]
            result[pref('e_vel')][i] = ev[j]
            result[pref('disp')][i] = d[j]
            result[pref('e_disp')][i] = ed[j]
            result[pref('EW')][i] = ew[j]
            result[pref('e_EW')][i] = eew[j]
        pm_mod = gaussian_model(wave[a:b], rest, f, v, d)
        if pm_mod is not None:
            model[i, a:b] += pm_mod
        return True

    for i in range(nf):
        if verbose:
            print(f"\n### Fitting PEK lines {i+1}/{nf}")
        else:
            print_progress_bar(i, nf, label='Fitting PEK lines', bar_width=40)

        row_lsf = lsf_row(i)
        if (row_lsf.shape != wave.shape or
                np.any(~np.isfinite(row_lsf)) or np.any(row_lsf <= 0)):
            continue

        for cluster in groups:
            rest = w_lines[cluster]
            names = [line_names[k] for k in cluster]
            pm_values = pm_cluster_arrays(i, names, rest)
            pm_valid, pm_flux, pm_eflux, pm_vel, pm_evel, pm_disp, pm_edisp, pm_ew, pm_eew = pm_values

            finite_pm_v = pm_vel[np.isfinite(pm_vel)]
            initial_velocity = (float(np.median(finite_pm_v)) if finite_pm_v.size
                                else float(vs[i]))
            local_bounds = None
            if map_velocity_bounds is not None and np.all(np.isfinite(map_velocity_bounds[i])) and not fixed_now[i]:
                local_bounds = tuple(map_velocity_bounds[i])
                initial_velocity = (np.clip(initial_velocity, *local_bounds)
                                    if np.isfinite(initial_velocity) else np.mean(local_bounds))
            weighted_bounds = None
            if cluster[0] in weighted and not fixed_now[i]:
                weighted_vel, weighted_error = weighted[cluster[0]]
                if np.isfinite(weighted_vel[i]):
                    weighted_bounds, conflict = _weighted_velocity_bounds(
                        weighted_vel[i], weighted_error[i], np_weighted_halfwidth, local_bounds)
                    initial_velocity = float(np.clip(weighted_vel[i], *weighted_bounds))
                    if conflict:
                        for key in names:
                            result[f'vel_w_conflict_{key}_pek'][i] = True
            if fixed_now[i]:
                initial_velocity = float(fixed_vel[i])
            if not np.isfinite(initial_velocity):
                continue

            intrinsic_starts = []
            for key, rw, vv, dd in zip(names, rest, pm_vel, pm_disp):
                if not (np.isfinite(vv) and np.isfinite(dd) and dd > 0):
                    continue
                center = rw * (1.0 + vv / _C)
                inst = float(np.interp(center, wave, row_lsf, left=np.nan, right=np.nan))
                if not (np.isfinite(inst) and inst > 0 and center > 0):
                    continue
                intrinsic_starts.append(
                    _C * np.sqrt(max(0.0, dd**2 - inst**2)) / center)

            if intrinsic_starts:
                initial_sigma_kms = float(np.median(intrinsic_starts))
            elif np.isfinite(ss[i]) and ss[i] > 0:
                center = reference_wavelength * (1.0 + initial_velocity / _C)
                inst = float(np.interp(center, wave, row_lsf, left=np.nan, right=np.nan))
                if not (np.isfinite(inst) and inst > 0):
                    if pm_valid:
                        low = rest.min() * (1.0 + initial_velocity/_C) - fit_padding
                        high = rest.max() * (1.0 + initial_velocity/_C) + fit_padding
                        a, b = np.searchsorted(wave, [low, high])
                        store_pm(i, names, rest, pm_values, a, b)
                    continue
                initial_sigma_kms = (
                    _C * np.sqrt(max(0.0, ss[i]**2 - inst**2)) / center)
            else:
                if pm_valid:
                    low = rest.min() * (1.0 + initial_velocity/_C) - fit_padding
                    high = rest.max() * (1.0 + initial_velocity/_C) + fit_padding
                    a, b = np.searchsorted(wave, [low, high])
                    store_pm(i, names, rest, pm_values, a, b)
                continue

            # PM/NP +/- their own 1-sigma errors define the velocity interval.
            # The YAML tolerance is used only when an individual reported
            # uncertainty is unavailable/nonpositive.
            velocity_ranges = []
            if weighted_bounds is not None:
                velocity_bounds = weighted_bounds
            elif local_bounds is not None:
                velocity_bounds = local_bounds
            elif not fixed_now[i]:
                for key in names:
                    for table, suffix in ((tab_PM, '_pm'), (tab_fe, '')):
                        measurement = measurement_margin(
                            table, i, key, suffix, 'vel', velocity_tolerance)
                        if measurement is not None:
                            value, margin = measurement
                            velocity_ranges.append((value-margin, value+margin))
                if not velocity_ranges and np.isfinite(vs[i]):
                    velocity_ranges.append(
                        (vs[i]-velocity_tolerance, vs[i]+velocity_tolerance))
                if not velocity_ranges:
                    if pm_valid:
                        low = rest.min() * (1.0 + initial_velocity/_C) - fit_padding
                        high = rest.max() * (1.0 + initial_velocity/_C) + fit_padding
                        a, b = np.searchsorted(wave, [low, high])
                        store_pm(i, names, rest, pm_values, a, b)
                    continue
                velocity_bounds = (
                    min(lo for lo, _ in velocity_ranges),
                    max(hi for _, hi in velocity_ranges))
                # The PM starting point must remain inside the allowed domain.
                velocity_bounds = (
                    min(velocity_bounds[0], initial_velocity),
                    max(velocity_bounds[1], initial_velocity))
            else:
                velocity_bounds = None

            # Build the analogous intrinsic-sigma interval from observed
            # PM/NP widths and their e_disp values, deconvolving the actual LSF
            # at each measurement's observed centroid.
            sigma_ranges = []
            for key, rw in zip(names, rest):
                for table, suffix in ((tab_PM, '_pm'), (tab_fe, '')):
                    disp_measurement = measurement_margin(
                        table, i, key, suffix, 'disp', dispersion_tolerance)
                    if disp_measurement is None:
                        continue
                    disp_value, disp_margin = disp_measurement
                    vel_value = _table_value(table, i, 'vel', key, suffix)
                    if not np.isfinite(vel_value):
                        vel_value = initial_velocity
                    center = rw * (1.0 + vel_value / _C)
                    inst = float(np.interp(
                        center, wave, row_lsf, left=np.nan, right=np.nan))
                    interval = intrinsic_sigma_interval(
                        disp_value, disp_margin, center, inst)
                    if interval is not None:
                        sigma_ranges.append(interval)

            if not sigma_ranges and np.isfinite(ss[i]) and ss[i] > 0:
                center = reference_wavelength * (1.0 + initial_velocity / _C)
                inst = float(np.interp(
                    center, wave, row_lsf, left=np.nan, right=np.nan))
                interval = intrinsic_sigma_interval(
                    ss[i], dispersion_tolerance, center, inst)
                if interval is not None:
                    sigma_ranges.append(interval)

            if not sigma_ranges:
                if pm_valid:
                    low = rest.min() * (1.0 + initial_velocity/_C) - fit_padding
                    high = rest.max() * (1.0 + initial_velocity/_C) + fit_padding
                    a, b = np.searchsorted(wave, [low, high])
                    store_pm(i, names, rest, pm_values, a, b)
                continue

            sigma_bounds = (
                min(lo for lo, _ in sigma_ranges),
                max(hi for _, hi in sigma_ranges))
            sigma_bounds = (
                min(sigma_bounds[0], initial_sigma_kms),
                max(sigma_bounds[1], initial_sigma_kms))
            if not np.isfinite(sigma_bounds[1]) or sigma_bounds[1] <= sigma_bounds[0]:
                if pm_valid:
                    low = rest.min() * (1.0 + initial_velocity/_C) - fit_padding
                    high = rest.max() * (1.0 + initial_velocity/_C) + fit_padding
                    a, b = np.searchsorted(wave, [low, high])
                    store_pm(i, names, rest, pm_values, a, b)
                continue

            # Preserve the pre-existing geometric requirement: every line in
            # the cluster must have at least one valid sample in its predicted
            # half-maximum core.  When PM is valid but PEK cannot be evaluated,
            # retain PM rather than creating a partial hybrid PEK cluster.
            window_low = rest.min() * (1.0 + initial_velocity/_C) - fit_padding
            window_high = rest.max() * (1.0 + initial_velocity/_C) + fit_padding
            a, b = np.searchsorted(wave, [window_low, window_high])
            if b-a < min_valid_pixels:
                if pm_valid:
                    store_pm(i, names, rest, pm_values, a, b)
                continue
            usable = (~bad[i, a:b] & np.isfinite(flux[i, a:b]) &
                      np.isfinite(err[i, a:b]) & (err[i, a:b] > 0) &
                      np.isfinite(row_lsf[a:b]) & (row_lsf[a:b] > 0))
            if usable.sum() < min_valid_pixels:
                if pm_valid:
                    store_pm(i, names, rest, pm_values, a, b)
                continue

            grid = wave[a:b][usable]
            centers0 = rest * (1.0 + initial_velocity/_C)
            inst0 = np.interp(centers0, wave, row_lsf)
            sig0 = np.sqrt(inst0**2 + (centers0*initial_sigma_kms/_C)**2)
            core_halfwidth = np.sqrt(2.0*np.log(2.0)) * sig0
            has_core = np.any(
                np.abs(grid[:, None]-centers0[None, :]) <= core_halfwidth[None, :],
                axis=0)
            # Preserve the old allowance for an NP-shifted core.
            for j, key in enumerate(names):
                np_v = _table_value(tab_fe, i, 'vel', key)
                if np.isfinite(np_v):
                    np_center = rest[j] * (1.0 + np_v/_C)
                    has_core[j] |= np.any(
                        np.abs(grid-np_center) <= core_halfwidth[j])
            if not np.all(has_core):
                if pm_valid:
                    store_pm(i, names, rest, pm_values, a, b)
                continue

            fixed = ({0: float(fixed_vel[i])} if fixed_now[i] else None)
            fit_options = dict(
                fixed_velocities=fixed,
                initial_sigma_kms=initial_sigma_kms,
                velocity_bounds=velocity_bounds,
                sigma_kms_bounds=sigma_bounds,
                bad_pixel_mask=bad[i],
                fit_padding=fit_padding,
                max_nfev=max_nfev,
                background_degree=-1,
            )
            candidate = fit_linked_lines(
                wave, flux[i], err[i], rest, row_lsf,
                initial_velocity=initial_velocity, **fit_options)

            def valid_fit(fit):
                return (fit.get('success', False) and
                        np.all(np.isfinite(fit['flux'])) and
                        np.all(np.isfinite(fit['vel'])) and
                        np.all(np.isfinite(fit['disp'])) and
                        np.all(fit['disp'] > 0))

            candidate_valid = valid_fit(candidate)

            # Compare candidate and PM on the SAME +/-3-sigma common window
            # later used by BEST/X_sq.  Include NP in the window definition
            # when its whole cluster is measurable, because BEST also uses the
            # union of every valid NP/PM/PEK candidate to define those pixels.
            # The PM and PEK models are still compared with identical errors,
            # masks and degrees of freedom, so raw chi-square is sufficient for
            # deciding whether the SciPy refinement improves PM.
            candidate_chi = np.inf
            pm_chi = np.inf
            lo_score = np.inf
            hi_score = -np.inf

            def expand_score_window(velocities, dispersions):
                nonlocal lo_score, hi_score
                for rest_wave, velocity_now, dispersion_now in zip(
                        rest, velocities, dispersions):
                    if (not np.isfinite(velocity_now) or
                            not np.isfinite(dispersion_now) or
                            dispersion_now <= 0):
                        return False
                    center_now = rest_wave * (1.0 + velocity_now/_C)
                    lo_score = min(lo_score, center_now - 3.0*dispersion_now)
                    hi_score = max(hi_score, center_now + 3.0*dispersion_now)
                return True

            if pm_valid:
                expand_score_window(pm_vel, pm_disp)
            if candidate_valid:
                expand_score_window(candidate['vel'], candidate['disp'])

            np_vel = np.array([
                _table_value(tab_fe, i, 'vel', key) for key in names], float)
            np_disp = np.array([
                _table_value(tab_fe, i, 'disp', key) for key in names], float)
            np_flux = np.array([
                _table_value(tab_fe, i, 'flux', key) for key in names], float)
            np_valid = (np.all(np.isfinite(np_flux)) and
                        np.all(np.isfinite(np_vel)) and
                        np.all(np.isfinite(np_disp)) and
                        np.all(np_disp > 0))
            if np_valid:
                expand_score_window(np_vel, np_disp)

            common = (
                (wave >= lo_score) & (wave <= hi_score) &
                ~bad[i] & np.isfinite(flux[i]) & np.isfinite(err[i]) &
                (err[i] > 0))
            if candidate_valid and np.any(common):
                residual = (
                    flux[i, common] - candidate['model'][common]) / err[i, common]
                candidate_chi = float(np.dot(residual, residual))
            if pm_valid and np.any(common):
                pm_model_common = gaussian_model(
                    wave[common], rest, pm_flux, pm_vel, pm_disp)
                if pm_model_common is not None:
                    residual = (
                        flux[i, common] - pm_model_common) / err[i, common]
                    pm_chi = float(np.dot(residual, residual))

            use_candidate = candidate_valid and (
                not pm_valid or candidate_chi < pm_chi)

            if restart is not None and local_bounds is not None and np.isfinite(restart[i]):
                new_start = float(np.clip(restart[i], *velocity_bounds))
                if 'Halpha_6562.85' in names and abs(new_start - initial_velocity) >= 10.0:
                    j = names.index('Halpha_6562.85')
                    retry = fit_linked_lines(
                        wave, flux[i], err[i], rest, row_lsf,
                        initial_velocity=new_start, fit_window_velocity=initial_velocity,
                        **fit_options)
                    if valid_fit(retry):
                        if use_candidate or pm_valid:
                            base_vel = candidate['vel'] if use_candidate else pm_vel
                            base_disp = candidate['disp'] if use_candidate else pm_disp
                            base_model = (candidate['model'] if use_candidate else
                                          gaussian_model(wave, rest, pm_flux, pm_vel, pm_disp))
                            expand_score_window(retry['vel'], retry['disp'])
                            compare = ((wave >= max(lo_score, window_low)) &
                                       (wave <= min(hi_score, window_high)) &
                                       ~bad[i] & np.isfinite(flux[i]) &
                                       np.isfinite(err[i]) & (err[i] > 0) &
                                       np.isfinite(row_lsf) & (row_lsf > 0))
                            residual_base = (flux[i, compare] - base_model[compare]) / err[i, compare]
                            residual_retry = (flux[i, compare] - retry['model'][compare]) / err[i, compare]
                            improvement = residual_base**2 - residual_retry**2
                            accepted = improvement.sum() > 1.0
                            centers = rest[j] * (1.0 + np.array([base_vel[j], retry['vel'][j]])/_C)
                            sigmas = np.array([base_disp[j], retry['disp'][j]])
                            local_wave = wave[compare]
                            local = ((local_wave >= np.min(centers - 3.0*sigmas)) &
                                     (local_wave <= np.max(centers + 3.0*sigmas)))
                            accepted &= improvement[local].sum() > 0
                        else:
                            accepted = True
                        if accepted:
                            candidate = retry
                            use_candidate = True
                            if verbose:
                                print(f'PEK fiber {i} cluster {names[0]}..{names[-1]}: '
                                      'accept spatial velocity restart')

            if not use_candidate:
                if pm_valid:
                    store_pm(i, names, rest, pm_values, a, b)
                    if verbose:
                        print(f'PEK fiber {i} cluster {names[0]}..{names[-1]}: '
                              f'retain PM (chi2_PM={pm_chi:.4g}, '
                              f'chi2_SciPy={candidate_chi:.4g})')
                continue

            for j, key in enumerate(names):
                pref = lambda field: f'{field}_{key}_pek'
                result[pref('flux')][i] = candidate['flux'][j]
                result[pref('e_flux')][i] = candidate['e_flux'][j]
                result[pref('vel')][i] = candidate['vel'][j]
                result[pref('e_vel')][i] = candidate['e_vel'][j]
                result[pref('disp')][i] = candidate['disp'][j]
                result[pref('e_disp')][i] = candidate['e_disp'][j]

                continua = []
                for table, suff in ((tab_PM, '_pm'), (tab_fe, '')):
                    original_f = _table_value(table, i, 'flux', key, suff)
                    original_ew = _table_value(table, i, 'EW', key, suff)
                    if (np.isfinite(original_f) and np.isfinite(original_ew)
                            and abs(original_ew) > 1e-20):
                        continua.append(-original_f/original_ew)
                if continua:
                    continuum = float(np.median(continua))
                    if abs(continuum) > 1e-20:
                        result[pref('EW')][i] = -candidate['flux'][j]/abs(continuum)
                        if np.isfinite(candidate['e_flux'][j]):
                            result[pref('e_EW')][i] = (
                                candidate['e_flux'][j]/abs(continuum))

            model[i] += candidate['model']
            if verbose:
                print(f'PEK fiber {i} cluster {names[0]}..{names[-1]}: '
                      f'SciPy improves PM '
                      f'(chi2_PM={pm_chi:.4g}, chi2_SciPy={candidate_chi:.4g})')

    model[bad | ~np.isfinite(flux) | ~np.isfinite(err) | (err <= 0)] = np.nan
    return result, model[0] if single else model
