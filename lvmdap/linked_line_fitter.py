"""Linked/fixed Gaussian emission-line fit with empirical instrumental LSF.

Independent of FIT3D and astropy. The default flux coefficients are signed,
independent integrated line fluxes; ties/fixed values are explicit options.
Noise weights and masked samples are always those provided by the caller.
"""

import numpy as np
from scipy.optimize import least_squares

_C = 299792.458
_ROOT2PI = np.sqrt(2.0 * np.pi)


def _flux_mapping(n, links, fixed):
    """Linear map F = M @ a + b for fixed and ratio-linked line fluxes."""
    links = {} if links is None else dict(links)
    fixed = {} if fixed is None else dict(fixed)
    for i in list(links) + list(fixed):
        if not isinstance(i, (int, np.integer)) or not 0 <= i < n:
            raise IndexError(f'Invalid line index {i!r}')
    if set(links) & set(fixed):
        raise ValueError('A line cannot be both fixed and ratio-linked')
    memo = {}
    def resolve(i, active):
        if i in active:
            raise ValueError('Cycle in flux_links')
        if i in memo:
            return memo[i]
        if i in fixed:
            value = float(fixed[i])
            if not np.isfinite(value):
                raise ValueError('Fixed flux must be finite')
            result = (None, 0.0, value)
        elif i in links:
            j, ratio = links[i]
            j = int(j)
            if not 0 <= j < n:
                raise IndexError('Flux link references nonexistent line')
            ratio = float(ratio)
            if not np.isfinite(ratio):
                raise ValueError('Flux link ratio must be finite')
            root, scale, constant = resolve(j, active | {i})
            result = (root, ratio * scale, ratio * constant)
        else:
            result = (i, 1.0, 0.0)
        memo[i] = result
        return result
    resolved = [resolve(i, set()) for i in range(n)]
    roots = sorted({item[0] for item in resolved if item[0] is not None})
    M = np.zeros((n, len(roots)), dtype=float)
    b = np.zeros(n, dtype=float)
    for i, (root, scale, constant) in enumerate(resolved):
        if root is not None:
            M[i, roots.index(root)] = scale
        b[i] = constant
    return M, b


def fit_linked_lines(wl, flux, eflux, rest_waves, lsf_spectrum, *,
                     initial_velocity, initial_sigma_kms=30.0,
                     velocity_groups=None, dispersion_groups=None,
                     fixed_velocities=None, fixed_sigmas_kms=None,
                     velocity_bounds=None, sigma_kms_bounds=None,
                     flux_links=None, fixed_fluxes=None,
                     background_degree=0, fit_padding=30.0,
                     bad_pixel_mask=None, max_nfev=150):
    """Fit one blend/cluster using separable weighted least squares.

    Emission lines have Gaussian profiles normalized to integrated flux.
    `velocity_groups` and `dispersion_groups` are per-line group identifiers;
    the same ID ties the corresponding parameters (defaults: all tied).
    `fixed_velocities` and `fixed_sigmas_kms` map group IDs to fixed values.
    Optional velocity/sigma bounds are pairs applied to *all* free groups.
    `flux_links={i:(j,ratio)}` enforces F_i=ratio*F_j; `fixed_fluxes`
    maps line indices to known integrated fluxes. Fluxes remain signed.
    Instrumental sigma(Angstrom) is interpolated from `lsf_spectrum` at
    each trial *observed* line wavelength. Intrinsic dispersion is in km/s.
    `background_degree=-1` disables nuisance background; 0 and 1 include
    a constant or affine background, not returned in the gas-only model.

    Returns a dict with per-line flux/velocity/observed sigma and 1-sigma
    errors, local gas-only model, full model with nuisance background,
    chi-square, degrees of freedom, and solver status. Failed/degenerate
    fits return NaNs rather than inserting fabricated measurements.
    """
    wave = np.asarray(wl, dtype=float)
    data = np.asarray(flux, dtype=float)
    errs = np.asarray(eflux, dtype=float)
    lsf = np.asarray(lsf_spectrum, dtype=float)
    rest = np.asarray(rest_waves, dtype=float)
    n = rest.size
    if not n or rest.ndim != 1 or not np.all(np.isfinite(rest)) or np.any(rest <= 0):
        raise ValueError('rest_waves must be a nonempty 1D array of positive wavelengths')
    if any(a.ndim != 1 or a.shape != wave.shape for a in (data, errs, lsf)):
        raise ValueError('wl, flux, eflux, lsf_spectrum must be same-length 1D arrays')
    if not np.all(np.isfinite(wave)) or not np.all(np.diff(wave) > 0):
        raise ValueError('wl must be finite and strictly increasing')
    bad = (np.zeros(wave.shape, bool) if bad_pixel_mask is None
           else np.asarray(bad_pixel_mask, bool))
    if bad.shape != wave.shape:
        raise ValueError('bad_pixel_mask shape mismatch')
    if background_degree not in (-1, 0, 1):
        raise ValueError('background_degree must be -1, 0 or 1')
    if not np.isfinite(fit_padding) or fit_padding <= 0:
        raise ValueError('fit_padding must be positive')

    vg = [0] * n if velocity_groups is None else list(velocity_groups)
    sg = [0] * n if dispersion_groups is None else list(dispersion_groups)
    if len(vg) != n or len(sg) != n:
        raise ValueError('Group arrays must have one entry per line')
    vnames = list(dict.fromkeys(vg))
    snames = list(dict.fromkeys(sg))
    vfix = {} if fixed_velocities is None else dict(fixed_velocities)
    sfix = {} if fixed_sigmas_kms is None else dict(fixed_sigmas_kms)
    if set(vfix) - set(vnames) or set(sfix) - set(snames):
        raise ValueError('Fixed parameter references an unknown group')
    for group, value in sfix.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError('Fixed intrinsic velocity dispersion must be finite and nonnegative')
    for group, value in vfix.items():
        if not np.isfinite(value):
            raise ValueError('Fixed velocity must be finite')

    mapping, offset = _flux_mapping(n, flux_links, fixed_fluxes)
    nf = mapping.shape[1]
    vfree = [name for name in vnames if name not in vfix]
    sfree = [name for name in snames if name not in sfix]
    nb = background_degree + 1
    v0 = float(initial_velocity)
    s0 = float(initial_sigma_kms)
    if not np.isfinite(v0) or not np.isfinite(s0) or s0 < 0:
        raise ValueError('Initial kinematics must be finite, sigma >= 0')

    # Preserve original mask/errors and select a fixed fit window once.
    centers_initial = rest * (1 + np.array([vfix.get(g, v0) for g in vg]) / _C)
    window = ((wave >= np.min(centers_initial) - fit_padding)
              & (wave <= np.max(centers_initial) + fit_padding))
    usable = (window & ~bad & np.isfinite(data) & np.isfinite(errs)
              & (errs > 0) & np.isfinite(lsf) & (lsf > 0))
    x = wave[usable]
    y = data[usable]
    ey = errs[usable]
    result = dict(flux=np.full(n, np.nan), e_flux=np.full(n, np.nan),
                  vel=np.full(n, np.nan), e_vel=np.full(n, np.nan),
                  disp=np.full(n, np.nan), e_disp=np.full(n, np.nan),
                  model=np.zeros_like(wave), model_full=np.zeros_like(wave),
                  success=False, chi2=np.nan, chi2_red=np.nan,
                  dof=0, n_pixels=len(x), message='')
    npar = len(vfree) + len(sfree) + nf + nb
    if len(x) <= npar:
        result['message'] = 'Insufficient valid pixels'
        return result

    # Only restrict centroids to the measured wavelength range, with no
    # additional astrophysical velocity prior unless explicitly supplied.
    vlow, vhigh = (-np.inf, np.inf) if velocity_bounds is None else map(float, velocity_bounds)
    slow, shigh = (0.0, np.inf) if sigma_kms_bounds is None else map(float, sigma_kms_bounds)
    slow = max(0.0, slow)
    if slow >= shigh or vlow >= vhigh:
        raise ValueError('Invalid kinematic bounds')
    lower, upper = [], []
    for group in vfree:
        selected = rest[np.array([g == group for g in vg])]
        bound_lo = max(vlow, float(np.max(_C * (wave[0] / selected - 1))))
        bound_hi = min(vhigh, float(np.min(_C * (wave[-1] / selected - 1))))
        if bound_lo >= bound_hi:
            result['message'] = 'No velocity interval within wavelength grid'
            return result
        lower.append(bound_lo)
        upper.append(bound_hi)
    lower.extend([slow] * len(sfree))
    upper.extend([shigh] * len(sfree))
    lower = np.array(lower, float)
    upper = np.array(upper, float)
    start = np.array([v0] * len(vfree) + [s0] * len(sfree), float)
    if len(start):
        start = np.minimum(np.maximum(start, lower + 1e-8), upper - 1e-8)

    def kinematics(pars):
        vs = dict(vfix)
        ss = dict(sfix)
        vs.update(zip(vfree, pars[:len(vfree)]))
        ss.update(zip(sfree, pars[len(vfree):]))
        return (np.array([vs[g] for g in vg], float),
                np.array([ss[g] for g in sg], float))

    def profiles(grid, pars):
        v, s = kinematics(pars)
        cen = rest * (1 + v / _C)
        instrumental = np.interp(cen, wave, lsf)
        observed = np.sqrt(instrumental**2 + (cen * s / _C)**2)
        dx = (grid[:, None] - cen[None, :]) / observed[None, :]
        return np.exp(-0.5 * dx * dx) / (_ROOT2PI * observed[None, :]), observed

    xscale = max(float(np.ptp(x)), 1.0)
    def local_matrices(pars):
        p, _ = profiles(x, pars)
        t = [np.ones(len(x))] if nb else []
        if nb == 2:
            t.append((x - np.mean(x)) / xscale)
        bkg = np.column_stack(t) if nb else np.empty((len(x), 0))
        A = np.column_stack((p @ mapping, bkg))
        constant = p @ offset
        return A, constant

    def linear_solution(pars):
        A, constant = local_matrices(pars)
        wA = A / ey[:, None]
        rhs = (y - constant) / ey
        if not A.shape[1]:
            return np.empty(0), (constant - y) / ey, 0
        # Normalize every weighted design column and the right-hand side.
        # This does NOT alter the least-squares objective or the signed flux
        # solution, but prevents a mix of tiny Gaussian basis functions and
        # high-contrast continua from destabilizing the SVD.
        col_scale = np.max(np.abs(wA), axis=0)
        if np.any(~np.isfinite(col_scale)) or np.any(col_scale == 0):
            raise FloatingPointError('Nonfinite or zero weighted design column')
        rhs_scale = max(float(np.max(np.abs(rhs))), 1.0)
        if not np.isfinite(rhs_scale):
            raise FloatingPointError('Nonfinite weighted data')
        normalized = wA / col_scale[None, :]
        coeff_norm, _, rank, _ = np.linalg.lstsq(
            normalized, rhs / rhs_scale, rcond=None)
        with np.errstate(over='raise', invalid='raise'):
            coefs = (coeff_norm * rhs_scale) / col_scale
            resid = (A @ coefs + constant - y) / ey
        if not np.all(np.isfinite(resid)):
            raise FloatingPointError('Nonfinite weighted model residual')
        return coefs, resid, rank

    def objective(pars):
        return linear_solution(pars)[1]

    try:
        if len(start):
            opt = least_squares(objective, start, bounds=(lower, upper),
                                method='trf', loss='linear',
                                max_nfev=int(max_nfev), x_scale='jac')
            solution = opt.x
            result['message'] = str(opt.message)
            result['success'] = bool(opt.success)
        else:
            solution = start
            result['success'] = True
            result['message'] = 'All kinematics fixed'
        coeffs, residuals, rank = linear_solution(solution)
        if rank != (nf + nb):
            result['success'] = False
            result['message'] = 'Linearly dependent flux/background columns'
            return result
        amps = coeffs[:nf]
        f = mapping @ amps + offset
        vel, sig_v = kinematics(solution)
        prof_all, obs_sigma = profiles(wave, solution)
        gas_model = prof_all @ f
        background = np.zeros_like(wave)
        if nb:
            background += coeffs[nf]
        if nb == 2:
            background += coeffs[nf+1] * (wave - np.mean(x)) / xscale
        result.update(flux=f, vel=vel, disp=obs_sigma,
                      model=gas_model, model_full=gas_model + background,
                      chi2=float(np.dot(residuals, residuals)),
                      dof=int(len(x) - npar))
        if result['dof'] > 0:
            result['chi2_red'] = result['chi2'] / result['dof']

        if npar == 0:
            # Fully fixed models have no inferred parameter errors.
            return result

        # Full local covariance: kinetic/line-flux/background correlations.
        # Unlike a conditional linear covariance, this includes the uncertainty
        # of fitted shared centroids/widths (when locally identifiable).
        A, _ = local_matrices(solution)
        jac_cols = [A / ey[:, None]]
        if len(solution):
            derivs = []
            for j in range(len(solution)):
                step = 1e-4 * max(abs(solution[j]), 1.0)
                hi = solution.copy()
                lo = solution.copy()
                hi[j] = min(hi[j] + step, upper[j])
                lo[j] = max(lo[j] - step, lower[j])
                delta = hi[j] - lo[j]
                if delta == 0:
                    derivs.append(np.zeros(len(x)))
                    continue
                ph, _ = profiles(x, hi)
                pl, _ = profiles(x, lo)
                derivs.append(((ph - pl) @ f / delta) / ey)
            jac_cols.append(np.column_stack(derivs))
        # Columns correspond to (independent flux, background, nonlinear).
        J = np.column_stack(jac_cols)
        _, sval, vt = np.linalg.svd(J, full_matrices=False)
        if len(sval) == npar and sval[-1] > sval[0] * 1e-10:
            cov = (vt.T * (1.0 / sval**2)) @ vt
            var_f = np.einsum('ik,kl,il->i', mapping, cov[:nf, :nf], mapping)
            result['e_flux'] = np.sqrt(np.maximum(var_f, 0))
            result['e_flux'][~np.any(mapping != 0, axis=1)] = np.nan
            # Numerical Jacobians of output velocities and observed sigmas.
            Dv = np.zeros((n, len(solution)))
            Ds = np.zeros((n, len(solution)))
            for j in range(len(solution)):
                step = 1e-4 * max(abs(solution[j]), 1.0)
                hi = solution.copy(); lo = solution.copy()
                hi[j] = min(hi[j]+step, upper[j]); lo[j] = max(lo[j]-step, lower[j])
                delta = hi[j] - lo[j]
                if delta:
                    vh, _ = kinematics(hi); vl, _ = kinematics(lo)
                    _, sh = profiles(x[:1], hi); _, sl = profiles(x[:1], lo)
                    Dv[:, j] = (vh-vl)/delta
                    Ds[:, j] = (sh-sl)/delta
            Cnon = cov[nf+nb:, nf+nb:]
            if len(solution):
                result['e_vel'] = np.sqrt(np.maximum(np.einsum('ik,kl,il->i', Dv, Cnon, Dv), 0))
                result['e_disp'] = np.sqrt(np.maximum(np.einsum('ik,kl,il->i', Ds, Cnon, Ds), 0))
                result['e_vel'][np.array([g in vfix for g in vg])] = np.nan
                result['e_disp'][np.array([g in sfix for g in sg])] = np.nan
            else:
                # Exact fixed parameters are not fitted: zero uncertainty is
                # a mathematical conditional error, not a measurement error.
                result['e_vel'] = np.full(n, np.nan)
                result['e_disp'] = np.full(n, np.nan)
        else:
            result['message'] += '; joint covariance rank deficient'
        return result
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        result['message'] = f'Fit failed: {exc}'
        result['success'] = False
        return result
