import numpy as np
from copy import deepcopy as copy
from scipy.signal import medfilt
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d

# local imports
from .io import print_verbose
from .constants import __c__, __shift_convolve_lnwl_frac__, __sigma_to_FWHM__

_STATS_POS = {
    'mean': 0,
    'pRMS': 1,
    'median': 2,
    'min': 3,
    'max': 4,
    'adev': 5,
    'RMS': 6,
}

def pdl_stats(x, ddof=1, bad=np.nan):
    # should return ($mean,$prms,$median,$min,$max,$adev,$rms)
    # MEAN = sum (x)/ N
    # PRMS = sqrt( sum( (x-mean(x))^2 )/(N-1))
    # AADEV = sum( abs(x-mean(x)) )/N
    # RMS = sqrt(sum( (x-mean(x))^2 )/N)
    [xmean, xpRMS, xmedian, xmin, xmax, xadev, xRMS] = [bad]*7
    if np.isfinite(x).any():
        x = x[np.isfinite(x)]
        N = x.size
        Np = N - ddof
        xmean = x.mean()
        d = x-xmean
        sdd = (d**2).sum()
        xRMS = np.sqrt(sdd/N)
        xpRMS = np.sqrt(sdd/Np)
        xmedian = np.median(x)
        xmin = np.min(x)
        xmax = np.max(x)
        xadev = np.sum(np.abs(d))/N
    return [xmean, xpRMS, xmedian, xmin, xmax, xadev, xRMS]


def WLS_invmat(y_obs, y_mod__m, dy=None):
    """
    Fit `y_obs` with linear combination of `y_mod__m` using Weighted
    Least Squares (WLS). An error `dy` (e.g. 1/sigma^2) could be set.

    TODO: Math stuff...

    Parameters
    ----------
    y_obs : array like
        Data to be fitted.
    y_mod__m : array like
        Set of models to fit `y_obs`.
    dy : array like, optional
        Error of `y_obs`. Default is None.

    Returns
    -------
    array like
        Best model of `y_obs`.
    array like
        Best coefficients of `y_mod__m`.
    """
    A = np.asarray(y_mod__m)
    y_obs_mean = y_obs.mean()
    B = y_obs/y_obs_mean
    # from sklearn.linear_model import LinearRegression
    # WLS = LinearRegression()
    # print(A.T, B.shape, dy.shape)
    # WLS.fit(A.T, B, sample_weight=dy)
    # coeffs = WLS.coef_[0]*y_obs_mean
    # return A*coeffs, coeffs
    # XXX: This function tries to bypass the Singular Matrix
    # problem inverting the matrix using pinv which computes
    # the (Moore-Penrose) pseudo-inverse of a matrix
    if dy is not None:
        AwAT = np.dot(A, np.diag(dy).dot(A.T))
        try:
            AwAT_inv = np.linalg.inv(AwAT)
        except np.linalg.LinAlgError:
            AwAT_inv = np.linalg.pinv(AwAT)
        AwB = A.dot(np.diag(dy).dot(B))
        p = np.dot(AwAT_inv, AwB)
    else:
        AAT = A.dot(A.T)
        try:
            AAT_inv = np.linalg.inv(AAT)
        except np.linalg.LinAlgError:
            AAT_inv = np.linalg.pinv(AAT)
        p = AAT_inv.dot(A).dot(B)
    p *= y_obs_mean
    return A.T.dot(p), p

# Hector median_filter copy of SFS median_filter in perl
def median_filter(box, x, verbose=False):
    """ Apply a median filter to `x` with box size `box`.

    Parameters
    ----------
    box: int
        Box size of the median filter.
    x: array like
        Array to be filtered.

    Returns
    -------
    array like
        `x` input array filtered
    """
    x = np.asarray(x)
    box_size = 2*box
    box_size = round_up_to_odd(box_size)
    if box_size >= x.size:
        print_verbose(f'[median_filter]: box_size ({box_size}) greater than x.size ({x.size}).', verbose=verbose)
        return x
    val = copy(x)
    for i in range(box, val.size - box):
        val[i] = np.median([x[i - box + j] for j in range(0, 2*box)])
    for i in range(1, box):
        val[i] = np.median([x[j] for j in range(0, 2*i)])
    for i in range(val.size - box, val.size - 1):
        val[i] = np.median([x[i - (val.size - i) + j] for j in range(0, 2*(val.size - i))])
    return val

def round_up_to_odd(n):
    """ Rounds up `n` to the next odd integer.

    Parameters
    ----------
    n : float
        A number to be rounded to the next odd integer.

    Returns
    -------
    int
        `n` rounded to the next odd integer.
    """
    n = int(np.ceil(n))
    return n + 1 if n % 2 == 0 else n

def smooth_ratio(flux_ratio, sigma, kernel_size_factor=None):
    """ Create a smooth factor using the ratio r = `flux_a`/`flux_b`
    through a median filter.

    Parameters
    ----------
    flux_ratio : array like
        The flux_ratio which will be passed through a median_filter
    sigma : float
        Sigma in angstroms.
    kernel_size_factor : float
        Will define, together with `sigma`, the kernel_size.

        kernel_size = next odd integer from int(kernel_size_factor * sigma)

    Returns
    -------
    array like
        The smooth ratio.
    """
    if sigma < 1:
        sigma = 1

    if kernel_size_factor is None:
        kernel_size_factor = 7*__sigma_to_FWHM__

    # setbadtoval(1)
    flux_ratio[~(np.isfinite(flux_ratio))] = 1

    kernel_size = round_up_to_odd(np.int(kernel_size_factor*sigma))
    # return median_filter(kernel_size, flux_ratio)
    sm_ratio = median_filter(kernel_size, flux_ratio)
    # if (sm_ratio == flux_ratio).all():
    #     sm_ratio = np.ones_like(flux_ratio)
    sm_ratio = np.clip(sm_ratio, 0, None)
    # sm_ratio = np.clip(sm_ratio, 0, 1)
    sm_ratio[~(np.isfinite(sm_ratio))] = 1
    return sm_ratio
    # return medfilt(flux_ratio, kernel_size)

def convolve_sigma(flux, sigma, side_box=None):
    """
    Convolves `flux` using a Gaussian-kernel with standard deviation `sigma`.
    The kernel have dimension 2*`side_box` + 1.

    Parameters
    ----------
    flux : array like
        Spectrum to be convolved.
    sigma : float
        Sigma of the Gaussian-kernel.
    N_side: float
        Will define the range size of the Gaussian-kernel.

    Returns
    -------
    array like
        Convolved `flux` by the weights defined by the Gaussian-kernel.
    """
    kernel_function = lambda x: np.exp(-0.5*(((x - side_box)/sigma)**2))
    N = 2*side_box + 1
    kernel = np.array(list(map(kernel_function, np.arange(N))))
    norm = kernel.sum()
    kernel = kernel/norm
    return convolve1d(flux, kernel, mode='nearest')

def shift_convolve(wave_obs, wave_in, flux_in, redshift, sigma, sigma_inst=None):
    """
    Shift and convolve spectrum.

    Shift the spectrum `flux_in` at `wave_obs` wavelenghts to `wave_in`
    corrected in redshift. Also convolves the spectrum to `sigma` +
    `sigma_inst`. If `sigma_inst` is None, the shift + convolution of
    `flux_in` is simplier and faster.


    Parameters
    ----------
    wave_obs : array like
        Observed wavelenghts.

    wave_in : array like
        Input wavelengts of SSP models at observed frame.

    flux_in : array like
        Flux in `wave_obs` wavelenghts.

    redshift : float
        Input redshift.

    sigma : float
        Velocity dispersion of data in km/s.

    sigma_inst : float, optional
        Instrumental velocity dispersion in Angstrom.
        Defaults to None.

    Returns
    -------
    array like
        The `flux_in` convolved by `sigma` + `sigma_int` and
        shifted (interpoled) to `wave_in` wavelenghts.

    See also
    --------
    `pyFIT3D.common.stats.convolve_sigma`, `scipy.interpolate.interp1d`
    """



    def conv_interp(w_out, w_in, f_in, sigma=None, box=0, conv=True):
        f = interp1d(
            w_in,
            convolve_sigma(f_in, sigma, 3 if box < 3 else box) if conv else f_in,
            assume_sorted=True, kind='linear', bounds_error=False, fill_value='extrapolate'
        )
        return f(w_out)

    # bring everything to the observed frame
    wave_in_of = wave_in*(1 + redshift)
    dpix_ini = wave_in_of[1] - wave_in_of[0]
    # in the case that the instrumental sigma is not avaible,
    # the shift+convolution process is simplier.
    if sigma_inst is None:
        rsigma = sigma/dpix_ini
        return conv_interp(wave_obs, wave_in_of, flux_in, rsigma, int(3*rsigma))

    dpix = dpix_ini/(1 + np.random.rand())
    # dpix = dpix_ini
    # interpolate flux_in to the new wavelenghts.
    w_min = wave_in_of.min()
    w_max = wave_in_of.max()
    N_sub = (w_max - w_min)/dpix
    wave = w_min + dpix*np.arange(N_sub)
    # resample wave in ln(wavelenght)
    ln_wave = np.log(wave)
    # interpolation factor
    f_fine = __shift_convolve_lnwl_frac__
    ln_dpix = (ln_wave[1] - ln_wave[0])/f_fine
    ln_w_min = ln_wave.min()
    ln_w_max = ln_wave.max()
    N_ln_wave = (ln_w_max - ln_w_min)/ln_dpix
    new_ln_wave = ln_w_min + ln_dpix*np.arange(N_ln_wave)
    # instrumental resolution in pixels
    rsigma_inst = sigma_inst/dpix
    new_wave_inst = np.exp(new_ln_wave)
    #########################################
    # Shift + convolve sigma_inst and sigma #
    #########################################
    # interpolated and sigma_inst convolved flux_in convolved to sigma    
    return conv_interp(
            w_out=wave_obs, w_in=new_wave_inst,

            # interpolated flux_in convolved to sigma_inst.
            f_in=conv_interp(
                w_out=new_wave_inst, w_in=wave,

                # interpolate flux_in to the new wavelenghts.
                f_in=conv_interp(w_out=wave, w_in=wave_in_of, f_in=flux_in, conv=False),

                sigma=rsigma_inst, box=int(6*rsigma_inst), conv=(rsigma_inst > 0.5)
            ),

            sigma=((1 if sigma == 0 else sigma)/__c__)/ln_dpix,
            box=int(5*(500/__c__)/ln_dpix), conv=True,
        )

def calc_chi_sq(f_obs, f_mod, ef_obs, ddof=0):
    """
    Calculates the Chi Square of a fitted model.

    Parameters
    ----------
    f_obs : array like
        Observed spectrum
    f_mod : array like
        Modeled spectrum
    ef_obs : array like
        Error of observed spectrum.

    Returns
    -------
    float
        The Chi Square of the fit.
    int
        The number of observations.
    """
    mask = ef_obs != 0
    N_obs = mask.sum()
    chi = np.divide(
        f_obs - f_mod, ef_obs,
        where=mask,
        out=np.zeros_like(f_obs)
    )
    chi_sq = np.sum(chi**2)
    chi_sq_red = ((chi_sq / (N_obs - ddof)) if N_obs - ddof > 0 else chi_sq)
    return chi_sq_red, N_obs

def hyperbolic_fit_par(x, y, verbose=False):
    """
    Calculates the parameter `x` considering `y` with an hyperbolic model function fit.
    This function assumes that the three last values of x and y are forming an hyperbole.

    Parameters
    ----------
    x : array like
        The x values.
    y : array like
        The y(x) values.
    verbose : bool, optional
        Print output errors. Default is False.
    """
    par = None
    error_par = None
    if len(y) > 2:
        a, b, c = x[-3:]
        fa, fb, fc = y[-3:]
        delta_ba = b - a
        delta_fba = fb - fa
        delta_bc = c - b
        delta_fcb = fc - fb
        den = fa + fc - 2*fb
        if (den != 0):
            par = c - delta_ba*(delta_fcb/den + 0.5)
        else:
            par = 0
        slope = np.abs(0.5*delta_fcb/delta_bc) + np.abs(0.5*delta_fba/delta_ba)
        if slope != 0:
            error_par = 0.01 * par/slope
    else:
        print_verbose(f'[hyperbolic_fit: n={len(y)}] Impossible to calculate hyperbolic fit!', verbose=verbose)
    return par, error_par

def median_box(box_init, x):
    """
    Creates a box with spans the same range of `x` evenly spaced with a median box.
    The box size is 2*`box_init`. If `box_init` is even, it will be rewritten to the
    next odd integer.

    Parameters
    ----------
    box_init : int
        The pixel where the box begins.
    x : array like
        The array which will be 'boxed'.

    Returns
    -------
    array like
        Evenly spaced box with approximately the same range of `x`.
    """
    box_init = round_up_to_odd(box_init)
    end_box = len(x) - box_init
    box_size = 2*box_init
    xbox = np.array([
        np.median([x[i - box_init + j] for j in range(box_size)])
        for i in range(box_init, end_box, box_size)
    ])
    return xbox

def std_m(data):
    mean = np.median(data)
    sum_res = 0
    if len(data) > 0:
        for dat in data:
            sum_res += (dat - mean)**2
        sum_res /= (len(data)-1)
        result = np.sqrt(sum_res)
    else:
        result = 0
    return result
