
from copy import deepcopy as copy
import numpy as np
from tqdm import tqdm

from scipy.signal import convolve


def weighted_pdf(pdf_params_hdus, ihdu, coeffs):
    """Return the weighted PDF given the HDU list of basis PDFs and the fitting coefficients"""
    h = pdf_params_hdus[ihdu].header
    PDF = np.asarray(pdf_params_hdus[ihdu].data).T

    x_scale = np.array([h["CRVAL1"] + i*h["CDELT1"] for i in range(h["NAXIS1"])])
    y_scale = np.array([h["CRVAL2"] + i*h["CDELT2"] for i in range(h["NAXIS2"])])
    wPDF = (coeffs[None,None,:] * PDF).sum(axis=-1)

    return wPDF.T, x_scale, y_scale

def normalize_to_pdf(pdf, x):
    """return the PDF given the function of a distribution and its support"""
    if callable(pdf):
        pdf_ = pdf(x)
    else:
        pdf_ = copy(pdf)
    
    if np.all(pdf_==0): return np.nan

    return pdf_ / np.trapz(pdf_, x)

def get_nth_moment(x, pdf, nth, mu=None):
    """Return the nth moment of the given PDF

    Parameters
    ----------
    x: array-like
        The support of the given PDF
    pdf: a callable function
        The PDF from which to calculate the moment
    nth: integer
        The order of the moment to calculate
    mu: float
        The value of the support around which the moment will be calculated.
        If not given defaults to the first moment of the distribution

    Returns
    -------
    moment: float
        The computed moment for the PDF
    """
    if callable(pdf):
        pdf_ = pdf(x)
    else:
        pdf_ = copy(pdf)

    if np.all(pdf_==0): return np.nan

    if mu is None:
        mu_ = np.trapz(x * pdf_, x) / np.trapz(pdf_, x)
        if nth == 1: return mu_
    else:
        if mu < x.min() or mu > x.max():
            raise ValueError("the passed value of 'mu' is out of the given support range")
        else:
            mu_ = mu

    moment = np.trapz((x - mu_)**nth * pdf_, x) / np.trapz(pdf_, x)

    return moment

def get_nth_percentile(x, pdf, percent=50):
    """Return the n-th percentile of the given PDF"""

    if hasattr(percent, "__len__") and not isinstance(percent, str):
        percent_ = sorted(percent)
    else:
        percent_ = [percent]

    if not np.all((0 <= np.asarray(percent_)) & (np.asarray(percent_) <= 100)):
        raise ValueError("[get_nth_percentile] you must provide percent values between 0 and 100")

    if callable(pdf):
        pdf_ = pdf(x)
    else:
        pdf_ = copy(pdf)

    if np.all(np.isnan(pdf_)) or np.all(pdf_==0):
        return np.full_like(percent_, np.nan, dtype=np.double)

    norm = np.trapz(pdf_, x)
    if not np.isclose(norm, 1.0):
        raise ValueError(f"[get_nth_percentile] the PDF you provided does not normalize to one ({norm})")

    x_delt = np.diff(x)

    i, j, i_pct, prob = 0, 0, [], 0.0
    while len(i_pct) != len(percent_):
        prob += x_delt[i] * pdf_[i]

        if prob == percent_[j]/100.0:
            i_pct.append(i)
            j += 1
        elif prob > percent_[j]/100.0:
            i_pct.append(i-1)
            j += 1

        i += 1

    return np.asarray(x)[i_pct]

def gaussian_kernel(sigma, half_box=50):
    N = 2*half_box + 1
    kernel = np.exp(-0.5*(((np.arange(N) - half_box)/sigma)**2))
    return kernel / kernel.sum()

def downgrade_resolution(wavelength, spectrum, sigma, verbose=True):

    # assuming the sampling is uniform
    dwl = np.diff(wavelength)[0]

    if np.isscalar(sigma):
        sigma_ = sigma/dwl

        if 2.355*sigma_ < 2:
            return spectrum

        kernel = gaussian_kernel(sigma_)
        dwn_spectrum = convolve(spectrum, kernel, mode="same", method="fft")
    else:
        dwn_spectrum = []
        if verbose:
            iterator = tqdm(range(wavelength.size), desc="downgrading resolution", unit="pixel", ascii=True)
        else:
            iterator = range(wavelength.size)
        for j in iterator:
            sigma_j = sigma[j]/dwl
            if 2.355*sigma_j < 2:
                dwn_spectrum.append(spectrum[j])
                continue

            kernel = gaussian_kernel(sigma_j)
            conv_spectrum = convolve(spectrum, kernel, mode="same", method="fft")

            dwn_spectrum.append(conv_spectrum[j])
        dwn_spectrum = np.asarray(dwn_spectrum)

    return dwn_spectrum
